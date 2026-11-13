"""A clean-EOF partial-stream stub that repeats the previous stub fragment verbatim is a
deterministic provider failure: the continuation trail is dropped (never stitched into the
reply) and the fallback chain is tried, else the turn ends with an explicit notice."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.error_classifier import FailoverReason
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import _REPEATED_STUB_FINAL, recover_from_truncation
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID

GLITCH = "évrier"


def _agent(fallback_chain=()):
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.log_prefix = ""
    agent._hold_stream = False
    agent._fallback_index = 0
    agent._fallback_chain = list(fallback_chain)
    agent._try_activate_fallback = MagicMock(return_value=bool(fallback_chain))
    agent._strip_think_blocks = lambda text: text
    agent._has_content_after_think_block = lambda text: True
    agent._get_transport.return_value.normalize_response = lambda r: r.choices[0].message
    agent._build_assistant_message = lambda msg, fr: {
        "role": "assistant", "content": msg.content, "finish_reason": fr,
    }
    return agent


def _response(content, *, stub=True, clean_eof=True):
    resp = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID if stub else "chatcmpl-1",
        _dropped_tool_names=None,
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(role="assistant", content=content, tool_calls=None,
                                    reasoning_content=None),
            finish_reason=FINISH_REASON_LENGTH,
        )],
    )
    if stub:
        resp._clean_eof = clean_eof
    return resp


def _recover(agent, response, messages, parts, retries, retry=None):
    return recover_from_truncation(
        agent, response, FINISH_REASON_LENGTH, retry or TurnRetryState(),
        messages=messages, conversation_history=None, api_kwargs={}, api_call_count=1,
        effective_task_id=None, current_turn_user_idx=0, length_continue_retries=retries,
        truncated_response_parts=parts, truncated_tool_call_retries=0, retry_count=0,
        compression_attempts=0,
    )


def _first_attempt(agent, content, *, stub=True):
    """Run the first truncation so messages/parts hold one fragment + nudge."""
    messages = [{"role": "user", "content": "hi"}]
    verdict = _recover(agent, _response(content, stub=stub), messages, [], 0)
    assert verdict.action == "break"
    return verdict


def _has_trail(messages):
    return any(m.get("_length_continuation_fragment") or m.get("_length_continuation_nudge")
               for m in messages)


def test_first_stub_continues():
    agent = _agent()
    verdict = _first_attempt(agent, GLITCH)
    assert verdict.length_continue_retries == 1
    assert verdict.truncated_response_parts == [(GLITCH, True)]
    assert verdict.messages[1]["_length_continuation_fragment"] is True
    assert verdict.messages[2]["_length_continuation_nudge"] is True


def test_repeated_stub_with_fallback_switches_provider():
    agent = _agent(fallback_chain=[{"provider": "other"}])
    first = _first_attempt(agent, GLITCH)
    retry = TurnRetryState()
    retry.primary_recovery_attempted = True
    verdict = _recover(agent, _response(GLITCH), first.messages,
                       first.truncated_response_parts, first.length_continue_retries, retry)

    assert verdict.action == "break"
    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.incomplete_response)
    assert retry.restart_with_rebuilt_messages is True
    assert retry.primary_recovery_attempted is False
    assert verdict.messages == [{"role": "user", "content": "hi"}]
    assert verdict.truncated_response_parts == []
    assert verdict.length_continue_retries == 0
    assert agent._ephemeral_reasoning_off is False


def test_repeated_stub_without_fallback_ends_turn_with_notice():
    agent = _agent()
    first = _first_attempt(agent, GLITCH)
    verdict = _recover(agent, _response(GLITCH), first.messages,
                       first.truncated_response_parts, first.length_continue_retries)

    assert verdict.action == "return"
    agent._try_activate_fallback.assert_not_called()
    result = verdict.result
    assert result["final_response"] == _REPEATED_STUB_FINAL
    assert not _has_trail(result["messages"])
    assert all(GLITCH not in (m.get("content") or "") for m in result["messages"])


def test_different_stub_fragments_keep_continuing():
    agent = _agent(fallback_chain=[{"provider": "other"}])
    first = _first_attempt(agent, "The answer starts here")
    verdict = _recover(agent, _response("and continues here"), first.messages,
                       first.truncated_response_parts, first.length_continue_retries)

    assert verdict.action == "break"
    agent._try_activate_fallback.assert_not_called()
    assert verdict.length_continue_retries == 2
    assert [p[0] for p in verdict.truncated_response_parts] == [
        "The answer starts here", "and continues here",
    ]


def test_repeated_real_length_truncation_is_unchanged():
    agent = _agent(fallback_chain=[{"provider": "other"}])
    first = _first_attempt(agent, GLITCH, stub=False)
    verdict = _recover(agent, _response(GLITCH, stub=False), first.messages,
                       first.truncated_response_parts, first.length_continue_retries)

    assert verdict.action == "break"
    agent._try_activate_fallback.assert_not_called()
    assert verdict.length_continue_retries == 2
    assert verdict.truncated_response_parts == [(GLITCH, False), (GLITCH, False)]
