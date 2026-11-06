"""Shared/forwarded and cross-channel Slack message unfurls reach the agent.

Slack sets ``is_msg_unfurl`` on every Slack-message unfurl, not only on echoes of the current
message. A forward (``is_share``) carries the forwarded message solely in ``attachments``: the
event ``text`` is just the sharer's note, so skipping the attachment left the agent with "WDYT?".
"""

from plugins.platforms.slack.adapter_unfurls import (
    SLACK_LINK_UNFURL_MAX_CHARS,
    SLACK_SHARED_MESSAGE_MAX_CHARS,
    append_link_unfurls,
)

_PERMALINK = "https://example.slack.com/archives/C0000000001/p1700000000000100"


def _forward(text: str, **overrides) -> dict:
    att = {
        "author_id": "U0000000001",
        "author_subname": "alice",
        "author_name": "Alice Example",
        "is_msg_unfurl": True,
        "is_share": True,
        "from_url": _PERMALINK,
        "channel_id": "D0000000001",
        "ts": "1700000000.000100",
        "text": text,
        "fallback": f"[Nov 14th, 2023 10:13 PM] alice: {text}",
        "footer": "Slack Conversation",
    }
    att.update(overrides)
    return att


def test_forwarded_message_text_author_and_permalink_are_appended():
    out = append_link_unfurls("WDYT?", [_forward("Deploy is blocked on the schema migration.")])

    assert out.startswith("WDYT?")
    assert f"📎 Shared message from Alice Example ({_PERMALINK})" in out
    assert "Deploy is blocked on the schema migration." in out
    assert "_Slack Conversation_" in out


def test_forward_header_falls_back_to_subname_then_bare_permalink():
    out = append_link_unfurls("WDYT?", [_forward("body", author_name="")])
    assert f"📎 Shared message from alice ({_PERMALINK})" in out

    out = append_link_unfurls("WDYT?", [_forward("body", author_name="", author_subname="")])
    assert f"📎 Shared message ({_PERMALINK})" in out


def test_forward_with_title_keeps_the_link_header():
    out = append_link_unfurls("WDYT?", [_forward("body", title="Standup notes", title_link=_PERMALINK)])
    assert f"📎 [Standup notes]({_PERMALINK})" in out


def test_forward_is_kept_even_when_its_text_appears_in_the_note():
    # A share is never an echo, even when the sharer quoted part of it.
    out = append_link_unfurls("ship it", [_forward("ship it")])
    assert "📎 Shared message from Alice Example" in out


def test_multi_paragraph_forward_is_not_cut_at_the_link_preview_cap():
    paragraph = ("The rollout plan has three phases and each needs sign-off. " * 5).strip()
    body = "\n\n".join([paragraph] * 5)
    assert SLACK_LINK_UNFURL_MAX_CHARS < len(body) < SLACK_SHARED_MESSAGE_MAX_CHARS

    out = append_link_unfurls("WDYT?", [_forward(body)])

    assert body in out


def test_oversized_forward_is_truncated_at_the_shared_message_cap():
    body = "y" * (SLACK_SHARED_MESSAGE_MAX_CHARS + 1000)

    out = append_link_unfurls("WDYT?", [_forward(body)])

    kept = "y" * (SLACK_SHARED_MESSAGE_MAX_CHARS - 3) + "..."
    assert kept in out
    assert "y" * (SLACK_SHARED_MESSAGE_MAX_CHARS - 2) not in out


def test_ordinary_link_unfurl_is_still_capped_at_the_link_preview_cap():
    body = "z" * 1500
    att = {"title": "Docs", "title_link": "https://example.com/docs", "text": body}

    out = append_link_unfurls("see https://example.com/docs", [att])

    assert "📎 [Docs](https://example.com/docs)" in out
    assert "z" * (SLACK_LINK_UNFURL_MAX_CHARS - 3) + "..." in out
    assert "z" * (SLACK_LINK_UNFURL_MAX_CHARS - 2) not in out


def test_cross_channel_message_unfurl_not_in_text_is_appended():
    att = {"is_msg_unfurl": True, "from_url": _PERMALINK, "author_name": "Bob Example",
           "text": "The incident review is moved to Thursday."}

    out = append_link_unfurls(f"<{_PERMALINK}> thoughts?", [att])

    assert f"📎 Shared message from Bob Example ({_PERMALINK})" in out
    assert "The incident review is moved to Thursday." in out


def test_message_unfurl_echoing_the_current_text_is_skipped():
    text = "The incident review is moved to Thursday."
    att = {"is_msg_unfurl": True, "from_url": _PERMALINK, "text": text, "fallback": text}

    assert append_link_unfurls(text, [att]) == text
    assert append_link_unfurls(text, [{"is_msg_unfurl": True, "from_url": _PERMALINK}]) == text


def test_slack_adapter_method_and_hydration_preserve_forwarded_share():
    from plugins.platforms.slack.adapter import SlackAdapter, _extract_text_from_slack_attachments

    share = _forward("Forwarded incident timeline for review.")
    assert "Forwarded incident timeline for review." in SlackAdapter._append_link_unfurls("WDYT?", [share])
    assert "Forwarded incident timeline for review." in _extract_text_from_slack_attachments([share])
    assert _extract_text_from_slack_attachments([{"is_msg_unfurl": True, "text": "echoed"}]) == ""

