"""Slack thinking task cards: live 2-block context while running, folded plan on completion."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

META = {"thread_ts": "100.1", "slack_team_id": "T1"}


def make_adapter(monkeypatch, extra=None, *, update_side_effect=None):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {"thinking_task_cards": True}))
    adapter._app = MagicMock()
    client = SimpleNamespace(
        api_call=AsyncMock(return_value={"ts": "stream-1"}),
        chat_postMessage=AsyncMock(return_value={"ts": "200.1"}),
        chat_update=AsyncMock(side_effect=update_side_effect, return_value={"ok": True}),
    )
    monkeypatch.setattr(adapter, "_get_client", Mock(return_value=client))
    return adapter, client


def block_texts(blocks):
    return [block["elements"][0]["text"] for block in blocks]


@pytest.mark.parametrize("extra,thinking,native", [
    ({"thinking_task_cards": True}, True, True),
    ({"thinkingTaskCards": "yes"}, True, True),
    ({"streaming": {"progress": {"thinking_task_cards": "true"}}}, True, True),
    ({"native_task_cards": True}, False, True),
    ({"thinking_task_cards": False}, False, False),
    ({}, False, False),
])
def test_thinking_flag_implies_native_gate(extra, thinking, native):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=extra))
    assert adapter.thinking_task_cards_enabled() is thinking
    assert adapter.native_task_cards_enabled() is native


@pytest.mark.asyncio
async def test_two_phase_lifecycle_posts_updates_then_folds_to_plan(monkeypatch):
    adapter, client = make_adapter(monkeypatch)
    running = [
        {"id": "thought_1", "kind": "thought", "title": "Check the read_file helper", "status": "complete",
         "details": "Check the read_file helper. It may cache stale paths."},
        {"id": "call-1", "kind": "tool", "title": "terminal - ls", "status": "in_progress"},
    ]
    assert (await adapter.send_native_task_card_progress("C1", running, metadata=META)).success
    post = client.chat_postMessage.await_args.kwargs
    assert post["thread_ts"] == "100.1" and post["channel"] == "C1"
    assert [block["type"] for block in post["blocks"]] == ["context", "context"]
    window, footer = block_texts(post["blocks"])
    # snake_case survives: no italic wrapper around the thought line
    assert window == "• Check the read_file helper"
    assert footer.startswith("⏳ *2 steps · ") and footer.endswith("Running: `terminal - ls`")

    thinking = running[:1] + [dict(running[1], status="complete"),
                              {"id": "thought_2", "kind": "thought", "title": "Now compare", "status": "in_progress"}]
    await adapter.send_native_task_card_progress("C1", thinking, metadata=META)
    update = client.chat_update.await_args.kwargs
    assert update["ts"] == "200.1"
    assert block_texts(update["blocks"])[1].endswith("Thinking...")
    client.api_call.assert_not_awaited()  # no chat.startStream / appendStream

    await adapter.stop_native_task_card_progress("C1", metadata=META)
    fold = client.chat_update.await_args.kwargs
    (plan,) = fold["blocks"]
    assert plan["type"] == "plan"
    assert plan["title"].startswith("✓ Thought for ") and plan["title"].endswith("· 3 steps")
    assert fold["text"] == plan["title"]
    assert [task["title"] for task in plan["tasks"]] == [
        "💭 Check the read_file helper", "🔧 terminal - ls", "💭 Now compare"]
    assert all(task["status"] == "complete" for task in plan["tasks"])
    details = plan["tasks"][0]["details"]
    assert details["type"] == "rich_text"
    assert details["elements"][0]["elements"][0]["text"].startswith("Check the read_file helper. It may")
    assert "details" not in plan["tasks"][1]
    assert adapter._native_task_card_streams == {}


@pytest.mark.asyncio
async def test_window_prefers_last_six_thoughts_and_falls_back_to_tools(monkeypatch):
    adapter, client = make_adapter(monkeypatch)
    tools = [{"id": f"call-{i}", "title": f"tool_{i}", "status": "complete"} for i in range(8)]
    await adapter.send_native_task_card_progress("C1", tools, metadata=META)
    window = block_texts(client.chat_postMessage.await_args.kwargs["blocks"])[0]
    assert window.splitlines() == [f"• `tool_{i}`" for i in range(2, 8)]

    thoughts = [{"id": f"thought_{i}", "kind": "thought", "title": f"idea {i}", "status": "complete"} for i in range(9)]
    await adapter.send_native_task_card_progress("C1", tools + thoughts, metadata=META)
    window = block_texts(client.chat_update.await_args.kwargs["blocks"])[0]
    assert window.splitlines() == [f"• idea {i}" for i in range(3, 9)]


@pytest.mark.asyncio
async def test_fold_caps_plan_at_fifty_tasks_and_keeps_errors(monkeypatch):
    adapter, client = make_adapter(monkeypatch)
    tasks = [{"id": f"call-{i}", "title": f"step {i}", "status": "complete"} for i in range(60)]
    tasks[-1]["status"] = "error"
    await adapter.send_native_task_card_progress("C1", tasks, metadata=META)
    await adapter.stop_native_task_card_progress("C1", metadata=META)
    (plan,) = client.chat_update.await_args.kwargs["blocks"]
    assert len(plan["tasks"]) == 50
    assert plan["tasks"][0] == {"task_id": "omitted", "title": "(11 earlier steps omitted)", "status": "complete"}
    assert plan["tasks"][1]["task_id"] == "call-11"
    assert plan["tasks"][-1]["status"] == "error"


@pytest.mark.asyncio
async def test_rejected_plan_folds_to_one_line_context(monkeypatch):
    calls = []

    async def update(**kwargs):
        calls.append(kwargs)
        if kwargs["blocks"][0]["type"] == "plan":
            raise RuntimeError("invalid_blocks")
        return {"ok": True}

    adapter, client = make_adapter(monkeypatch, update_side_effect=update)
    await adapter.send_native_task_card_progress("C1", [{"id": "a", "title": "x", "status": "complete"}], metadata=META)
    await adapter.stop_native_task_card_progress("C1", metadata=META)
    assert [c["blocks"][0]["type"] for c in calls] == ["plan", "context"]
    assert block_texts(calls[-1]["blocks"]) == [calls[-1]["text"]]


@pytest.mark.asyncio
async def test_background_stream_is_a_separate_card_with_its_own_fold(monkeypatch):
    adapter, client = make_adapter(monkeypatch)
    client.chat_postMessage.side_effect = [{"ts": "200.1"}, {"ts": "300.1"}]
    tasks = [{"id": "sub_1_tool_1", "title": "web_search - docs", "status": "in_progress"}]
    await adapter.send_native_task_card_progress("C1", tasks, metadata=META)
    await adapter.send_native_task_card_progress(
        "C1", tasks, metadata=META, stream_suffix="d1", background_label="Background subagent")
    assert len(adapter._native_task_card_streams) == 2
    bg_post = client.chat_postMessage.await_args.kwargs
    assert bg_post["thread_ts"] == "100.1"
    assert block_texts(bg_post["blocks"])[1].startswith("🔄 *Background subagent*   ·   1 step · ")

    await adapter.stop_native_task_card_progress("C1", metadata=META, stream_suffix="d1")
    fold = client.chat_update.await_args.kwargs
    assert fold["ts"] == "300.1"
    assert fold["text"].startswith("✓ Background subagent finished · ") and fold["text"].endswith("· 1 step")
    assert len(adapter._native_task_card_streams) == 1


@pytest.mark.asyncio
async def test_native_only_flag_keeps_stream_transport(monkeypatch):
    adapter, client = make_adapter(monkeypatch, {"native_task_cards": True})
    await adapter.send_native_task_card_progress("C1", [{"id": "a", "title": "x"}], metadata=META)
    await adapter.stop_native_task_card_progress("C1", metadata=META)
    assert [c.args[0] for c in client.api_call.await_args_list] == [
        "chat.startStream", "chat.appendStream", "chat.stopStream"]
    client.chat_postMessage.assert_not_awaited()


def test_thought_accumulator_and_task_card_state_events():
    from gateway.run_turn_progress import GatewayTurnProgressMixin
    from gateway.run_turn_thinking import ThoughtAccumulator

    acc = ThoughtAccumulator()
    evt1 = acc.feed("**Inspecting config**\nChecking how `show_reasoning` is resolved.")
    assert evt1["thought_id"] == "thought_1"
    assert evt1["title"].startswith("Inspecting config:")
    evt1b = acc.feed(" Also checking `interim_assistant_messages`.")
    assert evt1b["thought_id"] == "thought_1"
    acc.boundary()
    evt2 = acc.feed("Second thought segment after tool boundary.")
    assert evt2["thought_id"] == "thought_2"
    interim = acc.interim("Brief interim commentary before tool run.")
    assert interim["thought_id"] == "interim_1"

    st = GatewayTurnProgressMixin._TaskCardState(adapter=None, thinking=True)
    assert st.apply_event(evt1b) is True
    assert st.tasks["thought_1"]["status"] == "in_progress"
    assert st.apply_event({"type": "tool.started", "tool_call_id": "c1", "tool_name": "terminal", "preview": "ls"})
    assert st.tasks["thought_1"]["status"] == "complete"
    assert st.tasks["c1"]["status"] == "in_progress"
    assert st.apply_event({"type": "tool.completed", "tool_call_id": "c1", "tool_name": "terminal", "is_error": False})
    assert st.tasks["c1"]["status"] == "complete"

    assert st.apply_event({"type": "subagent.start", "subagent_id": "sa1", "delegation_id": "d1", "goal": "verify build"})
    assert "sa1" in st.subagents_running
    assert st.apply_event({"type": "subagent.thinking", "subagent_id": "sa1", "preview": "Checking build logs."})
    assert st.tasks["thought_sub_sa1_1"]["kind"] == "thought"
    assert st.apply_event({"type": "subagent.tool", "subagent_id": "sa1", "tool_name": "terminal", "preview": "make test"})
    assert st.tasks["thought_sub_sa1_1"]["status"] == "complete"
    assert st.tasks["sub_sa1_tool_1"]["status"] == "in_progress"
    assert st.apply_event({"type": "subagent.complete", "subagent_id": "sa1", "status": "completed"})
    assert "sa1" not in st.subagents_running
    assert st.tasks["sub_sa1_tool_1"]["status"] == "complete"

