"""Integration tests: SlackAdapter wiring of Block Kit into send paths.

Verifies the opt-in behaviour contract:
  * rich_blocks off (default)  => no ``blocks`` kwarg, plain ``text`` only
  * rich_blocks on             => ``blocks`` present AND ``text`` fallback set
  * edit_message: blocks only on finalize (streaming edits stay plain)
  * multi-chunk (>39k) messages fall back to plain text
"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {})
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_update = AsyncMock(return_value={"ts": "111.222"})
    a._get_client = MagicMock(return_value=client)
    a.stop_typing = AsyncMock()
    a._running = True
    return a, client


RICH_MD = "# Title\n\n- a\n  - nested\n\n---\n\nbody text"
RICH_TABLE_MD = (
    "| Item | Status | Note |\n"
    "|---|---:|---|\n"
    "| Hermes | ok | table |"
)


class SlackRejectedBlocks(Exception):
    def __init__(self, error="invalid_blocks"):
        super().__init__(f"Slack API rejected blocks: {error}")
        self.response = {"error": error}


def _slack_connection_key():
    from aiohttp.client_reqrep import ConnectionKey

    return ConnectionKey(
        host="slack.com",
        port=443,
        is_ssl=True,
        ssl=True,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )


class TestSendMessageBlocks:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_blocks(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", RICH_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]  # plain text still sent

    @pytest.mark.asyncio
    async def test_unfurl_config_suppresses_previews_without_changing_link_text(self):
        adapter, client = _make_adapter(
            {"unfurl_links": False, "unfurl_media": False}
        )
        content = "[Hermes](https://example.com/hermes)"

        await adapter.send("C1", content)

        kwargs = client.chat_postMessage.await_args.kwargs
        assert kwargs["text"] == "<https://example.com/hermes|Hermes>"
        assert kwargs["unfurl_links"] is False
        assert kwargs["unfurl_media"] is False


    @pytest.mark.asyncio
    async def test_enabled_but_unrenderable_falls_back_to_text(self):
        # 60 dividers -> renderer returns None -> no blocks kwarg, text stands
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", "\n\n".join(["---"] * 60))
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]


    @pytest.mark.asyncio
    async def test_feedback_buttons_opt_in_appended_to_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True, "feedback_buttons": True})

        await adapter.send("C1", "final answer")

        blocks = client.chat_postMessage.await_args.kwargs["blocks"]
        feedback = blocks[-1]
        assert feedback["type"] == "context_actions"
        assert feedback["elements"][0]["type"] == "feedback_buttons"
        assert feedback["elements"][0]["action_id"] == "hermes_feedback"


class TestEditMessageBlocks:
    @pytest.mark.asyncio
    async def test_intermediate_edit_no_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=False)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_finalize_edit_gets_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert kwargs.get("blocks")
        assert kwargs["text"]


    @pytest.mark.asyncio
    async def test_block_rejection_retries_edit_without_blocks_using_workspace_client(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_update = AsyncMock(
            side_effect=[SlackRejectedBlocks("invalid_blocks"), {"ts": "111.222"}]
        )

        result = await adapter.edit_message(
            "C1",
            "111.222",
            RICH_TABLE_MD,
            finalize=True,
            metadata={"team_id": "T_SECONDARY"},
        )

        assert result.success is True
        assert adapter._get_client.call_args_list == [
            call("C1", team_id="T_SECONDARY"),
            call("C1", team_id="T_SECONDARY"),
        ]
        assert client.chat_update.await_count == 2
        first = client.chat_update.await_args_list[0].kwargs
        second = client.chat_update.await_args_list[1].kwargs
        assert first.get("blocks")
        assert second["blocks"] == []
        assert second["text"]

    @pytest.mark.asyncio
    async def test_msg_blocks_too_long_retries_send_without_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_postMessage = AsyncMock(
            side_effect=[SlackRejectedBlocks("msg_blocks_too_long"), {"ts": "111.222"}]
        )

        result = await adapter.send("C1", RICH_MD)

        assert result.success is True
        assert client.chat_postMessage.await_count == 2
        first = client.chat_postMessage.await_args_list[0].kwargs
        second = client.chat_postMessage.await_args_list[1].kwargs
        assert first.get("blocks")
        assert not second.get("blocks")
        assert second["text"] == first["text"]

    def test_msg_blocks_too_long_is_block_payload_rejection(self):
        assert SlackAdapter._is_block_payload_rejection(SlackRejectedBlocks("msg_blocks_too_long"))
        assert not SlackAdapter._is_block_payload_rejection(SlackRejectedBlocks("channel_not_found"))

    @pytest.mark.asyncio
    async def test_timeout_error_on_edit_is_retryable_transient(self):
        adapter, client = _make_adapter()
        client.chat_update = AsyncMock(side_effect=TimeoutError("timed out"))

        result = await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)

        assert result.success is False
        assert result.retryable is True
        assert result.error_kind == "transient"


# ---------------------------------------------------------------------------
# markdown_blocks mode — Slack's native ``markdown`` Block Kit block (#8552)
# ---------------------------------------------------------------------------


class TestMarkdownBlockMode:
    """Opt-in ``markdown_blocks`` renders raw standard markdown via Slack's
    native ``markdown`` block, keeping the mrkdwn ``text`` fallback."""


    @pytest.mark.asyncio
    async def test_enabled_sends_markdown_block_with_raw_content(self):
        adapter, client = _make_adapter({"markdown_blocks": True})
        await adapter.send("C1", RICH_TABLE_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        blocks = kwargs["blocks"]
        assert blocks[0]["type"] == "markdown"
        # RAW standard markdown, not mrkdwn-converted — Slack translates it
        assert blocks[0]["text"] == RICH_TABLE_MD
        # mrkdwn fallback text is still present for notifications/search
        assert kwargs["text"]


    @pytest.mark.asyncio
    async def test_edit_finalize_uses_markdown_block(self):
        adapter, client = _make_adapter({"markdown_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_TABLE_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert kwargs["blocks"][0]["type"] == "markdown"
        assert kwargs["blocks"][0]["text"] == RICH_TABLE_MD




# ---------------------------------------------------------------------------
# rich_blocks over one message's limits — consecutive native Block Kit messages
# ---------------------------------------------------------------------------


def _long_ordered_list_md(n=40):
    return "\n".join(
        f"{i}. **Item {i}** see [ticket {i}](https://example.com/tracker/issues/{i:05d}?view=full)"
        f" and the long explanatory description of what changed and why it matters {'z' * 200}"
        for i in range(1, n + 1))


class TestMultiMessageBlocks:
    @pytest.mark.asyncio
    async def test_over_budget_reply_posts_each_batch_natively(self):
        from plugins.platforms.slack.block_kit import MAX_RICH_TEXT_MESSAGE_CHARS, estimate_text_chars

        adapter, client = _make_adapter(
            {"rich_blocks": True, "feedback_buttons": True, "reply_broadcast": True})
        result = await adapter.send("C1", _long_ordered_list_md(), metadata={"thread_id": "9.9"})

        assert result.success is True
        calls = [c.kwargs for c in client.chat_postMessage.await_args_list]
        assert len(calls) > 1
        for kwargs in calls:
            content = [b for b in kwargs["blocks"] if b["type"] != "context_actions"]
            assert all(b["type"] == "rich_text" for b in content)
            assert estimate_text_chars(content) <= MAX_RICH_TEXT_MESSAGE_CHARS
            assert kwargs["thread_ts"] == "9.9"
        # Each message's text fallback carries only its own batch.
        assert "Item 1*" in calls[0]["text"] and "Item 40*" not in calls[0]["text"]
        assert "Item 40*" in calls[-1]["text"]
        assert [c.get("reply_broadcast", False) for c in calls] == [True] + [False] * (len(calls) - 1)
        assert [c["blocks"][-1]["type"] == "context_actions" for c in calls] == (
            [False] * (len(calls) - 1) + [True])

    @pytest.mark.asyncio
    async def test_two_tables_post_as_two_native_messages(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", f"{RICH_TABLE_MD}\n\nbetween\n\n{RICH_TABLE_MD}")
        calls = [c.kwargs for c in client.chat_postMessage.await_args_list]
        assert [[b["type"] for b in c["blocks"]] for c in calls] == [
            ["table", "section"], ["table"]]
        assert all("Hermes | ok | table" in c["text"] for c in calls)

    @pytest.mark.asyncio
    async def test_rejected_batch_retries_with_its_own_text(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_postMessage = AsyncMock(side_effect=[
            {"ts": "1"}, SlackRejectedBlocks("invalid_blocks"), {"ts": "2"}])
        await adapter.send("C1", f"{RICH_TABLE_MD}\n\n{RICH_TABLE_MD.replace('Hermes', 'Second')}")
        calls = [c.kwargs for c in client.chat_postMessage.await_args_list]
        assert len(calls) == 3
        assert "blocks" not in calls[2]
        assert "Second" in calls[2]["text"] and "Hermes" not in calls[2]["text"]

    @pytest.mark.asyncio
    async def test_finalize_edit_over_one_message_keeps_plain_text(self):
        # chat.update cannot split a message, so a multi-message layout is declined.
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", _long_ordered_list_md(), finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]


class TestFormatMessageBoldLinkLabel:
    @pytest.mark.parametrize("label, clean", [
        ("**Open Opps** sheet", "Open Opps sheet"), ("__All__", "All"), ("**a** and **b**", "a and b")])
    def test_emphasis_in_label_bolds_whole_link(self, label, clean):
        adapter, _ = _make_adapter()
        assert adapter.format_message(f"see [{label}](https://example.com/s)") == (
            f"see *<https://example.com/s|{clean}>​*")

    def test_plain_label_unchanged(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("[docs](https://example.com)") == "<https://example.com|docs>"


class TestLongReplyKeepsBlocks:
    @pytest.mark.asyncio
    async def test_over_39k_multi_batch_reply_posts_all_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        para = "word " * 199 + "end"
        content = "\n\n".join(f"{para} https://example.com/{i}" for i in range(55))
        content += "\n\n" + RICH_TABLE_MD
        assert len(content) > 39000
        assert len(adapter.truncate_message(content, adapter.MAX_MESSAGE_LENGTH)) == 2
        await adapter.send("C1", content)
        calls = [c.kwargs for c in client.chat_postMessage.await_args_list]
        assert len(calls) > 1
        assert all(c.get("blocks") for c in calls)
        assert "table" in [b["type"] for b in calls[-1]["blocks"]]


class TestFormatMessageBacktickLinks:
    def test_backtick_slack_link(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("`<https://example.com|FUSE-2651>`") == (
            "<https://example.com|FUSE-2651>")

    def test_backtick_markdown_link(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("`[doitbse/resolve](https://github.com/doitbse/resolve)`") == (
            "<https://github.com/doitbse/resolve|doitbse/resolve>")

    def test_plain_inline_code_untouched(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("`foo`") == "`foo`"


class TestInboundHeaders:
    def test_header_block_and_rich_text_header_render_as_markdown_headings(self):
        from plugins.platforms.slack.adapter import _extract_text_from_slack_blocks

        blocks = [
            {"type": "header", "level": 2, "text": {"type": "plain_text", "text": "Plan"}},
            {"type": "header", "text": {"type": "plain_text", "text": "Default"}},
            {"type": "rich_text", "elements": [
                {"type": "rich_text_header", "level": 3,
                 "elements": [{"type": "text", "text": "Step one"}]},
                {"type": "rich_text_section", "elements": [{"type": "text", "text": "body"}]},
            ]},
            {"type": "header", "level": 9, "text": {"type": "plain_text", "text": "Deep"}},
        ]
        assert _extract_text_from_slack_blocks(blocks).splitlines() == [
            "## Plan", "# Default", "### Step one", "body", "###### Deep"]

    def test_serialized_header_keeps_level(self):
        import json

        from plugins.platforms.slack.adapter import _serialize_slack_blocks_for_agent

        dumped = _serialize_slack_blocks_for_agent(
            [{"type": "header", "level": 2, "text": {"type": "plain_text", "text": "Plan"}}])
        assert '"level": 2' in dumped
        assert json.loads(dumped[dumped.index("\n[") + 1:dumped.rindex("]") + 1])[0]["level"] == 2


class TestFormatMessageHtmlLineBreaks:
    @pytest.mark.parametrize("tag", ["<br>", "<br/>", "<br />", "<BR>"])
    def test_br_becomes_newline_not_escaped(self, tag):
        adapter, _ = _make_adapter()
        out = adapter.format_message(f"line 1{tag}line 2")
        assert out == "line 1\nline 2"
        assert "&lt;br" not in out

    def test_other_angle_brackets_still_escaped(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("x < y<br>z") == "x &lt; y\nz"

    def test_br_after_bare_url(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("see https://example.com/path<br>2) next") == (
            "see https://example.com/path\n2) next")

    def test_paragraph_section_block_gets_newline(self):
        from plugins.platforms.slack.block_kit import render_blocks

        adapter, _ = _make_adapter()
        blocks = render_blocks("see https://example.com/path<br>2) next", mrkdwn_fn=adapter.format_message)
        assert blocks == [{"type": "section", "text": {
            "type": "mrkdwn", "text": "see https://example.com/path\n2) next"}}]

    def test_emphasis_spanning_br_wraps_each_line(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("**bold line 1<br>bold line 2**") == (
            "*bold line 1*\n*bold line 2*")
        assert adapter.format_message("~~a<br>b~~") == "~a~\n~b~"

    def test_inline_and_fenced_code_br_stays_literal(self):
        adapter, _ = _make_adapter()
        assert adapter.format_message("a `<br>` b") == "a `<br>` b"
        assert adapter.format_message("```\nx<br>y\n```") == "```\nx<br>y\n```"

    def test_wrap_markdown_tables_uses_slash(self):
        from plugins.platforms.slack.adapter import _wrap_markdown_tables

        assert _wrap_markdown_tables("| a | b |\n|---|---|\n| x<br/>y | `<br>` |") == (
            "```\n"
            "| a     | b      |\n"
            "| ----- | ------ |\n"
            "| x / y | `<br>` |\n"
            "```"
        )
