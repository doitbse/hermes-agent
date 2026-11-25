"""Unit tests for the Slack Block Kit renderer (pure function, no adapter)."""

import pytest

from plugins.platforms.slack.block_kit import (
    MAX_BLOCKS,
    MAX_RICH_TEXT_MESSAGE_CHARS,
    MAX_SECTION_TEXT,
    _header_block,
    _render_markdown,
    blocks_fallback_text,
    estimate_text_chars,
    plan_messages,
    render_block_batches,
    render_blocks,
    sanitize_blocks,
    split_block_batches,
)


def _types(blocks):
    return [b["type"] for b in blocks]


class TestRenderBlocksBasics:
    def test_empty_returns_none(self):
        assert render_blocks("") is None
        assert render_blocks("   \n  ") is None


    def test_header_becomes_header_block(self):
        blocks = render_blocks("# Title")
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["type"] == "plain_text"
        assert blocks[0]["text"]["text"] == "Title"


class TestNestedLists:
    def test_nested_bullets_produce_increasing_indent(self):
        md = "- a\n  - b\n    - c"
        blocks = render_blocks(md)
        rich = next(b for b in blocks if b["type"] == "rich_text")
        indents = [e["indent"] for e in rich["elements"] if e["type"] == "rich_text_list"]
        # true nesting: indent levels must strictly increase across the run
        assert indents == sorted(indents)
        assert max(indents) >= 2
        assert min(indents) == 0


class TestListFollowedByIndentedBlock:
    def test_indented_table_not_inlined_into_bullet(self):
        md = (
            "- **Encryption tiers:**\n"
            "  | Tier | File | KMS Key | Contents |\n"
            "  |---|---|---|---|\n"
            "  | 1 | a.yaml | key-a | secrets |\n"
            "- **Control-plane IAM hardening:** restrict roles"
        )
        blocks = render_blocks(md)
        assert _types(blocks) == ["rich_text", "table", "rich_text"]
        first = blocks[0]["elements"][0]
        assert first["type"] == "rich_text_list"
        assert len(first["elements"]) == 1
        assert "|" not in str(first)
        assert "Control-plane IAM hardening" in str(blocks[2])

    def test_indented_fence_emitted_as_preformatted(self):
        md = (
            "- Run this:\n"
            "  ```bash\n"
            "  echo hi\n"
            "    nested\n"
            "  ```\n"
            "- Next step"
        )
        blocks = render_blocks(md)
        assert _types(blocks) == ["rich_text", "rich_text", "rich_text"]
        pre = blocks[1]["elements"][0]
        assert pre["type"] == "rich_text_preformatted"
        assert pre["elements"][0]["text"] == "echo hi\n  nested"
        assert "```" not in str(blocks[0])
        assert "echo" not in str(blocks[0])
        assert "Next step" in str(blocks[2])


class TestInlineFormatting:

    def test_slack_mrkdwn_link_in_bullet_becomes_link_element(self):
        """rich_text lists must parse Slack <url|label>, not emit it as text.

        rich_blocks turns markdown bullets into rich_text_list. rich_text does
        not interpret mrkdwn, so <url|text> has to become type=link.
        """
        blocks = render_blocks(
            "- <https://example.com/x|GitLab #1> — allow `in_progress`"
        )
        assert blocks is not None
        rich = next(b for b in blocks if b["type"] == "rich_text")
        els = rich["elements"][0]["elements"][0]["elements"]
        links = [e for e in els if e.get("type") == "link"]
        assert len(links) == 1
        assert links[0]["url"] == "https://example.com/x"
        assert links[0]["text"] == "GitLab #1"
        assert not any("<https://" in (e.get("text") or "") for e in els)
        assert any(e.get("style", {}).get("code") for e in els)

    def test_slack_mentions_in_bullet_are_not_links(self):
        """Mentions must become mention elements, never link elements.

        rich_text does not interpret mrkdwn, so a token left as a text element
        renders literally in Slack (see #133315).
        """
        blocks = render_blocks("- ping <@U123> in <#C456>")
        assert blocks is not None
        els = blocks[0]["elements"][0]["elements"][0]["elements"]
        assert all(e.get("type") != "link" for e in els)
        assert {"type": "user", "user_id": "U123"} in els
        assert {"type": "channel", "channel_id": "C456"} in els
        assert not any("<@" in (e.get("text") or "") or "<#" in (e.get("text") or "") for e in els)


    def test_blank_line_separated_ordered_items_stay_in_one_list(self):
        """Regression: blank lines between ordered items must not reset numbering.

        Slack numbers each rich_text_list independently.  If blank lines break
        the list run, N items produce N separate lists each starting at 1.
        See: https://github.com/NousResearch/hermes-agent/issues/57076
        """
        md = "1. alpha\n\n1. beta\n\n1. gamma"
        blocks = render_blocks(md)
        rich = next(b for b in blocks if b["type"] == "rich_text")
        lists = [e for e in rich["elements"] if e["type"] == "rich_text_list"]
        # Must be ONE list with 3 items, not 3 separate single-item lists
        assert len(lists) == 1
        items = lists[0]["elements"]
        assert len(items) == 3


class TestSlackMentions:
    """Mention tokens must become rich_text user/channel/broadcast elements.

    Regression coverage for #133315: mentions inside rich_text (lists, quotes,
    table cells) were emitted as text elements, so Slack showed the raw token
    literally instead of a mention.
    """

    @staticmethod
    def _item_elements(blocks):
        rich = next(b for b in blocks if b["type"] == "rich_text")
        return rich["elements"][0]["elements"][0]["elements"]

    def test_channel_mention_with_label_keeps_only_the_id(self):
        blocks = render_blocks("- see <#C0C6N2RF5HS|general>")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert {"type": "channel", "channel_id": "C0C6N2RF5HS"} in els
        assert not any("general" in (e.get("text") or "") for e in els)

    def test_broadcast_mentions_map_to_range(self):
        blocks = render_blocks("- <!here> <!channel> <!everyone>")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert [e["range"] for e in els if e.get("type") == "broadcast"] == [
            "here",
            "channel",
            "everyone",
        ]

    def test_usergroup_mention_becomes_usergroup_element(self):
        blocks = render_blocks("- notify <!subteam^S0123ABC|@marketing> and <!subteam^S987XYZ>")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert {"type": "usergroup", "usergroup_id": "S0123ABC"} in els
        assert {"type": "usergroup", "usergroup_id": "S987XYZ"} in els
        assert not any("marketing" in (e.get("text") or "") for e in els)

    def test_mention_label_with_underscores_is_not_split_by_italic(self):
        blocks = render_blocks("- see <#C1|dev_ops_alerts> and <!subteam^S1|@eng_team_x>")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert {"type": "channel", "channel_id": "C1"} in els
        assert {"type": "usergroup", "usergroup_id": "S1"} in els
        assert not any(e.get("style") for e in els)
        assert not any("<" in (e.get("text") or "") for e in els)

    def test_mention_in_quote_becomes_mention_element(self):
        blocks = render_blocks("> ping <@U123> now")
        assert blocks is not None
        quote = blocks[0]["elements"][0]
        assert quote["type"] == "rich_text_quote"
        assert {"type": "user", "user_id": "U123"} in quote["elements"]

    def test_mention_in_table_cell_becomes_mention_element(self):
        blocks = render_blocks("| who | note |\n| --- | --- |\n| <@U123> | ok |")
        assert blocks is not None
        assert blocks[0]["type"] == "table"
        cell = blocks[0]["rows"][1][0]
        assert {"type": "user", "user_id": "U123"} in cell["elements"][0]["elements"]

    def test_unmatched_angle_token_is_preserved_as_text(self):
        # Not a mention and not an autolink: must survive verbatim, not vanish.
        blocks = render_blocks("- a <b> c")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert "a <b> c" in "".join(e.get("text") or "" for e in els)

    def test_mention_inside_emphasis_preserves_styling_and_mention(self):
        # Regression: emphasis wrapping a mention must not be destroyed into literal delimiters
        blocks = render_blocks("- **ping <@U123> now**")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert not any("**" in (e.get("text") or "") for e in els)
        assert {"type": "user", "user_id": "U123"} in els
        assert any(e.get("text") == "ping " and e.get("style", {}).get("bold") for e in els)
        assert any(e.get("text") == " now" and e.get("style", {}).get("bold") for e in els)

        # Italic
        blocks = render_blocks("- *ping <#C456>*")
        els = self._item_elements(blocks)
        assert not any("*" in (e.get("text") or "") for e in els)
        assert {"type": "channel", "channel_id": "C456"} in els
        assert any(e.get("text") == "ping " and e.get("style", {}).get("italic") for e in els)

        # Strike
        blocks = render_blocks("- ~~hi <!here>~~")
        els = self._item_elements(blocks)
        assert not any("~~" in (e.get("text") or "") for e in els)
        assert {"type": "broadcast", "range": "here"} in els
        assert any(e.get("text") == "hi " and e.get("style", {}).get("strike") for e in els)


def _list_item(md):
    blocks = render_blocks(md)
    assert blocks is not None
    return blocks[0]["elements"][0]["elements"][0]["elements"]


def _quote(md):
    blocks = render_blocks(md)
    assert blocks is not None and blocks[0]["elements"][0]["type"] == "rich_text_quote"
    return blocks[0]["elements"][0]["elements"]


def _table_cell(cell_md):
    # _TABLE_SEP_RE requires >= 2 columns, so pad with a throwaway second column.
    blocks = render_blocks(f"| h | h2 |\n| --- | --- |\n| {cell_md} | x |")
    assert blocks is not None and blocks[0]["type"] == "table"
    return blocks[0]["rows"][1][0]["elements"][0]["elements"]


def _no_literal_markup(els, marks=("**", "__", "~~", "`", "](", "<http")):
    for e in els:
        if e.get("type") in ("text", "link"):
            assert not any(m in e["text"] for m in marks), e


class TestEmphasisWrappingOpaqueTokens:
    """Emphasis must survive wrapping inline code, links and mentions.

    Code/link spans used to be split off before emphasis parsing, leaving the
    opening ``**`` in one slice and the closing ``**`` in the other, so both
    rendered as literal text.
    """

    def test_bold_wrapping_code_then_slack_link_in_bullet(self):
        els = _list_item(
            "- **Upstream Chromium / `browser_exec` bug filed:** "
            "<https://github.com/NousResearch/hermes-agent/issues/150314|NousResearch/hermes-agent#150314>"
        )
        assert els == [
            {"type": "text", "text": "Upstream Chromium / ", "style": {"bold": True}},
            {"type": "text", "text": "browser_exec", "style": {"bold": True, "code": True}},
            {"type": "text", "text": " bug filed:", "style": {"bold": True}},
            {"type": "text", "text": " "},
            {
                "type": "link",
                "url": "https://github.com/NousResearch/hermes-agent/issues/150314",
                "text": "NousResearch/hermes-agent#150314",
            },
        ]

    @pytest.mark.parametrize("extract", [_list_item, _quote, _table_cell], ids=["list", "quote", "cell"])
    @pytest.mark.parametrize(
        "md, key",
        [("**see `x_y` now**", "bold"), ("__see `x_y` now__", "bold"),
         ("*see `x_y` now*", "italic"), ("_see `x_y` now_", "italic"),
         ("~~see `x_y` now~~", "strike")],
    )
    def test_emphasis_wrapping_code_in_rich_text_containers(self, extract, md, key):
        prefix = {_list_item: "- ", _quote: "> ", _table_cell: ""}[extract]
        els = extract(prefix + md)
        assert els == [
            {"type": "text", "text": "see ", "style": {key: True}},
            {"type": "text", "text": "x_y", "style": {key: True, "code": True}},
            {"type": "text", "text": " now", "style": {key: True}},
        ]

    @pytest.mark.parametrize(
        "link", ["[the docs](https://example.com/a_b)", "<https://example.com/a_b|the docs>"],
        ids=["markdown", "slack"],
    )
    @pytest.mark.parametrize("delim, key", [("**", "bold"), ("*", "italic"), ("~~", "strike")])
    def test_emphasis_wrapping_sole_and_mixed_links(self, link, delim, key):
        expected_link = {
            "type": "link", "url": "https://example.com/a_b", "text": "the docs", "style": {key: True}}
        assert _list_item(f"- {delim}{link}{delim}") == [expected_link]
        assert _list_item(f"- {delim}See {link} now{delim}") == [
            {"type": "text", "text": "See ", "style": {key: True}},
            expected_link,
            {"type": "text", "text": " now", "style": {key: True}},
        ]

    def test_link_label_code_and_emphasis_move_into_link_style(self):
        els = _list_item(
            "- [`code`](https://e.com/1) [**bold**](https://e.com/2) "
            "<https://e.com/3|***both***> **[_it_ `x`](https://e.com/4)**"
        )
        links = [e for e in els if e["type"] == "link"]
        assert links == [
            {"type": "link", "url": "https://e.com/1", "text": "code", "style": {"code": True}},
            {"type": "link", "url": "https://e.com/2", "text": "bold", "style": {"bold": True}},
            {"type": "link", "url": "https://e.com/3", "text": "both",
             "style": {"bold": True, "italic": True}},
            {"type": "link", "url": "https://e.com/4", "text": "it",
             "style": {"bold": True, "italic": True}},
            {"type": "link", "url": "https://e.com/4", "text": " ", "style": {"bold": True}},
            {"type": "link", "url": "https://e.com/4", "text": "x",
             "style": {"bold": True, "code": True}},
        ]
        _no_literal_markup(els, marks=("**", "`", "]("))

    def test_triple_emphasis_wrapping_code(self):
        assert _list_item("- ***bold italic with `code`***") == [
            {"type": "text", "text": "bold italic with ", "style": {"bold": True, "italic": True}},
            {"type": "text", "text": "code", "style": {"bold": True, "italic": True, "code": True}},
        ]

    def test_nested_italic_and_code_inside_bold(self):
        assert _list_item("- **bold with _italic_ and `code`**") == [
            {"type": "text", "text": "bold with ", "style": {"bold": True}},
            {"type": "text", "text": "italic", "style": {"bold": True, "italic": True}},
            {"type": "text", "text": " and ", "style": {"bold": True}},
            {"type": "text", "text": "code", "style": {"bold": True, "code": True}},
        ]

    @pytest.mark.parametrize(
        "code", ["foo_bar_baz", "**not bold**", "a ** b", "<@U123>", "a[0](1)", "<https://x.com>"]
    )
    def test_code_span_contents_stay_literal_inside_and_outside_bold(self, code):
        assert _list_item(f"- `{code}`") == [{"type": "text", "text": code, "style": {"code": True}}]
        assert _list_item(f"- **see `{code}` here**") == [
            {"type": "text", "text": "see ", "style": {"bold": True}},
            {"type": "text", "text": code, "style": {"bold": True, "code": True}},
            {"type": "text", "text": " here", "style": {"bold": True}},
        ]

    def test_snake_case_beside_code_is_not_italicized(self):
        els = _list_item("- set `max_tokens` in my_config_file and foo_bar")
        assert not any(e.get("style", {}).get("italic") for e in els)
        assert "".join(e["text"] for e in els) == "set max_tokens in my_config_file and foo_bar"

    def test_emphasis_wrapping_mention_and_code_keeps_mention_unstyled(self):
        els = _list_item("- **hello <@U123> run `deploy` <#C9|ops_room> world**")
        assert els == [
            {"type": "text", "text": "hello ", "style": {"bold": True}},
            {"type": "user", "user_id": "U123"},
            {"type": "text", "text": " run ", "style": {"bold": True}},
            {"type": "text", "text": "deploy", "style": {"bold": True, "code": True}},
            {"type": "text", "text": " ", "style": {"bold": True}},
            {"type": "channel", "channel_id": "C9"},
            {"type": "text", "text": " world", "style": {"bold": True}},
        ]

    def test_unclosed_markup_and_nul_are_preserved_verbatim(self):
        md = "**open `code` [x](`u`) \x00 tail"
        els = _list_item(f"- {md}")
        rendered = "".join(
            f"`{e['text']}`" if e.get("style", {}).get("code") else e["text"] for e in els)
        assert rendered == md


class TestTables:
    def test_pipe_table_renders_native_table_block(self):
        md = (
            "| Name | Status |\n"
            "|------|--------|\n"
            "| a | ok |\n"
            "| b | fail |"
        )
        blocks = render_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "table"
        rows = blocks[0]["rows"]
        # header + 2 body rows, 2 columns each
        assert len(rows) == 3
        assert all(len(r) == 2 for r in rows)
        # cells are rich_text carrying the values
        assert str(rows[0]).count("Name") == 1
        assert "fail" in str(rows[2])


    def test_oversized_table_falls_back_to_monospace(self):
        # 120 rows > MAX_TABLE_ROWS -> monospace rich_text fallback, not a table
        big = "| a | b |\n|---|---|\n" + "\n".join(f"| x{i} | y |" for i in range(120))
        blocks = render_blocks(big)
        assert blocks[0]["type"] == "rich_text"  # preformatted fallback
        assert blocks[0]["elements"][0]["type"] == "rich_text_preformatted"


    def test_escaped_pipe_not_a_column_separator(self):
        md = (
            "| Expr | Meaning |\n"
            "|------|--------|\n"
            "| a \\| b | or |"
        )
        blocks = render_blocks(md)
        assert blocks[0]["type"] == "table"
        # the escaped-pipe cell stays a single cell containing a literal pipe
        body = blocks[0]["rows"][1]
        assert len(body) == 2
        assert "|" in str(body[0])


class TestLimits:

    def test_too_many_blocks_returns_none(self):
        # 60 dividers => 60 blocks > MAX_BLOCKS => decline (caller uses text)
        md = "\n\n".join(["---"] * (MAX_BLOCKS + 10))
        assert render_blocks(md) is None


class TestEmptyContentGuards:
    """Empty content must never produce a Slack-rejected (invalid_blocks) payload.

    Slack rejects a rich_text_section / rich_text_preformatted /
    rich_text_quote whose ``elements`` is empty or contains a zero-length
    ``text`` element, and a ``header`` whose plain_text is empty. Each guard
    below corresponds to a real chat.postMessage rejection observed in
    production ("missing element" / "must be more than 0 characters").
    """

    @staticmethod
    def _assert_schema_valid(blocks):
        def walk(o):
            if isinstance(o, dict):
                if o.get("type") in (
                    "rich_text_section", "rich_text_preformatted", "rich_text_quote"
                ):
                    assert o.get("elements"), f"empty {o['type']} elements"
                if o.get("type") == "text":
                    assert len(o.get("text", "")) > 0, "zero-length text element"
                if o.get("type") == "header":
                    assert o["text"]["text"], "empty plain_text header"
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(blocks)

    def test_ragged_and_empty_table_cells_are_schema_valid(self):
        # Blank middle cell + ragged short row (padded with "") must not emit
        # an empty section or a 0-char text element.
        md = (
            "| x | y | z |\n"
            "| --- | --- | --- |\n"
            "| 1 |  | 3 |\n"   # blank middle cell
            "| 4 |"           # ragged row -> padded with empty cells
        )
        blocks = render_blocks(md)
        assert blocks[0]["type"] == "table"
        self._assert_schema_valid(blocks)

    def test_empty_code_fence_quote_and_list_item_are_schema_valid(self):
        # Empty fenced code block (common around empty tool output), blank
        # quote line, and empty list item must all stay schema-valid.
        md = "```\n```\n\n> \n\n- \n- real item"
        blocks = render_blocks(md)
        assert blocks is not None
        self._assert_schema_valid(blocks)


class TestSanitizeBlocks:
    """Outbound boundary clamp: one bad block must never fail the whole call.

    Regression coverage for the invalid_blocks / msg_too_long bug class
    (#56615 null column_settings, #62054 / #53693 >3000-char sections on
    approval chat.update after HTML-escaping inflation).
    """


    def test_oversized_section_text_is_clamped(self):
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "x" * 3500}},
        ]
        out = sanitize_blocks(blocks)
        assert len(out[0]["text"]["text"]) <= MAX_SECTION_TEXT
        assert out[0]["text"]["text"].endswith("…")

    def test_html_escape_inflated_approval_update_is_clamped(self):
        # #53693 / #62054: send path budgeted the RAW text to <=3000, but the
        # interaction payload echoes it back HTML-escaped (& -> &amp;) so the
        # chat.update section exceeds the cap.
        inflated = "a" * 2990 + "&amp;" * 10  # 3040 chars
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": inflated}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "✅ ok"}]},
        ]
        out = sanitize_blocks(blocks)
        assert len(out[0]["text"]["text"]) <= MAX_SECTION_TEXT
        # context block untouched
        assert out[1] == blocks[1]

    def test_null_column_settings_entries_are_fixed(self):
        # #56615: Slack rejects null entries in table column_settings.
        table = {
            "type": "table",
            "rows": [[{"type": "rich_text", "elements": []}]],
            "column_settings": [None, {"align": "center"}, None],
        }
        out = sanitize_blocks([table])
        cs = out[0]["column_settings"]
        assert cs == [{}, {"align": "center"}]
        assert all(isinstance(c, dict) for c in cs)

    def test_all_null_column_settings_are_dropped(self):
        table = {
            "type": "table",
            "rows": [[{"type": "rich_text", "elements": []}]],
            "column_settings": [None, None],
        }
        out = sanitize_blocks([table])
        assert "column_settings" not in out[0]


class TestSplitTextFenceBalanced:
    """_split_text closes/reopens ``` fences at section chunk boundaries."""

    def test_fenced_split_every_chunk_balanced(self):
        from plugins.platforms.slack.block_kit import _split_text

        text = "```\n" + "\n".join("y" * 20 for _ in range(30)) + "\n```"
        chunks = _split_text(text, 100)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert chunk.count("```") % 2 == 0, (
                f"chunk {i} has unbalanced fences: {chunk[:60]!r}"
            )


def _long_nested_list_md(n=40):
    return "\n".join(
        f"{i}. **Item {i} title** see [ticket {i}](https://example.com/tracker/issues/{i:05d}?view=full)\n"
        f"   - detail one for item {i} with [docs](https://docs.example.com/guide/section-{i})"
        f" covering the owner, deadline and current status\n"
        f"   - detail two for item {i} with `code_{i}` and more explanatory words describing the"
        f" change, why it matters, and the follow-up actions the reviewer should take next"
        for i in range(1, n + 1)
    )




def _fmt():
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter

    return SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake")).format_message


def _assert_batches_fit(batches):
    for batch in batches:
        assert len(batch) <= MAX_BLOCKS
        assert estimate_text_chars(batch) <= MAX_RICH_TEXT_MESSAGE_CHARS
        assert sum(b["type"] == "table" for b in batch) <= 1


def _ordered_numbers(batches, indent=0):
    """The numbers Slack shows for ordered items at ``indent`` (each element counts from offset+1)."""
    numbers = []
    for batch in batches:
        for block in batch:
            for el in block.get("elements") or []:
                if (el.get("type") == "rich_text_list" and el["style"] == "ordered"
                        and el["indent"] == indent):
                    start = el.get("offset", 0)
                    numbers += range(start + 1, start + 1 + len(el["elements"]))
    return numbers


class TestRichTextMessageBudget:
    """Once any rich_text block is present Slack caps the aggregate text of the whole message
    (msg_blocks_too_long); over budget, the blocks are split across messages, staying native."""

    def test_estimate_counts_link_urls(self):
        url = "https://example.com/" + "x" * 80
        blocks = render_blocks(f"- [a]({url})")
        assert estimate_text_chars(blocks) >= len(url) + 1
        section = {"type": "section", "text": {"type": "mrkdwn", "text": "abc"}}
        assert estimate_text_chars([section]) == 4  # text + the line break Slack counts

    def test_estimate_counts_emoji_at_shortcode_length(self):
        # Slack counts 🚦 as :vertical_traffic_light: (24 chars) toward the budget.
        section = {"type": "section", "text": {"type": "mrkdwn", "text": "🚦" * 10}}
        assert estimate_text_chars([section]) >= 240

    def test_long_nested_list_splits_into_native_batches_with_continuous_numbering(self):
        md = _long_nested_list_md()
        assert estimate_text_chars(_render_markdown(md, _fmt())) > MAX_RICH_TEXT_MESSAGE_CHARS

        batches = render_block_batches(md, mrkdwn_fn=_fmt())
        assert len(batches) > 1
        _assert_batches_fit(batches)
        assert all(b["type"] == "rich_text" for batch in batches for b in batch)
        assert _ordered_numbers(batches) == list(range(1, 41))
        text = str(batches)
        for i in range(1, 41):
            assert f"Item {i} title" in text and f"detail two for item {i} " in text
        # A caller that cannot split (chat.update) gets None and uses plain text.
        assert render_blocks(md, mrkdwn_fn=_fmt()) is None

    def test_emoji_heavy_list_stays_rich_text(self):
        md = "\n".join(f"- item {i} 🚦🏗️✅ status 🚦" for i in range(200))
        batches = render_block_batches(md)
        assert len(batches) > 1
        _assert_batches_fit(batches)
        items = [
            item for batch in batches for block in batch for el in block["elements"]
            for item in el["elements"]]
        assert len(items) == 200
        assert all(el["style"] == "bullet" for batch in batches for block in batch
                   for el in block["elements"])

    def test_code_and_quote_stay_native_when_over_budget(self):
        md = (
            "```python\nprint('hi')\n```\n\n> quoted line\n> second\n\n"
            + _long_nested_list_md()
        )
        first = render_block_batches(md)[0]
        assert first[0]["elements"][0] == {
            "type": "rich_text_preformatted", "elements": [{"type": "text", "text": "print('hi')"}]}
        assert first[1]["elements"][0]["type"] == "rich_text_quote"

    def test_short_nested_list_keeps_single_rich_text_list(self):
        batches = render_block_batches(_long_nested_list_md(3))
        assert len(batches) == 1
        assert any(e["type"] == "rich_text_list" for e in batches[0][0]["elements"])

    def test_oversized_code_block_splits_on_line_boundaries(self):
        lines = [f"line {i:05d} " + "x" * 40 for i in range(600)]
        batches = render_block_batches("```\n" + "\n".join(lines) + "\n```")
        assert len(batches) > 1
        _assert_batches_fit(batches)
        texts = []
        for batch in batches:
            for block in batch:
                el = block["elements"][0]
                assert el["type"] == "rich_text_preformatted"
                texts.append("".join(c["text"] for c in el["elements"]))
        assert all(not t.startswith("\n") and not t.endswith("\n") for t in texts)
        assert "\n".join(texts) == "\n".join(lines)

    def test_oversized_quote_splits_on_lines_and_keeps_links(self):
        md = "\n".join(f"> quote {i} see [ref](https://example.com/{i}) " + "y" * 60 for i in range(300))
        batches = render_block_batches(md)
        assert len(batches) > 1
        _assert_batches_fit(batches)
        quotes = [b["elements"][0] for batch in batches for b in batch]
        assert all(q["type"] == "rich_text_quote" for q in quotes)
        links = [c for q in quotes for c in q["elements"] if c["type"] == "link"]
        assert [c["url"] for c in links] == [f"https://example.com/{i}" for i in range(300)]

    def test_single_line_over_budget_is_cut_without_losing_text(self):
        line = "z" * (MAX_RICH_TEXT_MESSAGE_CHARS * 2 + 10)
        batches = render_block_batches(f"```\n{line}\n```")
        assert len(batches) == 3
        _assert_batches_fit(batches)
        assert "".join(
            b["elements"][0]["elements"][0]["text"] for batch in batches for b in batch) == line

    def test_more_than_fifty_blocks_split_into_batches(self):
        md = "\n\n".join(f"paragraph {i}" for i in range(MAX_BLOCKS + 10))
        batches = render_block_batches(md)
        assert [len(b) for b in batches] == [MAX_BLOCKS, 10]
        assert render_blocks(md) is None

    def test_sections_only_message_is_not_char_capped(self):
        # Slack caps aggregate text only once rich_text is present.
        md = "\n\n".join("w" * 2000 for _ in range(10))
        assert len(render_block_batches(md)) == 1

    def test_divider_only_content_declines(self):
        assert render_block_batches("\n\n".join(["---"] * (MAX_BLOCKS + 10))) is None

    def test_each_table_gets_its_own_message_and_keeps_its_heading(self):
        table = "| a | b |\n|---|---|\n| {0} | x |"
        md = f"## First\n\n{table.format(1)}\n\n## Second\n\n{table.format(2)}"
        batches = render_block_batches(md)
        assert [_types(b) for b in batches] == [["header", "table"], ["header", "table"]]
        assert batches[1][0]["text"]["text"] == "Second"
        assert render_blocks(md) is None

    def test_trailing_header_moves_to_next_batch(self):
        body = _rich_text_para("a" * 60)
        header = _header_block("Next", level=2)
        batches = split_block_batches([body, header, _rich_text_para("b" * 60)], max_chars=100)
        assert [_types(b) for b in batches] == [["rich_text"], ["header", "rich_text"]]

    def test_header_only_batch_is_not_emptied(self):
        h1, h2 = _header_block("One"), _header_block("Two")
        batches = split_block_batches([h1, h2, _rich_text_para("c" * 66)], max_chars=70)
        assert [_types(b) for b in batches] == [["header", "header"], ["rich_text"]]


def _rich_text_para(text):
    return {"type": "rich_text", "elements": [
        {"type": "rich_text_section", "elements": [{"type": "text", "text": text}]}]}


class TestOrderedListOffsets:
    def test_ordered_run_resumed_after_nested_list_continues_numbering(self):
        blocks = render_blocks("1. a\n   - nested\n2. b\n3. c\n   1. sub\n4. d")
        assert _ordered_numbers([blocks]) == [1, 2, 3, 4]
        assert _ordered_numbers([blocks], indent=1) == [1]

    def test_bullets_reset_ordered_numbering_at_same_indent(self):
        blocks = render_blocks("1. a\n2. b\n\n- x\n\n1. c")
        assert _ordered_numbers([blocks]) == [1, 2, 1]


class TestHeadings:
    @pytest.mark.parametrize("hashes, level", [("#", 1), ("##", 2), ("###", 3), ("####", 4),
                                               ("#####", 4), ("######", 4)])
    def test_heading_level_maps_to_header_level(self, hashes, level):
        block = render_blocks(f"{hashes} Title")[0]
        assert block["type"] == "header" and block["level"] == level

    def test_header_block_replaces_links_with_labels(self):
        block = _header_block("**See [Docs](https://x.com/d) and <https://y.com|Y> or <https://z.com>**")
        assert block["text"]["text"] == "See Docs and Y or https://z.com"

    def test_heading_with_link_renders_bold_rich_text_keeping_link(self):
        blocks = render_blocks("## **Status: [tracker](https://x.com/t) _today_**")
        assert blocks[0]["type"] == "rich_text"
        assert blocks[0]["elements"][0]["elements"] == [
            {"type": "text", "text": "Status: ", "style": {"bold": True}},
            {"type": "link", "url": "https://x.com/t", "text": "tracker", "style": {"bold": True}},
            {"type": "text", "text": " ", "style": {"bold": True}},
            {"type": "text", "text": "today", "style": {"bold": True, "italic": True}},
        ]


class TestBareUrls:
    def test_bare_url_becomes_link_without_duplicate_text(self):
        assert _list_item("- see https://example.com/a?b=1.") == [
            {"type": "text", "text": "see "},
            {"type": "link", "url": "https://example.com/a?b=1"},
            {"type": "text", "text": "."},
        ]

    @pytest.mark.parametrize("md, url, tail", [
        ("(https://x.com/a)", "https://x.com/a", ")"),
        ("https://en.wikipedia.org/wiki/Foo_(bar)", "https://en.wikipedia.org/wiki/Foo_(bar)", ""),
        ("https://en.wikipedia.org/wiki/Foo_(bar)).", "https://en.wikipedia.org/wiki/Foo_(bar)", ")."),
        ("[https://x.com/a]", "https://x.com/a", "]"),
        ("https://x.com/a!?", "https://x.com/a", "!?"),
    ])
    def test_trailing_punctuation_and_unbalanced_brackets_stay_outside(self, md, url, tail):
        els = _list_item(f"- {md}")
        link = next(e for e in els if e["type"] == "link")
        assert link == {"type": "link", "url": url}
        assert els[-1] == ({"type": "text", "text": tail} if tail else link)

    def test_slack_autolink_without_label_omits_text(self):
        assert _list_item("- <https://example.com/x>") == [
            {"type": "link", "url": "https://example.com/x"}]

    def test_bold_bare_url_keeps_style(self):
        assert _list_item("- **https://example.com/x**") == [
            {"type": "link", "url": "https://example.com/x", "style": {"bold": True}}]

    def test_url_in_code_or_markdown_link_is_not_relinked(self):
        assert _list_item("- `https://x.com/a`") == [
            {"type": "text", "text": "https://x.com/a", "style": {"code": True}}]
        assert _list_item("- [https://x.com/a](https://x.com/a)") == [
            {"type": "link", "url": "https://x.com/a"}]

    def test_bare_url_counted_once_in_budget(self):
        url = "https://example.com/" + "p" * 100
        link = _list_item(f"- {url}")[0]
        assert estimate_text_chars(link) == len(url) + 3

    def test_quote_and_table_cell_autolink(self):
        assert {"type": "link", "url": "https://x.com/q"} in _quote("> at https://x.com/q")
        assert _table_cell("https://x.com/c") == [{"type": "link", "url": "https://x.com/c"}]


class TestLinkLabelPartialEmphasis:
    def test_partially_bold_label_splits_into_styled_links(self):
        assert _list_item("- [**Open Opps** sheet](https://example.com/s)") == [
            {"type": "link", "url": "https://example.com/s", "text": "Open Opps",
             "style": {"bold": True}},
            {"type": "link", "url": "https://example.com/s", "text": " sheet"},
        ]

    def test_label_mixing_code_and_italic_under_outer_strike(self):
        els = _list_item("- ~~[run `make` _now_](https://e.com/r)~~")
        assert els == [
            {"type": "link", "url": "https://e.com/r", "text": "run ", "style": {"strike": True}},
            {"type": "link", "url": "https://e.com/r", "text": "make",
             "style": {"strike": True, "code": True}},
            {"type": "link", "url": "https://e.com/r", "text": " ", "style": {"strike": True}},
            {"type": "link", "url": "https://e.com/r", "text": "now",
             "style": {"strike": True, "italic": True}},
        ]
        _no_literal_markup(els)


class TestBatchFallbackText:
    def test_fallback_text_covers_every_block_kind(self):
        md = (
            "# Plan\n\nIntro para\n\n- see [docs](https://x.com/d) for <@U1>\n- a < b\n\n"
            "3. third\n4. **fourth**\n\n> quoted\n\n```\ncode <x>\n```\n\n"
            "| k | v |\n|---|---|\n| one | two |"
        )
        text = blocks_fallback_text(render_blocks(md))
        assert text.splitlines() == [
            "*Plan*", "Intro para", "• see <https://x.com/d|docs> for <@U1>", "• a &lt; b",
            "1. third", "2. *fourth*", "> quoted", "```", "code &lt;x&gt;", "```",
            "k | v", "one | two",
        ]

    def test_fallback_numbers_continue_after_split(self):
        batches = render_block_batches(_long_nested_list_md())
        numbered = [ln for ln in blocks_fallback_text(batches[1]).splitlines() if ln[0].isdigit()]
        assert int(numbered[0].split(".", 1)[0]) == _ordered_numbers(batches[1:])[0] > 1


class TestPlanMessagesAndCodeLinks:
    def test_multi_chunk_keeps_all_batches(self):
        batches = [[{"type": "divider"}], [{"type": "table", "rows": []}]]
        out = plan_messages(["a" * 5, "b" * 5], batches)
        assert [b for _, b in out] == batches

    def test_single_chunk_single_batch_keeps_chunk_text(self):
        batch = [{"type": "divider"}]
        assert plan_messages(["hello"], [batch]) == [("hello", batch)]

    def test_no_batches_is_text_only(self):
        assert plan_messages(["a", "b"], None) == [("a", None), ("b", None)]

    @pytest.mark.parametrize("md, url, label", [
        ("`<https://example.com|FUSE-2651>`", "https://example.com", "FUSE-2651"),
        ("`[doitbse/resolve](https://github.com/doitbse/resolve)`",
         "https://github.com/doitbse/resolve", "doitbse/resolve"),
    ])
    def test_backtick_wrapped_link_is_clickable_code(self, md, url, label):
        from plugins.platforms.slack.block_kit import _inline_elements
        assert _inline_elements(md) == [
            {"type": "link", "url": url, "text": label, "style": {"code": True}}]


class TestMarkdownLinkBalancedBracketLabels:
    URL = "https://github.com/o/r/blob/main/app/posts/%5BpostId%5D/route.ts"

    @staticmethod
    def _links(md):
        blocks = render_blocks(md)
        out = []

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "link":
                    out.append(node)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(blocks)
        return out

    def test_plain_label_with_inner_brackets(self):
        links = self._links(f"- [posts/[postId]/route.ts]({self.URL})")
        assert [(l["url"], l.get("text")) for l in links] == [(self.URL, "posts/[postId]/route.ts")]

    def test_code_label_with_inner_brackets(self):
        links = self._links(f"- [`posts/[postId]/route.ts`]({self.URL})")
        assert len(links) == 1 and links[0]["url"] == self.URL
        assert links[0]["text"] == "posts/[postId]/route.ts"
        assert links[0]["style"] == {"code": True}

    def test_multiple_links_on_one_line(self):
        u2 = "https://example.com/b"
        links = self._links(f"- [a/[x]/p.ts]({self.URL}) and [b]({u2}) and [c/[y]]({u2}/c)")
        assert [(l["url"], l["text"]) for l in links] == [
            (self.URL, "a/[x]/p.ts"), (u2, "b"), (u2 + "/c", "c/[y]")]

    def test_unrelated_brackets_before_link(self):
        links = self._links("- [note] see [link](https://example.com/z) [tail]")
        assert [(l["url"], l["text"]) for l in links] == [("https://example.com/z", "link")]


class TestFormatMessageBalancedBracketLabels:
    URL = "https://github.com/o/r/blob/main/app/posts/%5BpostId%5D/route.ts"

    def test_label_with_inner_brackets(self):
        out = _fmt()(f"[posts/[postId]/route.ts]({self.URL})")
        assert out == f"<{self.URL}|posts/[postId]/route.ts>"

    def test_backticked_label_with_inner_brackets(self):
        out = _fmt()(f"[`posts/[postId]/route.ts`]({self.URL})")
        assert self.URL in out and "posts/[postId]/route.ts" in out
        assert "](" not in out

    def test_multiple_links_one_line(self):
        out = _fmt()(f"[a/[x]]({self.URL}) and [b](https://example.com/b)")
        assert out == f"<{self.URL}|a/[x]> and <https://example.com/b|b>"

    def test_unrelated_brackets_before_link(self):
        out = _fmt()("[note] see [link](https://example.com/z)")
        assert out == "[note] see <https://example.com/z|link>"

    def test_code_span_wrapped_link_with_inner_brackets_is_unwrapped(self):
        out = _fmt()(f"`[posts/[postId]/route.ts]({self.URL})`")
        assert out == f"<{self.URL}|posts/[postId]/route.ts>"


class TestReviewRegressionsBlockKit:
    def test_two_level_bracket_label(self):
        from plugins.platforms.slack.block_kit import _LINK_RE
        m = _LINK_RE.search("[a/[b/[c]]/d.ts](https://example.com/x)")
        assert m and m.group(1) == "a/[b/[c]]/d.ts"

    def test_two_level_bracket_label_format_message(self):
        out = _fmt()("[a/[b/[c]]/d.ts](https://example.com/x)")
        assert out == "<https://example.com/x|a/[b/[c]]/d.ts>"

    def test_code_span_link_unwrap_balanced_paren_url(self):
        url = "https://en.wikipedia.org/wiki/Foo_(bar)"
        out = _fmt()(f"`[label]({url})`")
        assert out == f"<{url}|label>"

    def test_link_mrkdwn_bolds_whole_link_for_bold_label(self):
        from plugins.platforms.slack.block_kit import link_mrkdwn
        assert link_mrkdwn("https://e.com", "**hi**") == "*<https://e.com|hi>​*"
        assert link_mrkdwn("https://e.com", "__hi__") == "*<https://e.com|hi>​*"

    def test_link_mrkdwn_keeps_dunder_in_bracket_path(self):
        from plugins.platforms.slack.block_kit import link_mrkdwn
        out = link_mrkdwn("https://e.com", "[__slug__]/page")
        assert "__slug__" in out and not out.startswith("*")

    def test_link_mrkdwn_escapes_label_and_url(self):
        from plugins.platforms.slack.block_kit import link_mrkdwn
        out = link_mrkdwn("https://e.com/a|b>c<d", "x <!here> & y")
        assert out.startswith("<https://e.com/a%7Cb%3Ec%3Cd|")
        assert "<!here>" not in out and "&lt;!here&gt;" in out and "&amp;" in out

    @pytest.mark.parametrize("label", ["posts/[_id_]/route.ts", "app/[__slug__]/page.tsx"])
    def test_underscore_in_brackets_not_emphasis(self, label):
        url = "https://example.com/x"
        links = TestMarkdownLinkBalancedBracketLabels._links(f"- [{label}]({url})")
        assert [(l["url"], l["text"]) for l in links] == [(url, label)]
        for l in links:
            assert not (l.get("style") or {}).get("italic")
            assert not (l.get("style") or {}).get("bold")

    def test_underscore_emphasis_still_works(self):
        from plugins.platforms.slack.block_kit import _BOLD_RE, _ITALIC_RE
        assert _BOLD_RE.search("a __bold__ b")
        assert _ITALIC_RE.search("a _it_ b")
        assert not _ITALIC_RE.search("[_id_]")
        assert not _BOLD_RE.search("[__slug__]")


class TestG4CRound3Slack:
    def test_inline_code_link_label_escapes_angle_brackets(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        a = SlackAdapter.__new__(SlackAdapter)
        assert a.format_message("[`a > b`](https://example.com/a)") == "<https://example.com/a|`a &gt; b`>"

    def test_dunder_span_cannot_cross_internal_dunder(self):
        from plugins.platforms.slack.block_kit import _BOLD_RE, _LABEL_BOLD_RE, link_mrkdwn, render_blocks

        s = "__init__.py and __repr__()"
        for rx in (_BOLD_RE, _LABEL_BOLD_RE):
            assert not any("init__.py and" in (m.group(m.lastindex) or "") for m in rx.finditer(s))
        assert "__init__.py" in link_mrkdwn("https://example.com/a", s)
        flat = str(render_blocks("- " + s))
        assert "__init__.py" in flat


class TestG4CRound4Slack:
    @pytest.mark.parametrize("src,want", [
        ("[`__init__`](https://example.com/a)", "<https://example.com/a|`__init__`>"),
        ("[`__all__`](https://example.com/a)", "<https://example.com/a|`__all__`>"),
        ("[`x**2 + y**2`](https://example.com/a)", "<https://example.com/a|`x**2 + y**2`>"),
        ("[`foo(**kw)` and `bar(** args)`](https://example.com/a)",
         "<https://example.com/a|`foo(**kw)` and `bar(** args)`>"),
    ])
    def test_code_span_in_link_label_not_unbolded(self, src, want):
        from plugins.platforms.slack.adapter import SlackAdapter

        a = SlackAdapter.__new__(SlackAdapter)
        assert a.format_message(src) == want

    def test_label_bold_dunder_requires_nonspace_edges(self):
        from plugins.platforms.slack.block_kit import _LABEL_BOLD_RE

        assert not _LABEL_BOLD_RE.search("__ a __")
        assert _LABEL_BOLD_RE.search("__a__")
