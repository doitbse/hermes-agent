"""Unit tests for the Slack Block Kit renderer (pure function, no adapter)."""

import pytest

from plugins.platforms.slack.block_kit import (
    _RICH_BUILDERS,
    MAX_BLOCKS,
    MAX_RICH_TEXT_MESSAGE_CHARS,
    MAX_SECTION_TEXT,
    _render_markdown,
    estimate_text_chars,
    render_blocks,
    sanitize_blocks,
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
            {"type": "link", "url": "https://e.com/4", "text": "_it_ x",
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
        "code", ["foo_bar_baz", "**not bold**", "a ** b", "<@U123>", "a[0](1)", "<https://x.com|y>"]
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


def _section_text(blocks):
    return "\n".join(b["text"]["text"] for b in blocks if b["type"] == "section")


class TestRichTextMessageBudget:
    """Once any rich_text block is present Slack caps the aggregate text of the whole message
    (msg_blocks_too_long); over budget, the renderer degrades to mrkdwn sections."""

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

    def test_emoji_heavy_list_degrades(self):
        md = "\n".join(f"- item {i} 🚦🏗️✅ status 🚦" for i in range(200))
        blocks = render_blocks(md)
        assert blocks is not None
        assert "rich_text" not in _types(blocks)

    def test_long_nested_list_degrades_to_sections(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.slack.adapter import SlackAdapter

        fmt = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake")).format_message
        md = _long_nested_list_md()
        rich = _render_markdown(md, fmt, _RICH_BUILDERS)
        assert estimate_text_chars(rich) > MAX_RICH_TEXT_MESSAGE_CHARS  # premise

        blocks = render_blocks(md, mrkdwn_fn=fmt)
        assert blocks is not None
        assert "rich_text" not in _types(blocks)
        assert len(blocks) <= MAX_BLOCKS
        assert all(len(b["text"]["text"]) <= MAX_SECTION_TEXT for b in blocks)
        text = _section_text(blocks)
        for i in range(1, 41):
            assert f"{i}. *Item {i} title*" in text
            assert f"detail two for item {i}" in text
        assert "<https://example.com/tracker/issues/00007?view=full|ticket 7>" in text
        assert "\u2003\u2003◦ detail one for item 7" in text
        assert estimate_text_chars(blocks) < 2 * estimate_text_chars(rich)

    def test_short_nested_list_keeps_rich_text_list(self):
        blocks = render_blocks(_long_nested_list_md(3))
        rich = next(b for b in blocks if b["type"] == "rich_text")
        assert any(e["type"] == "rich_text_list" for e in rich["elements"])

    def test_code_and_quote_render_as_sections_when_degraded(self):
        md = (
            "```python\nprint('hi')\n```\n\n> quoted line\n> second\n\n"
            + _long_nested_list_md()
        )
        blocks = render_blocks(md)
        assert "rich_text" not in _types(blocks)
        assert blocks[0]["text"]["text"] == "```\nprint('hi')\n```"
        assert blocks[1]["text"]["text"] == "> quoted line\n> second"


