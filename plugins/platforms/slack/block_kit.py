"""Render agent markdown into Slack Block Kit blocks.

Opt-in (``slack.extra.rich_blocks: true``) alternative to the flat mrkdwn
``text`` payload produced by :meth:`SlackAdapter.format_message`.  Block Kit
gives us real structural primitives — section headers, dividers, and true
*nested* lists via ``rich_text`` — that plain mrkdwn can only approximate.

Design constraints (why this module is deliberately conservative):

* **Markdown pipe-tables render as native ``table`` blocks** — real grid
  cells with per-column alignment and inline-formatted ``rich_text`` content.
  A table that exceeds Slack's limits (100 rows / 20 cols / 10k aggregate
  cell chars) or won't parse falls back to aligned monospace
  ``rich_text_preformatted`` so a large table never breaks the message.
* **Slack caps a message at 50 blocks** and a ``section``/text object at 3000
  characters.  :func:`render_blocks` enforces both and, if the content simply
  cannot be expressed within them, returns ``None`` so the caller falls back
  to the plain-text path.  A rich render is a nice-to-have; it must never lose
  a message.
* **Every blocks payload MUST ship a ``text`` fallback.**  Slack uses it for
  notifications, screen readers, and old clients.  This module only builds the
  ``blocks`` list; the adapter pairs it with the existing mrkdwn string.

The renderer never raises: any unexpected input degrades to ``None`` (caller
uses plain text).  It is a pure function of its input — no Slack client, no
adapter state — so it is trivially unit-testable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Slack Block Kit hard limits (https://docs.slack.dev/reference/block-kit/blocks)
MAX_BLOCKS = 50
MAX_SECTION_TEXT = 3000
MAX_HEADER_TEXT = 150
# Native table block limits (https://docs.slack.dev/reference/block-kit/blocks/table-block)
MAX_TABLE_ROWS = 100
MAX_TABLE_COLS = 20
MAX_TABLE_CHARS = 10000  # aggregate across all cells
# Once any rich_text block is present, Slack rejects the message (msg_blocks_too_long) when the
# aggregate text of ALL blocks, link URLs included, exceeds ~13.2k chars; sections alone are not
# capped this way. Over budget, lists/quotes/code re-render as mrkdwn sections instead.
MAX_RICH_TEXT_MESSAGE_CHARS = 12000

Block = dict[str, Any]

# ----------------------------------------------------------------------------
# Line classification
# ----------------------------------------------------------------------------

_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")


def _is_list_line(line: str) -> bool:
    """True if ``line`` is a markdown list item (bullet or ordered)."""
    return bool(_BULLET_RE.match(line) or _ORDERED_RE.match(line))


def _indent_level(spaces: str) -> int:
    """Map leading whitespace to a nesting level (2 spaces or 1 tab per level)."""
    width = 0
    for ch in spaces:
        width += 4 if ch == "\t" else 1
    return min(width // 2, 5)  # Slack rich_text_list supports up to indent 5


# ----------------------------------------------------------------------------
# Inline markdown → rich_text elements
# ----------------------------------------------------------------------------

# Order matters: code first (opaque), then links, then emphasis.
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^()\s]+(?:\([^()]*\)[^()\s]*)*)\)")
# Slack mrkdwn autolink: <scheme:target> or <scheme:target|label>.
_SLACK_LINK_RE = re.compile(
    r"<([a-zA-Z][a-zA-Z0-9+.\-]*:[^>|]+)(?:\|([^>]+))?>"
)
# Slack mention tokens carry no scheme: <@U…> (user), <#C…> (channel, with an
# optional |name label), <!subteam^S…> (usergroup) and <!here|!channel|!everyone>
# (broadcast). rich_text does not interpret mrkdwn, so these must become
# user/channel/broadcast/usergroup elements — as plain text Slack renders the
# token literally.
_SLACK_MENTION_RE = re.compile(
    r"<(@[A-Z0-9]+|#[A-Z0-9]+|!subteam\^[A-Z0-9]+|!(?:here|channel|everyone))(?:\|[^>]*)?>"
)
# Emphasis regexes are alternations; the span is always ``m.group(m.lastindex)``.
# A closer never directly precedes another delimiter char, so ``***x***`` is bold(italic(x)).
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*(?!\*)|__(.+?)__(?!_)")
# `_` never opens/closes intraword (CommonMark), so snake_case stays literal.
_ITALIC_RE = re.compile(
    r"(?<![\*_])\*(?![\*_\s])(.+?)(?<![\*_\s])\*(?![\*_])"
    r"|(?<![\w\*_])_(?![\*_\s])(.+?)(?<![\*_\s])_(?![\w\*_])"
)
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_EMPHASIS = ((_BOLD_RE, "bold"), (_STRIKE_RE, "strike"), (_ITALIC_RE, "italic"))
# Opaque tokens are swapped for \x00<index>\x00 before emphasis parsing; a raw
# NUL in the input is shielded too so it can never forge a placeholder.
_CODE_OR_NUL_RE = re.compile(_INLINE_CODE_RE.pattern + r"|\x00")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")


def _mention_element(body: str) -> Optional[dict[str, Any]]:
    """Map a Slack mention token body (without the angle brackets) to a rich_text element.

    ``@U…`` -> user, ``#C…`` -> channel, ``!subteam^S…`` -> usergroup,
    ``!here``/``!channel``/``!everyone`` -> broadcast.
    Returns None for anything else, so the caller can emit the raw token as text."""
    if body.startswith("@"):
        return {"type": "user", "user_id": body[1:]}
    if body.startswith("#"):
        return {"type": "channel", "channel_id": body[1:]}
    if body.startswith("!subteam^"):
        return {"type": "usergroup", "usergroup_id": body[len("!subteam^"):]}
    if body.startswith("!"):
        return {"type": "broadcast", "range": body[1:]}
    return None


# Shielded token: (kind, raw source, *payload). kind is code/text/link/mention.
_Token = Tuple[str, ...]


def _shield_tokens(text: str) -> Tuple[str, list[_Token]]:
    """Swap opaque inline tokens for placeholders so emphasis can wrap them.
    Code goes first, so link/mention/emphasis syntax inside backticks stays literal.
    rich_text does not interpret mrkdwn, so Slack ``<url|label>`` and mentions are parsed here."""
    tokens: list[_Token] = []

    def put(token: _Token) -> str:
        tokens.append(token)
        return f"\x00{len(tokens) - 1}\x00"

    def link(url: str, label: str, raw: str) -> str:
        # A URL that swallowed a placeholder overlapped a code span: not a link.
        return raw if "\x00" in url else put(("link", raw, url, label))

    s = _CODE_OR_NUL_RE.sub(
        lambda m: put(("code", m.group(0), m.group(1)) if m.group(1) else ("text", m.group(0))),
        text)
    s = _LINK_RE.sub(lambda m: link(m.group(2), m.group(1), m.group(0)), s)
    s = _SLACK_LINK_RE.sub(lambda m: link(m.group(1), m.group(2) or m.group(1), m.group(0)), s)
    s = _SLACK_MENTION_RE.sub(lambda m: put(("mention", m.group(0), m.group(1))), s)
    return s, tokens


def _full_emphasis(s: str) -> Optional[Tuple[str, str]]:
    """``(style_key, inner)`` when one emphasis run spans all of ``s``, else None."""
    for rx, key in _EMPHASIS:
        m = rx.match(s)
        if m and m.end() == len(s):
            return key, m.group(m.lastindex)
    return None


def _link_label(label: str, tokens: list[_Token]) -> Tuple[str, dict[str, bool]]:
    """Plain label text + style for a link element (which takes one ``text`` and one ``style``).
    Wrapping emphasis runs and any code span inside the label become style flags."""
    style: dict[str, bool] = {}
    while (wrapped := _full_emphasis(label)) is not None and wrapped[0] not in style:
        style[wrapped[0]] = True
        label = wrapped[1]

    def unshield(m: re.Match) -> str:
        tok = tokens[int(m.group(1))]
        if tok[0] == "code":
            style["code"] = True
            return tok[2]
        return _PLACEHOLDER_RE.sub(unshield, tok[1])  # a nested link's raw may hold code

    return _PLACEHOLDER_RE.sub(unshield, label), style


def _emit_text(elements: list[dict[str, Any]], s: str, style: dict[str, bool]) -> None:
    if s:
        elements.append({"type": "text", "text": s, **({"style": dict(style)} if style else {})})


def _emit_token(elements: list[dict[str, Any]], tokens: list[_Token], tok: _Token,
                style: dict[str, bool]) -> None:
    kind = tok[0]
    if kind == "code":
        _emit_text(elements, tok[2], {**style, "code": True})
    elif kind == "link":
        text, label_style = _link_label(tok[3], tokens)
        merged = {**style, **label_style}
        elements.append({
            "type": "link", "url": tok[2], "text": text, **({"style": merged} if merged else {})})
    elif kind == "mention" and (mention := _mention_element(tok[2])) is not None:
        # Slack rejects ``style`` on user/channel/usergroup/broadcast elements.
        elements.append(mention)
    else:
        _emit_text(elements, tok[1], style)


def _walk_emphasis(elements: list[dict[str, Any]], tokens: list[_Token], s: str,
                   style: dict[str, bool]) -> None:
    """Split ``s`` on bold, then strike, then italic; expand placeholders at the leaves."""
    if not s:
        return
    for rx, key in _EMPHASIS:
        m = rx.search(s)
        if m:
            _walk_emphasis(elements, tokens, s[: m.start()], style)
            _walk_emphasis(elements, tokens, m.group(m.lastindex), {**style, key: True})
            _walk_emphasis(elements, tokens, s[m.end() :], style)
            return
    pos = 0
    for m in _PLACEHOLDER_RE.finditer(s):
        _emit_text(elements, s[pos : m.start()], style)
        _emit_token(elements, tokens, tokens[int(m.group(1))], style)
        pos = m.end()
    _emit_text(elements, s[pos:], style)


def _inline_elements(text: str) -> list[dict[str, Any]]:
    """Parse a run of inline markdown into rich_text section child elements.
    Produces ``text`` elements (optionally styled bold/italic/strike/code), ``link`` elements and
    unstyled mention elements. Unmatched markup is emitted verbatim as plain text, so this never
    loses characters."""
    elements: list[dict[str, Any]] = []
    shielded, tokens = _shield_tokens(text)
    _walk_emphasis(elements, tokens, shielded, {})
    return elements or [{"type": "text", "text": text}]


# ----------------------------------------------------------------------------
# Structural block builders
# ----------------------------------------------------------------------------


def _nonempty_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make a rich_text child-element list safe for Slack.
    Slack rejects any ``rich_text_section`` / ``rich_text_preformatted`` / ``rich_text_quote`` whose
    ``elements`` list is empty or contains a ``text`` element of zero length (``invalid_blocks``:
    "missing element" / "must be more than 0 characters"). Empty content is common — ragged table
    rows are padded with ``""``, agents emit empty code fences around empty tool output, blank quote
    lines and empty list items occur in the wild — so drop zero-length text elements and, if nothing
    remains, substitute a single space, which renders as blank yet stays schema-valid. Used by every
    rich_text builder so empty content can never poison the whole payload."""
    els = [e for e in elements if not (e.get("type") == "text" and not e.get("text"))]
    return els or [{"type": "text", "text": " "}]


def _header_block(text: str) -> Optional[Block]:
    # header blocks are plain_text only, 150 char cap.
    clean = re.sub(r"[*_~`]", "", text).strip()
    if not clean:
        # Emphasis-/whitespace-only header (e.g. "# ***" or "#   ") reduces to
        # empty; Slack rejects an empty plain_text with invalid_blocks. Skip it
        # (caller drops None) rather than poison the whole payload.
        return None
    if len(clean) > MAX_HEADER_TEXT:
        clean = clean[: MAX_HEADER_TEXT - 1] + "…"
    return {"type": "header", "text": {"type": "plain_text", "text": clean, "emoji": True}}


def _divider_block() -> Block:
    return {"type": "divider"}


def _rich_text_block(kind: str, children: list[dict[str, Any]]) -> Block:
    """One ``rich_text`` block wrapping a single ``kind`` element (section/quote/preformatted)."""
    return {
        "type": "rich_text", "elements": [{"type": kind, "elements": _nonempty_elements(children)}]}


def _preformatted_block(text: str) -> Block:
    # rich_text_preformatted renders monospace; used for code fences + tables.
    return _rich_text_block("rich_text_preformatted", [{"type": "text", "text": text.rstrip("\n")}])


def _quote_block(lines: list[str]) -> Block:
    section_children: list[dict[str, Any]] = []
    for i, ln in enumerate(lines):
        if i:
            section_children.append({"type": "text", "text": "\n"})
        section_children.extend(_inline_elements(ln))
    return _rich_text_block("rich_text_quote", section_children)


def _list_block(items: list[tuple[int, bool, str]]) -> Block:
    """Build ONE rich_text block from consecutive list items.
    ``items`` is a list of ``(indent, ordered, text)``. Each contiguous run sharing the same
    (indent, ordered) becomes a ``rich_text_list`` element; indentation changes start a new element,
    which is how Slack renders true nesting."""
    elements: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    cur_key: Optional[tuple[int, bool]] = None
    for indent, ordered, text in items:
        key = (indent, ordered)
        if key != cur_key:
            cur = {
                "type": "rich_text_list", "style": "ordered" if ordered else "bullet",
                "indent": indent, "elements": []}
            elements.append(cur)
            cur_key = key
        cur["elements"].append(
            {"type": "rich_text_section", "elements": _nonempty_elements(_inline_elements(text))})
    return {"type": "rich_text", "elements": elements}


def _section_block(text: str) -> Block:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


# ----------------------------------------------------------------------------
# mrkdwn-section builders — used when rich_text would exceed the message budget
# ----------------------------------------------------------------------------

_BULLET_MARKERS = ("•", "◦", "▪")
# Slack collapses leading ASCII spaces in mrkdwn; em spaces survive.
_MRKDWN_INDENT = "\u2003\u2003"


def _mrkdwn_sections(text: str) -> list[Block]:
    return [_section_block(chunk) for chunk in _split_text(text, MAX_SECTION_TEXT)]


def _mrkdwn_code(text: str, fmt) -> list[Block]:
    return _mrkdwn_sections(fmt("```\n" + (text.rstrip("\n") or " ") + "\n```"))


def _mrkdwn_quote(lines: list[str], fmt) -> list[Block]:
    return _mrkdwn_sections(fmt("\n".join(f"> {ln}" for ln in lines)))


def _mrkdwn_list(items: list[tuple[int, bool, str]], fmt) -> list[Block]:
    """Items as ``•``/``1.`` lines, nesting shown by em-space indentation and a per-level bullet."""
    levels: list[tuple[bool, int]] = []  # (ordered, count) per indent level
    lines: list[str] = []
    for indent, ordered, text in items:
        del levels[indent + 1:]
        levels.extend([(ordered, 0)] * (indent + 1 - len(levels)))
        kind, count = levels[indent]
        count = count + 1 if kind == ordered else 1
        levels[indent] = (ordered, count)
        marker = f"{count}." if ordered else _BULLET_MARKERS[indent % len(_BULLET_MARKERS)]
        lines.append(f"{_MRKDWN_INDENT * indent}{marker} {fmt(text)}")
    return _mrkdwn_sections("\n".join(lines))


_RICH_BUILDERS = {
    "code": lambda text, fmt: [_preformatted_block(text)],
    "quote": lambda lines, fmt: [_quote_block(lines)],
    "list": lambda items, fmt: [_list_block(items)],
}
_MRKDWN_BUILDERS = {"code": _mrkdwn_code, "quote": _mrkdwn_quote, "list": _mrkdwn_list}


# Slack measures the budget on its mrkdwn rendering: a unicode emoji counts as its ``:shortcode:``
# (up to ~24 chars, e.g. :vertical_traffic_light:), a link as ``<url|text>``, a list item/line +1.
_EMOJI_CHARS = 24
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u27BF\u2B00-\u2BFF\u3030\u303D]")


def _text_cost(text: str) -> int:
    return len(text) + len(_EMOJI_RE.findall(text)) * (_EMOJI_CHARS - 1)


def estimate_text_chars(node: Any) -> int:
    """Aggregate text Slack counts toward a message's block budget: every ``text`` string, link
    ``url``, table cell and nested element, with emoji at shortcode length (see ``_EMOJI_CHARS``)."""
    if isinstance(node, list):
        return sum(estimate_text_chars(n) for n in node)
    if not isinstance(node, dict):
        return 0
    btype = node.get("type")
    text = node.get("text")
    total = _text_cost(text) if isinstance(text, str) else estimate_text_chars(text)
    if btype == "link":
        total += len(node.get("url") or "") + 3
    elif btype == "emoji":
        total += _EMOJI_CHARS
    elif btype in ("user", "channel", "usergroup", "broadcast"):
        total += 12
    elif btype in ("rich_text_section", "rich_text_list", "section", "header"):
        total += 1
    return total + sum(estimate_text_chars(node.get(k)) for k in ("elements", "rows", "fields"))


def _over_rich_text_budget(blocks: list[Block]) -> bool:
    has_rich_text = any(b.get("type") == "rich_text" for b in blocks)
    return has_rich_text and estimate_text_chars(blocks) > MAX_RICH_TEXT_MESSAGE_CHARS


# ----------------------------------------------------------------------------
# Table handling — native Block Kit ``table`` block, monospace fallback
# ----------------------------------------------------------------------------


def _parse_alignment(sep_line: str) -> list[str]:
    """Parse a markdown separator row (``|:--|:-:|--:|``) into column aligns.
    Returns a list of ``"left"``/``"center"``/``"right"`` per column."""
    aligns: list[str] = []
    for cell in sep_line.strip().strip("|").split("|"):
        c = cell.strip()
        left, right = c.startswith(":"), c.endswith(":")
        aligns.append("center" if left and right else "right" if right else "left")
    return aligns


def _split_row(row: str) -> list[str]:
    """Split a markdown table row into trimmed cell strings.
    Respects backslash-escaped pipes (``\\|``) so they aren't treated as column separators."""
    # Temporarily protect escaped pipes, split on real ones, then restore.
    protected = row.strip().strip("|").replace(r"\|", "\x00PIPE\x00")
    return [c.strip().replace("\x00PIPE\x00", "|") for c in protected.split("|")]


def _rich_text_cell(text: str) -> dict[str, Any]:
    """A ``rich_text`` table cell carrying inline-formatted content.
    Empty cells are common (ragged rows are padded with ``""``); Slack rejects a cell whose section
    is empty or carries a zero-length text element, so the elements are routed through
    ``_nonempty_elements``."""
    return _rich_text_block("rich_text_section", _inline_elements(text))


def _table_block(rows: list[str], sep_line: str) -> Optional[Block]:
    """Build a native Slack ``table`` block from markdown pipe-table rows.
    ``rows`` includes the header row (index 0) and body rows; ``sep_line`` is the ``|---|``
    alignment row (already consumed by the caller). Returns ``None`` when the table exceeds Slack's
    limits (100 rows / 20 cols / 10,000 aggregate cell chars) or parses to nothing — the caller then
    falls back to the monospace preformatted rendering."""
    parsed = [_split_row(r) for r in rows if r.strip()]
    if not parsed:
        return None
    ncols = max(len(r) for r in parsed)
    # Reject rather than silently truncate beyond Slack's structural limits.
    if len(parsed) > MAX_TABLE_ROWS or ncols > MAX_TABLE_COLS:
        return None
    for r in parsed:
        r.extend([""] * (ncols - len(r)))
    total_chars = sum(len(c) for r in parsed for c in r)
    if total_chars > MAX_TABLE_CHARS:
        return None
    aligns = _parse_alignment(sep_line)
    # Slack requires every provided ``column_settings`` entry to be an object.
    # Missing trailing entries inherit defaults, so only emit settings through
    # the last non-default alignment. Earlier default-left placeholders still
    # need explicit valid objects to preserve positional alignment.
    padded = [aligns[c] if c < len(aligns) else "left" for c in range(ncols)]
    last_non_default = max((c for c, a in enumerate(padded) if a != "left"), default=-1)
    column_settings = [{"align": a} for a in padded[: last_non_default + 1]]
    block: Block = {
        "type": "table", "rows": [[_rich_text_cell(cell) for cell in row] for row in parsed]}
    if column_settings:
        block["column_settings"] = column_settings
    return block


def _render_table(rows: list[str]) -> str:
    """Render markdown pipe-table rows as aligned monospace text (fallback)."""
    parsed = [_split_row(r) for r in rows]
    if not parsed:
        return "\n".join(rows)
    ncols = max(len(r) for r in parsed)
    for r in parsed:
        r.extend([""] * (ncols - len(r)))
    widths = [max(len(r[c]) for r in parsed) for c in range(ncols)]
    out_lines = []
    for ri, r in enumerate(parsed):
        line = " | ".join(r[c].ljust(widths[c]) for c in range(ncols))
        out_lines.append(line.rstrip())
        if ri == 0:  # header underline
            out_lines.append("-+-".join("-" * widths[c] for c in range(ncols)))
    return "\n".join(out_lines)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def render_blocks(markdown: str, mrkdwn_fn=None) -> Optional[list[Block]]:
    """Convert agent markdown to a Slack Block Kit ``blocks`` list.
    Args:
        markdown: The agent's response text (standard markdown).
        mrkdwn_fn: Optional callable converting a markdown paragraph to Slack
            mrkdwn for ``section`` blocks (the adapter passes
            ``format_message``).  When ``None``, the raw paragraph text is used.

    Returns:
        A list of Block Kit block dicts, or ``None`` when the content is empty,
        exceeds Slack's structural limits, or hits an unexpected shape — the
        caller then falls back to the flat ``text`` payload.  Never raises."""
    if not markdown or not markdown.strip():
        return None
    fmt = mrkdwn_fn or (lambda s: s)
    blocks = _render_markdown(markdown, fmt, _RICH_BUILDERS)
    if blocks and _over_rich_text_budget(blocks):
        blocks = _render_markdown(markdown, fmt, _MRKDWN_BUILDERS)
    return blocks


def _render_markdown(markdown: str, fmt, build: dict) -> Optional[list[Block]]:
    """One render pass; ``build`` supplies the code/quote/list block builders."""
    try:
        blocks: list[Block] = []
        lines = markdown.replace("\r\n", "\n").split("\n")
        i = 0
        n = len(lines)
        para: list[str] = []

        def flush_para() -> None:
            if not para:
                return
            text = "\n".join(para).strip()
            para.clear()
            if not text:
                return
            rendered = fmt(text)
            # Split oversized sections on the 3000-char limit.
            for chunk in _split_text(rendered, MAX_SECTION_TEXT):
                blocks.append(_section_block(chunk))
        while i < n:
            line = lines[i]

            # Blank line: paragraph boundary
            if not line.strip():
                flush_para()
                i += 1
                continue

            # Fenced code block
            fence = _FENCE_RE.match(line)
            if fence:
                flush_para()
                marker = fence.group(1)
                body: list[str] = []
                i += 1
                while i < n and not lines[i].lstrip().startswith(marker):
                    body.append(lines[i])
                    i += 1
                i += 1  # consume closing fence
                blocks.extend(build["code"]("\n".join(body), fmt))
                continue

            # Horizontal rule → divider
            if _HR_RE.match(line):
                flush_para()
                blocks.append(_divider_block())
                i += 1
                continue

            # ATX header
            hm = _HEADER_RE.match(line)
            if hm:
                flush_para()
                header = _header_block(hm.group(2))
                if header is not None:
                    blocks.append(header)
                i += 1
                continue

            # Pipe table: current line has a pipe AND next line is a separator
            if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
                flush_para()
                header_row = line
                sep_line = lines[i + 1]
                trows = [header_row]
                i += 2  # skip header + separator
                while i < n and "|" in lines[i] and lines[i].strip():
                    trows.append(lines[i])
                    i += 1
                # Prefer a native Block Kit table; fall back to aligned
                # monospace when it exceeds Slack's table limits or won't parse.
                table = _table_block(trows, sep_line)
                if table is not None:
                    blocks.append(table)
                else:
                    blocks.extend(build["code"](_render_table(trows), fmt))
                continue

            # Blockquote group
            if _QUOTE_RE.match(line):
                flush_para()
                qlines: list[str] = []
                while i < n:
                    qm = _QUOTE_RE.match(lines[i])
                    if not qm:
                        break
                    qlines.append(qm.group(1))
                    i += 1
                blocks.extend(build["quote"](qlines, fmt))
                continue

            # List group (bullets + ordered, with nesting)
            if _is_list_line(line):
                flush_para()
                items: list[tuple[int, bool, str]] = []
                while i < n:
                    bm = _BULLET_RE.match(lines[i])
                    om = _ORDERED_RE.match(lines[i])
                    if bm:
                        items.append((_indent_level(bm.group(1)), False, bm.group(2)))
                        i += 1
                    elif om:
                        items.append((_indent_level(om.group(1)), True, om.group(3)))
                        i += 1
                    elif lines[i].strip() and lines[i].startswith((" ", "\t")) and items:
                        # continuation line of the previous item
                        indent, ordered, txt = items[-1]
                        items[-1] = (indent, ordered, txt + " " + lines[i].strip())
                        i += 1
                    elif not lines[i].strip() and items:
                        # Blank line inside a list run. LLM-authored ordered
                        # lists commonly separate items with a blank line; if
                        # the next non-blank line is another list item, treat
                        # the blank(s) as a soft separator and keep the run
                        # going so the items stay in one rich_text_list (Slack
                        # numbers each list independently, so splitting would
                        # restart every item at "1."). Otherwise the blank
                        # ends the list.
                        j = i + 1
                        while j < n and not lines[j].strip():
                            j += 1
                        if j < n and _is_list_line(lines[j]):
                            i = j
                        else:
                            break
                    else:
                        break
                blocks.extend(build["list"](items, fmt))
                continue

            # Default: accumulate into a paragraph
            para.append(line)
            i += 1
        flush_para()

        if not blocks:
            return None
        if len(blocks) > MAX_BLOCKS:
            # Too structurally complex to express safely — let the caller fall
            # back to plain text rather than truncating and losing content.
            return None
        return blocks
    except Exception:
        # Never let a rendering bug drop a message.
        return None


def _split_text(text: str, limit: int) -> list[str]:
    """Split ``text`` into <= ``limit``-char chunks on line, then hard, boundaries.
    Chunks are fence-balanced: when a split lands inside a ``` code span that survived into section
    text (the renderer normally routes fenced blocks to ``rich_text_preformatted``, but mrkdwn text
    can still carry fences), the fence is closed at the end of the chunk and reopened on the next so
    each section renders correctly on its own."""
    if len(text) <= limit:
        return [text]
    # Reserve headroom for the close/reopen markers the balancing pass adds.
    split_limit = max(limit - 8, limit // 2, 1) if "```" in text else limit
    out: list[str] = []
    remaining = text
    while len(remaining) > split_limit:
        cut = remaining.rfind("\n", 0, split_limit)
        if cut <= 0:
            cut = split_limit
        out.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        out.append(remaining)
    if len(out) > 1 and "```" in text:
        balanced: list[str] = []
        reopen = False
        for chunk in out:
            if reopen:
                chunk = "```\n" + chunk
            odd = chunk.count("```") % 2 == 1
            if odd:
                chunk += "\n```"
            reopen = odd
            balanced.append(chunk)
        out = balanced
    return out


# ----------------------------------------------------------------------------
# Outbound payload boundary — last-resort clamp before the Slack API
# ----------------------------------------------------------------------------


def _clamp_text_obj(text_obj: dict[str, Any], limit: int) -> dict[str, Any]:
    """Return ``text_obj`` with its ``text`` clamped to ``limit`` chars."""
    txt = text_obj.get("text") or ""
    if len(txt) <= limit:
        return text_obj
    clamped = dict(text_obj)
    clamped["text"] = txt[: limit - 1].rstrip() + "…"
    return clamped


def sanitize_blocks(blocks: Optional[list[Block]]) -> Optional[list[Block]]:
    """Clamp an outbound ``blocks`` payload to Slack's hard limits.
    Defensive boundary applied wherever the adapter attaches ``blocks`` to
    ``chat.postMessage`` / ``chat.update``.  One oversized or malformed block
    fails the WHOLE call with ``invalid_blocks`` — approval cards then never
    update and messages silently drop — so instead of trusting every builder,
    the payload is normalized just before the API call:

    * ``section`` / ``context`` text objects are truncated to the 3000-char
      cap with an ellipsis (Slack HTML-escapes ``< > &`` on storage, so text
      echoed back through interaction payloads can exceed the limit that the
      send path originally budgeted for — see #53693 / #62054).
    * ``header`` text is truncated to its 150-char cap.
    * Empty blocks (no text / no elements / no rows) are dropped — Slack
      rejects zero-length text objects and empty element lists.
    * ``table.column_settings`` entries must all be objects; ``null`` entries
      (emitted by older renderers, per the "use null to skip" misreading of
      the schema) are replaced with ``{}`` and default trailing entries are
      trimmed (#56615).
    * The payload is capped at Slack's 50-block maximum.

    Returns the sanitized list, or ``None`` when nothing valid remains — the
    caller then sends the plain ``text`` fallback alone.  Never raises."""
    if not blocks:
        return None
    try:
        out: list[Block] = []
        for block in blocks:
            if not isinstance(block, dict) or not block.get("type"):
                continue
            btype = block["type"]

            if btype == "section":
                text_obj = block.get("text")
                has_body = bool(block.get("fields")) or bool(block.get("accessory"))
                if isinstance(text_obj, dict):
                    if not (text_obj.get("text") or "").strip() and not has_body:
                        continue
                    clamped = _clamp_text_obj(text_obj, MAX_SECTION_TEXT)
                    if clamped is not text_obj:
                        block = dict(block)
                        block["text"] = clamped
                elif not has_body:
                    continue
            elif btype == "header":
                text_obj = block.get("text")
                if not isinstance(text_obj, dict) or not (text_obj.get("text") or "").strip():
                    continue
                clamped = _clamp_text_obj(text_obj, MAX_HEADER_TEXT)
                if clamped is not text_obj:
                    block = dict(block)
                    block["text"] = clamped
            elif btype == "context":
                elements = block.get("elements") or []
                if not elements:
                    continue
                clamped_els = [
                    _clamp_text_obj(el, MAX_SECTION_TEXT)
                    if isinstance(el, dict) and el.get("type") in ("mrkdwn", "plain_text")
                    else el
                    for el in elements]
                if any(c is not e for c, e in zip(clamped_els, elements)):
                    block = dict(block)
                    block["elements"] = clamped_els
            elif btype in ("rich_text", "actions", "context_actions"):
                if not block.get("elements"):
                    continue
            elif btype == "table":
                if not block.get("rows"):
                    continue
                settings = block.get("column_settings")
                if isinstance(settings, list) and any(not isinstance(cs, dict) for cs in settings):
                    fixed = [cs if isinstance(cs, dict) else {} for cs in settings]
                    while fixed and not fixed[-1]:
                        fixed.pop()
                    block = dict(block)
                    if fixed:
                        block["column_settings"] = fixed
                    else:
                        block.pop("column_settings", None)
            out.append(block)
        if not out:
            return None
        return out[:MAX_BLOCKS]
    except Exception:
        # A sanitizer bug must never take down the send path.
        return None
