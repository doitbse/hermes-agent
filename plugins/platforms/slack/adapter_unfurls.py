"""Inbound ``attachments`` (link previews and Slack-message unfurls) rendered into the agent's text."""

import logging
from typing import Optional

from agent.compression_marker import ELISION_MARKER_MAX_LEN, elide

logger = logging.getLogger(__name__)

#: One ceiling for every ``attachments[].blocks[]`` projection in a message. Slack allows 20
#: attachments each with its own blocks, so a per-attachment cap alone still grows 20x; the
#: top-level ``blocks`` path is capped once (``_serialize_slack_blocks_for_agent``) and this
#: keeps the unfurl path in the same order of magnitude.
SLACK_UNFURL_BLOCKS_MAX_CHARS = 6000
#: Body cap for a shared/forwarded or cross-channel message unfurl. Unlike a link preview, its
#: ``text`` is the whole message the user handed the bot (a forward carries nothing else).
SLACK_SHARED_MESSAGE_MAX_CHARS = 4000
#: Body cap for an ordinary link-preview unfurl.
SLACK_LINK_UNFURL_MAX_CHARS = 500


def _unfurl_header(att: dict, is_shared_msg: bool) -> Optional[str]:
    """``📎`` header line for one unfurl attachment, or None when it has nothing to name."""
    att_title = att.get("title", "")
    att_url = att.get("title_link", "") or att.get("from_url", "")
    if att_title and att_url:
        return f"📎 [{att_title}]({att_url})"
    if att_title or not is_shared_msg:
        return f"📎 {att_title or att_url}" if (att_title or att_url) else None
    author = att.get("author_name", "") or att.get("author_subname", "")
    header = f"📎 Shared message from {author}" if author else "📎 Shared message"
    return f"{header} ({att_url})" if att_url else header


def append_link_unfurls(text: str, slack_attachments: list) -> str:
    """Append unfurl previews (``attachments``) to ``text``. Slack sets ``is_msg_unfurl`` on
    every Slack-message unfurl: a forward/share (``is_share``, whose ``text`` holds only the
    sharer's note), a permalink to another conversation, or an echo of this message's own text;
    only the echo is skipped. Dedup matches the rendered section, not the bare URL (which is
    usually already in the user's text while the preview body is not)."""
    from plugins.platforms.slack.adapter import _extract_text_from_slack_blocks

    att_parts: list[str] = []
    blocks_budget = SLACK_UNFURL_BLOCKS_MAX_CHARS
    for att in slack_attachments:
        att_text = att.get("text", "")
        att_footer = att.get("footer", "")
        att_fallback = att.get("fallback", "")
        is_shared_msg = bool(att.get("is_share") or att.get("is_msg_unfurl"))
        body = (att_text or att_fallback or "").strip()
        if att.get("is_msg_unfurl") and not att.get("is_share") and (not body or body in text):
            continue
        header = _unfurl_header(att, is_shared_msg)
        max_body_chars = SLACK_SHARED_MESSAGE_MAX_CHARS if is_shared_msg else SLACK_LINK_UNFURL_MAX_CHARS
        if len(body) > max_body_chars:
            body = body[:max_body_chars - 3] + "..."
        # Pasted tables arrive as ``table`` blocks in ``attachments[].blocks[]``, absent from
        # ``text``/``fallback``/files; without this the agent sees only the sentence before them.
        # The budget is shared across the whole array: a 20-attachment alert must not project
        # 20x what a single one does, and a spent budget still leaves the header visible.
        nested_text = ""
        if blocks_budget > 0:
            nested_text = _extract_text_from_slack_blocks(att.get("blocks") or [])
            if len(nested_text) > blocks_budget and blocks_budget <= ELISION_MARKER_MAX_LEN:
                nested_text = ""  # leftover budget cannot hold marker + content: skip, don't overshoot
            nested_text = elide(nested_text, blocks_budget)
        if nested_text and nested_text not in body:
            blocks_budget -= len(nested_text)
            body = f"{body}\n{nested_text}".strip() if body else nested_text
        if header:
            section = f"{header}\n   {body}" if body else header
        elif body:
            section = f"📎 {body}"
        else:
            continue
        if section in text:
            continue
        if att_footer:
            section = f"{section}\n   _{att_footer}_"
        att_parts.append(section)
    if att_parts:
        text = (text.strip() + "\n\n" + "\n\n".join(att_parts)).strip()
        logger.debug("Slack: appended %d link unfurl(s) to message text", len(att_parts))
    return text
