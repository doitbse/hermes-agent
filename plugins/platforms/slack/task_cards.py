"""Slack progress cards for one turn's thread.

Two transports share ``_NativeTaskCardStream``:

* ``native_task_cards``: a ``chat.startStream`` plan card fed ``task_update`` chunks.
* ``thinking_task_cards``: a regular message holding a live 2-block ``context`` view
  (recent thoughts/tools + status footer) while running, folded with ``chat.update`` into
  one ``plan`` block on completion. Regular messages never expire server-side, unlike streams.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

# Slack rejects a plan block with more than 50 tasks (invalid_blocks).
MAX_PLAN_TASKS = 50
_WINDOW_ITEMS = 6
_WINDOW_LINE_CHARS = 400
# A context mrkdwn element is capped at 3000 chars.
_WINDOW_CHARS = 2800
_TITLE_CHARS = 256
_DETAILS_CHARS = 2000
DEFAULT_BACKGROUND_LABEL = "Background subagent"
_THOUGHT_PREFIX, _TOOL_PREFIX, _WARNING_PREFIX = "💭 ", "🔧 ", "⚠️ "
# Characters a ``plan`` block (plain text) or a ``task_update`` title must never carry raw.
_TITLE_MARKS_RE = re.compile(r"`|\*\*|__")
# Markdown inline spans tokenized into ``rich_text`` styled runs.
_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+")
_INLINE_TOKEN_RE = re.compile(
    r"`([^`\n]+)`|\*\*([^*\n]+)\*\*|__([^_\n]+)__|(?<!\*)\*([^\s*\n](?:[^*\n]*[^\s*\n])?)\*(?!\*)"
)


@dataclass
class _NativeTaskCardStream:
    """Serialized state for one workspace-scoped Slack progress stream."""

    team_id: str
    channel: str
    thread_ts: str
    stream_ts: str = ""
    stopped: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started_at: float = field(default_factory=time.monotonic)
    all_tasks: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    active_label: str = ""
    is_background: bool = False
    label_prefix: str = ""


async def start_native_stream(
    client, stream: _NativeTaskCardStream, metadata: Optional[dict[str, Any]]
) -> None:
    """chat.startStream a plan-mode card in ``stream``'s thread; sets ``stream.stream_ts``."""
    start_payload: dict[str, Any] = {
        "channel": stream.channel, "thread_ts": stream.thread_ts,
        "task_display_mode": "plan"}
    md = metadata or {}
    recipients = (
        ("recipient_team_id", ("recipient_team_id", "team_id", "slack_team_id")),
        ("recipient_user_id", ("recipient_user_id", "user_id")))
    for key, sources in recipients:
        value = next((md[src] for src in sources if md.get(src)), None)
        if value:
            start_payload[key] = value
    result = await client.api_call("chat.startStream", json=start_payload)
    if hasattr(result, "get"):
        stream.stream_ts = str(result.get("ts") or result.get("message_ts") or "")
    if not stream.stream_ts:
        raise RuntimeError("Slack startStream returned no stream timestamp")


def task_update_chunk(task: dict[str, Any]) -> dict[str, Any]:
    """One ``task_update`` stream chunk; unknown statuses coerce to ``in_progress``."""
    status = str(task.get("status") or "in_progress")
    status = status if status in {"in_progress", "complete", "error"} else "in_progress"
    task_id = str(task.get("id") or task.get("task_id") or "task")
    return {
        "type": "task_update", "id": task_id, "title": str(task.get("title") or task_id)[:256],
        "status": status}


# ── thinking cards ──────────────────────────────────────────────────────────────────────────

def is_thought(task: dict[str, Any]) -> bool:
    return task.get("kind") == "thought" or str(task.get("id", "")).startswith("thought")


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60}s"


def _step_str(count: int) -> str:
    return f"{count} step" if count == 1 else f"{count} steps"


def _mrkdwn_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _one_line(text: Any, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1].rstrip()
    if cut.count("`") % 2 == 1:
        cut += "`"
    return cut + "…"


def _window_text(tasks: list[dict[str, Any]]) -> str:
    """Last few thoughts (or tools, before the first thought) as plain grey bullets.
    No ``_..._`` wrapper: an underscore inside a snake_case identifier would end the italics."""
    thoughts = [task for task in tasks if is_thought(task)]
    if thoughts:
        lines = [f"• {_mrkdwn_escape(_one_line(task.get('live_title') or task.get('title'), _WINDOW_LINE_CHARS))}"
                 for task in thoughts[-_WINDOW_ITEMS:]]
    else:
        lines = [f"• `{_mrkdwn_escape(_one_line(task.get('title'), _WINDOW_LINE_CHARS)).replace('`', chr(39))}`"
                 for task in tasks[-_WINDOW_ITEMS:]]
    return "\n".join(lines)[:_WINDOW_CHARS]


def _active_tool(tasks: list[dict[str, Any]]) -> str:
    """Title of the running tool, or "" when the newest running item is a thought (or nothing runs)."""
    running = [task for task in tasks if task.get("status") == "in_progress"]
    if not running or is_thought(running[-1]):
        return ""
    return _one_line(running[-1].get("title"), 80).replace("`", "'")


def _background_label(stream: _NativeTaskCardStream) -> str:
    return stream.label_prefix or DEFAULT_BACKGROUND_LABEL


def live_blocks(stream: _NativeTaskCardStream, now: Optional[float] = None) -> tuple[list[dict[str, Any]], str]:
    """Blocks + notification fallback text for the running view of ``stream.all_tasks``."""
    tasks = stream.all_tasks
    elapsed = format_elapsed((time.monotonic() if now is None else now) - stream.started_at)
    steps = _step_str(stream.step_count or len(tasks))
    stream.active_label = _active_tool(tasks)
    activity = f"Running: `{_mrkdwn_escape(stream.active_label)}`" if stream.active_label else "Thinking..."
    if stream.is_background:
        footer = f"🔄 *{_mrkdwn_escape(_background_label(stream))}*   ·   {steps} · {elapsed}   ·   {activity}"
    else:
        footer = f"⏳ *{steps} · {elapsed}*   ·   {activity}"
    blocks: list[dict[str, Any]] = []
    window = _window_text(tasks)
    if window:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": window, "verbatim": False}]})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": footer, "verbatim": False}]})
    return blocks, f"{steps} · {elapsed}"


def folded_title(stream: _NativeTaskCardStream, now: Optional[float] = None) -> str:
    """No leading checkmark: Slack's ``plan`` block already renders its own icon."""
    elapsed = format_elapsed((time.monotonic() if now is None else now) - stream.started_at)
    steps = _step_str(stream.step_count or len(stream.all_tasks))
    if stream.is_background:
        return f"{_background_label(stream)} finished · {elapsed} · {steps}"[:_TITLE_CHARS]
    return f"Thought for {elapsed} · {steps}"[:_TITLE_CHARS]


def markdown_to_rich_text(text: str) -> dict[str, Any]:
    """Tokenize a bounded slice of markdown into a Slack ``rich_text`` block's inline spans."""
    body = _BULLET_RE.sub("• ", str(text or ""))[:_DETAILS_CHARS]
    elements: list[dict[str, Any]] = []
    pos = 0
    for m in _INLINE_TOKEN_RE.finditer(body):
        if m.start() > pos:
            elements.append({"type": "text", "text": body[pos:m.start()]})
        if m.group(1) is not None:
            elements.append({"type": "text", "text": m.group(1), "style": {"code": True}})
        elif m.group(2) is not None or m.group(3) is not None:
            elements.append({"type": "text", "text": m.group(2) or m.group(3), "style": {"bold": True}})
        else:
            elements.append({"type": "text", "text": m.group(4), "style": {"italic": True}})
        pos = m.end()
    if pos < len(body):
        elements.append({"type": "text", "text": body[pos:]})
    elements = [el for el in elements if el["text"]]
    return {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": elements}]}


def _plan_task(task: dict[str, Any], idx: int) -> dict[str, Any]:
    """A ``plan`` task row never carries the error status (that turns the plan's own icon into
    a warning triangle); failed tools stay visible via a ``⚠️`` title prefix instead."""
    raw_title = str(task.get("title") or "").strip()
    plain_body = _TITLE_MARKS_RE.sub("", raw_title).strip()
    errored = task.get("status") == "error"
    prefix = _THOUGHT_PREFIX if is_thought(task) else (_WARNING_PREFIX if errored else _TOOL_PREFIX)
    title = raw_title if raw_title.startswith(prefix.strip()) else prefix + raw_title
    # Slack renders plan task titles as plain text: drop any leftover markdown characters.
    title = _TITLE_MARKS_RE.sub("", title)[:_TITLE_CHARS]
    entry: dict[str, Any] = {
        "task_id": str(task.get("id") or task.get("task_id") or f"step_{idx}"),
        "title": title,
        "status": "complete",
    }
    details = task.get("details")
    if isinstance(details, str) and details.strip() and details.strip() != plain_body:
        entry["details"] = markdown_to_rich_text(details)
    return entry


def plan_block(stream: _NativeTaskCardStream, title: str) -> dict[str, Any]:
    tasks = stream.all_tasks
    head: list[dict[str, Any]] = []
    if len(tasks) > MAX_PLAN_TASKS:
        keep = MAX_PLAN_TASKS - 1
        head = [{"task_id": "omitted", "title": f"({len(tasks) - keep} earlier steps omitted)", "status": "complete"}]
        tasks = tasks[-keep:]
    return {"type": "plan", "title": title, "tasks": head + [_plan_task(task, idx) for idx, task in enumerate(tasks)]}


async def publish_live(
    client, stream: _NativeTaskCardStream, chat_id: str, tasks: list[dict[str, Any]],
    unfurl_kwargs: dict[str, bool],
) -> SendResult:
    """Post (first frame) or edit the running view. Caller holds ``stream.lock``."""
    stream.all_tasks = [dict(task) for task in tasks]
    stream.step_count = len(stream.all_tasks)
    blocks, fallback = live_blocks(stream)
    if not stream.stream_ts:
        res = await client.chat_postMessage(
            channel=chat_id, thread_ts=stream.thread_ts, text=fallback, blocks=blocks, **unfurl_kwargs)
        stream.stream_ts = str((res.get("ts") if hasattr(res, "get") else "") or "")
        if not stream.stream_ts:
            raise RuntimeError("Slack chat.postMessage returned no message timestamp")
    else:
        # chat.update takes no unfurl arguments; unfurling is fixed when the message is posted.
        await client.chat_update(channel=chat_id, ts=stream.stream_ts, text=fallback, blocks=blocks)
    return SendResult(success=True, message_id=stream.stream_ts)


async def fold(client, stream: _NativeTaskCardStream) -> None:
    """Replace the running view with the collapsed plan; a rejected plan degrades to one line."""
    if not stream.stream_ts:
        return
    title = folded_title(stream)
    try:
        await client.chat_update(
            channel=stream.channel, ts=stream.stream_ts, text=title, blocks=[plan_block(stream, title)])
    except Exception as exc:
        logger.info("[Slack] Thinking card plan fold rejected (%s); folding to a one-line summary", exc)
        # No ``plan`` block here to render its own icon, so this one-line fallback adds its own.
        fallback = f"✓ {title}"
        await client.chat_update(
            channel=stream.channel, ts=stream.stream_ts, text=fallback,
            blocks=[{"type": "context", "elements": [{"type": "mrkdwn", "text": fallback, "verbatim": False}]}])
