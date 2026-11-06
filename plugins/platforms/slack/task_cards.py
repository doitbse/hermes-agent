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
_THOUGHT_PREFIX, _TOOL_PREFIX = "💭 ", "🔧 "


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
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _window_text(tasks: list[dict[str, Any]]) -> str:
    """Last few thoughts (or tools, before the first thought) as plain grey bullets.
    No ``_..._`` wrapper: an underscore inside a snake_case identifier would end the italics."""
    thoughts = [task for task in tasks if is_thought(task)]
    if thoughts:
        lines = [f"• {_mrkdwn_escape(_one_line(task.get('title'), _WINDOW_LINE_CHARS))}"
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
    elapsed = format_elapsed((time.monotonic() if now is None else now) - stream.started_at)
    steps = _step_str(stream.step_count or len(stream.all_tasks))
    if stream.is_background:
        return f"✓ {_background_label(stream)} finished · {elapsed} · {steps}"[:_TITLE_CHARS]
    return f"✓ Thought for {elapsed} · {steps}"[:_TITLE_CHARS]


def _plan_task(task: dict[str, Any], idx: int) -> dict[str, Any]:
    title = str(task.get("title") or "").strip()
    prefix = _THOUGHT_PREFIX if is_thought(task) else _TOOL_PREFIX
    if not title.startswith(prefix.strip()):
        title = prefix + title
    title = title[:_TITLE_CHARS]
    entry: dict[str, Any] = {
        "task_id": str(task.get("id") or task.get("task_id") or f"step_{idx}"),
        "title": title,
        "status": "error" if task.get("status") == "error" else "complete",
    }
    details = task.get("details")
    if isinstance(details, str) and details.strip() and details.strip() != title:
        entry["details"] = {"type": "rich_text", "elements": [{
            "type": "rich_text_section", "elements": [{"type": "text", "text": details[:_DETAILS_CHARS]}]}]}
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
        await client.chat_update(
            channel=stream.channel, ts=stream.stream_ts, text=title,
            blocks=[{"type": "context", "elements": [{"type": "mrkdwn", "text": title, "verbatim": False}]}])
