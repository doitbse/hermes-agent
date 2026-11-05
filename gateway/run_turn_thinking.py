"""Thought and subagent events for Slack thinking task cards (``thinking_task_cards``).

Agent-thread callbacks enqueue ``thinking.updated`` / ``subagent.*`` dicts onto the turn's
progress queue; ``_TaskCardState`` folds them into task rows via ``apply_thinking_event``.
A background follower keeps a second card live for ``delegate_task(background=true)``
children after the parent turn ends.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.run")

PUBLISH_INTERVAL = 1.0
_TITLE_CHARS = 180
# Title extraction reads a bounded head so per-token re-extraction stays O(1).
_TITLE_SCAN_CHARS = 600
_DETAILS_CHARS = 4000
BACKGROUND_LABEL = "Background subagent"
# The parent's reply is delivered after the turn's cleanup; hold the first background frame
# so the card lands below it rather than above.
_BACKGROUND_FIRST_FRAME_DELAY = 2.0
_BACKGROUND_POLL = 0.25
SUBAGENT_EVENTS = frozenset({"subagent.start", "subagent.thinking", "subagent.tool", "subagent.complete"})

_BOLD_HEAD = re.compile(r"^\s*(\*\*|\*|__)(.+?)\1\s*[:.\-–—]?\s*(.*)$", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s|\n")
_CLAUSE_END = re.compile(r"[.!?;,]\s|\n")

_background_tasks: set[asyncio.Task] = set()


def _strip_marks(text: str) -> str:
    text = re.sub(r"^\s*#+\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    while len(text) > 1 and text[0] == text[-1] and text[0] in "_*":
        text = text[1:-1].strip()
    return text


def _cap(text: str) -> str:
    return text if len(text) <= _TITLE_CHARS else text[: _TITLE_CHARS - 1].rstrip() + "…"


def thought_title(text: str) -> str:
    """One-line summary: a leading bold heading (+ first clause), else the first sentence."""
    head = str(text or "")[:_TITLE_SCAN_CHARS].strip()
    lines = [line for line in head.splitlines() if line.strip()]
    if not lines:
        return ""
    first = re.sub(r"^\s*#+\s*", "", lines[0])
    bold = _BOLD_HEAD.match(first if len(lines) == 1 else first + "\n" + "\n".join(lines[1:]))
    if bold:
        heading, rest = _strip_marks(bold.group(2)), bold.group(3).strip()
        clause = _CLAUSE_END.split(rest, 1)[0].strip() if rest else ""
        return _cap(f"{heading}: {_strip_marks(clause)}" if clause else heading)
    return _cap(_strip_marks(_SENTENCE_END.split(" ".join(lines), 1)[0]))


class ThoughtAccumulator:
    """Streamed reasoning per segment; a tool start opens the next ``thought_<seq>`` row."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 1
        self._buf = ""
        self._interim_seq = 0

    def feed(self, text: str) -> Optional[dict[str, Any]]:
        if not text:
            return None
        with self._lock:
            self._buf += text
            full, seq = self._buf, self._seq
        title = thought_title(full)
        if not title:
            return None
        return {"type": "thinking.updated", "thought_id": f"thought_{seq}", "title": title,
                "details": full[:_DETAILS_CHARS]}

    def boundary(self) -> None:
        with self._lock:
            if self._buf:
                self._seq += 1
                self._buf = ""

    def interim(self, text: str) -> Optional[dict[str, Any]]:
        title = thought_title(text)
        if not title:
            return None
        with self._lock:
            self._interim_seq += 1
            seq = self._interim_seq
        return {"type": "thinking.updated", "thought_id": f"interim_{seq}", "title": title,
                "details": str(text)[:_DETAILS_CHARS]}


def chain_callbacks(*callbacks: Optional[Callable]) -> Optional[Callable]:
    """Fan one agent callback out to several sinks; a failing sink never starves the others."""
    sinks = [cb for cb in callbacks if cb is not None]
    if len(sinks) <= 1:
        return sinks[0] if sinks else None

    def fanout(*args, **kwargs):
        for sink in sinks:
            try:
                sink(*args, **kwargs)
            except Exception:
                logger.debug("chained callback sink failed", exc_info=True)
    return fanout


def subagent_card_event(event_type: str, tool_name, preview, kwargs: dict) -> dict[str, Any]:
    return {
        "type": event_type, "subagent_id": str(kwargs.get("subagent_id") or kwargs.get("task_index") or "0"),
        "delegation_id": str(kwargs.get("delegation_id") or ""), "goal": str(kwargs.get("goal") or ""),
        "tool_name": str(tool_name or ""), "preview": str(preview or ""), "status": str(kwargs.get("status") or ""),
    }


# ── _TaskCardState extensions ──────────────────────────────────────────────────────────────

def _settle_running(st, *, kind: Optional[str] = None, owner: Optional[str] = None) -> None:
    for task in st.tasks.values():
        if task.get("status") != "in_progress":
            continue
        if (kind is None or task.get("kind") == kind) and (owner is None or task.get("owner") == owner):
            task["status"] = "complete"


def _put(st, task_id: str, **fields) -> dict[str, Any]:
    if task_id not in st.tasks:
        st.task_order.append(task_id)
        st.tasks[task_id] = {"id": task_id, "status": "in_progress"}
    st.tasks[task_id].update(fields)
    return st.tasks[task_id]


def _apply_thought(st, raw: dict) -> bool:
    task_id = str(raw.get("thought_id") or "thought")
    if task_id not in st.tasks:
        _settle_running(st, kind="thought", owner="")
    _put(st, task_id, kind="thought", owner="", title=st._compact(raw.get("title"), 180),
         details=str(raw.get("details") or ""))
    return True


def _apply_subagent(st, raw: dict) -> bool:
    event_type, sid = raw["type"], raw["subagent_id"]
    goal = st._compact(raw.get("goal"), 80)
    if event_type == "subagent.start":
        st.subagents_running.add(sid)
        st.delegation_id = st.delegation_id or raw.get("delegation_id") or ""
        _put(st, f"sub_{sid}", kind="tool", owner=sid, title=f"subagent - {goal}" if goal else "subagent")
    elif event_type == "subagent.thinking":
        seq = st.subagent_segments.setdefault(sid, 1)
        task_id = f"thought_sub_{sid}_{seq}"
        if task_id not in st.tasks:
            _settle_running(st, kind="thought", owner=sid)
        _put(st, task_id, kind="thought", owner=sid, title=thought_title(raw.get("preview") or "") or "…",
             details=str(raw.get("preview") or "")[:_DETAILS_CHARS])
    elif event_type == "subagent.tool":
        st.subagent_segments[sid] = st.subagent_segments.get(sid, 1) + 1
        st.subagent_tools[sid] = st.subagent_tools.get(sid, 0) + 1
        _settle_running(st, owner=sid)
        st.tasks.get(f"sub_{sid}", {}).update(status="in_progress")
        preview = st._compact(raw.get("preview"), 64)
        name = raw.get("tool_name") or "tool"
        _put(st, f"sub_{sid}_tool_{st.subagent_tools[sid]}", kind="tool", owner=sid,
             title=f"{name} - {preview}" if preview else name)
    else:  # subagent.complete
        st.subagents_running.discard(sid)
        _settle_running(st, owner=sid)
        from tools.delegate_tool import SUBAGENT_FAILURE_STATUSES
        if raw.get("status") in SUBAGENT_FAILURE_STATUSES:
            st.tasks.get(f"sub_{sid}", {}).update(status="error")
    return True


def apply_thinking_event(st, raw: Any) -> Optional[bool]:
    """Handle thought/subagent events; None when ``raw`` is not one (caller falls through)."""
    event_type = raw.get("type") if isinstance(raw, dict) else None
    if event_type == "thinking.updated":
        return _apply_thought(st, raw)
    if event_type in SUBAGENT_EVENTS:
        return _apply_subagent(st, raw)
    if event_type == "tool.started":
        _settle_running(st, kind="thought", owner="")
    return None


# ── background follower ────────────────────────────────────────────────────────────────────

def active_background_children(session_id: Optional[str]) -> list[dict[str, Any]]:
    if not session_id:
        return []
    from tools.delegate_tool_registry import list_active_subagents
    return [rec for rec in list_active_subagents()
            if rec.get("owner_agent_session_id") == session_id and rec.get("status") == "running"]


def spawn_background_follower(runner, adapter, foreground_st, session_id: Optional[str]) -> Optional[asyncio.Task]:
    """Keep a background card live (and fold it) while this turn's background children run."""
    task = asyncio.get_running_loop().create_task(_follow(runner, adapter, foreground_st, session_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _seed_background_state(runner, adapter, foreground_st):
    st = runner._TaskCardState(adapter, thinking=True)
    st.subagents_running = set(foreground_st.subagents_running)
    st.delegation_id = foreground_st.delegation_id
    for task_id in foreground_st.task_order:
        task = foreground_st.tasks[task_id]
        if task.get("owner") in st.subagents_running:
            st.task_order.append(task_id)
            st.tasks[task_id] = dict(task)
    return st


async def _follow(runner, adapter, foreground_st, session_id: Optional[str]) -> None:
    ctx = runner._ctx
    st = _seed_background_state(runner, adapter, foreground_st)
    suffix = st.delegation_id or "bg"
    route = dict(reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata)
    dirty, last_publish = bool(st.tasks), time.monotonic() - PUBLISH_INTERVAL + _BACKGROUND_FIRST_FRAME_DELAY
    try:
        while st.subagents_running or active_background_children(session_id):
            dirty = _drain_into(ctx.progress_queue, st) or dirty
            if dirty and time.monotonic() - last_publish >= PUBLISH_INTERVAL:
                await _publish_background(adapter, ctx, st, suffix, route)
                dirty, last_publish = False, time.monotonic()
            await asyncio.sleep(_BACKGROUND_POLL)
        if (_drain_into(ctx.progress_queue, st) or dirty) and st.tasks:
            await _publish_background(adapter, ctx, st, suffix, route)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("background task-card follower failed", exc_info=True)
    finally:
        runner._background_card_follow = False
        runner._drain_progress_queue()
        try:
            await adapter.stop_native_task_card_progress(ctx.source.chat_id, stream_suffix=suffix, **route)
        except Exception:
            logger.debug("background task-card fold failed", exc_info=True)


def _drain_into(progress_queue, st) -> bool:
    changed = False
    try:
        while True:
            raw = progress_queue.get_nowait()
            if isinstance(raw, dict) and raw.get("type") in SUBAGENT_EVENTS:
                changed = _apply_subagent(st, raw) or changed
    except queue.Empty:
        pass
    return changed


async def _publish_background(adapter, ctx, st, suffix: str, route: dict) -> None:
    if st.tasks:
        await adapter.send_native_task_card_progress(
            ctx.source.chat_id, st.visible_tasks(), fallback_text=st.fallback_text(),
            stream_suffix=suffix, background_label=BACKGROUND_LABEL, **route)
