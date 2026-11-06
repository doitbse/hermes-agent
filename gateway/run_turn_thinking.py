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
# A heading line of its own (bold/underline/markdown), or a bold heading immediately followed
# by ``:``/``-`` and same-line body text (e.g. Gemini's ``**Finding root causes**: ...``).
_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"\*\*(?P<h1>[^*\n]+)\*\*[ \t]*$"
    r"|__(?P<h2>[^_\n]+)__[ \t]*$"
    r"|#+[ \t]*(?P<h3>[^\n]+)$"
    r"|\*\*(?P<h4>[^*\n]+)\*\*[ \t]*[:.\-–—][ \t]*(?P<b4>.*)$"
    r")"
)
_TITLE_MARKS_RE = re.compile(r"(?m)`|\*\*|__|^\s*#+\s*")

_background_tasks: set[asyncio.Task] = set()


def _strip_marks(text: str) -> str:
    text = re.sub(r"^\s*#+\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    while len(text) > 1 and text[0] == text[-1] and text[0] in "_*":
        text = text[1:-1].strip()
    return text


def _cap(text: str) -> str:
    if len(text) <= _TITLE_CHARS:
        return text
    cut = text[: _TITLE_CHARS - 1].rstrip()
    if cut.count("`") % 2 == 1:
        cut += "`"
    return cut + "…"


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


def _clean_plain(text: str) -> str:
    """Plain-text title: markdown/code marks removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", _TITLE_MARKS_RE.sub("", text)).strip()


def _headed_section(heading: str, body: str) -> dict[str, str]:
    """Case A: a bold/markdown heading followed by body text (Gemini-style thoughts)."""
    heading = re.sub(r"\s+", " ", heading).strip()
    details = re.sub(r"^[\s:\-–—]+", "", body).strip()[:_DETAILS_CHARS]
    title = _cap(_clean_plain(heading))
    if details:
        first_sent = _SENTENCE_END.split(details, 1)[0].strip()
        live_title = _cap(f"*{heading}*: {first_sent}")
    else:
        live_title = _cap(f"*{heading}*")
    return {"title": title, "details": details, "live_title": live_title}


def _prose_section(text: str) -> dict[str, str]:
    """Case B: unheaded prose (Claude Opus-style thoughts), split at the first sentence."""
    stripped = text.strip()
    parts = _SENTENCE_END.split(stripped, 1)
    first_sent = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return {"title": _cap(_clean_plain(first_sent)), "details": rest[:_DETAILS_CHARS],
            "live_title": _cap(first_sent)}


_CONTINUATION_BLOCK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|```)\s*")


def _unheaded_sections(text: str) -> list[dict[str, str]]:
    """Split unheaded prose on blank-line paragraph boundaries, keeping list/code blocks attached."""
    paragraphs: list[str] = []
    for raw_block in re.split(r"\n[ \t]*\n+", text.strip()):
        block = raw_block.strip()
        if not block:
            continue
        if paragraphs and _CONTINUATION_BLOCK_RE.match(block):
            paragraphs[-1] = f"{paragraphs[-1]}\n\n{block}"
        else:
            paragraphs.append(block)
    return [_prose_section(p) for p in (paragraphs or [text])]


def parse_thought_sections(text: str) -> list[dict[str, str]]:
    """Split a raw thought into sections at each bold/markdown heading (Gemini); absent any
    heading, split on blank-line prose paragraph boundaries (Claude Opus thinking)."""
    text = str(text or "")
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return _unheaded_sections(text)
    sections: list[dict[str, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.extend(_unheaded_sections(text[: matches[0].start()]))
    for i, m in enumerate(matches):
        heading = next(g for g in (m.group("h1"), m.group("h2"), m.group("h3"), m.group("h4"))
                       if g is not None)
        same_line = m.group("b4") or ""
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        rest = text[m.end():end]
        body = f"{same_line}\n{rest}" if same_line else rest
        sections.append(_headed_section(heading, body))
    return sections or _unheaded_sections(text)


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
            # _assistant_reasoning_text replays the full reasoning block after streaming when
            # stream_delta_callback is unset; replace (rather than double) on full-buffer replay.
            if self._buf and re.sub(r"\s+", " ", self._buf).strip() == re.sub(r"\s+", " ", text).strip():
                self._buf = text
            else:
                self._buf += text
            full, seq = self._buf, self._seq
        sections = parse_thought_sections(full)
        if not sections or not sections[0]["title"]:
            return None
        return {"type": "thinking.updated", "thought_id": f"thought_{seq}", "title": sections[0]["title"],
                "live_title": sections[0]["live_title"], "details": sections[0]["details"], "sections": sections}

    def boundary(self) -> None:
        with self._lock:
            if self._buf:
                self._seq += 1
                self._buf = ""

    def interim(self, text: str) -> Optional[dict[str, Any]]:
        sections = parse_thought_sections(text)
        if not sections or not sections[0]["title"]:
            return None
        with self._lock:
            self._interim_seq += 1
            seq = self._interim_seq
        return {"type": "thinking.updated", "thought_id": f"interim_{seq}", "title": sections[0]["title"],
                "live_title": sections[0]["live_title"], "details": sections[0]["details"], "sections": sections}


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


def thinking_cards_enabled(adapter) -> bool:
    # ``is True``: duck-typed adapters (mocks, relay connectors) must not opt in by accident.
    probe = getattr(adapter, "thinking_task_cards_enabled", None)
    return callable(probe) and probe() is True


def arm_thinking_cards(turn, agent, want_interim_messages: bool) -> None:
    """Route reasoning (and commentary the chat would drop) into this turn's thinking card.

    Module-level so ``TurnRunner._wire_turn_agent_callbacks`` works for any ``turn`` exposing
    ``_ctx``/``_runner`` (including bare holders). Every gate is read defensively and checked
    BEFORE ``agent`` is touched: when cards are off (API, non-Slack, thinking cards disabled) the
    agent's existing ``reasoning_callback`` (e.g. an API admission's SSE observer) is left as-is.
    """
    ctx = getattr(turn, "_ctx", None)
    if ctx is None or not getattr(ctx, "_native_slack_task_cards", False):
        return
    if getattr(ctx, "progress_queue", None) is None or getattr(ctx, "mute_notification_reply", False):
        return
    delivery_adapter_for = getattr(getattr(turn, "_runner", None), "_delivery_adapter_for", None)
    if not callable(delivery_adapter_for) or not thinking_cards_enabled(delivery_adapter_for(ctx.source)):
        return
    reasoning_sink = getattr(turn, "native_reasoning_callback", None)
    if not callable(reasoning_sink):
        return
    turn._thoughts, turn._background_card_follow = ThoughtAccumulator(), True
    # Chained, not replaced: an API admission's observer may already hold this slot.
    agent.reasoning_callback = chain_callbacks(getattr(agent, "reasoning_callback", None), reasoning_sink)
    interim_sink = getattr(turn, "native_interim_callback", None)
    if not want_interim_messages and callable(interim_sink):
        agent.interim_assistant_callback = interim_sink


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


def _apply_sections(st, base_id: str, sections: list[dict[str, str]], owner: str) -> None:
    """Upsert one task row per thought section; only the last section stays ``in_progress``."""
    if base_id not in st.tasks:
        _settle_running(st, kind="thought", owner=owner)
    last = len(sections) - 1
    for idx, section in enumerate(sections):
        task_id = base_id if idx == 0 else f"{base_id}_{idx + 1}"
        _put(st, task_id, kind="thought", owner=owner, status="complete" if idx != last else "in_progress",
             title=st._compact(section.get("title") or "…", 180),
             live_title=str(section.get("live_title") or section.get("title") or "…"),
             details=str(section.get("details") or ""))


def _apply_thought(st, raw: dict) -> bool:
    base_id = str(raw.get("thought_id") or "thought")
    sections = raw.get("sections") or parse_thought_sections(raw.get("title") or raw.get("details") or "")
    _apply_sections(st, base_id, sections, owner="")
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
        base_id = f"thought_sub_{sid}_{seq}"
        _apply_sections(st, base_id, parse_thought_sections(raw.get("preview") or ""), owner=sid)
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
