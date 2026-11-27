"""Behaviour contracts for the ``platform_connected`` / ``platform_inbound_event`` plugin hooks
and the Slack adapter's dispatch helpers (dedup release, active-turn probe, deferred dispatch)."""

import asyncio
import contextlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig
from hermes_cli.plugins import VALID_HOOKS, PluginContext, PluginManager, PluginManifest
from plugins.platforms.slack.adapter import SlackAdapter

CHANNEL = "D_TEST"
TEAM = "T_TEST"


class HookRecorder:
    """Stands in for ``hermes_cli.lifecycle.ainvoke_hook``; records calls, returns ``results``."""

    def __init__(self, results=None):
        self.calls: list[tuple[str, dict]] = []
        self.results = results if results is not None else []

    async def __call__(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return list(self.results)

    def of(self, name):
        return [kw for n, kw in self.calls if n == name]


@contextlib.contextmanager
def _hooks(recorder):
    with patch("hermes_cli.lifecycle.ainvoke_hook", recorder), \
            patch("hermes_cli.lifecycle.hook_known_unused", return_value=False), \
            patch("hermes_cli.lifecycle.has_hook", return_value=True):
        yield recorder


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={
        "user": {"is_bot": False, "profile": {"display_name": "Test User"}, "real_name": "Test User"}})
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


def _dm(ts="100.000", text="hello"):
    return {"text": text, "user": "U_HUMAN", "channel": CHANNEL, "channel_type": "im", "ts": ts}


# --- VALID_HOOKS / registration ---------------------------------------------------------------

@pytest.mark.parametrize("hook", ["platform_connected", "platform_inbound_event"])
def test_hooks_are_valid_and_registrable(hook):
    assert hook in VALID_HOOKS
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="fixture-hooks", version="1.0.0", description="fixture",
                                       source="user", key="fixture-hooks"), manager)

    def cb(**kwargs):
        return None

    ctx.register_hook(hook, cb)
    assert cb in manager._hooks.get(hook, [])


# --- platform_connected -----------------------------------------------------------------------

def _connect_patches():
    class FakeHandler:
        def __init__(self, app, app_token, proxy=None):
            self.client = MagicMock()
            self.client.proxy = proxy
            self.client.is_connected = lambda: True
            self._stop = asyncio.Event()

        async def start_async(self):
            await self._stop.wait()

        async def close_async(self):
            self._stop.set()

    def _noop_decorator(_):
        return lambda fn: fn

    mock_app = MagicMock()
    mock_app.event = mock_app.command = mock_app.action = _noop_decorator
    mock_app.client = AsyncMock()
    web = AsyncMock()
    web.auth_test = AsyncMock(return_value={
        "user_id": "U_BOT", "user": "testbot", "team_id": TEAM, "team": "TestTeam"})
    return [
        patch.object(_slack_mod, "AsyncApp", return_value=mock_app),
        patch.object(_slack_mod, "AsyncWebClient", return_value=web),
        patch.object(_slack_mod, "AsyncSocketModeHandler", FakeHandler),
        patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}),
        patch("gateway.status.acquire_scoped_lock", return_value=(True, None)),
        patch("gateway.status_scoped_locks.release_scoped_lock"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_reconnect", [False, True])
async def test_connect_fires_platform_connected(is_reconnect):
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    with contextlib.ExitStack() as stack, _hooks(HookRecorder()) as rec:
        for p in _connect_patches():
            stack.enter_context(p)
        try:
            assert await a.connect(is_reconnect=is_reconnect) is True
        finally:
            await a.disconnect()
    assert rec.of("platform_connected") == [
        {"platform": "slack", "adapter": a, "is_reconnect": is_reconnect}]


@pytest.mark.asyncio
async def test_connect_survives_failing_platform_connected_hook():
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    with contextlib.ExitStack() as stack:
        for p in _connect_patches():
            stack.enter_context(p)
        stack.enter_context(patch("hermes_cli.lifecycle.hook_known_unused", return_value=False))
        stack.enter_context(patch("hermes_cli.lifecycle.has_hook", return_value=True))
        stack.enter_context(patch("hermes_cli.lifecycle.ainvoke_hook",
                                  AsyncMock(side_effect=RuntimeError("boom"))))
        try:
            assert await a.connect() is True
        finally:
            await a.disconnect()


# --- platform_inbound_event: message ----------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("results", [[], [None], [{"action": "allow"}]])
async def test_message_passthrough(adapter, results):
    with _hooks(HookRecorder(results)) as rec:
        await adapter._handle_slack_message(_dm(), {"team_id": TEAM})
    (call,) = rec.of("platform_inbound_event")
    assert call["platform"] == "slack" and call["kind"] == "message"
    assert call["event"]["ts"] == "100.000" and call["adapter"] is adapter
    assert callable(call["dispatch"]) and call["removed"] is False
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_message_drop_skips_prefilter_and_dedup(adapter):
    adapter._prefilter_inbound = AsyncMock()
    with _hooks(HookRecorder([{"action": "drop"}])):
        assert await adapter._handle_slack_message(_dm(), {"team_id": TEAM}) is None
    adapter._prefilter_inbound.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
    assert "100.000" not in adapter._processed_message_ts
    assert not adapter._dedup._seen


@pytest.mark.asyncio
async def test_hook_failure_fails_open(adapter):
    with patch("hermes_cli.lifecycle.hook_known_unused", return_value=False), \
            patch("hermes_cli.lifecycle.has_hook", return_value=True), \
            patch("hermes_cli.lifecycle.ainvoke_hook", AsyncMock(side_effect=RuntimeError("boom"))):
        await adapter._handle_slack_message(_dm(), {"team_id": TEAM})
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_defer_then_dispatch_coalesced_event_returns_turn_task(adapter):
    """A deferring plugin buffers two messages, then dispatches one coalesced event: buffered
    timestamps are released from dedup, the hook is not re-fired, and the turn task comes back."""
    release = asyncio.Event()

    async def _background(event, session_key):
        await release.wait()

    adapter._process_message_background = _background

    async def _handle(event):
        adapter._start_session_processing(event, f"agent:main:slack:dm:{CHANNEL}")

    adapter.handle_message = AsyncMock(side_effect=_handle)

    with _hooks(HookRecorder([{"action": "defer"}])) as rec:
        await adapter._handle_slack_message(_dm("100.000", "first"), {"team_id": TEAM})
        await adapter._handle_slack_message(_dm("101.000", "second"), {"team_id": TEAM})
        assert "100.000" not in adapter._processed_message_ts
        assert "101.000" not in adapter._processed_message_ts
        adapter.handle_message.assert_not_awaited()
        dispatch = rec.of("platform_inbound_event")[-1]["dispatch"]

        # A stale claim on a buffered ts (e.g. from a redelivery) must not block the dispatch.
        adapter._processed_message_ts["100.000"] = 1.0
        adapter._dedup._seen[f"{TEAM}:100.000"] = 1.0
        coalesced = _dm("101.000", "first\nsecond")
        task = await dispatch(coalesced, clear_ts=["100.000"])

        assert len(rec.of("platform_inbound_event")) == 2
    try:
        assert isinstance(task, asyncio.Task) and not task.done()
        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text.endswith("first\nsecond")
        assert "100.000" not in adapter._processed_message_ts
        assert "101.000" in adapter._processed_message_ts  # re-claimed by the real dispatch
        assert adapter.has_active_thread_turn(CHANNEL)
    finally:
        release.set()
        await task
    assert not adapter.has_active_thread_turn(CHANNEL)


@pytest.mark.asyncio
async def test_dispatch_finds_new_task_without_event_marker(adapter):
    release = asyncio.Event()
    key = f"agent:main:slack:dm:{CHANNEL}:100.000"  # a top-level DM opens its own thread session

    async def _handle(event):
        adapter._session_tasks[key] = asyncio.create_task(release.wait())

    adapter.handle_message = AsyncMock(side_effect=_handle)
    with _hooks(HookRecorder()):
        task = await adapter.dispatch_inbound_message(_dm(), {"team_id": TEAM})
    assert task is adapter._session_tasks[key]
    release.set()
    await task


@pytest.mark.asyncio
async def test_monitor_ready_event_bypasses_hook(adapter):
    event = {**_dm(), "_hermes_inbound_hook_ready": True}
    with _hooks(HookRecorder([{"action": "drop"}])) as rec:
        await adapter._handle_slack_message(event, {"team_id": TEAM})
    assert rec.of("platform_inbound_event") == []
    adapter.handle_message.assert_awaited_once()


# --- platform_inbound_event: reaction / slash command -----------------------------------------

def _reaction():
    return {"type": "reaction_added", "user": "U_HUMAN", "reaction": "eyes",
            "item": {"type": "message", "channel": CHANNEL, "ts": "100.000"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [False, True])
async def test_reaction_fires_and_drops(adapter, removed):
    adapter._slack_reaction_triggers = MagicMock(return_value=set())
    with _hooks(HookRecorder([{"action": "drop"}])) as rec:
        assert await adapter._handle_slack_reaction(_reaction(), removed=removed) is None
    (call,) = rec.of("platform_inbound_event")
    assert call["kind"] == "reaction" and call["removed"] is removed
    assert call["event"]["reaction"] == "eyes" and call["payload"] is None
    adapter._slack_reaction_triggers.assert_not_called()
    adapter.handle_message.assert_not_awaited()


def _slash():
    return {"command": "/hermes", "text": "help", "user_id": "U_HUMAN", "channel_id": CHANNEL,
            "team_id": TEAM}


@pytest.mark.asyncio
async def test_slash_command_fires_and_drops(adapter):
    adapter._remember_channel_team = MagicMock()
    with _hooks(HookRecorder([{"action": "drop"}])) as rec:
        assert await adapter._handle_slash_command(_slash()) is None
    (call,) = rec.of("platform_inbound_event")
    assert call["kind"] == "slash_command" and call["event"]["command"] == "/hermes"
    adapter._remember_channel_team.assert_not_called()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_slash_command_passthrough(adapter):
    adapter._remember_channel_team = MagicMock()
    with _hooks(HookRecorder([{"action": "allow"}])):
        await adapter._handle_slash_command(_slash())
    adapter._remember_channel_team.assert_called_once_with(CHANNEL, TEAM)


@pytest.mark.asyncio
async def test_undecided_hook_without_consumer_skips_invocation(adapter):
    """Plugins not yet loaded: ``has_hook`` (off the loop) decides there is no consumer."""
    rec = HookRecorder([{"action": "drop"}])
    with patch("hermes_cli.lifecycle.ainvoke_hook", rec), \
            patch("hermes_cli.lifecycle.hook_known_unused", return_value=False), \
            patch("hermes_cli.lifecycle.has_hook", return_value=False):
        await adapter._handle_slack_message(_dm(), {"team_id": TEAM})
    assert rec.calls == []
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_hook_known_unused_skips_invocation(adapter):
    rec = HookRecorder([{"action": "drop"}])
    with patch("hermes_cli.lifecycle.ainvoke_hook", rec), \
            patch("hermes_cli.lifecycle.hook_known_unused", return_value=True):
        await adapter._handle_slack_message(_dm(), {"team_id": TEAM})
    assert rec.calls == []
    adapter.handle_message.assert_awaited_once()


# --- helpers on bare test doubles ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_has_active_thread_turn_initialized_and_bare(adapter):
    bare = object.__new__(SlackAdapter)
    assert bare.has_active_thread_turn(CHANNEL) is False
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    for a in (adapter, bare):
        a._session_tasks = {f"agent:main:slack:thread:{CHANNEL}:200.000": task}
        assert a.has_active_thread_turn(CHANNEL, "200.000") is True
        assert a.has_active_thread_turn(CHANNEL) is True
        assert a.has_active_thread_turn(CHANNEL, "999.000") is False
        assert a.has_active_thread_turn("C_OTHER") is False
        assert a.has_active_thread_turn("") is False
    release.set()
    await task
    assert adapter.has_active_thread_turn(CHANNEL, "200.000") is False


def test_clear_processed_dedup_initialized(adapter):
    adapter._processed_message_ts.update({"1.0": 1.0, "2.0": 1.0, "3.0": 1.0})
    adapter._dedup._seen.update({f"{TEAM}:1.0": 1.0, "2.0": 1.0, f"{TEAM}:3.0": 1.0})
    adapter.clear_processed_dedup(["1.0", "2.0", "", None], team_id=TEAM, channel_id=CHANNEL)
    assert set(adapter._processed_message_ts) == {"3.0"}
    assert set(adapter._dedup._seen) == {f"{TEAM}:3.0"}


@pytest.mark.parametrize("container", [dict, set, list])
def test_clear_processed_dedup_bare_double(container):
    bare = object.__new__(SlackAdapter)
    keys = [(TEAM, CHANNEL, "1.0"), (CHANNEL, "1.0"), "1.0", "keep"]
    bare._processed_message_ts = dict.fromkeys(keys, 1.0) if container is dict else container(keys)
    bare.clear_processed_dedup(["1.0"], team_id=TEAM, channel_id=CHANNEL)
    assert list(bare._processed_message_ts) == ["keep"]
    bare_without_state = object.__new__(SlackAdapter)
    bare_without_state.clear_processed_dedup(["1.0"])  # no maps: a no-op, not an error
