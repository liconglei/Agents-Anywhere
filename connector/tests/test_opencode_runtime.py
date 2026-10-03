"""Tests for the OpenCode runtime adapter (provider config, timeline, runtime).

All tests run against fakes — no live ``opencode serve`` required.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import pytest

from connector.runtime_protocol import (
    CAPABILITY_SESSION_SEND_MESSAGE,
    RuntimeConfig,
    RuntimeInvalidRequestError,
    RuntimeUnavailableError,
)
from connector.runtimes.opencode import provider_config, serve_process, timeline
from connector.runtimes.opencode.pending_messages import (
    OpenCodePendingClientMessageRegistry,
)
from connector.runtimes.opencode.provider import OpenCodeProvider
from connector.runtimes.opencode.runtime import OpenCodeRuntime
from connector.runtimes.opencode.sdk.client import (
    OpenCodeClient,
    OpenCodeClientError,
    append_directory_param,
)

# --------------------------------------------------------------------------- #
# provider config
# --------------------------------------------------------------------------- #


def test_config_schema_defaults() -> None:
    defaults = provider_config.default_config_values()
    assert defaults["serverUrl"] == provider_config.DEFAULT_SERVER_URL
    assert defaults["requestTimeoutSeconds"] == provider_config.DEFAULT_REQUEST_TIMEOUT_S


def test_normalized_config_values_trims_url() -> None:
    values = provider_config.normalized_config_values(
        {"serverUrl": "http://127.0.0.1:4096/"}
    )
    assert values["serverUrl"] == "http://127.0.0.1:4096"


def test_normalized_config_values_rejects_out_of_range_timeout() -> None:
    with pytest.raises(RuntimeInvalidRequestError):
        provider_config.normalized_config_values({"requestTimeoutSeconds": 99999})


def test_normalized_config_values_rejects_bad_url() -> None:
    with pytest.raises(RuntimeInvalidRequestError):
        provider_config.normalized_config_values({"serverUrl": "not-a-url"})


def test_validate_config_round_trips_through_provider() -> None:
    async def run() -> None:
        provider = OpenCodeProvider()
        config = await provider.validate_config({"serverUrl": "http://127.0.0.1:4096"})
        assert config.runtime == "opencode"
        assert config.values["serverUrl"] == "http://127.0.0.1:4096"

    asyncio.run(run())


def test_capabilities_declare_core_surface() -> None:
    caps = provider_config.opencode_capabilities()
    assert caps["modelCatalog"] is True
    assert caps["sessionDiscovery"] is True
    assert caps["startTurn"] is True
    assert caps["interruptTurn"] is True


# --------------------------------------------------------------------------- #
# timeline mapping
# --------------------------------------------------------------------------- #

USER_MSG = {
    "info": {
        "id": "msg_user_1",
        "role": "user",
        "sessionID": "ses_x",
        "time": {"created": 1_000},
    },
    "parts": [{"type": "text", "text": "hello", "time": {"created": 1_000}}],
}

ASSISTANT_MSG = {
    "info": {
        "id": "msg_asst_1",
        "role": "assistant",
        "sessionID": "ses_x",
        "parentID": "msg_user_1",
        "time": {"created": 2_000, "completed": 3_000},
    },
    "parts": [
        {"type": "reasoning", "text": "thinking..."},
        {"type": "step-start"},
        {"type": "tool", "callID": "call_1", "tool": "bash", "state": {
            "status": "completed",
            "input": {"command": "ls"},
            "output": "file.txt",
            "title": "Run ls",
        }},
        {"type": "text", "text": "hi there", "time": {"created": 2_500}},
        {"type": "step-finish"},
    ],
}


def test_map_messages_user_and_assistant() -> None:
    items = timeline.map_messages_to_timeline("ses_x", [USER_MSG, ASSISTANT_MSG])
    kinds = [(item.type, item.role) for item in items]
    assert ("message", "user") in kinds
    assert ("message", "assistant") in kinds
    # reasoning -> system item, tool -> tool item, step-* dropped
    assert ("system", "system") in kinds
    assert ("tool", "tool") in kinds
    assert len(items) == 4


def test_map_messages_tool_status_mapping() -> None:
    items = timeline.map_messages_to_timeline("ses_x", [USER_MSG, ASSISTANT_MSG])
    tool_items = [item for item in items if item.type == "tool"]
    assert len(tool_items) == 1
    assert tool_items[0].status == "done"
    assert tool_items[0].content.title == "bash"
    assert tool_items[0].content.output == "file.txt"

    running = {
        "info": dict(ASSISTANT_MSG["info"]),
        "parts": [{
            "type": "tool",
            "callID": "call_2",
            "tool": "bash",
            "state": {"status": "running", "input": {}, "title": "running tool"},
        }],
    }
    items = timeline.map_messages_to_timeline("ses_x", [running])
    assert [i.status for i in items if i.type == "tool"] == ["running"]


def test_derive_session_status() -> None:
    # last message is a completed assistant -> idle
    assert timeline.derive_session_status([USER_MSG, ASSISTANT_MSG]) == "idle"
    # in-flight assistant (no completed time) -> running
    inflight = {
        "info": {
            "id": "msg_asst_2",
            "role": "assistant",
            "sessionID": "ses_x",
            "time": {"created": 2_000},
        },
        "parts": [],
    }
    assert timeline.derive_session_status([USER_MSG, inflight]) == "running"
    # no messages -> idle
    assert timeline.derive_session_status([]) == "idle"


# --------------------------------------------------------------------------- #
# runtime with a fake client
# --------------------------------------------------------------------------- #


class FakeClient:
    """Mimics the OpenCodeClient surface used by the runtime."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.status: dict[str, dict[str, str]] = {}
        self.closed = False
        self.prompted: list[tuple[str, str, dict[str, str] | None, str | None]] = []
        self.created_directories: list[tuple[str, str | None]] = []
        self.aborted: list[str] = []
        self.api_key: str | None = None
        self.api_user: str | None = None
        self._events: list[dict[str, Any]] = []

    def set_credentials(self, api_user: str, api_key: str) -> None:
        if api_key:
            self.api_user = api_user
            self.api_key = api_key

    async def health(self) -> dict[str, Any]:
        return {"version": "1.18.34"}

    async def list_sessions(self) -> list[dict[str, Any]]:
        return list(self.sessions.values())

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return self.sessions[session_id]

    async def list_messages(
        self, session_id: str, directory: str | None = None
    ) -> list[dict[str, Any]]:
        return self.messages.get(session_id, [])

    async def create_session(
        self, title: str | None = None, directory: str | None = None
    ) -> dict[str, Any]:
        sid = f"ses_new_{len(self.sessions)}"
        session = {
            "id": sid,
            "title": title,
            "directory": directory or "/work",
            "time": {"created": 100, "updated": 100},
        }
        self.sessions[sid] = session
        self.messages[sid] = []
        self.created_directories.append((sid, directory))
        return session

    async def prompt_async(
        self,
        session_id: str,
        content: str,
        model: dict[str, str] | None = None,
        agent: str | None = None,
        directory: str | None = None,
    ) -> None:
        self.prompted.append((session_id, content, model, directory))
        self.status[session_id] = {"type": "busy"}

    async def abort(self, session_id: str, directory: str | None = None) -> None:
        self.aborted.append(session_id)
        self.status[session_id] = {"type": "idle"}

    async def session_status_map(self, directory: str | None = None) -> dict[str, Any]:
        return self.status

    async def provider_overview(self) -> dict[str, Any]:
        return {
            "connected": ["local"],
            "all": [
                {
                    "id": "local",
                    "name": "Local",
                    "models": {
                        "nemotron35-dspark": {
                            "id": "nemotron35-dspark",
                            "name": "Nemotron 35 DSpark",
                            "limit": {"context": 262144, "output": 32768},
                        }
                    },
                }
            ],
        }

    def push_event(self, event: dict[str, Any]) -> None:
        self._events.append(event)

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            if self._events:
                yield self._events.pop(0)
            else:
                await asyncio.sleep(0.01)

    async def close(self) -> None:
        self.closed = True


class FakeHost:
    def __init__(self) -> None:
        self.timeline_syncs: list[tuple[str, int]] = []
        self.meta_upserts: list[str] = []
        self.state_updates: list[tuple[str, str]] = []
        self.turn_ended: list[tuple[str, str]] = []
        self.health: list[tuple[str, dict[str, Any] | None]] = []

    async def timeline_sync(
        self, session_id: str, runtime: str, items: tuple, external_session_id: str | None = None, complete: bool = False
    ) -> None:
        self.timeline_syncs.append((session_id, len(items)))

    async def session_meta_upsert(
        self, session_id: str, runtime: str, external_session_id: str | None = None, **_: Any
    ) -> None:
        self.meta_upserts.append(session_id)

    async def session_state_update(
        self, session_id: str, runtime: str, status: str, external_session_id: str | None = None
    ) -> None:
        self.state_updates.append((session_id, status))

    async def session_turn_ended(
        self, session_id: str, runtime: str, external_session_id: str | None = None, outcome: str = "completed"
    ) -> None:
        self.turn_ended.append((session_id, outcome))

    async def runtime_health_update(self, status: str, detail: dict[str, Any] | None = None) -> None:
        self.health.append((status, detail))


def _make_runtime(
    client: FakeClient, host: FakeHost, values: dict[str, Any] | None = None
) -> OpenCodeRuntime:
    provider = OpenCodeProvider()

    async def run() -> OpenCodeRuntime:
        config = await provider.validate_config(values or {"serverUrl": "http://127.0.0.1:4096"})
        return OpenCodeRuntime(config=config, host=host, client=client)

    return asyncio.run(run())


def test_runtime_identity_from_health() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        await runtime.start()
        try:
            assert runtime.identity.runtime_version == "1.18.34"
            assert runtime.identity.runtime == "opencode"
        finally:
            await runtime.stop()

    asyncio.run(run())


def test_default_client_streams_with_provider_timeout() -> None:
    """The SSE stream must not inherit the short REST request timeout."""
    provider = OpenCodeProvider()

    async def run() -> None:
        config = await provider.validate_config({"serverUrl": "http://127.0.0.1:4096"})
        runtime = OpenCodeRuntime(config=config, host=FakeHost())
        assert runtime._client.stream_timeout == provider_config.DEFAULT_STREAM_TIMEOUT_S
        assert runtime._client.timeout == provider_config.DEFAULT_REQUEST_TIMEOUT_S

    asyncio.run(run())


def test_client_sends_http_basic_credentials() -> None:
    """serve expects Basic auth (username default "opencode"), not Bearer."""
    import base64

    def expected(user: str, key: str) -> str:
        return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()

    async def run() -> None:
        client = OpenCodeClient("http://127.0.0.1:4096", "pw", api_user="lcl")
        http = await client._client()
        assert http.headers["Authorization"] == expected("lcl", "pw")
        await client.close()

        quiet = OpenCodeClient("http://127.0.0.1:4096")
        http2 = await quiet._client()
        assert "Authorization" not in http2.headers
        quiet.set_credentials("lcl", "new")
        assert http2.headers["Authorization"] == expected("lcl", "new")
        await quiet.close()

    asyncio.run(run())


class _FakeSSEResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""


class _FakeSSEContext:
    def __init__(self, response: _FakeSSEResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeSSEResponse:
        return self._response

    async def __aexit__(self, *_: object) -> None:
        return None


class _FakeStreamingHTTP:
    def __init__(self, response: _FakeSSEResponse) -> None:
        self._response = response
        self.requested_url: str | None = None

    def stream(self, _method: str, url: str, **_kwargs: Any) -> _FakeSSEContext:
        self.requested_url = url
        return _FakeSSEContext(self._response)


def test_client_stream_events_unwraps_global_frames() -> None:
    """``/global/event`` wraps every event and replays ``sync`` duplicates.

    ``/event`` is instance-scoped (only the serve cwd's project), so sessions
    in any other directory never surface their events there — the platform
    session then never leaves "running" and the send button stays disabled.
    The client must subscribe to the global stream, unwrap the envelope, and
    drop sync replay frames.
    """

    client = OpenCodeClient("http://127.0.0.1:4096")
    http = _FakeStreamingHTTP(
        _FakeSSEResponse(
            [
                'data: {"payload": {"type": "server.connected", "properties": {}}}',
                'data: {"directory": "d", "project": "p", "payload": {"type": "sync", "syncEvent": {}}}',
                'data: {"directory": "d", "project": "p", "payload": {"type": "session.idle", "properties": {"sessionID": "ses_1"}}}',
                'data: {"type": "session.status", "properties": {"sessionID": "ses_2", "status": {"type": "idle"}}}',
                ": keep-alive comment",
                "event: message",
            ]
        )
    )
    client._http = http  # type: ignore[assignment]

    async def run() -> list[dict[str, Any]]:
        return [event async for event in client.stream_events()]

    events = asyncio.run(run())
    assert [event["type"] for event in events] == [
        "server.connected",
        "session.idle",
        "session.status",
    ]
    assert http.requested_url == "/global/event"


def test_probe_url_sends_http_basic_and_flags_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credentialed serve answers even /global/health with 401; the probe
    must authenticate like the client and not report that as "unreachable"."""
    import base64

    import httpx

    from connector.runtimes.opencode import provider as provider_module

    captured: dict[str, Any] = {}

    class _Resp:
        def __init__(self, status_code: int, body: dict[str, Any] | None = None) -> None:
            self.status_code = status_code
            self.content = b"{}" if body is not None else b""

        def json(self) -> dict[str, Any]:
            return {"version": "1.18.34"}

    def fake_get(url: str, headers: dict[str, str] | None = None, timeout: float = 0.0):
        captured["url"] = url
        captured["headers"] = dict(headers or {})
        return _Resp(captured.get("_next_status", 200), {"version": "1.18.34"})

    monkeypatch.setattr(httpx, "get", fake_get)

    token = base64.b64encode(b"lcl:pw").decode()
    result = provider_module._probe_url("http://127.0.0.1:4096", api_user="lcl", api_key="pw")
    assert result == {"available": True, "version": "1.18.34"}
    assert captured["headers"]["Authorization"] == f"Basic {token}"

    captured["_next_status"] = 401
    denied = provider_module._probe_url("http://127.0.0.1:4096")
    assert denied["available"] is False
    assert denied["status"] == 401
    assert denied["reason"] == "authentication required"


def test_default_probe_adopts_env_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-started serve inherits OPENCODE_SERVER_PASSWORD; discovery must reuse it."""
    import base64

    import httpx

    from connector.runtimes.opencode import provider as provider_module

    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "secret")
    monkeypatch.setenv("OPENCODE_SERVER_USERNAME", "licon")

    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict[str, Any]:
            return {"version": "1.18.34"}

    def fake_get(url: str, headers: dict[str, str] | None = None, timeout: float = 0.0):
        captured["headers"] = dict(headers or {})
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)

    result = provider_module.default_probe()
    assert result["available"] is True
    token = base64.b64encode(b"licon:secret").decode()
    assert captured["headers"]["Authorization"] == f"Basic {token}"


def test_start_raises_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    class DeadClient(FakeClient):
        async def health(self) -> dict[str, Any]:
            raise OpenCodeClientError("connection refused", status_code=None)

    # No resolvable opencode binary -> auto-start fails fast, nothing spawned.
    def _missing(*_a: Any, **_k: Any) -> list[str]:
        raise RuntimeError("opencode executable not found on PATH (and no npx fallback)")

    monkeypatch.setattr(serve_process, "resolve_serve_command", _missing)
    runtime = _make_runtime(DeadClient(), FakeHost())

    async def run() -> None:
        with pytest.raises(RuntimeUnavailableError):
            await runtime.start()

    asyncio.run(run())


def test_list_sessions_maps_meta() -> None:
    client, host = FakeClient(), FakeHost()
    client.sessions["ses_a"] = {
        "id": "ses_a",
        "title": "demo",
        "directory": "/work",
        "time": {"created": 1, "updated": 2},
        "model": {"providerID": "local", "id": "nemotron35-dspark"},
    }
    runtime = _make_runtime(client, host)

    async def run() -> None:
        metas = await runtime.list_sessions()
        assert len(metas) == 1
        assert metas[0].session_id == "ses_a"
        assert metas[0].external_session_id == "ses_a"
        assert metas[0].title == "demo"

    asyncio.run(run())


def test_snapshot_returns_platform_items() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        client.messages["ses_a"] = [USER_MSG, ASSISTANT_MSG]
        snap = await runtime.get_session_snapshot("ses_a", "ses_a")
        assert snap.complete is True
        assert len(snap.items) == 4
        assert snap.items[0].type == "message"
        assert snap.items[0].content.get("text") == "hello"

    asyncio.run(run())


def test_create_and_start_session_binds_ids() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session(
            "plat_1", "do it", selections={"model": "local/nemotron35-dspark"}
        )
        assert result.ok is True
        native = result.result["externalSessionId"]
        assert native.startswith("ses_new_")
        # turn was started with the model ref
        assert client.prompted == [
            ("ses_new_0", "do it", {"providerID": "local", "modelID": "nemotron35-dspark"}, "/work")
        ]
        assert client.created_directories == [("ses_new_0", None)]
        # meta upsert under the platform id
        assert "plat_1" in host.meta_upserts
        # interrupt via platform id resolves to the native id
        interrupt = await runtime.interrupt_session("plat_1")
        assert interrupt.ok is True
        assert client.aborted == [native]

    asyncio.run(run())


def test_start_turn_without_selections_sends_no_model() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        client.sessions["ses_b"] = {"id": "ses_b"}
        result = await runtime.start_turn("ses_b", "ses_b", "plain prompt")
        assert result.ok is True
        assert client.prompted == [("ses_b", "plain prompt", None, None)]

    asyncio.run(run())


def test_continue_session_after_create_via_platform_id() -> None:
    """The desktop app continues a created session by platform id alone.

    The server forwards ``externalSessionId`` when it is stored, but after a
    connector restart the in-memory mapping is rebuilt from zero, so the
    platform-id-only path must also resolve correctly.
    """
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session(
            "plat_continue", "first prompt", selections=None
        )
        assert result.ok is True
        native = result.result["externalSessionId"]

        # Server-side continuation with the stored external id.
        cont = await runtime.start_turn("plat_continue", native, "second prompt")
        assert cont.ok is True, cont.message
        assert client.prompted[-1] == (native, "second prompt", None, "/work")

        # Connector-restart path: mapping lost, platform id must self-resolve
        # through the identity fallback for discovered-style ids.
        assert runtime._platform_to_native["plat_continue"] == native

        # A discovered session (platform id == native id) resolves itself.
        client.sessions["ses_disc"] = {"id": "ses_disc"}
        disc = await runtime.start_turn("ses_disc", "ses_disc", "discovered turn")
        assert disc.ok is True
        assert client.prompted[-1] == ("ses_disc", "discovered turn", None, None)

    asyncio.run(run())


def test_get_session_state_busy() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        client.status["ses_c"] = {"type": "busy"}
        state = await runtime.get_session_state("ses_c", "ses_c")
        assert state.status == "running"
        client.status["ses_c"] = {"type": "idle"}
        state = await runtime.get_session_state("ses_c", "ses_c")
        assert state.status == "idle"

    asyncio.run(run())


def test_get_session_state_reasserts_idle_once() -> None:
    """Opening a session re-pushes idle once, unlocking stale running states."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        client.status["ses_h"] = {"type": "idle"}
        state = await runtime.get_session_state("plat_h", "ses_h")
        assert state.status == "idle"
        assert ("plat_h", "idle") in host.state_updates

        # only one push until a running transition is seen again
        await runtime.get_session_state("plat_h", "ses_h")
        assert host.state_updates.count(("plat_h", "idle")) == 1

        client.status["ses_h"] = {"type": "busy"}
        await runtime.get_session_state("plat_h", "ses_h")
        client.status["ses_h"] = {"type": "idle"}
        await runtime.get_session_state("plat_h", "ses_h")
        assert host.state_updates.count(("plat_h", "idle")) == 2

    asyncio.run(run())


def test_event_session_idle_publishes_and_ends_turn() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_d", "hi", selections=None)
        native = result.result["externalSessionId"]
        client.messages[native] = [USER_MSG, ASSISTANT_MSG]

        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        assert ("plat_d", "completed") in host.turn_ended
        assert ("plat_d", "idle") in host.state_updates
        assert ("plat_d", 4) in host.timeline_syncs

    asyncio.run(run())


def test_event_busy_ignored_without_active_turn() -> None:
    """A stray/replayed busy for an untracked session must not force "running".

    ``opencode serve`` replays session status on every SSE reconnect, so a busy
    can arrive after the turn's idle was already reported. Reflecting it would
    leave the platform session stuck in "running" (nothing reconciles untracked
    sessions), disabling the client send button after the turn showed as ended.
    """

    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        await runtime._handle_event(
            {"type": "session.status", "properties": {"sessionID": "ses_e", "status": {"type": "busy"}}}
        )
        assert ("ses_e", "running") not in host.state_updates

    asyncio.run(run())


def test_event_busy_updates_state_for_active_turn() -> None:
    """A busy belonging to a tracked active turn still marks the session running."""

    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_e", "hi", selections=None)
        native = result.result["externalSessionId"]
        host.state_updates.clear()

        await runtime._handle_event(
            {"type": "session.status", "properties": {"sessionID": native, "status": {"type": "busy"}}}
        )
        assert ("plat_e", "running") in host.state_updates

    asyncio.run(run())


def test_event_status_idle_ends_tracked_turn() -> None:
    """``session.status idle`` precedes ``session.idle`` on the wire; ending
    the turn on it unlocks the client a beat earlier, and the later duplicate
    is a no-op rather than a second turn-end."""

    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_si", "hi", selections=None)
        native = result.result["externalSessionId"]
        client.messages[native] = [USER_MSG, ASSISTANT_MSG]

        await runtime._handle_event(
            {
                "type": "session.status",
                "properties": {"sessionID": native, "status": {"type": "idle"}},
            }
        )
        assert ("plat_si", "completed") in host.turn_ended
        assert ("plat_si", "idle") in host.state_updates

        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        assert host.turn_ended.count(("plat_si", "completed")) == 1

    asyncio.run(run())


def test_event_status_idle_ignored_for_untracked_session() -> None:
    """A replayed idle for a session this process never started must not
    fabricate a turn-end or push state."""

    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        await runtime._handle_event(
            {
                "type": "session.status",
                "properties": {"sessionID": "ses_untracked", "status": {"type": "idle"}},
            }
        )
        assert host.turn_ended == []
        assert host.state_updates == []

    asyncio.run(run())


def test_session_capabilities_carry_platform_session_id() -> None:
    """``session.send_message`` must be a session-scoped entry that carries the
    platform session id: the Server matches session-scoped capabilities by
    ``(runtime, scope, sessionId, runtimeId)``, and an entry without a session
    id (or one declared only on the runtime-scoped set) matches nothing —
    the send button then stays disabled forever after the first turn."""

    runtime = _make_runtime(FakeClient(), FakeHost())

    async def run() -> None:
        caps = await runtime.get_session_capabilities("sess_caps", "ses_native_caps")
        by_id = {cap.capability_id: cap for cap in caps.capabilities}
        send = by_id[CAPABILITY_SESSION_SEND_MESSAGE]
        assert send.scope == "session"
        assert send.session_id == "sess_caps"
        assert send.supported is True
        assert send.available is True

    asyncio.run(run())


def test_start_turn_rebuilds_binding_after_restart() -> None:
    """After a restart the in-memory maps are empty; a Server-driven call that
    carries both ids must rebuild them so every push addresses the platform
    session instead of an unresolvable native id."""

    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.start_turn("plat_r", "ses_native_r", "hi", selections=None)
        assert result.ok
        assert runtime._platform_id("ses_native_r") == "plat_r"
        assert ("plat_r", "running") in host.state_updates

    asyncio.run(run())


def test_active_turn_tracked_until_idle() -> None:
    """A started turn is tracked as active and cleared by session.idle."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_t", "hi", selections=None)
        native = result.result["externalSessionId"]
        assert native in runtime._active_turns

        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        assert native not in runtime._active_turns
        assert ("plat_t", "completed") in host.turn_ended

    asyncio.run(run())


def test_lost_idle_reconciled_against_status_map() -> None:
    """A missed session.idle is recovered by reconciling against /session/status.

    This is the failure mode behind the 'send button stuck disabled after one
    turn' report: if the SSE stream drops mid-turn, the idle event is lost and
    the platform session would stay 'running' forever.
    """
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_r", "hi", selections=None)
        native = result.result["externalSessionId"]
        assert native in runtime._active_turns

        # opencode finished the turn, but the idle event was lost
        # (the status map no longer lists the session as busy).
        del client.status[native]
        await runtime._reconcile_active_turns()
        assert native not in runtime._active_turns
        assert ("plat_r", "completed") in host.turn_ended
        assert ("plat_r", 0) in host.timeline_syncs

        # a genuinely busy turn is NOT reconciled away
        client.status[native] = {"type": "busy"}
        runtime._active_turns.add(native)
        await runtime._reconcile_active_turns()
        assert native in runtime._active_turns

    asyncio.run(run())


def test_reconcile_queries_session_directory_scope() -> None:
    """Reconcile must query /session/status in the session's own directory.

    ``/session/status`` is scoped by ``?directory=``. An unscoped query omits a
    workspace-bound session, so it looked idle and the turn was "recovered"
    seconds after it started — before ``opencode serve`` produced any reply —
    leaving the answer blank. Codex ends turns only on a terminal event, never on
    a missing status, so scoping the poll is both correct and consistent.
    """
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    scope: dict[str, str | None] = {"dir": None}

    async def scoped_status_map(directory: str | None = None) -> dict[str, Any]:
        if directory is not None and directory == scope["dir"]:
            return dict(client.status)
        return {}

    client.session_status_map = scoped_status_map  # type: ignore[method-assign]

    async def run() -> None:
        result = await runtime.create_and_start_session(
            "plat_scope", "hi", cwd="C:\\work\\proj", selections=None
        )
        native = result.result["externalSessionId"]
        scope["dir"] = runtime._directory_for(native)
        assert scope["dir"]

        # serve reports it busy in its own scope: reconcile must not end the turn.
        client.status[native] = {"type": "busy"}
        await runtime._reconcile_active_turns()
        assert native in runtime._active_turns
        assert ("plat_scope", "completed") not in host.turn_ended

        # once serve reports it idle in that scope, reconcile recovers the end.
        client.status[native] = {"type": "idle"}
        await runtime._reconcile_active_turns()
        assert native not in runtime._active_turns
        assert ("plat_scope", "completed") in host.turn_ended

    asyncio.run(run())


def test_abort_reports_interrupted_once_without_duplicate_idle() -> None:
    """Abort ends the tracked turn as interrupted; the later idle does not re-report."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_i", "hi", selections=None)
        native = result.result["externalSessionId"]
        ir = await runtime.interrupt_session("plat_i")
        assert ir.ok
        # FakeClient.abort flips the status to idle, so the inline reconcile
        # (or a watchdog pass) ends the turn immediately.
        assert ("plat_i", "interrupted") in host.turn_ended

        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        ends = [t for t in host.turn_ended if t[0] == "plat_i"]
        assert ends == [("plat_i", "interrupted")]  # no duplicate completed end
        assert native not in runtime._active_turns
        assert native not in runtime._aborted_turns

        # a new turn clears the memo, so its idle reports completed again
        await runtime.start_turn("plat_i", None, "more")
        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        assert host.turn_ended[-1] == ("plat_i", "completed")

    asyncio.run(run())


def test_reconcile_skipped_when_no_active_turns() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        assert runtime._active_turns == set()
        await runtime._reconcile_active_turns()  # must not call the status map
        assert host.turn_ended == []

    asyncio.run(run())


def test_turn_end_retried_after_host_delivery_failure() -> None:
    """A turn-end the host refuses (backend blip) is queued and retried.

    Swallowing it would leave the platform session stuck in "running": the
    server never forwards follow-up messages and the client cannot interact
    again even after reopening the history session.
    """

    class FlakyHost(FakeHost):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 2

        async def session_turn_ended(
            self,
            session_id: str,
            runtime: str,
            external_session_id: str | None = None,
            outcome: str = "completed",
        ) -> None:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("backend websocket closed")
            self.turn_ended.append((session_id, outcome))

    client, host = FakeClient(), FlakyHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_f", "hi", selections=None)
        native = result.result["externalSessionId"]

        # idle must not raise even though the first turn-end delivery fails
        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        assert host.turn_ended == []
        assert runtime._unended_turns[native] == "completed"

        await runtime._flush_unended_turns()
        assert host.turn_ended == [("plat_f", "completed")]
        assert ("plat_f", "idle") in host.state_updates
        assert native not in runtime._unended_turns

    asyncio.run(run())


def test_watchdog_flush_retries_unended_turns() -> None:
    """Queued turn-ends are retried, so recovery is automatic once backend returns."""

    class DownHost(FakeHost):
        def __init__(self) -> None:
            super().__init__()
            self.down = True

        async def session_turn_ended(
            self,
            session_id: str,
            runtime: str,
            external_session_id: str | None = None,
            outcome: str = "completed",
        ) -> None:
            if self.down:
                raise ConnectionError("offline")
            self.turn_ended.append((session_id, outcome))

    client, host = FakeClient(), DownHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_w", "hi", selections=None)
        native = result.result["externalSessionId"]
        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        assert runtime._unended_turns.get(native) == "completed"

        host.down = False
        await runtime._flush_unended_turns()
        assert ("plat_w", "completed") in host.turn_ended
        assert native not in runtime._unended_turns

    asyncio.run(run())


def test_watchdog_task_lifecycle() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        await runtime.start()
        assert runtime._event_task is not None
        assert runtime._watchdog_task is not None
        await runtime.stop()
        assert runtime._event_task is None
        assert runtime._watchdog_task is None
        assert client.closed is True

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# auto-start of a local `opencode serve`
# --------------------------------------------------------------------------- #


class _FakeStream:
    """StreamReader-like: yields queued lines, then EOF."""

    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeProc:
    """A live child; wait() returns once terminate/kill/crash set returncode."""

    def __init__(self, stdout: _FakeStream | None = None) -> None:
        self.returncode: int | None = None
        self.pid = 42424
        self.terminated = False
        self.killed = False
        self.stdout = stdout or _FakeStream()
        self.stderr = None

    def terminate(self) -> None:
        self.terminated = True
        if self.returncode is None:
            self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9

    def crash(self, code: int, *lines: str) -> None:
        self.stdout._lines.extend(line.encode() + b"\n" for line in lines)
        self.returncode = code

    async def wait(self) -> int:
        while self.returncode is None:
            await asyncio.sleep(0.005)
        return self.returncode


class _ExitedProc(_FakeProc):
    def __init__(self, message: str) -> None:
        super().__init__(stdout=_FakeStream([message.encode() + b"\n"]))
        self.returncode = 1


_REAL_IS_LOCK_HELD = serve_process.is_serve_log_lock_held


@pytest.fixture(autouse=True)
def _no_real_log_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must never probe the developer's real opencode.log."""

    monkeypatch.setattr(serve_process, "is_serve_log_lock_held", lambda environment: False)


def test_describe_exit_code_decodes_fatal_ntstatus() -> None:
    described = serve_process.describe_exit_code(3221226505)
    assert "0xC0000409" in described
    assert serve_process.describe_exit_code(1) == "code=1"
    assert serve_process.describe_exit_code(None) == "code=unknown"


def test_is_serve_log_lock_held_false_off_win32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve_process, "is_serve_log_lock_held", _REAL_IS_LOCK_HELD)
    monkeypatch.setattr(sys, "platform", "linux")
    assert _REAL_IS_LOCK_HELD({"XDG_DATA_HOME": "/tmp/does-not-matter"}) is False


@pytest.mark.skipif(sys.platform != "win32", reason="share-mode probe is Windows-only")
def test_is_serve_log_lock_held_detects_exclusive_holder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import ctypes
    from ctypes import wintypes

    monkeypatch.setattr(serve_process, "is_serve_log_lock_held", _REAL_IS_LOCK_HELD)
    log_dir = tmp_path / "opencode" / "log"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "opencode.log"
    log_file.write_text("x")
    env = {"XDG_DATA_HOME": str(tmp_path)}
    assert serve_process.opencode_log_path(env) == str(log_file)
    assert _REAL_IS_LOCK_HELD(env) is False

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(str(log_file), 0x80000000, 0, None, 3, 0, None)
    assert handle is not None and handle != ctypes.c_void_p(-1).value
    try:
        assert _REAL_IS_LOCK_HELD(env) is True
    finally:
        kernel32.CloseHandle(handle)
    assert _REAL_IS_LOCK_HELD(env) is False


def test_wait_healthy_message_decodes_fatal_exit_code() -> None:
    async def run() -> None:
        proc = _FakeProc()
        proc.crash(3221226505)
        serve = serve_process.ServeProcess()
        serve.process = proc

        async def dead() -> dict:
            raise OpenCodeClientError("connection refused")

        with pytest.raises(RuntimeError, match="0xC0000409"):
            await serve.wait_healthy(dead, OpenCodeClientError, 0.01)

    asyncio.run(run())


def _patch_serve(monkeypatch: pytest.MonkeyPatch, spawn_impl: Any) -> None:
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_impl)
    monkeypatch.setattr(serve_process, "serve_environment", dict)

    def fake_tree_kill(proc: Any) -> None:  # stand-in for taskkill /T /F
        proc.terminate()

    monkeypatch.setattr(serve_process, "_kill_process_tree", fake_tree_kill)

    async def no_listener(host: str, port: int) -> bool:
        return False

    monkeypatch.setattr(OpenCodeRuntime, "_port_listening", staticmethod(no_listener))
    monkeypatch.setattr(
        serve_process,
        "resolve_serve_command",
        lambda port, environment=None, hostname="127.0.0.1": [
            "opencode",
            "serve",
            "--hostname",
            hostname,
            "--port",
            str(port),
        ],
    )


def test_auto_start_resolution_does_not_block_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a slow synchronous version check must not freeze the
    connector event loop (heartbeats stopping = supervisor kills the RPC)."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    def slow_resolve(port: int, environment: Any = None, hostname: str = "127.0.0.1") -> list[str]:
        time.sleep(0.3)
        raise RuntimeError("opencode executable not found on PATH (and no npx fallback)")

    async def down() -> dict[str, Any]:
        raise OpenCodeClientError("connection refused")

    monkeypatch.setattr(client, "health", down)
    monkeypatch.setattr(serve_process, "resolve_serve_command", slow_resolve)

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    async def run() -> None:
        task = asyncio.create_task(ticker())
        with suppress(RuntimeError):
            await runtime._auto_start_serve("http://127.0.0.1:4096")
        task.cancel()
        assert ticks >= 4  # loop kept ticking while resolution ran in a thread

    asyncio.run(run())


def test_serve_command_prefers_opencode_exe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_process,
        "find_executable_on_path",
        lambda name, path: rf"C:\bin\{name}.cmd" if name == "opencode" else None,
    )
    monkeypatch.setattr(serve_process, "check_version_output", lambda candidate, env: None)
    assert OpenCodeRuntime._serve_command(4096) == [
        r"C:\bin\opencode.cmd", "serve", "--hostname", "127.0.0.1", "--port", "4096"
    ]


def test_serve_command_falls_back_to_npx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_process,
        "find_executable_on_path",
        lambda name, path: rf"C:\node\{name}.cmd" if name == "npx" else None,
    )
    monkeypatch.setattr(serve_process, "check_version_output", lambda candidate, env: None)
    assert OpenCodeRuntime._serve_command(5000) == [
        r"C:\node\npx.cmd", "opencode", "serve", "--hostname", "127.0.0.1", "--port", "5000"
    ]


def test_serve_command_falls_back_when_version_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_process,
        "find_executable_on_path",
        lambda name, path: rf"C:\bin\{name}.cmd",
    )
    monkeypatch.setattr(
        serve_process, "check_version_output", lambda candidate, env: "exited with code 1"
    )
    assert OpenCodeRuntime._serve_command(4096) == [
        r"C:\bin\npx.cmd", "opencode", "serve", "--hostname", "127.0.0.1", "--port", "4096"
    ]


def test_serve_command_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve_process, "find_executable_on_path", lambda name, path: None)
    with pytest.raises(RuntimeError, match="not found"):
        OpenCodeRuntime._serve_command(4096)


def test_auto_start_allowed_only_for_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    assert runtime._should_auto_start("http://127.0.0.1:4096") is True
    assert runtime._should_auto_start("http://localhost:4096") is True
    assert runtime._should_auto_start("http://nas.liconglei.top:4096") is False
    assert runtime._should_auto_start("http://10.0.0.5:4096") is False

    # disabled by config
    values = dict(runtime.config.values)
    values["autoStart"] = False
    cfg = provider_config.normalized_config_values(values)
    cfg["autoStart"] = False
    disabled = OpenCodeRuntime(RuntimeConfig("opencode", 1, cfg), host)
    assert disabled._should_auto_start("http://127.0.0.1:4096") is False


def test_auto_start_waits_until_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    state = {"calls": 0}

    async def flaky_health() -> dict:
        state["calls"] += 1
        if state["calls"] < 3:
            raise OpenCodeClientError("connection refused")
        return {"healthy": True, "version": "1.99"}

    monkeypatch.setattr(client, "health", flaky_health)
    spawned: list = []

    async def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        spawned.append((args, kwargs))
        return _FakeProc()

    _patch_serve(monkeypatch, fake_spawn)

    async def run() -> None:
        await runtime._auto_start_serve("http://127.0.0.1:4096")
        assert runtime._serve.process is not None
        assert state["calls"] == 3
        assert len(spawned) == 1
        await runtime.stop()

    asyncio.run(run())


def test_auto_start_reports_early_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def dead_health() -> dict:
        raise OpenCodeClientError("connection refused")

    monkeypatch.setattr(client, "health", dead_health)

    async def fake_spawn(*args: Any, **kwargs: Any) -> _ExitedProc:
        return _ExitedProc("fatal: boom")

    _patch_serve(monkeypatch, fake_spawn)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="exited early"):
            await runtime._auto_start_serve("http://127.0.0.1:4096")
        assert runtime._serve is None
        await runtime.stop()

    asyncio.run(run())


def test_auto_start_adopts_external_serve_on_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Port busy while our child exits -> keep the externally-started serve."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    state = {"calls": 0}

    async def booting_health() -> dict:
        state["calls"] += 1
        if state["calls"] <= 2:  # precheck + adoption poll fail, then external serve is up
            raise OpenCodeClientError("connection refused")
        return {"healthy": True, "version": "1.18"}

    monkeypatch.setattr(client, "health", booting_health)

    async def fake_spawn(*args: Any, **kwargs: Any) -> _ExitedProc:
        return _ExitedProc("listen EADDRINUSE 127.0.0.1:4096")

    _patch_serve(monkeypatch, fake_spawn)

    async def run() -> None:
        await runtime._auto_start_serve("http://127.0.0.1:4096")
        assert runtime._serve is None  # external serve adopted, nothing owned

    asyncio.run(run())


def test_auto_start_adopts_env_server_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """serve inherits OPENCODE_SERVER_PASSWORD/USERNAME -> health needs Basic auth."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    monkeypatch.setattr(
        serve_process,
        "serve_environment",
        lambda: {"OPENCODE_SERVER_PASSWORD": "pw", "OPENCODE_SERVER_USERNAME": "lcl"},
    )

    async def run() -> None:
        await runtime._auto_start_serve("http://127.0.0.1:4096")
        assert client.api_key == "pw"
        assert client.api_user == "lcl"
        assert runtime._serve is None  # precheck health passed once the key was applied

    asyncio.run(run())


def test_auto_start_reports_401_without_spawning_second_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401 on the precheck means a foreign serve is up with other credentials;
    spawning a second serve would only die on the shared log-file lock."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def denied() -> dict:
        raise OpenCodeClientError("opencode GET /global/health failed: 401", status_code=401)

    monkeypatch.setattr(client, "health", denied)

    async def no_spawn(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not spawn while a 401 serve owns the port")

    _patch_serve(monkeypatch, no_spawn)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="401"):
            await runtime._auto_start_serve("http://127.0.0.1:4096")

    asyncio.run(run())


def test_auto_start_adopts_serve_when_port_has_unlisted_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production failure: child died with 'ServeError' (no EADDRINUSE marker)
    while a foreign serve already held the port -> must adopt, not give up."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    state = {"calls": 0}

    async def booting_health() -> dict:
        state["calls"] += 1
        if state["calls"] <= 2:  # precheck + adoption poll fail, then serve answers
            raise OpenCodeClientError("connection refused")
        return {"healthy": True, "version": "1.18"}

    monkeypatch.setattr(client, "health", booting_health)

    async def fake_spawn(*args: Any, **kwargs: Any) -> _ExitedProc:
        return _ExitedProc("Error: Unexpected error\nServeError")

    _patch_serve(monkeypatch, fake_spawn)

    async def listening(host_: str, port_: int) -> bool:
        return True

    monkeypatch.setattr(OpenCodeRuntime, "_port_listening", staticmethod(listening))

    async def run() -> None:
        await runtime._auto_start_serve("http://127.0.0.1:4096")
        assert runtime._serve is None

    asyncio.run(run())


def test_auto_start_refuses_spawn_when_log_lock_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production failure: every spawn died fastfail because a live opencode
    instance held the log lock -> skip the doomed spawn, say why, spawn=never."""
    from connector.runtimes.opencode import runtime as runtime_module

    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def always_down() -> dict:
        raise OpenCodeClientError("connection refused")

    monkeypatch.setattr(client, "health", always_down)
    monkeypatch.setattr(serve_process, "is_serve_log_lock_held", lambda environment: True)
    monkeypatch.setattr(runtime_module, "PORT_ADOPTION_WAIT_S", 0.01)

    async def no_spawn(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not spawn while the single-instance log lock is held")

    _patch_serve(monkeypatch, no_spawn)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="日志锁"):
            await runtime._auto_start_serve("http://127.0.0.1:4096")
        assert runtime._serve is None

    asyncio.run(run())


def test_auto_start_adopts_serve_when_only_log_lock_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child died silently (no EADDRINUSE, no listener) while the log lock was
    held by a serve that is still booting -> adoption must still succeed."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    state = {"calls": 0}

    async def booting_health() -> dict:
        state["calls"] += 1
        if state["calls"] <= 2:  # precheck + first adoption poll fail, then serve answers
            raise OpenCodeClientError("connection refused")
        return {"healthy": True, "version": "1.18"}

    monkeypatch.setattr(client, "health", booting_health)
    lock_calls = {"n": 0}

    def lock_probe(environment: Any) -> bool:
        lock_calls["n"] += 1
        return lock_calls["n"] > 1  # free before our spawn, held after the crash

    monkeypatch.setattr(serve_process, "is_serve_log_lock_held", lock_probe)

    async def fake_spawn(*args: Any, **kwargs: Any) -> _ExitedProc:
        return _ExitedProc("Error: Unexpected error\nUnknown: FileSystem.open")

    _patch_serve(monkeypatch, fake_spawn)

    async def run() -> None:
        await runtime._auto_start_serve("http://127.0.0.1:4096")
        assert runtime._serve is None

    asyncio.run(run())


def test_start_auto_starts_local_serve_when_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported bug: connector started while serve was down -> runtime never
    became 'running'. With auto-start, start() spawns serve and recovers."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    state = {"calls": 0}

    async def down_then_up() -> dict:
        state["calls"] += 1
        if state["calls"] < 3:
            raise OpenCodeClientError("All connection attempts failed")
        return {"healthy": True, "version": "1.18"}

    monkeypatch.setattr(client, "health", down_then_up)
    proc = _FakeProc()

    async def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    _patch_serve(monkeypatch, fake_spawn)

    async def run() -> None:
        await runtime.start()  # must not raise
        assert runtime._serve.process is proc
        assert runtime.identity.runtime_version == "1.18"
        await runtime.stop()
        assert proc.terminated is True
        assert runtime._serve is None

    asyncio.run(run())


def test_start_remote_url_fails_without_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    client, host = FakeClient(), FakeHost()
    values = provider_config.default_config_values()
    values["serverUrl"] = "http://10.0.0.5:4096"
    runtime = _make_runtime(client, host, values=values)

    async def always_down() -> dict:
        raise OpenCodeClientError("connection refused")

    monkeypatch.setattr(client, "health", always_down)

    async def no_spawn(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not spawn for a remote serverUrl")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_spawn)

    async def run() -> None:
        with pytest.raises(RuntimeUnavailableError):
            await runtime.start()
        assert runtime._serve is None

    asyncio.run(run())


def test_serve_exit_reports_health_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected child exit surfaces as an error health update and the
    recovery pass marks the runtime running again once serve answers."""
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)
    state = {"calls": 0}

    async def down_twice() -> dict:
        state["calls"] += 1
        if state["calls"] <= 2:  # start() probe + auto-start pre-check
            raise OpenCodeClientError("connection refused")
        return {"healthy": True, "version": "1.18"}

    monkeypatch.setattr(client, "health", down_twice)
    proc = _FakeProc()

    async def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    _patch_serve(monkeypatch, fake_spawn)

    async def run() -> None:
        await runtime.start()
        assert runtime._serve is not None and runtime._serve.process is proc
        proc.crash(2, "fatal: serve died")
        for _ in range(400):
            if any(h[0] == "running" for h in host.health):
                break
            await asyncio.sleep(0.01)
        assert any(
            h[0] == "error" and (h[1] or {}).get("code") == "opencode_serve_exited"
            for h in host.health
        )
        assert any(h[0] == "running" for h in host.health)
        await runtime.stop()

    asyncio.run(run())


def test_recovery_respawns_child_while_serve_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """While serve stays unreachable, recovery keeps spawning attempts."""
    from connector.runtimes.opencode import runtime as runtime_module

    host = FakeHost()
    state = {"healthy": False}

    async def down_health() -> dict:
        if state["healthy"]:
            return {"healthy": True, "version": "1.18"}
        raise OpenCodeClientError("connection refused")

    client = FakeClient()
    monkeypatch.setattr(client, "health", down_health)
    procs: list[_FakeProc] = []

    async def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        proc = _FakeProc()
        procs.append(proc)
        return proc

    _patch_serve(monkeypatch, fake_spawn)
    monkeypatch.setattr(runtime_module, "SERVE_RESTART_INITIAL_BACKOFF_S", 0.01)

    config = asyncio.run(
        OpenCodeProvider().validate_config({"serverUrl": "http://127.0.0.1:4096"})
    )
    runtime = OpenCodeRuntime(
        config=config, host=host, client=client, auto_start_serve_timeout_s=0.05
    )

    async def run() -> None:
        runtime._schedule_recovery("unit-respawn")
        for _ in range(200):
            if procs:
                break
            await asyncio.sleep(0.01)
        assert procs, "recovery spawned a serve child"
        state["healthy"] = True
        for _ in range(400):
            if any(h[0] == "running" for h in host.health):
                break
            await asyncio.sleep(0.01)
        assert any(h[0] == "running" for h in host.health)
        await runtime.stop()

    asyncio.run(run())


def test_stop_closes_client_and_cancels_events() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        await runtime.start()
        assert runtime._event_task is not None
        await runtime.stop()
        assert client.closed is True
        assert runtime._event_task is None

    asyncio.run(run())


def test_model_catalog_from_provider_overview() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        catalog = await runtime.list_model_catalog()
        assert len(catalog.models) == 1
        item = catalog.models[0]
        assert item.id == "local/nemotron35-dspark"
        assert "262144" in (item.description or "")

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# e2e against a live opencode serve (auto-skipped when unreachable)
# --------------------------------------------------------------------------- #


def _server_reachable() -> bool:
    import httpx

    try:
        r = httpx.get("http://127.0.0.1:4096/global/health", timeout=3.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(
    not _server_reachable(),
    reason="opencode serve not reachable at 127.0.0.1:4096",
)
def test_e2e_full_lifecycle() -> None:
    """create -> turn -> wait -> snapshot -> interrupt against a live server."""

    class CollectingHost:
        connector_id = "e2e"

        def __init__(self) -> None:
            self.timeline_items: list = []
            self.turn_ended: list[str] = []

        async def timeline_sync(self, sid, rt, items, **kw):
            self.timeline_items.extend(items)

        async def session_meta_upsert(self, sid, rt, **kw):
            pass

        async def session_state_update(self, sid, rt, status, **kw):
            pass

        async def session_turn_ended(self, sid, rt, **kw):
            self.turn_ended.append(sid)

        async def runtime_health_update(self, status, detail=None):
            pass

    async def run() -> None:
        provider = OpenCodeProvider()
        config = await provider.validate_config({"serverUrl": "http://127.0.0.1:4096"})
        host = CollectingHost()
        rt = OpenCodeRuntime(config=config, host=host)
        await rt.start()
        try:
            res = await rt.create_and_start_session(
                "e2e-session",
                "只回复两个字:收到",
                selections={"model": "local/nemotron35-dspark"},
            )
            assert res.ok, res.message
            native = res.result["external_session_id"]

            for _ in range(90):
                st = await rt.get_session_state("e2e-session", native)
                if st and st.status == "idle":
                    break
                await asyncio.sleep(1)
            else:
                pytest.fail("turn did not finish within 90s")

            snap = await rt.get_session_snapshot("e2e-session", native)
            assert snap.complete
            roles = [i.role for i in snap.items]
            assert "user" in roles
            assert "assistant" in roles

            ir = await rt.interrupt_session(native)
            assert ir.ok

            # SSE-driven turn end should have been observed
            assert "e2e-session" in host.turn_ended or host.timeline_items
        finally:
            await rt.stop()

    asyncio.run(run())


@pytest.mark.skipif(
    not _server_reachable(),
    reason="opencode serve not reachable at 127.0.0.1:4096",
)
def test_e2e_continue_session_second_turn() -> None:
    """Two consecutive turns on one live session — the user-reported failure.

    Turn 1 goes through create_and_start_session; turn 2 goes through
    start_turn with the server-forwarded external id, then again with the
    platform id alone (connector-restart shape where the mapping is fresh
    but the server still knows the external id).
    """

    class CollectingHost:
        def __init__(self) -> None:
            self.turn_ended: list[str] = []

        async def timeline_sync(self, sid, rt, items, **kw):
            pass

        async def session_meta_upsert(self, sid, rt, **kw):
            pass

        async def session_state_update(self, sid, rt, status, **kw):
            pass

        async def session_turn_ended(self, sid, rt, **kw):
            self.turn_ended.append(sid)

        async def runtime_health_update(self, status, detail=None):
            pass

    async def wait_idle(rt, platform_id, native_id, timeout_s=120) -> None:
        for _ in range(timeout_s):
            st = await rt.get_session_state(platform_id, native_id)
            if st and st.status == "idle":
                return
            await asyncio.sleep(1)
        pytest.fail(f"turn did not finish within {timeout_s}s")

    async def run() -> None:
        provider = OpenCodeProvider()
        config = await provider.validate_config({"serverUrl": "http://127.0.0.1:4096"})
        host = CollectingHost()
        rt = OpenCodeRuntime(config=config, host=host)
        await rt.start()
        try:
            res = await rt.create_and_start_session(
                "e2e-cont",
                "只回复两个字:收到",
                selections={"model": "local/nemotron35-dspark"},
            )
            assert res.ok, res.message
            native = res.result["external_session_id"]
            await wait_idle(rt, "e2e-cont", native)

            # Turn 2: server-shaped call (external id forwarded by the server).
            res2 = await rt.start_turn("e2e-cont", native, "再只回复两个字:明白")
            assert res2.ok, res2.message
            await wait_idle(rt, "e2e-cont", native)

            # Turn 3: platform id only, mapping already bound in this process.
            res3 = await rt.start_turn("e2e-cont", None, "第三轮:好")
            assert res3.ok, res3.message
            await wait_idle(rt, "e2e-cont", native)

            snap = await rt.get_session_snapshot("e2e-cont", native)
            user_texts = [
                i.content.get("text") for i in snap.items
                if i.role == "user" and i.type == "message"
            ]
            assert any("收到" in (t or "") for t in user_texts)
            assert any("明白" in (t or "") for t in user_texts)
            assert any("第三轮" in (t or "") for t in user_texts)
        finally:
            await rt.stop()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# pending client-message registry (echo dedup)
# --------------------------------------------------------------------------- #


def _user_message(message_id: str, text: str) -> dict[str, Any]:
    return {
        "info": {"id": message_id, "role": "user", "sessionID": "ses_x"},
        "parts": [{"type": "text", "text": text}],
    }


def test_registry_binds_by_text_in_send_order() -> None:
    registry = OpenCodePendingClientMessageRegistry()
    registry.register(native_session_id="ses_x", client_message_id="cm_1", text="你好")
    registry.register(native_session_id="ses_x", client_message_id="cm_2", text="你好")
    matched = registry.resolve(
        "ses_x",
        [_user_message("msg_1", "你好"), _user_message("msg_2", "你好\n")],
    )
    assert matched == {"msg_1": "cm_1", "msg_2": "cm_2"}
    assert registry.resolve("ses_x", [_user_message("msg_1", "你好")]) == {}


def test_registry_keeps_unmatched_until_turn_end() -> None:
    registry = OpenCodePendingClientMessageRegistry()
    registry.register(native_session_id="ses_x", client_message_id="cm_1", text="later")
    assert registry.resolve("ses_x", []) == {}
    assert registry.resolve("ses_x", [_user_message("msg_9", "later")]) == {"msg_9": "cm_1"}
    registry.register(native_session_id="ses_x", client_message_id="cm_2", text="gone")
    registry.unresolve("ses_x")
    assert registry.resolve("ses_x", [_user_message("msg_8", "gone")]) == {}


def test_registry_ignores_empty_ids_and_text() -> None:
    registry = OpenCodePendingClientMessageRegistry()
    registry.register(native_session_id="ses_x", client_message_id=None, text="hi")
    registry.register(native_session_id="ses_x", client_message_id="cm_1", text="")
    assert registry.resolve("ses_x", [_user_message("msg_1", "hi")]) == {}


def test_map_messages_sets_client_message_id_on_source() -> None:
    items = timeline.map_messages_to_timeline("ses_x", [USER_MSG], {"msg_user_1": "cm_9"})
    user_items = [item for item in items if item.role == "user"]
    assert user_items[0].source.client_message_id == "cm_9"
    payload = user_items[0].to_platform_item("plat_1", 0).source
    assert payload.get("clientMessageId") == "cm_9"
    other = timeline.map_messages_to_timeline("ses_x", [USER_MSG])
    assert other[0].to_platform_item("plat_1", 0).source.get("clientMessageId") is None


def test_snapshot_user_item_carries_client_message_id_once() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session(
            "plat_echo", "hello", selections=None, client_message_id="cm_echo"
        )
        assert result.ok is True
        native = result.result["externalSessionId"]
        client.messages[native] = [_user_message("msg_h", "hello")]

        snap = await runtime.get_session_snapshot("plat_echo", native)
        user = next(i for i in snap.items if i.role == "user")
        assert user.source.get("clientMessageId") == "cm_echo"

        # Bindings are one-shot; a rebuild no longer stamps the id (the
        # optimistic echo is already resolved by then).
        snap2 = await runtime.get_session_snapshot("plat_echo", native)
        user2 = next(i for i in snap2.items if i.role == "user")
        assert user2.source.get("clientMessageId") is None

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# workspace directory (serve multiplexes per project directory)
# --------------------------------------------------------------------------- #


def test_append_directory_param() -> None:
    assert (
        append_directory_param("/session", "D:/work space")
        == "/session?directory=D%3A%2Fwork%20space"
    )
    assert append_directory_param("/x?a=1", "/tmp").endswith("?a=1&directory=%2Ftmp")


def test_create_session_uses_selected_workspace_directory() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session(
            "plat_ws", "hi", selections=None, cwd="D:/projects/demo"
        )
        assert result.ok is True
        native = result.result["externalSessionId"]
        assert client.created_directories == [(native, "D:/projects/demo")]
        # follow-up traffic for the session keeps the same directory
        await runtime.start_turn("plat_ws", native, "again", cwd=None)
        assert client.prompted[-1] == (native, "again", None, "D:/projects/demo")
        await runtime.interrupt_session("plat_ws")
        assert client.aborted == [native]

    asyncio.run(run())
