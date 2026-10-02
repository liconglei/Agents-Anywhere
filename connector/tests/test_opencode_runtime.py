"""Tests for the OpenCode runtime adapter (provider config, timeline, runtime).

All tests run against fakes — no live ``opencode serve`` required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from connector.runtime_protocol import (
    RuntimeInvalidRequestError,
    RuntimeUnavailableError,
)
from connector.runtimes.opencode import provider_config, timeline
from connector.runtimes.opencode.provider import OpenCodeProvider
from connector.runtimes.opencode.runtime import OpenCodeRuntime
from connector.runtimes.opencode.sdk.client import OpenCodeClientError

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
        self.prompted: list[tuple[str, str, dict[str, str] | None]] = []
        self.aborted: list[str] = []
        self._events: list[dict[str, Any]] = []

    async def health(self) -> dict[str, Any]:
        return {"version": "1.18.34"}

    async def list_sessions(self) -> list[dict[str, Any]]:
        return list(self.sessions.values())

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return self.sessions[session_id]

    async def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self.messages.get(session_id, [])

    async def create_session(self, title: str | None = None) -> dict[str, Any]:
        sid = f"ses_new_{len(self.sessions)}"
        session = {
            "id": sid,
            "title": title,
            "directory": "/work",
            "time": {"created": 100, "updated": 100},
        }
        self.sessions[sid] = session
        self.messages[sid] = []
        return session

    async def prompt_async(
        self, session_id: str, content: str, model: dict[str, str] | None = None, agent: str | None = None
    ) -> None:
        self.prompted.append((session_id, content, model))
        self.status[session_id] = {"type": "busy"}

    async def abort(self, session_id: str) -> None:
        self.aborted.append(session_id)
        self.status[session_id] = {"type": "idle"}

    async def session_status_map(self) -> dict[str, Any]:
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
        pass


def _make_runtime(client: FakeClient, host: FakeHost) -> OpenCodeRuntime:
    provider = OpenCodeProvider()

    async def run() -> OpenCodeRuntime:
        config = await provider.validate_config({"serverUrl": "http://127.0.0.1:4096"})
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


def test_start_raises_when_unreachable() -> None:
    class DeadClient(FakeClient):
        async def health(self) -> dict[str, Any]:
            raise OpenCodeClientError("connection refused", status_code=None)

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
        native = result.result["external_session_id"]
        assert native.startswith("ses_new_")
        # turn was started with the model ref
        assert client.prompted == [("ses_new_0", "do it", {"providerID": "local", "modelID": "nemotron35-dspark"})]
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
        assert client.prompted == [("ses_b", "plain prompt", None)]

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


def test_event_session_idle_publishes_and_ends_turn() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        result = await runtime.create_and_start_session("plat_d", "hi", selections=None)
        native = result.result["external_session_id"]
        client.messages[native] = [USER_MSG, ASSISTANT_MSG]

        await runtime._handle_event(
            {"type": "session.idle", "properties": {"sessionID": native}}
        )
        assert ("plat_d", "completed") in host.turn_ended
        assert ("plat_d", 4) in host.timeline_syncs

    asyncio.run(run())


def test_event_busy_status_updates_state() -> None:
    client, host = FakeClient(), FakeHost()
    runtime = _make_runtime(client, host)

    async def run() -> None:
        await runtime._handle_event(
            {"type": "session.status", "properties": {"sessionID": "ses_e", "status": {"type": "busy"}}}
        )
        assert ("ses_e", "running") in host.state_updates

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
