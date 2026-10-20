from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from connector.runtime_protocol import RuntimeConfig, RuntimeUnsupportedError
from connector.runtimes.dsh.bridge.client import BridgeClient
from connector.runtimes.dsh.discovery import BridgeEndpoint
from connector.runtimes.dsh.provider_config import dsh_capabilities
from connector.runtimes.dsh.runtime import DshRuntime

MALFORMED_ACKS: dict[str, Any] = {
    "empty": {},
    "no-command": {"ok": True, "result": {"kind": "success", "commandId": "one"}},
    "null-command": {"command": None, "ok": True, "result": {"kind": "success", "commandId": "one"}},
    "numeric-command": {"command": 7, "ok": True, "result": {"kind": "success", "commandId": "one"}},
    "empty-command": {"command": "", "ok": True, "result": {"kind": "success", "commandId": "one"}},
    "no-ok": {"command": "compact", "result": {"kind": "success", "commandId": "one"}},
    "null-ok": {"command": "compact", "ok": None, "result": {"kind": "success", "commandId": "one"}},
    "numeric-ok": {"command": "compact", "ok": 1, "result": {"kind": "success", "commandId": "one"}},
    "no-result": {"command": "compact", "ok": True},
    "empty-result": {"command": "compact", "ok": True, "result": {}},
    "wrong-result-type": {"command": "compact", "ok": True, "result": []},
    "wrong-kind": {"command": "compact", "ok": True, "result": {"kind": "error", "commandId": "one"}},
    "array-kind": {"command": "compact", "ok": True, "result": {"kind": [], "commandId": "one"}},
    "no-kind": {"command": "compact", "ok": True, "result": {"commandId": "one"}},
    "no-id": {"command": "compact", "ok": True, "result": {"kind": "success"}},
    "no-success-state": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": "one"}},
    "numeric-id": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": 7}},
    "empty-id": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": ""}},
    "negative-seq": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": "one", "sourceEventSeq": -1}},
    "boolean-seq": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": "one", "sourceEventSeq": True}},
    "invalid-text": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": "one", "text": 8}},
    "invalid-state": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": "one", "executionState": "unknown"}},
    "array-state": {"command": "compact", "ok": True, "result": {"kind": "success", "commandId": "one", "executionState": []}},
    "error-source-seq": {"command": "compact", "ok": False, "code": "command_error", "result": {"kind": "error", "commandId": "one", "sourceEventSeq": 1}},
    "false-success": {"command": "compact", "ok": False, "code": "command_error", "result": {"kind": "success", "commandId": "one"}},
    "error-without-id": {"command": "compact", "ok": False, "code": "command_error", "result": {"kind": "error"}},
    "no-error-state": {"command": "compact", "ok": False, "code": "command_error", "result": {"kind": "error", "commandId": "one"}},
    "failed-without-code": {"command": "compact", "ok": False, "result": {}},
    "invalid-retryability": {"command": "compact", "ok": False, "code": "command_outcome_unknown", "result": {"executionState": "unknown", "retryable": True}},
    "missing-unknown-state": {"command": "compact", "ok": False, "code": "command_outcome_unknown", "result": {}},
    "missing-no-retry": {"command": "compact", "ok": False, "code": "command_failed", "result": {"executionState": "unknown"}},
    "validation-with-unknown-state": {"command": "compact", "ok": False, "code": "invalid_command", "result": {"executionState": "unknown", "retryable": False}},
    "native-error-wrong-code": {"command": "compact", "ok": False, "code": "invalid_command", "result": {"kind": "error", "commandId": "one", "executionState": "completed"}},
    "missing-native-error-kind": {"command": "compact", "ok": False, "code": "command_error", "result": {}},
    "unrecognized-validation-code": {"command": "compact", "ok": False, "code": "unexpected_code", "result": {}},
}


@asynccontextmanager
async def bridge(tmp_path: Path, *, supported: bool = True, mode: str = "success") -> AsyncIterator[tuple[DshRuntime, list[dict[str, Any]], asyncio.Event]]:
    """Real JSON-RPC client/transport around the public adapter; native semantics are tested in TS."""
    requests: list[dict[str, Any]] = []
    cancelled = asyncio.Event()
    entered = asyncio.Event()
    writers: list[asyncio.StreamWriter] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writers.append(writer)
        try:
            while line := await reader.readline():
                request = json.loads(line)
                requests.append(request)
                method, params = request["method"], request.get("params", {})
                if method == "$/cancelRequest":
                    cancelled.set()
                    continue
                if method == "initialize":
                    result = {"identity": {"runtime": "dsh", "runtimeVersion": "test", "protocolVersion": "1.0"}}
                elif method == "session.getCapabilities":
                    result = {"runtime": "dsh", "sessionId": "session", "revision": 7, "capabilities": [{
                        "capabilityId": "session.commands", "supported": supported, "available": supported, "allowed": supported,
                        "unavailableReason": None if supported else "This DSH capability is not available.",
                        "metadata": {"catalogRevision": "host:2"} if supported else {},
                    }]}
                    if mode == "old-empty":
                        result["capabilities"] = []
                    elif mode == "partial-capability":
                        result["capabilities"] = [{"capabilityId": "session.commands"}]
                    elif mode == "unavailable":
                        result["capabilities"][0].update({"available": False, "allowed": False, "unavailableReason": "DSH session is archived."})
                elif method == "session.listCommands":
                    assert params == {"sessionId": "session", "externalSessionId": "native", "query": "compact", "limit": 1}
                    result = {"commands": [{"id": "compact", "title": "compact", "description": "native compact", "acceptsArgs": True,
                        "metadata": {"definitionId": "compact-plugin", "input": {"hint": "text", "attachments": True}, "attachmentsAvailable": False,
                            "ui": {"kind": "execute", "acceptsMultiline": True, "allowedStatuses": ["idle", "running"]}}}]}
                    if mode == "wide-catalog":
                        result["commands"].extend({"id": name, "title": name} for name in ["export", "goal", "permission"])
                elif method == "session.executeCommand":
                    entered.set()
                    if mode == "timeout":
                        continue
                    if mode == "disconnect":
                        writer.close()
                        return
                    if mode == "rpc-timeout":
                        writer.write(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32006, "message": "cancelled", "data": {"code": "REQUEST_TIMEOUT", "retryable": True}}}).encode() + b"\n")
                        await writer.drain()
                        continue
                    result = {"command": params["command"], "ok": mode != "error", "message": "native acknowledgement",
                        **({"code": "command_error"} if mode == "error" else {}),
                        "result": {"commandId": "native-command-7", "kind": "error" if mode == "error" else "success", "text": "native acknowledgement",
                            **({} if mode == "error" else {"sourceEventSeq": 12}), "executionState": "completed" if mode == "error" else "accepted"}}
                    if mode == "wrong-command":
                        result["command"] = "another"
                    elif mode == "error-kind":
                        result["result"]["kind"] = "error"
                    elif mode in MALFORMED_ACKS:
                        result = MALFORMED_ACKS[mode]
                    elif mode == "validation-failed":
                        result = {"command": params["command"], "ok": False, "code": "invalid_command", "message": "raw mismatch", "result": {}}
                else:
                    raise AssertionError(f"unexpected request {method}")
                writer.write(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def ignore(*_args: Any) -> None:
        return None

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    client = BridgeClient(endpoint=BridgeEndpoint("127.0.0.1", server.sockets[0].getsockname()[1], "token", 1, tmp_path / "endpoint.json"),
        connector_id="test", client_version="test", startup_timeout=1, request_timeout=0.05,
        notification_handler=ignore, exit_handler=ignore)
    runtime = DshRuntime(RuntimeConfig("dsh", 2), SimpleNamespace(connector_id="test"))
    await client.start()
    runtime._client = client
    try:
        yield runtime, requests, entered
        if mode == "timeout":
            await asyncio.wait_for(cancelled.wait(), 1)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
        for writer in writers:
            writer.close()
            await writer.wait_closed()


def test_public_catalog_execution_preserves_native_descriptor_and_raw_correlations(tmp_path: Path) -> None:
    async def run() -> None:
        async with bridge(tmp_path) as (runtime, requests, _):
            catalog = await runtime.list_commands("session", "native", query="compact", limit=1)
            assert len(catalog) == 1
            assert catalog[0].id == "compact" and catalog[0].accepts_args
            assert catalog[0].metadata["input"] == {"hint": "text", "attachments": True}
            assert catalog[0].metadata["attachmentsAvailable"] is False
            assert catalog[0].metadata["ui"]["acceptsMultiline"] is True
            raw = "/compact  first\nsecond  "
            result = await runtime.execute_command("session", "compact", "native", raw=raw, args=("ignored",))
            assert result.ok and result.message == "native acknowledgement"
            assert result.result == {"commandId": "native-command-7", "kind": "success", "text": "native acknowledgement", "sourceEventSeq": 12, "executionState": "accepted"}
            execution = [r for r in requests if r["method"] == "session.executeCommand"]
            assert len(execution) == 1
            assert execution[0]["params"] == {"sessionId": "session", "externalSessionId": "native", "command": "compact", "raw": raw, "args": ["ignored"]}
    asyncio.run(run())


@pytest.mark.parametrize("command", ["goal", "plan", "export", "feedback", "permission", "model", "file", "compact-thread"])
def test_non_compact_commands_never_reach_the_bridge(tmp_path: Path, command: str) -> None:
    async def run() -> None:
        async with bridge(tmp_path) as (runtime, requests, _):
            result = await runtime.execute_command("session", command, "native", raw=f"/{command}")
            assert not result.ok and result.code == "unknown_command"
            assert not any(r["method"] == "session.executeCommand" for r in requests)
    asyncio.run(run())


def test_catalog_filters_non_compact_commands_from_an_older_bridge(tmp_path: Path) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode="wide-catalog") as (runtime, _, _):
            catalog = await runtime.list_commands("session", "native", query="compact", limit=1)
            assert [command.id for command in catalog] == ["compact"]
    asyncio.run(run())


def test_old_bridge_remains_unavailable_with_upgrade_reason_and_no_command_rpc(tmp_path: Path) -> None:
    async def run() -> None:
        async with bridge(tmp_path, supported=False) as (runtime, requests, _):
            capability = (await runtime.get_session_capabilities("session", "native")).capabilities[0]
            assert not capability.available
            assert "upgrade" in (capability.unavailable_reason or "").lower()
            result = await runtime.execute_command("session", "compact", "native", raw="/compact hi")
            assert not result.ok and result.code == "bridge_upgrade_required"
            assert not any(r["method"] == "session.executeCommand" for r in requests)
    asyncio.run(run())
    assert not dsh_capabilities({"capabilities": []})["commands"]
    assert dsh_capabilities({"capabilities": [{"capabilityId": "session.commands", "supported": True, "available": True, "allowed": True}]})["commands"]
    assert dsh_capabilities({"capabilities": []})["sessionNotices"] is True
    assert not dsh_capabilities({"capabilities": []})["interactions"]
    assert dsh_capabilities({"capabilities": [{"capabilityId": "session.interaction.approval", "supported": True, "available": True, "allowed": True}]})["interactions"]


def test_public_native_error_is_not_success(tmp_path: Path) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode="error") as (runtime, _, _):
            result = await runtime.execute_command("session", "compact", "native", raw="/compact invalid")
            assert not result.ok
            assert result.result["kind"] == "error"
            assert result.result["commandId"] == "native-command-7"
            assert result.result["executionState"] == "completed"
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["timeout", "rpc-timeout", "disconnect"])
def test_ambiguous_execution_never_retries_and_exposes_unknown_state(tmp_path: Path, mode: str) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode=mode) as (runtime, requests, _):
            result = await runtime.execute_command("session", "compact", "native", raw="/compact once")
            assert not result.ok and result.code == "command_outcome_unknown"
            assert result.result == {"executionState": "unknown", "retryable": False}
            assert len([r for r in requests if r["method"] == "session.executeCommand"]) == 1
    asyncio.run(run())


def test_python_cancellation_propagates_and_sends_transport_cancel(tmp_path: Path) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode="timeout") as (runtime, requests, entered):
            task = asyncio.create_task(runtime.execute_command("session", "compact", "native", raw="/compact once"))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len([r for r in requests if r["method"] == "session.executeCommand"]) == 1
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["old-empty", "partial-capability", "unavailable"])
def test_missing_partial_or_session_unavailable_capability_never_dispatches(tmp_path: Path, mode: str) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode=mode) as (runtime, requests, _):
            with pytest.raises(RuntimeUnsupportedError):
                await runtime.list_commands("session", "native", query="compact", limit=1)
            result = await runtime.execute_command("session", "compact", "native", raw="/compact hi")
            assert not result.ok
            assert result.code == ("commands_unavailable" if mode == "unavailable" else "bridge_upgrade_required")
            assert not any(r["method"] in {"session.executeCommand", "session.listCommands"} for r in requests)
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["wrong-command", "error-kind"])
def test_result_identity_or_error_kind_cannot_be_reported_as_success(tmp_path: Path, mode: str) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode=mode) as (runtime, _, _):
            result = await runtime.execute_command("session", "compact", "native", raw="/compact hi")
            assert not result.ok
            assert result.command == "compact"
            assert result.result == {"executionState": "unknown", "retryable": False}
    asyncio.run(run())


@pytest.mark.parametrize("mode", MALFORMED_ACKS)
def test_authenticated_public_execution_rejects_incomplete_or_malformed_ack_without_retry(tmp_path: Path, mode: str) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode=mode) as (runtime, requests, _):
            result = await runtime.execute_command("session", "compact", "native", raw="/compact once")
            assert result.command == "compact"
            assert result.ok is False and result.code == "command_outcome_unknown"
            assert result.result == {"executionState": "unknown", "retryable": False}
            assert len([r for r in requests if r["method"] == "session.executeCommand"]) == 1
    asyncio.run(run())


def test_authenticated_public_execution_preserves_known_validation_without_correlation(tmp_path: Path) -> None:
    async def run() -> None:
        async with bridge(tmp_path, mode="validation-failed") as (runtime, requests, _):
            result = await runtime.execute_command("session", "compact", "native", raw="/permission")
            assert not result.ok and result.code == "invalid_command"
            assert result.result == {} and result.message == "raw mismatch"
            assert len([r for r in requests if r["method"] == "session.executeCommand"]) == 1
    asyncio.run(run())
