"""OpenCode runtime adapter.

Self-contained: talks directly to a local ``opencode serve`` instance over
REST + SSE. No plugin bridge required (unlike DSH).

Session model:
- The opencode native id (``ses_...``) is used as both the platform
  ``session_id`` and ``external_session_id`` (single-instance runtime).
- Turns are driven by ``POST /session/{id}/prompt_async``; the global SSE
  stream reports progress and ``session.idle`` marks turn completion.
- Timeline snapshots are rebuilt from ``GET /session/{id}/message``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from connector.logging import logger
from connector.runtime_protocol import (
    CAPABILITY_CATALOG_MODEL,
    CAPABILITY_SESSION_INTERRUPT,
    CAPABILITY_SESSION_SEND_MESSAGE,
    AgentRuntime,
    RuntimeCapability,
    RuntimeCapabilitySet,
    RuntimeConfig,
    RuntimeIdentity,
    RuntimeModelCatalog,
    RuntimeModelItem,
    RuntimeOperationResult,
    RuntimeTimelineSnapshot,
    RuntimeUnavailableError,
    SessionMeta,
    SessionSourceState,
    SessionState,
)
from connector.runtime_protocol.host import RuntimeHostClient
from connector.runtimes.opencode import provider_config, timeline
from connector.runtimes.opencode.sdk.client import OpenCodeClient, OpenCodeClientError

MODEL_SELECTION_KEY = "model"


def _iso_ms(ms: float | None) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


class OpenCodeRuntime(AgentRuntime):
    def __init__(
        self,
        config: RuntimeConfig,
        host: RuntimeHostClient,
        client_version: str = "1.0",
        client: OpenCodeClient | None = None,
    ) -> None:
        self.config = config
        self.host = host
        self.client_version = client_version
        values = dict(config.values)
        self._client = client or OpenCodeClient(
            values["serverUrl"],
            values.get("apiKey"),
            float(values.get("requestTimeoutSeconds", provider_config.DEFAULT_REQUEST_TIMEOUT_S)),
        )
        self._identity = RuntimeIdentity("opencode", "unknown", "OpenCode")
        self._stopping = False
        self._event_task: asyncio.Task[None] | None = None
        self._models: dict[str, dict[str, Any]] | None = None
        self._model_lock = asyncio.Lock()
        # Bidirectional platform <-> native session id mapping. Discovered
        # sessions are identity-mapped; created sessions carry a connector-
        # assigned platform id separate from the native ``ses_...`` id.
        self._platform_to_native: dict[str, str] = {}
        self._native_to_platform: dict[str, str] = {}

    def _native_id(self, session_id: str, external_session_id: str | None) -> str:
        if external_session_id:
            return external_session_id
        return self._platform_to_native.get(session_id, session_id)

    def _platform_id(self, native_id: str) -> str:
        return self._native_to_platform.get(native_id, native_id)

    def _bind_session(self, platform_id: str, native_id: str) -> None:
        self._platform_to_native[platform_id] = native_id
        self._native_to_platform[native_id] = platform_id

    # ------------------------------------------------------------------ #
    # identity / lifecycle
    # ------------------------------------------------------------------ #

    @property
    def sync_mode(self) -> str:
        return "events"

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    async def start(self) -> None:
        self._stopping = False
        try:
            health = await self._client.health()
        except OpenCodeClientError as exc:
            with suppress(Exception):
                await self.host.runtime_health_update(
                    "starting",
                    {
                        "code": "opencode_unreachable",
                        "message": f"无法连接 opencode serve: {exc}",
                        "retryable": True,
                    },
                )
            raise RuntimeUnavailableError(str(exc)) from exc
        version = str((health or {}).get("version") or "unknown")
        self._identity = RuntimeIdentity(
            "opencode", version, "OpenCode", protocol_version=self.client_version
        )
        if self._event_task is None or self._event_task.done():
            self._event_task = asyncio.create_task(self._event_loop(), name="opencode-events")

    async def stop(self) -> None:
        self._stopping = True
        task, self._event_task = self._event_task, None
        if task is not None:
            task.cancel()
            with suppress(BaseException):
                await task
        await self._client.close()

    async def get_config(self) -> RuntimeConfig:
        return self.config

    async def resynchronize(self, session_id: str | None = None, external_session_id: str | None = None) -> None:
        """Rebuild the timeline from the server for one session (or all known)."""

        if session_id:
            await self._publish_timeline(session_id, external_session_id)
        # Full resync is driven by the connector's background scanner via
        # get_session_snapshot(); nothing to calibrate on the transport itself.

    # ------------------------------------------------------------------ #
    # capabilities / catalogs
    # ------------------------------------------------------------------ #

    async def get_runtime_capabilities(self) -> RuntimeCapabilitySet:
        return RuntimeCapabilitySet(
            runtime="opencode",
            revision=1,
            capabilities=(
                RuntimeCapability(
                    capability_id=CAPABILITY_SESSION_SEND_MESSAGE,
                    scope="session",
                    runtime="opencode",
                ),
                RuntimeCapability(
                    capability_id=CAPABILITY_SESSION_INTERRUPT,
                    scope="session",
                    runtime="opencode",
                ),
                RuntimeCapability(
                    capability_id=CAPABILITY_CATALOG_MODEL,
                    scope="runtime",
                    runtime="opencode",
                ),
            ),
            metadata={"source": "opencode.static"},
        )

    async def list_model_catalog(
        self,
        query: str | None = None,
        limit: int = 100,
    ) -> RuntimeModelCatalog:
        models = await self._connected_models()
        items: list[RuntimeModelItem] = []
        for key, model in models.items():
            provider_id, _, model_id = key.partition("/")
            title = str(model.get("name") or model_id)
            if query and query.lower() not in key.lower():
                continue
            context = (model.get("limit") or {}).get("context")
            items.append(
                RuntimeModelItem(
                    id=key,
                    title=f"{title} ({provider_id})",
                    selection_id=key,
                    description=f"context {context}" if context else None,
                    metadata={"providerID": provider_id, "modelID": model_id},
                )
            )
            if len(items) >= limit:
                break
        return RuntimeModelCatalog(
            runtime="opencode", revision=1, models=tuple(items)
        )

    async def _connected_models(self) -> dict[str, dict[str, Any]]:
        async with self._model_lock:
            if self._models is None:
                try:
                    data = await self._client.provider_overview()
                except OpenCodeClientError:
                    data = {}
                connected = data.get("connected") if isinstance(data, dict) else []
                all_providers = data.get("all") if isinstance(data, dict) else []
                by_id = {p.get("id"): p for p in all_providers if isinstance(p, dict)}
                models: dict[str, dict[str, Any]] = {}
                for provider_id in connected or []:
                    provider = by_id.get(provider_id)
                    if not provider:
                        continue
                    for model_id, model in (provider.get("models") or {}).items():
                        if isinstance(model, dict):
                            models[f"{provider_id}/{model_id}"] = model
                self._models = models
            return self._models

    # ------------------------------------------------------------------ #
    # sessions
    # ------------------------------------------------------------------ #

    async def list_sessions(
        self,
        limit: int = 100,
        cursor: str | None = None,
        force: bool = False,
    ) -> tuple[SessionMeta, ...]:
        sessions = await self._client.list_sessions()
        metas: list[SessionMeta] = []
        for session in sessions[: max(limit, 0)]:
            sid = str(session.get("id") or "")
            if not sid:
                continue
            model = session.get("model") or {}
            metas.append(
                SessionMeta(
                    session_id=sid,
                    external_session_id=sid,
                    runtime="opencode",
                    title=session.get("title"),
                    cwd=session.get("directory"),
                    ordering_time=_iso_ms((session.get("time") or {}).get("updated")),
                    source_state=SessionSourceState(availability="available"),
                    metadata={
                        "model": f"{model.get('providerID')}/{model.get('id')}" if model else None,
                        "version": session.get("version"),
                    },
                )
            )
        return tuple(metas)

    async def get_session_snapshot(
        self,
        session_id: str,
        external_session_id: str | None = None,
        limit: int | None = None,
    ) -> RuntimeTimelineSnapshot:
        native_id = external_session_id or session_id
        messages = await self._client.list_messages(native_id)
        items = [
            item.to_platform_item(session_id, seq)
            for seq, item in enumerate(timeline.map_messages_to_timeline(native_id, messages))
        ]
        if limit is not None:
            items = items[-limit:]
        return RuntimeTimelineSnapshot(
            session_id=session_id,
            external_session_id=native_id,
            runtime="opencode",
            items=tuple(items),
            complete=True,
            metadata={"messageCount": len(messages)},
        )

    async def sync_session_timeline(
        self,
        session_id: str,
        external_session_id: str | None = None,
    ) -> bool:
        """Publish a fresh snapshot ourselves; the connector need not re-poll."""

        with suppress(Exception):
            await self._publish_timeline(session_id, external_session_id)
        return True

    async def _publish_timeline(
        self, session_id: str, external_session_id: str | None
    ) -> None:
        native_id = external_session_id or session_id
        messages = await self._client.list_messages(native_id)
        items = [
            item.to_platform_item(session_id, seq)
            for seq, item in enumerate(timeline.map_messages_to_timeline(native_id, messages))
        ]
        await self.host.timeline_sync(
            session_id,
            "opencode",
            tuple(items),
            external_session_id=native_id,
            complete=True,
        )

    async def get_session_state(
        self,
        session_id: str,
        external_session_id: str | None = None,
    ) -> SessionState | None:
        native_id = external_session_id or session_id
        status_map = await self._client.session_status_map()
        status = status_map.get(native_id) or {}
        status_type = status.get("type")
        if status_type == "busy":
            runtime_status = "running"
        elif status_type == "idle":
            runtime_status = "idle"
        else:
            runtime_status = "idle"
        return SessionState(
            session_id=session_id,
            external_session_id=native_id,
            runtime="opencode",
            status=runtime_status,
        )

    # ------------------------------------------------------------------ #
    # turns
    # ------------------------------------------------------------------ #

    async def create_and_start_session(
        self,
        session_id: str,
        content: str,
        title: str | None = None,
        cwd: str | None = None,
        selections: Mapping[str, str | None] | None = None,
        attachments: tuple = (),
        client_message_id: str | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> RuntimeOperationResult:
        _ = cwd, attachments, client_message_id, runtime_options
        try:
            session = await self._client.create_session(title)
        except OpenCodeClientError as exc:
            return RuntimeOperationResult(ok=False, code="create_failed", message=str(exc))
        native_id = str(session["id"])
        self._bind_session(session_id, native_id)
        await self.host.session_meta_upsert(
            session_id,
            "opencode",
            external_session_id=native_id,
            title=title or (content[:60] if content else None),
            cwd=session.get("directory"),
            ordering_time=_iso_ms((session.get("time") or {}).get("created")),
        )
        await self._start_turn(native_id, content, selections)
        return RuntimeOperationResult(
            ok=True,
            result={"external_session_id": native_id},
        )

    async def start_turn(
        self,
        session_id: str,
        external_session_id: str | None,
        content: str,
        selections: Mapping[str, str | None] | None = None,
        attachments: tuple = (),
        client_message_id: str | None = None,
        cwd: str | None = None,
    ) -> RuntimeOperationResult:
        _ = attachments, client_message_id, cwd
        native_id = self._native_id(session_id, external_session_id)
        try:
            await self._start_turn(native_id, content, selections)
        except OpenCodeClientError as exc:
            return RuntimeOperationResult(ok=False, code="turn_failed", message=str(exc))
        return RuntimeOperationResult(ok=True)

    async def _start_turn(
        self,
        native_id: str,
        content: str,
        selections: Mapping[str, str | None] | None,
    ) -> None:
        model = self._model_ref(selections)
        await self._client.prompt_async(native_id, content, model=model)
        with suppress(Exception):
            await self.host.session_state_update(
                self._platform_id(native_id),
                "opencode",
                status="running",
                external_session_id=native_id,
            )

    @staticmethod
    def _model_ref(selections: Mapping[str, str | None] | None) -> dict[str, str] | None:
        raw = (selections or {}).get(MODEL_SELECTION_KEY)
        if not raw or "/" not in raw:
            return None
        provider_id, _, model_id = raw.partition("/")
        return {"providerID": provider_id, "modelID": model_id}

    async def interrupt_session(
        self,
        session_id: str,
        reason: str | None = None,
    ) -> RuntimeOperationResult:
        _ = reason
        native_id = self._native_id(session_id, None)
        try:
            await self._client.abort(native_id)
        except OpenCodeClientError as exc:
            return RuntimeOperationResult(ok=False, code="abort_failed", message=str(exc))
        return RuntimeOperationResult(ok=True)

    # ------------------------------------------------------------------ #
    # SSE event loop
    # ------------------------------------------------------------------ #

    async def _event_loop(self) -> None:
        """Consume the global /event stream and push state to the host.

        Transport-level resilience: on disconnect, back off and reconnect.
        """

        backoff = 1.0
        while not self._stopping:
            try:
                async for event in self._client.stream_events():
                    backoff = 1.0
                    await self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors are expected
                if self._stopping:
                    break
                logger.warning("opencode event stream error: {}; reconnecting", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        props = event.get("properties") or {}
        native_id = props.get("sessionID")
        if not native_id:
            return
        platform_id = self._platform_id(native_id)
        if etype == "session.idle":
            with suppress(Exception):
                await self._publish_timeline(platform_id, native_id)
            with suppress(Exception):
                await self.host.session_turn_ended(
                    platform_id, "opencode", external_session_id=native_id, outcome="completed"
                )
        elif etype == "session.status":
            status_type = (props.get("status") or {}).get("type")
            if status_type == "busy":
                with suppress(Exception):
                    await self.host.session_state_update(
                        platform_id,
                        "opencode",
                        status="running",
                        external_session_id=native_id,
                    )
        elif etype == "session.updated":
            info = props.get("info") or {}
            with suppress(Exception):
                await self.host.session_meta_upsert(
                    platform_id,
                    "opencode",
                    external_session_id=native_id,
                    title=info.get("title"),
                    cwd=info.get("directory"),
                    ordering_time=_iso_ms((info.get("time") or {}).get("updated")),
                )
        # message.part.delta / message.part.updated / session.diff: live
        # updates are covered by snapshot rebuilds at turn end (session.idle).
