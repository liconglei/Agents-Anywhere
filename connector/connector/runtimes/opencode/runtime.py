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
from urllib.parse import urlparse

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
from connector.runtimes.opencode import provider_config, serve_process, timeline
from connector.runtimes.opencode.pending_messages import (
    OpenCodePendingClientMessageRegistry,
)
from connector.runtimes.opencode.sdk.client import OpenCodeClient, OpenCodeClientError

MODEL_SELECTION_KEY = "model"
# Reconnect failures tolerated by the event loop before a recovery pass runs.
STREAM_FAILURE_RECOVERY_THRESHOLD = 3
SERVE_RESTART_INITIAL_BACKOFF_S = 1.0
SERVE_RESTART_MAX_BACKOFF_S = 30.0
PORT_ADOPTION_WAIT_S = 15.0
# A live-but-unhealthy child within this window is still booting; recovery
# waits instead of killing and respawning it. opencode cold starts (especially
# npx-launched on Windows) routinely exceed 60s, so a tight grace creates a
# kill-respawn loop that looks like "serve started several times".
SERVE_BOOT_GRACE_S = 180.0
# How many times start() retries the auto-start + health-check cycle before
# giving up. The supervisor does not retry a failed start, so a transient
# failure (serve still booting, npx mid-download) would otherwise permanently
# mark the runtime as error and disable the client send button.
START_MAX_ATTEMPTS = 3
START_RETRY_BACKOFF_S = 3.0


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
        state_reconcile_interval_s: float = 20.0,
        auto_start_serve_timeout_s: float = 180.0,
    ) -> None:
        self.config = config
        self.host = host
        self.client_version = client_version
        values = dict(config.values)
        self._client = client or OpenCodeClient(
            values["serverUrl"],
            values.get("apiKey"),
            float(values.get("requestTimeoutSeconds", provider_config.DEFAULT_REQUEST_TIMEOUT_S)),
            stream_timeout=provider_config.DEFAULT_STREAM_TIMEOUT_S,
            api_user=str(values.get("apiUser") or "opencode"),
        )
        self._identity = RuntimeIdentity("opencode", "unknown", "OpenCode")
        self._pending_messages = OpenCodePendingClientMessageRegistry()
        # Native session id -> workspace directory. serve binds requests to the
        # instance for that directory; without it sessions land in the serve
        # process cwd regardless of the workspace the user selected.
        self._session_dirs: dict[str, str] = {}
        # Turn-ends the host refused (backend blip); retried on every
        # watchdog tick, snapshot sync and idle event.
        self._unended_turns: dict[str, str] = {}
        # Native sessions whose idle state was asserted to the host since their
        # last running update (see get_session_state self-heal).
        self._pushed_idle: set[str] = set()
        self._stopping = False
        self._started = False
        self._state_reconcile_interval_s = state_reconcile_interval_s
        self._auto_start_serve_timeout_s = auto_start_serve_timeout_s
        # Native session ids with a turn in flight on the platform side.
        # If the ``session.idle`` event is lost (SSE drop mid-turn), the
        # watchdog reconciles against /session/status and emits the missing
        # turn-end so the platform session cannot stay "running" forever.
        self._active_turns: set[str] = set()
        # Native ids whose abort was requested; the turn-end is reported as
        # interrupted when serve finally confirms the session left busy.
        self._aborted_turns: set[str] = set()
        # Last outcome reported per native id, so a late duplicate idle event
        # cannot re-report a turn the reconciler already ended.
        self._recently_ended: dict[str, str] = {}
        # Supervised child ``opencode serve`` process, when auto-start owns one.
        self._serve: serve_process.ServeProcess | None = None
        self._serve_env: dict[str, str] | None = None
        self._recovery_lock = asyncio.Lock()
        self._recovery_task: asyncio.Task[None] | None = None
        # Cached ``opencode serve`` argv so the (slow, up to 15s) --version
        # probe runs at most once per runtime instead of on every spawn.
        self._cached_serve_command: list[str] | None = None
        self._stream_failures = 0
        self._health_degraded = False
        # --- diagnostics (temporary; no behavioral effect) ---
        # How many ``opencode serve`` children this runtime has spawned; a value
        # above 1 across a single start reveals repeated/spurious spawns.
        self._serve_spawn_count = 0
        # Monotonic counter of idle notifications per native session. Used to
        # detect the turn-start race: an idle arriving while prompt_async is
        # still in flight (before the turn is marked active).
        self._idle_seq: dict[str, int] = {}
        # Whether the SSE stream has delivered its first event since the last
        # disconnect; distinguishes "never connected" from "idle lost".
        self._event_stream_connected = False
        self._event_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
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

    def _adopt_session_binding(self, session_id: str, external_session_id: str | None) -> str:
        """Rebuild the platform<->native pairing when the Server supplies both ids.

        The maps are in-memory: after a connector restart they are empty, and a
        Server-driven call (send message, state read) is the only place the
        pairing reappears. Without rebuilding here, every push for a
        pre-restart session is addressed with the *native* id; the Server can
        then only resolve it through the session row's externalSessionId —
        which sessions created before the contract fix carry as NULL — so the
        update is dropped and the platform session stays stuck in "running"
        with the send button permanently disabled.
        """

        native_id = self._native_id(session_id, external_session_id)
        if external_session_id:
            self._bind_session(session_id, native_id)
        return native_id

    def _remember_directory(self, native_id: str, directory: str | None) -> None:
        if directory:
            self._session_dirs[native_id] = directory

    def _directory_for(self, native_id: str) -> str | None:
        return self._session_dirs.get(native_id)

    # ------------------------------------------------------------------ #
    # identity / lifecycle
    # ------------------------------------------------------------------ #

    @property
    def sync_mode(self) -> str:
        return "events"

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    @staticmethod
    def _auth_hint_message(exc: BaseException) -> str | None:
        """Descriptive guidance when serve answers 401 (Basic auth mismatch)."""

        if getattr(exc, "status_code", None) != 401:
            return None
        return (
            "opencode serve 已运行但认证失败(401)：serve 使用 HTTP Basic 认证。"
            "请在运行时配置中填写 apiKey(密码)和 apiUser(用户名, 默认 opencode)；"
            "连接器自动拉起的 serve 会自动继承环境变量 "
            "OPENCODE_SERVER_PASSWORD / OPENCODE_SERVER_USERNAME。"
        )

    async def start(self) -> None:
        self._stopping = False
        server_url = str(self.config.values.get("serverUrl") or "")
        logger.info("opencode start: connecting to {}", server_url)
        last_error: Exception | None = None
        for attempt in range(1, START_MAX_ATTEMPTS + 1):
            try:
                health = await self._try_start_once(server_url)
                break
            except RuntimeUnavailableError:
                # Surface auth/configuration failures immediately; retrying
                # them only delays the inevitable error message.
                raise
            except Exception as exc:  # noqa: BLE001 - transient startup failures
                last_error = exc
                if attempt >= START_MAX_ATTEMPTS:
                    break
                logger.warning(
                    "opencode start attempt {}/{} failed ({}); retrying in {}s",
                    attempt,
                    START_MAX_ATTEMPTS,
                    exc,
                    START_RETRY_BACKOFF_S,
                )
                await asyncio.sleep(START_RETRY_BACKOFF_S)
        else:
            # Unreachable: the for loop always breaks or raises.
            health = None  # pragma: no cover
        if last_error is not None:
            hint = self._auth_hint_message(last_error)
            detail = hint or str(last_error)
            logger.warning("opencode start failed after {} attempts: {}", START_MAX_ATTEMPTS, detail)
            with suppress(Exception):
                await self.host.runtime_health_update(
                    "starting",
                    {
                        "code": "opencode_auth_failed" if hint else "opencode_unreachable",
                        "message": f"无法连接 opencode serve: {detail}",
                        "retryable": True,
                    },
                )
            raise RuntimeUnavailableError(detail) from last_error
        version = str((health or {}).get("version") or "unknown")
        self._identity = RuntimeIdentity(
            "opencode", version, "OpenCode", protocol_version=self.client_version
        )
        logger.info("opencode start ok: serve version={}", version)
        if self._event_task is None or self._event_task.done():
            self._event_task = asyncio.create_task(self._event_loop(), name="opencode-events")
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(
                self._state_watchdog(), name="opencode-state-watchdog"
            )
        self._started = True

    async def _try_start_once(self, server_url: str) -> dict[str, Any] | None:
        """One attempt to reach (or auto-start) serve and return its health."""

        try:
            return await self._client.health()
        except OpenCodeClientError as exc:
            logger.warning("opencode start: server unreachable ({})", exc)
            if not self._should_auto_start(server_url):
                hint = self._auth_hint_message(exc)
                with suppress(Exception):
                    await self.host.runtime_health_update(
                        "starting",
                        {
                            "code": "opencode_auth_failed" if hint else "opencode_unreachable",
                            "message": hint or f"无法连接 opencode serve: {exc}",
                            "retryable": True,
                        },
                    )
                raise RuntimeUnavailableError(hint or str(exc)) from exc
            try:
                await self._auto_start_serve(server_url)
                return await self._client.health()
            except Exception as start_exc:
                hint = self._auth_hint_message(start_exc) or self._auth_hint_message(exc)
                detail = hint or str(start_exc)
                logger.warning("opencode start: auto-start did not recover ({})", detail)
                if hint:
                    # Auth failures are not transient: surface them so the outer
                    # loop does not waste attempts.
                    raise RuntimeUnavailableError(detail) from start_exc
                raise

    async def stop(self) -> None:
        self._stopping = True
        self._started = False
        tasks = [t for t in (self._event_task, self._watchdog_task, self._recovery_task) if t is not None]
        self._event_task = None
        self._watchdog_task = None
        self._recovery_task = None
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(BaseException):
                await task
        await self._stop_serve_process()
        await self._client.close()
        logger.info("opencode stop: tasks cancelled, client closed")

    async def _stop_serve_process(self) -> None:
        """Terminate the auto-started ``opencode serve`` child, if any."""
        serve = self._serve
        self._serve = None
        if serve is not None:
            await serve.stop()

    def _should_auto_start(self, server_url: str) -> bool:
        """Auto-start is allowed only for loopback URLs when not disabled."""
        if not bool(self.config.values.get("autoStart", True)):
            return False
        try:
            parsed = urlparse(server_url)
        except (ValueError, TypeError):
            return False
        return parsed.hostname in ("127.0.0.1", "localhost", "::1")

    @staticmethod
    def _serve_command(port: int) -> list[str]:
        return serve_process.resolve_serve_command(port)

    def _serve_target(self, server_url: str) -> tuple[str, int]:
        parsed = urlparse(server_url)
        return parsed.hostname or "127.0.0.1", parsed.port or 4096

    @staticmethod
    async def _port_listening(host: str, port: int) -> bool:
        """True when something accepts TCP on host:port (a foreign serve)."""
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=1.0)
        except OSError:  # includes TimeoutError, the builtin subclasses OSError
            return False
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()
        return True

    async def _auto_start_serve(self, server_url: str) -> None:
        """Spawn a local ``opencode serve`` child and wait for it to become healthy."""

        async with self._recovery_lock:
            await self._auto_start_serve_locked(server_url)

    async def _adoption_wait(self) -> tuple[bool, OpenCodeClientError | None]:
        """Poll health for PORT_ADOPTION_WAIT_S; (adopted?, last error)."""

        deadline = asyncio.get_running_loop().time() + PORT_ADOPTION_WAIT_S
        last_exc: OpenCodeClientError | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                await self._client.health()
            except OpenCodeClientError as poll_exc:
                last_exc = poll_exc
                await asyncio.sleep(1.0)
                continue
            return True, None
        return False, last_exc

    async def _auto_start_serve_locked(self, server_url: str) -> None:
        host, port = self._serve_target(server_url)
        if self._serve_env is None:
            self._serve_env = await asyncio.to_thread(serve_process.serve_environment)
        # A spawned serve inherits OPENCODE_SERVER_PASSWORD (and optionally
        # OPENCODE_SERVER_USERNAME) from the user environment; without the
        # matching Basic credentials every request, including health, is
        # answered with 401.
        if not self._client.api_key:
            env_password = str(self._serve_env.get("OPENCODE_SERVER_PASSWORD") or "")
            if env_password:
                env_user = str(self._serve_env.get("OPENCODE_SERVER_USERNAME") or "opencode")
                self._client.set_credentials(env_user, env_password)
                logger.info("opencode auto-start: inherited serve Basic auth user={}", env_user)
        # A concurrent recovery pass may already have brought serve up. A 401
        # here means a serve is already listening with credentials we do not
        # have; spawning another instance would only die on the shared log
        # file lock, so surface the auth hint instead.
        try:
            await self._client.health()
            return
        except OpenCodeClientError as exc:
            hint = self._auth_hint_message(exc)
            if hint is not None:
                raise RuntimeError(hint) from exc
        # Another live opencode instance (TUI, foreign serve, CherryStudio, or
        # a dying orphan serve from a previous connector) holds the
        # single-instance log lock: any spawn we attempt dies during boot,
        # often with a native fastfail and zero output. Wait briefly for that
        # instance to answer health (adopt) or release the lock (re-spawn),
        # instead of burning start attempts on doomed spawns.
        if await asyncio.to_thread(serve_process.is_serve_log_lock_held, self._serve_env):
            logger.warning("opencode auto-start: opencode.log lock held elsewhere; not spawning")
            adopted, last_exc = await self._adoption_wait()
            if adopted:
                logger.info("opencode auto-start: adopted external serve on port {}", port)
                return
            if await asyncio.to_thread(serve_process.is_serve_log_lock_held, self._serve_env):
                hint = self._auth_hint_message(last_exc) or serve_process.LOCK_HOLD_HINT
                raise RuntimeError(hint) from last_exc
            logger.info("opencode auto-start: log lock released; proceeding to spawn")
        if self._cached_serve_command is None:
            self._cached_serve_command = await asyncio.to_thread(
                serve_process.resolve_serve_command,
                port,
                self._serve_env,
                hostname="127.0.0.1" if host in ("localhost", "::1") else host,
            )
        command = self._cached_serve_command
        self._serve_spawn_count += 1
        logger.info(
            "opencode auto-start: launching (#{}): {}",
            self._serve_spawn_count,
            " ".join(command),
        )
        serve = serve_process.ServeProcess(on_exit=self._on_serve_exit)
        self._serve = serve
        await serve.spawn(command, self._serve_env)
        boot_started = asyncio.get_running_loop().time()
        try:
            await serve.wait_healthy(
                self._client.health, OpenCodeClientError, self._auto_start_serve_timeout_s
            )
        except RuntimeError as exc:
            # A serve started elsewhere may be booting on the same port (or
            # our spawn died because the port or the log lock was taken by
            # a transient holder); if something answers, adoption beats
            # treating the failed spawn as fatal.
            port_busy = serve_process.is_port_in_use(str(exc)) or await self._port_listening(
                host, port
            )
            lock_busy = await asyncio.to_thread(
                serve_process.is_serve_log_lock_held, self._serve_env
            )
            if not port_busy and not lock_busy:
                await serve.stop()
                if self._serve is serve:
                    self._serve = None
                raise
            logger.warning(
                "opencode auto-start: startup collision ({}); waiting for other serve", exc
            )
            adopted, last_poll_exc = await self._adoption_wait()
            await serve.stop()
            if self._serve is serve:
                self._serve = None
            if adopted:
                logger.info("opencode auto-start: adopted external serve on port {}", port)
                return
            hint = self._auth_hint_message(last_poll_exc) if last_poll_exc else None
            if hint is not None:
                raise RuntimeError(hint) from exc
            if lock_busy:
                raise RuntimeError(f"{exc} [{serve_process.LOCK_HOLD_HINT}]") from exc
            # Non-lock collision (port in use by a foreign process we cannot
            # adopt): surface the failure; the recovery loop retries with
            # backoff rather than stacking spawns here.
            raise
        logger.info(
            "opencode auto-start: serve ready on {}:{} after {:.1f}s (spawn #{})",
            host,
            port,
            asyncio.get_running_loop().time() - boot_started,
            self._serve_spawn_count,
        )
        return

    async def _on_serve_exit(self, code: int | None, tail: str) -> None:
        """ServeProcess monitor callback: the child died while we were running."""

        if self._stopping:
            return
        dead = self._serve
        if dead is not None and dead.process is not None and dead.process.returncode is not None:
            self._serve = None
        if not self._started:
            # Still inside start(): wait_healthy surfaces the failure to the
            # caller, who decides whether to retry. No background loop yet.
            return
        logger.warning(
            "opencode serve exited unexpectedly (code={}); scheduling recovery", code
        )
        with suppress(Exception):
            await self.host.runtime_health_update(
                "error",
                {
                    "code": "opencode_serve_exited",
                    "message": f"opencode serve 意外退出 (code={code})，正在自动重启。",
                    "retryable": True,
                },
            )
        self._schedule_recovery("serve-exited")

    def _schedule_recovery(self, reason: str) -> None:
        if self._stopping:
            return
        if self._recovery_task is not None and not self._recovery_task.done():
            return
        self._recovery_task = asyncio.create_task(
            self._recover(reason), name="opencode-serve-recovery"
        )

    async def _recover(self, reason: str) -> None:
        """Keep probing/restoring serve reachability until it is healthy."""

        server_url = str(self.config.values.get("serverUrl") or "")
        attempt = 0
        while not self._stopping:
            async with self._recovery_lock:
                try:
                    await self._client.health()
                except OpenCodeClientError as exc:
                    logger.debug("opencode recovery ({}) probe failed: {}", reason, exc)
                else:
                    logger.info("opencode recovery ({}) complete: serve reachable", reason)
                    self._health_degraded = False
                    self._stream_failures = 0
                    with suppress(Exception):
                        await self.host.runtime_health_update("running", None)
                    with suppress(Exception):
                        await self._reconcile_active_turns()
                    return
                serve = self._serve
                if (
                    serve is not None
                    and serve.process is not None
                    and serve.process.returncode is None
                    and not serve.boot_grace_expired(SERVE_BOOT_GRACE_S)
                ):
                    # The child is still booting; re-probe next cycle instead
                    # of killing it or stacking duplicate spawns.
                    logger.info(
                        "opencode recovery ({}) decision=wait (child pid={} still within {}s boot grace)",
                        reason,
                        serve.process.pid,
                        int(SERVE_BOOT_GRACE_S),
                    )
                elif self._should_auto_start(server_url):
                    if serve is not None and serve.process is not None:
                        # Wedged long enough: replace it rather than wait.
                        logger.info(
                            "opencode recovery ({}) decision=kill-and-respawn (pid={} exceeded {}s boot grace)",
                            reason,
                            serve.process.pid,
                            int(SERVE_BOOT_GRACE_S),
                        )
                        await serve.stop()
                        self._serve = None
                    else:
                        logger.info(
                            "opencode recovery ({}) decision=auto-start (no live child)",
                            reason,
                        )
                    try:
                        await self._auto_start_serve_locked(server_url)
                    except Exception as exc:  # noqa: BLE001 - keep retrying with backoff
                        logger.warning("opencode recovery ({}) spawn failed: {}", reason, exc)
                else:
                    logger.info(
                        "opencode recovery ({}) decision=give-up (auto-start disabled or non-loopback)",
                        reason,
                    )
            if self._stopping:
                return
            attempt += 1
            delay = SERVE_RESTART_INITIAL_BACKOFF_S * 2**min(attempt, 5)
            await asyncio.sleep(min(delay, SERVE_RESTART_MAX_BACKOFF_S))

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
                    capability_id=CAPABILITY_CATALOG_MODEL,
                    scope="runtime",
                    runtime="opencode",
                ),
            ),
            metadata={"source": "opencode.static"},
        )

    async def get_session_capabilities(
        self,
        session_id: str,
        external_session_id: str | None = None,
    ) -> RuntimeCapabilitySet:
        """Session-scoped facts; send/interrupt must carry the platform session id.

        The Server's capability projection matches session-scoped entries by
        ``(runtime, scope="session", sessionId, runtimeId)``. A session-scoped
        entry without ``session_id`` — or one declared only on the
        runtime-scoped set — matches no projection group, so
        ``session.send_message`` resolves to "unsupported" and the client's
        send button stays disabled forever after the first turn.
        """

        return RuntimeCapabilitySet(
            runtime="opencode",
            revision=1,
            session_id=session_id,
            capabilities=(
                RuntimeCapability(
                    capability_id=CAPABILITY_SESSION_SEND_MESSAGE,
                    scope="session",
                    runtime="opencode",
                    session_id=session_id,
                ),
                RuntimeCapability(
                    capability_id=CAPABILITY_SESSION_INTERRUPT,
                    scope="session",
                    runtime="opencode",
                    session_id=session_id,
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
            self._remember_directory(sid, session.get("directory"))
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
        native_id = self._adopt_session_binding(session_id, external_session_id)
        messages = await self._client.list_messages(native_id, self._directory_for(native_id))
        client_message_ids = self._pending_messages.resolve(native_id, messages)
        items = [
            item.to_platform_item(session_id, seq)
            for seq, item in enumerate(
                timeline.map_messages_to_timeline(native_id, messages, client_message_ids)
            )
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
        native_id = self._adopt_session_binding(session_id, external_session_id)
        messages = await self._client.list_messages(native_id, self._directory_for(native_id))
        client_message_ids = self._pending_messages.resolve(native_id, messages)
        items = [
            item.to_platform_item(session_id, seq)
            for seq, item in enumerate(
                timeline.map_messages_to_timeline(native_id, messages, client_message_ids)
            )
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
        native_id = self._adopt_session_binding(session_id, external_session_id)
        status_map = await self._client.session_status_map(self._directory_for(native_id))
        status = status_map.get(native_id) or {}
        status_type = status.get("type")
        if status_type == "busy":
            runtime_status = "running"
        elif status_type == "idle":
            runtime_status = "idle"
        else:
            runtime_status = "idle"
        if runtime_status == "running":
            self._pushed_idle.discard(native_id)
        elif native_id not in self._pushed_idle and native_id not in self._active_turns:
            # Self-heal sessions that got stuck in "running" (e.g. a turn ended
            # before this push existed): reopening them re-queries serve, so
            # re-asserting idle unlocks the client send button.
            self._pushed_idle.add(native_id)
            logger.info(
                "opencode get_session_state self-heal: pushing idle platform={} native={}",
                session_id,
                native_id,
            )
            with suppress(Exception):
                await self.host.session_state_update(
                    session_id, "opencode", status="idle", external_session_id=native_id
                )
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
        _ = attachments, runtime_options
        try:
            session = await self._client.create_session(title, directory=cwd)
        except OpenCodeClientError as exc:
            logger.warning("opencode create session failed: {}", exc)
            return RuntimeOperationResult(ok=False, code="create_failed", message=str(exc))
        native_id = str(session["id"])
        self._bind_session(session_id, native_id)
        self._remember_directory(
            native_id, str(session.get("directory") or cwd) if session.get("directory") or cwd else None
        )
        self._pending_messages.register(
            native_session_id=native_id,
            client_message_id=client_message_id,
            text=content,
        )
        logger.info(
            "opencode session created: platform={} native={}", session_id, native_id
        )
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
            # Contract key is camelCase (same as codex/claude/dsh). A snake_case
            # key here is silently unreadable by the Server: session.create
            # responses then bind externalSessionId=None, and every later
            # notification that arrives with the native id (e.g. after a
            # connector restart, when the platform<->native map is gone) can
            # never resolve back to the platform session, leaving it stuck in
            # "running" (send button permanently disabled).
            result={"externalSessionId": native_id},
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
        _ = attachments
        native_id = self._adopt_session_binding(session_id, external_session_id)
        self._remember_directory(native_id, cwd)
        self._pending_messages.register(
            native_session_id=native_id,
            client_message_id=client_message_id,
            text=content,
        )
        logger.info(
            "opencode start_turn: platform={} external={} -> native={}",
            session_id,
            external_session_id,
            native_id,
        )
        try:
            await self._start_turn(native_id, content, selections)
        except OpenCodeClientError as exc:
            logger.warning("opencode start_turn failed: native={} error={}", native_id, exc)
            return RuntimeOperationResult(ok=False, code="turn_failed", message=str(exc))
        return RuntimeOperationResult(ok=True)

    async def _start_turn(
        self,
        native_id: str,
        content: str,
        selections: Mapping[str, str | None] | None,
    ) -> None:
        model = self._model_ref(selections)
        logger.info(
            "opencode turn start: platform={} native={} model={}",
            self._platform_id(native_id),
            native_id,
            (model or {}).get("modelID"),
        )
        idle_before = self._idle_seq.get(native_id, 0)
        # Mark the turn active BEFORE awaiting prompt_async. If serve emits
        # session.idle while this await is in flight (a fast turn or a stale
        # idle replayed on SSE connect), _finish_turn must see it in
        # _active_turns so the turn is properly ended instead of being dropped
        # as "untracked" and leaving the platform session stuck in "running".
        self._active_turns.add(native_id)
        self._pushed_idle.discard(native_id)
        await self._client.prompt_async(
            native_id, content, model=model, directory=self._directory_for(native_id)
        )
        idle_after = self._idle_seq.get(native_id, 0)
        if idle_after != idle_before:
            logger.warning(
                "opencode turn-start race: idle arrived while prompt in flight "
                "platform={} native={} (idle_seq {}->{})",
                self._platform_id(native_id),
                native_id,
                idle_before,
                idle_after,
            )
        # If an idle arrived during the await, _finish_turn already ran and
        # removed this session from _active_turns; do not re-push running.
        if native_id not in self._active_turns:
            return
        self._aborted_turns.discard(native_id)
        self._recently_ended.pop(native_id, None)
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
            await self._client.abort(native_id, self._directory_for(native_id))
        except OpenCodeClientError as exc:
            return RuntimeOperationResult(ok=False, code="abort_failed", message=str(exc))
        if native_id in self._active_turns:
            self._aborted_turns.add(native_id)
            # Serve normally emits session.idle after an abort; reconcile now
            # so a missed idle event does not leave the session stuck running.
            with suppress(Exception):
                await self._reconcile_active_turns()
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
                    self._stream_failures = 0
                    if not self._event_stream_connected:
                        self._event_stream_connected = True
                        logger.info("opencode event stream connected (first event received)")
                    await self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors are expected
                if self._stopping:
                    break
                self._event_stream_connected = False
                self._stream_failures += 1
                logger.warning("opencode event stream error: {}; reconnecting", exc)
                with suppress(Exception):
                    await self._reconcile_active_turns()
                if self._stream_failures >= STREAM_FAILURE_RECOVERY_THRESHOLD:
                    if not self._health_degraded:
                        self._health_degraded = True
                        with suppress(Exception):
                            await self.host.runtime_health_update(
                                "error",
                                {
                                    "code": "opencode_unreachable",
                                    "message": f"opencode serve 持续不可达: {exc}",
                                    "retryable": True,
                                },
                            )
                    self._schedule_recovery("stream-down")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _state_watchdog(self) -> None:
        """Periodically reconcile in-flight turns against ``/session/status``.

        A ``session.idle`` event lost while the SSE stream was down (or before
        this process started) would otherwise leave the platform session stuck
        in "running" forever, disabling the client send button.
        """

        while not self._stopping:
            await asyncio.sleep(self._state_reconcile_interval_s)
            if self._stopping:
                break
            with suppress(Exception):
                await self._reconcile_active_turns()
            with suppress(Exception):
                await self._flush_unended_turns()

    async def _reconcile_active_turns(self) -> None:
        """Emit the missing turn-end for turns opencode no longer reports busy.

        The status query must run in the same ``directory`` scope the prompt
        used (exactly like ``get_session_state``): ``/session/status`` is scoped
        by ``?directory=``, so an unscoped query omits a workspace-bound session,
        it looks idle, and the turn is "recovered" seconds after it started —
        before ``opencode serve`` produced any reply, so nothing was rendered.
        """

        # Primary set: turns this process started and has not ended.
        candidates = set(self._active_turns)
        # Safety net: known sessions that we have not yet confirmed idle. These
        # cover the case where the connector restarted mid-turn (so _active_turns
        # is empty) or where an idle event was lost and the active-turn bookkeeping
        # somehow desynced. Without this, a stuck "running" session can only be
        # recovered when the user reopens it (get_session_state self-heal).
        for native_id in self._session_dirs:
            if native_id not in self._pushed_idle:
                candidates.add(native_id)
        if not candidates:
            return
        recovered = []
        for native_id in list(candidates):
            try:
                status_map = await self._client.session_status_map(
                    self._directory_for(native_id)
                )
            except OpenCodeClientError as exc:
                logger.debug("opencode reconcile skipped native={}: {}", native_id, exc)
                continue
            entry = status_map.get(native_id)
            if isinstance(entry, dict) and entry.get("type") == "busy":
                continue
            recovered.append(native_id)
        for native_id in recovered:
            platform_id = self._platform_id(native_id)
            if native_id in self._active_turns or native_id in self._recently_ended:
                logger.info(
                    "opencode reconcile: turn end recovered for platform={} native={} (idle event missed)",
                    platform_id,
                    native_id,
                )
                await self._finish_turn(native_id)
            else:
                # Safety-net session (not tracked by this process, not recently
                # ended). Serve says it is idle; just nudge the platform back to
                # idle without emitting turnEnded, which would be wrong for a
                # turn this connector never started.
                logger.info(
                    "opencode reconcile: idle safety-net push for platform={} native={}",
                    platform_id,
                    native_id,
                )
                self._pushed_idle.add(native_id)
                with suppress(Exception):
                    await self.host.session_state_update(
                        platform_id, "opencode", status="idle", external_session_id=native_id
                    )

    async def _finish_turn(self, native_id: str) -> None:
        """End the tracked turn; aborts report ``interrupted``, rest ``completed``.

        An idle event for a turn this process never tracked still ends the
        platform turn: that is what frees sessions that were mid-turn across a
        connector restart. A turn already reported (via the abort reconciler or
        an earlier idle) is not reported twice.
        """

        interrupted = native_id in self._aborted_turns
        was_tracked = interrupted or native_id in self._active_turns
        self._active_turns.discard(native_id)
        self._aborted_turns.discard(native_id)
        platform_id = self._platform_id(native_id)
        logger.debug(
            "opencode finish_turn: platform={} native={} was_tracked={} interrupted={} recently_ended={}",
            platform_id,
            native_id,
            was_tracked,
            interrupted,
            self._recently_ended.get(native_id),
        )
        if was_tracked:
            self._recently_ended[native_id] = "interrupted" if interrupted else "completed"
        elif self._recently_ended.get(native_id) is not None:
            await self._publish_timeline_safe(platform_id, native_id)
            self._pending_messages.unresolve(native_id)
            await self._flush_unended_turns()
            return
        outcome = self._recently_ended.get(native_id, "completed")
        await self._publish_timeline_safe(platform_id, native_id)
        self._pending_messages.unresolve(native_id)
        await self._report_turn_end(platform_id, native_id, outcome)
        await self._flush_unended_turns()

    async def _report_turn_end(self, platform_id: str, native_id: str, outcome: str) -> None:
        """Deliver turn-end + idle state to the host, or remember it for retry.

        The client's "working" indicator tracks the pushed session state, so
        the idle update is as load-bearing as ``turnEnded`` itself (the codex
        adapter sends both on every turn end). A lost pair leaves the platform
        session stuck in "running": the server then never forwards follow-up
        messages, and the client cannot interact again even after reopening
        the session.
        """

        try:
            await self.host.session_turn_ended(
                platform_id, "opencode", external_session_id=native_id, outcome=outcome
            )
            await self.host.session_state_update(
                platform_id, "opencode", status="idle", external_session_id=native_id
            )
            self._pushed_idle.add(native_id)
            logger.info(
                "opencode turn-end delivered: platform={} native={} outcome={}",
                platform_id,
                native_id,
                outcome,
            )
        except Exception as exc:  # noqa: BLE001 - retry through the watchdog
            logger.warning(
                "opencode turn-end delivery failed; will retry: platform={} native={} error={}",
                platform_id,
                native_id,
                exc,
            )
            self._unended_turns[native_id] = outcome

    async def _flush_unended_turns(self) -> None:
        if not self._unended_turns:
            return
        for native_id, outcome in list(self._unended_turns.items()):
            try:
                await self.host.session_turn_ended(
                    self._platform_id(native_id),
                    "opencode",
                    external_session_id=native_id,
                    outcome=outcome,
                )
                await self.host.session_state_update(
                    self._platform_id(native_id),
                    "opencode",
                    status="idle",
                    external_session_id=native_id,
                )
            except Exception as exc:  # noqa: BLE001 - keep retrying
                logger.debug("opencode turn-end retry failed native={} error={}", native_id, exc)
                continue
            logger.info(
                "opencode turn-end retried successfully: platform={} native={}",
                self._platform_id(native_id),
                native_id,
            )
            self._pushed_idle.add(native_id)
            self._unended_turns.pop(native_id, None)

    async def _publish_timeline_safe(
        self, session_id: str, external_session_id: str
    ) -> None:
        try:
            await self._publish_timeline(session_id, external_session_id)
        except Exception as exc:  # noqa: BLE001 - snapshot failures must not block turn-end
            logger.warning(
                "opencode timeline publish failed: platform={} native={} error={}",
                session_id,
                external_session_id,
                exc,
            )

    async def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        props = event.get("properties") or {}
        native_id = props.get("sessionID")
        if not native_id:
            return
        platform_id = self._platform_id(native_id)
        logger.debug(
            "opencode event: type={} platform={} native={}", etype, platform_id, native_id
        )
        if etype == "session.idle":
            self._idle_seq[native_id] = self._idle_seq.get(native_id, 0) + 1
            logger.info(
                "opencode session.idle: platform={} native={}", platform_id, native_id
            )
            await self._finish_turn(native_id)
        elif etype == "session.status":
            status_type = (props.get("status") or {}).get("type")
            logger.debug(
                "opencode event session.status: platform={} native={} status={} active={}",
                platform_id,
                native_id,
                status_type,
                native_id in self._active_turns,
            )
            if status_type == "busy":
                # Only a turn this process started and has not yet ended may
                # drive the platform session into "running". ``opencode serve``
                # replays a session's current status on every SSE (re)connect,
                # so an untracked busy can arrive long after the turn finished
                # (its idle already reported). Reflecting it would flip the
                # session back to "running" and, because the watchdog only
                # reconciles tracked turns, nothing would ever restore idle —
                # the client's send button would stay disabled after the turn
                # already showed as ended. Codex ties running to an active turn
                # id for the same reason.
                if native_id not in self._active_turns:
                    logger.debug(
                        "opencode session.status busy ignored (no active turn): platform={} native={}",
                        platform_id,
                        native_id,
                    )
                    return
                logger.info(
                    "opencode session.status busy: platform={} native={}", platform_id, native_id
                )
                self._pushed_idle.discard(native_id)
                with suppress(Exception):
                    await self.host.session_state_update(
                        platform_id,
                        "opencode",
                        status="running",
                        external_session_id=native_id,
                    )
            elif status_type == "idle":
                # ``session.status idle`` precedes ``session.idle`` on the wire
                # (verified against 1.18.34), so it ends the turn a beat
                # earlier. Only tracked turns are ended here: a replayed idle
                # for an unrelated session must not fabricate a turn-end, and
                # untracked-but-stale sessions are covered by the watchdog's
                # safety-net reconcile.
                if native_id in self._active_turns or native_id in self._recently_ended:
                    logger.info(
                        "opencode session.status idle: platform={} native={}",
                        platform_id,
                        native_id,
                    )
                    self._idle_seq[native_id] = self._idle_seq.get(native_id, 0) + 1
                    await self._finish_turn(native_id)
                else:
                    logger.debug(
                        "opencode session.status idle ignored (untracked): platform={} native={}",
                        platform_id,
                        native_id,
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
