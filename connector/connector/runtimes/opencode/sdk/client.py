"""Async HTTP + SSE client for a local ``opencode serve`` instance.

Endpoint map (verified against opencode 1.18.34). opencode serves two
namespaces: most routes live at the root, while question replies and the
per-session model switch only exist under ``/api``. A route that does not exist
falls through to the web UI, which answers **HTTP 200 with an HTML body** — so
a typo in a path looks like success. Both variants of the paths below were
probed; the ones not listed here return the SPA fallback:

- ``GET  /global/health``               health
- ``GET  /session``                     list sessions
- ``POST /session``                     create session (body optional)
- ``GET  /session/{id}``                session info (carries ``agent``/``model``)
- ``PATCH /session/{id}``               update (title; ``permission`` merges, see below)
- ``GET  /session/{id}/message``        message list (info + parts)
- ``POST /session/{id}/message``        synchronous prompt (blocks until done)
- ``POST /session/{id}/prompt_async``   fire-and-forget prompt (204)
- ``POST /session/{id}/abort``          abort the active turn
- ``POST /session/{id}/command``        run a user-defined skill
- ``POST /session/{id}/summarize``      compact the conversation
- ``POST /session/{id}/fork``           fork the session
- ``POST /session/{id}/share``          create a share link
- ``POST /session/{id}/revert``         revert one message (needs ``messageID``)
- ``POST /session/{id}/permissions/{id}`` answer a permission request
- ``GET  /session/status``              busy/idle map for all sessions
- ``GET  /agent``                       agents (primary/subagent + permissions)
- ``GET  /provider``                    providers and their models
- ``GET  /command``                     user-defined commands (**needs ``directory``**)
- ``POST /api/session/{id}/model``      switch the session model (V2 only)
- ``POST /api/session/{id}/question/{id}/reply`` answer a question (V2 only)
- ``GET  /global/event``                global SSE stream (all project instances)
- ``GET  /event``                       instance-scoped SSE (only the serve cwd's project)
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from urllib.parse import quote

import httpx


class OpenCodeClientError(RuntimeError):
    """Raised when the opencode server rejects a request or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def append_directory_param(path: str, directory: str) -> str:
    """Bind a request to one project directory.

    ``opencode serve`` multiplexes one instance per working directory; without
    the ``directory`` parameter every request resolves to the serve process
    cwd, silently ignoring the workspace the user selected. Some routes *require*
    it: ``GET /command`` answers 503 without a directory.
    """

    separator = "&" if "?" in path else "?"
    return f"{path}{separator}directory={quote(directory, safe='')}"


def _is_html(content_type: str, body: str) -> bool:
    """True when a response is the opencode web UI instead of an API payload."""

    if "html" in content_type.lower():
        return True
    head = body.lstrip()[:64].lower()
    return head.startswith(("<!doctype html", "<html"))


# Read budget for the auto-start "is a serve already there?" precheck. The
# client's default connect timeout is generous on purpose (a busy serve can be
# slow to answer), but paying it while nothing is listening only delays the spawn.
HEALTH_FAST_TIMEOUT_S = 1.5


class OpenCodeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        stream_timeout: float | None = None,
        api_user: str = "opencode",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_user = api_user
        self.timeout = timeout
        # /event can legitimately go quiet for a whole turn; request timeouts
        # must not cut the stream and drop events in the gap.
        self.stream_timeout = stream_timeout if stream_timeout is not None else timeout
        self._http: httpx.AsyncClient | None = None

    def _authorization(self) -> str:
        # opencode serve protects itself with HTTP Basic (username defaults to
        # "opencode", overridable via OPENCODE_SERVER_USERNAME).
        token = base64.b64encode(f"{self.api_user}:{self.api_key}".encode()).decode()
        return f"Basic {token}"

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = self._authorization()
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def set_credentials(self, api_user: str, api_key: str) -> None:
        """Adopt credentials discovered after construction.

        ``opencode serve`` inherits ``OPENCODE_SERVER_PASSWORD`` (and possibly
        ``OPENCODE_SERVER_USERNAME``) from the environment, so an auto-started
        instance requires Basic auth even when the runtime config left
        ``apiKey`` empty.
        """

        if not api_key:
            return
        self.api_user = api_user
        self.api_key = api_key
        if self._http is not None:
            self._http.headers["Authorization"] = self._authorization()

    # ------------------------------------------------------------------ #
    # REST
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        directory: str | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> Any:
        client = await self._client()
        if directory:
            path = append_directory_param(path, directory)
        try:
            response = await client.request(method, path, json=json_body, timeout=timeout)
        except httpx.HTTPError as exc:
            raise OpenCodeClientError(f"opencode server unreachable: {exc}") from exc
        if response.status_code >= 400:
            # Surface the response body so callers see opencode's actual error
            # detail (e.g. ``{"_tag":"BadRequest","data":{"message":"..."}}``)
            # instead of just the bare status code, which is useless for triage.
            body_excerpt = (response.text or "").strip()
            if len(body_excerpt) > 400:
                body_excerpt = body_excerpt[:400] + "..."
            raise OpenCodeClientError(
                f"opencode {method} {path} failed: {response.status_code}"
                + (f" body={body_excerpt}" if body_excerpt else ""),
                status_code=response.status_code,
            )
        # 204 / empty bodies are normal (prompt_async, model switch).
        if response.status_code == 204 or not response.content:
            return None
        # opencode answers unknown routes with the web UI (HTTP 200 + HTML), so
        # a mistyped path would otherwise look like a successful call.
        if _is_html(response.headers.get("content-type", ""), response.text):
            raise OpenCodeClientError(
                f"opencode {method} {path} answered {response.status_code} with the "
                "web UI instead of JSON; the route does not exist on this opencode "
                "version"
            )
        return response.json()

    async def health(self, *, fast: bool = False) -> dict[str, Any] | None:
        """``GET /global/health``.

        ``fast`` uses a short connect timeout for the "is anything listening?"
        precheck on the auto-start path: with the client's default (10s connect)
        a dead port costs several seconds of ADE attempts before the runtime is
        even allowed to spawn, and the client shows ``starting`` throughout.
        """

        timeout = httpx.Timeout(HEALTH_FAST_TIMEOUT_S, connect=1.0) if fast else None
        return await self._request("GET", "/global/health", timeout=timeout)

    async def list_sessions(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/session")
        return data if isinstance(data, list) else []

    async def list_agents(self, directory: str | None = None) -> list[dict[str, Any]]:
        """List opencode agents (build/plan/compaction/...).

        Each ``primary`` agent is a mode a user can switch to (``build`` is the
        read-write default, ``plan`` is read-only) and carries the
        ``permission`` ruleset opencode applies when that agent runs. The
        connector uses the list to expose Build/Plan as permission presets and
        to check that a selected mode exists in this project. Agents resolve
        per project config, so ``directory`` selects which project's agents are
        returned (without it, opencode answers for the serve cwd).
        """

        data = await self._request("GET", "/agent", directory=directory)
        return data if isinstance(data, list) else []

    # NOTE: there is deliberately no "switch session permission mode" method.
    # ``PATCH /session/{id}`` with ``{"permission": [...]}`` looks like the
    # obvious API (it is accepted and echoed back), but on serve 1.18.34 the
    # array is appended to the session's existing rules instead of replacing
    # it: switching Plan -> Build left Plan's ``edit * deny`` behind and grew
    # the ruleset from 142 to 186 entries. The mode is switched by sending
    # ``agent`` on the prompt instead (see ``prompt_async``), and opencode
    # records it as ``session["agent"]``.

    async def get_session(self, session_id: str, directory: str | None = None) -> dict[str, Any]:
        data = await self._request("GET", f"/session/{session_id}", directory=directory)
        if not isinstance(data, dict):
            raise OpenCodeClientError(f"session {session_id} not found")
        return data

    async def list_messages(self, session_id: str, directory: str | None = None) -> list[dict[str, Any]]:
        data = await self._request("GET", f"/session/{session_id}/message", directory=directory)
        return data if isinstance(data, list) else []

    async def create_session(
        self,
        title: str | None = None,
        directory: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        data = await self._request("POST", "/session", json_body=body or None, directory=directory)
        if not isinstance(data, dict) or "id" not in data:
            raise OpenCodeClientError("session create returned no session id")
        return data

    async def prompt_async(
        self,
        session_id: str,
        content: str,
        model: dict[str, str] | None = None,
        agent: str | None = None,
        directory: str | None = None,
        extra_parts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        parts: list[dict[str, Any]] = []
        if content:
            parts.append({"type": "text", "text": content})
        parts.extend(dict(part) for part in extra_parts)
        if not parts:
            return
        body: dict[str, Any] = {"parts": parts}
        if model:
            body["model"] = model
        if agent:
            body["agent"] = agent
        await self._request(
            "POST",
            f"/session/{session_id}/prompt_async",
            json_body=body,
            directory=directory,
        )

    async def abort(self, session_id: str, directory: str | None = None) -> None:
        await self._request("POST", f"/session/{session_id}/abort", directory=directory)

    async def reply_permission(
        self,
        session_id: str,
        permission_id: str,
        response: str,
        directory: str | None = None,
    ) -> None:
        """Respond to a permission request.

        ``response`` is one of ``once`` (approve this once), ``always``
        (approve and remember for the session) or ``reject``.
        """

        await self._request(
            "POST",
            f"/session/{session_id}/permissions/{permission_id}",
            json_body={"response": response},
            directory=directory,
        )

    async def reply_question(
        self,
        session_id: str,
        question_id: str,
        reply: str,
        directory: str | None = None,
    ) -> None:
        """Answer a question the model asked mid-turn.

        Only the V2 route exists (``/api/session/{id}/question/{id}/reply``);
        the V1 path falls through to the web UI with HTTP 200. The payload is
        ``{"answers": [[...]]}`` — one array of answer labels per question, in
        the order opencode asked them. A free-form reply is just a one-label
        answer (opencode's own UI does the same for its "type your own answer"
        option); sending ``{"reply": ...}`` is rejected with
        ``Missing key at ["answers"]``.
        """

        await self._request(
            "POST",
            f"/api/session/{session_id}/question/{question_id}/reply",
            json_body={"answers": [[reply]]},
            directory=directory,
        )

    async def reject_question(
        self,
        session_id: str,
        question_id: str,
        directory: str | None = None,
    ) -> None:
        """Dismiss a question without answering it (V2 route)."""

        await self._request(
            "POST",
            f"/api/session/{session_id}/question/{question_id}/reject",
            json_body={},
            directory=directory,
        )

    async def list_commands(self, directory: str | None = None) -> list[dict[str, Any]]:
        data = await self._request("GET", "/command", directory=directory)
        return data if isinstance(data, list) else []

    async def execute_command(
        self,
        session_id: str,
        command: str,
        arguments: str = "",
        directory: str | None = None,
    ) -> dict[str, Any] | None:
        body = {"command": command, "arguments": arguments}
        data = await self._request(
            "POST",
            f"/session/{session_id}/command",
            json_body=body,
            directory=directory,
        )
        return data if isinstance(data, dict) else None

    async def switch_session_model(
        self,
        session_id: str,
        model_id: str,
        provider_id: str,
        variant: str | None = None,
        directory: str | None = None,
    ) -> None:
        """Switch the model used by subsequent turns (``POST /api/session/{id}/model``).

        ``model_id`` is the bare model id without the provider prefix
        (e.g. ``qwen3.8-27b``), and ``provider_id`` is the provider key
        from ``opencode.json`` (e.g. ``local``). The opencode web UI's
        ``/model`` slash command maps to this endpoint.
        """

        model: dict[str, Any] = {"id": model_id, "providerID": provider_id}
        if variant:
            model["variant"] = variant
        await self._request(
            "POST",
            f"/api/session/{session_id}/model",
            json_body={"model": model},
            directory=directory,
        )

    async def summarize_session(
        self,
        session_id: str,
        provider_id: str,
        model_id: str,
        directory: str | None = None,
    ) -> bool:
        """Compact the session conversation (``POST /session/{id}/summarize``).

        The V2 ``POST /api/session/{id}/compact`` exists in the OpenAPI
        document but answers 503 "Session compact is not available yet" on
        1.18.34; summarize is the V1 endpoint the web UI's ``/compact``
        drives. It requires the model that should write the summary.
        """

        data = await self._request(
            "POST",
            f"/session/{session_id}/summarize",
            json_body={"providerID": provider_id, "modelID": model_id},
            directory=directory,
        )
        return bool(data)

    async def fork_session(
        self,
        session_id: str,
        message_id: str | None = None,
        directory: str | None = None,
    ) -> dict[str, Any] | None:
        """Fork a session at ``message_id`` (``POST /session/{id}/fork``).

        Returns the new session object. ``message_id`` is optional; when
        omitted opencode forks at the latest message.
        """

        body: dict[str, Any] = {}
        if message_id:
            body["messageID"] = message_id
        data = await self._request(
            "POST",
            f"/session/{session_id}/fork",
            json_body=body,
            directory=directory,
        )
        return data if isinstance(data, dict) else None

    async def share_session(
        self, session_id: str, directory: str | None = None
    ) -> dict[str, Any] | None:
        """Create a shareable link for the session (``POST /session/{id}/share``)."""

        data = await self._request(
            "POST", f"/session/{session_id}/share", directory=directory
        )
        return data if isinstance(data, dict) else None

    async def revert_latest_message(
        self,
        session_id: str,
        message_id: str | None = None,
        directory: str | None = None,
    ) -> bool:
        """Revert a model message (``POST /session/{id}/revert``).

        Mirrors opencode's ``/undo`` slash command. ``message_id`` is the
        target message to revert to; if omitted, opencode requires the id of
        the latest model message, so this method falls back to the most
        recent assistant message found via ``GET /session/{id}/message``.
        Returns ``True`` if the revert actually ran, ``False`` if there was
        no model message to revert (clean session).
        """

        target = message_id
        if not target:
            messages = await self.list_messages(session_id, directory=directory)
            latest_assistant = None
            for msg in reversed(messages):
                if not isinstance(msg, dict):
                    continue
                # opencode's message shape: ``{"info": {"id", "role", ...},
                # "parts": [...]}``. The role lives under ``info.role``.
                info = msg.get("info") if isinstance(msg.get("info"), dict) else msg
                role = info.get("role")
                mid = info.get("id") or info.get("messageID")
                if mid and role != "user":
                    latest_assistant = mid
                    break
            if not latest_assistant:
                return False
            target = latest_assistant
        await self._request(
            "POST",
            f"/session/{session_id}/revert",
            json_body={"messageID": target},
            directory=directory,
        )
        return True

    async def provider_overview(self) -> dict[str, Any]:
        data = await self._request("GET", "/provider")
        return data if isinstance(data, dict) else {}

    async def session_status_map(self, directory: str | None = None) -> dict[str, Any]:
        data = await self._request("GET", "/session/status", directory=directory)
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ #
    # SSE
    # ------------------------------------------------------------------ #

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed SSE events from the global ``/global/event`` stream.

        ``/event`` is instance-scoped: it only carries events for the project
        the serve process was launched in. Sessions bound to any other
        directory (the common case — the workspace the user picked) publish
        their ``session.status`` / ``session.idle`` events to *their own*
        instance, so an instance-scoped subscription never sees them and the
        platform session can never leave "running" on its own.

        ``/global/event`` carries every instance's events, each wrapped as
        ``{"directory", "project", "payload": {...}}``. The inner ``payload``
        is the same event object the instance-scoped stream yields, so both
        shapes are handled here and callers only ever see the bare event.

        The underlying connection is long-lived; callers must iterate this
        generator inside a task and handle ``OpenCodeClientError`` to
        reconnect.
        """
        client = await self._client()
        async with client.stream(
            "GET",
            "/global/event",
            timeout=httpx.Timeout(self.stream_timeout, connect=10.0),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise OpenCodeClientError(
                    f"opencode /global/event failed: {response.status_code}",
                    status_code=response.status_code,
                )
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                inner = event.get("payload")
                if not isinstance(inner, dict):
                    # Instance-scoped shape (no wrapper) — accept as-is.
                    inner = event
                if inner.get("type") == "sync":
                    # Replay frames duplicate the primary events; handling
                    # both would double-process every state change.
                    continue
                yield inner
