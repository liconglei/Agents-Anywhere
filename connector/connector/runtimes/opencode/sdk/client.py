"""Async HTTP + SSE client for a local ``opencode serve`` instance.

Endpoint map (verified against opencode 1.18.34):

- ``GET  /global/health``               health
- ``GET  /session``                     list sessions
- ``GET  /session/{id}``                session info
- ``GET  /session/{id}/message``        message list (info + parts)
- ``POST /session``                     create session (body optional)
- ``POST /session/{id}/message``        synchronous prompt (blocks until done)
- ``POST /session/{id}/prompt_async``   fire-and-forget prompt (204)
- ``POST /session/{id}/abort``          abort the active turn
- ``GET  /session/status``              busy/idle map for all sessions
- ``GET  /global/event``                global SSE stream (all project instances)
- ``GET  /event``                       instance-scoped SSE (only the serve cwd's project)
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
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
    cwd, silently ignoring the workspace the user selected.
    """

    separator = "&" if "?" in path else "?"
    return f"{path}{separator}directory={quote(directory, safe='')}"


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
    ) -> Any:
        client = await self._client()
        if directory:
            path = append_directory_param(path, directory)
        try:
            response = await client.request(method, path, json=json_body)
        except httpx.HTTPError as exc:
            raise OpenCodeClientError(f"opencode server unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise OpenCodeClientError(
                f"opencode {method} {path} failed: {response.status_code}",
                status_code=response.status_code,
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def health(self) -> dict[str, Any] | None:
        return await self._request("GET", "/global/health")

    async def list_sessions(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/session")
        return data if isinstance(data, list) else []

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
    ) -> None:
        body: dict[str, Any] = {"parts": [{"type": "text", "text": content}]}
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
