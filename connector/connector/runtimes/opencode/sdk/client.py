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
- ``GET  /event``                       global SSE stream
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OpenCodeClientError(RuntimeError):
    """Raised when the opencode server rejects a request or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenCodeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
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

    # ------------------------------------------------------------------ #
    # REST
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
    ) -> Any:
        client = await self._client()
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

    async def get_session(self, session_id: str) -> dict[str, Any]:
        data = await self._request("GET", f"/session/{session_id}")
        if not isinstance(data, dict):
            raise OpenCodeClientError(f"session {session_id} not found")
        return data

    async def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        data = await self._request("GET", f"/session/{session_id}/message")
        return data if isinstance(data, list) else []

    async def create_session(self, title: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        data = await self._request("POST", "/session", json_body=body or None)
        if not isinstance(data, dict) or "id" not in data:
            raise OpenCodeClientError("session create returned no session id")
        return data

    async def prompt_async(
        self,
        session_id: str,
        content: str,
        model: dict[str, str] | None = None,
        agent: str | None = None,
    ) -> None:
        body: dict[str, Any] = {"parts": [{"type": "text", "text": content}]}
        if model:
            body["model"] = model
        if agent:
            body["agent"] = agent
        await self._request("POST", f"/session/{session_id}/prompt_async", json_body=body)

    async def abort(self, session_id: str) -> None:
        await self._request("POST", f"/session/{session_id}/abort")

    async def provider_overview(self) -> dict[str, Any]:
        data = await self._request("GET", "/provider")
        return data if isinstance(data, dict) else {}

    async def session_status_map(self) -> dict[str, Any]:
        data = await self._request("GET", "/session/status")
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ #
    # SSE
    # ------------------------------------------------------------------ #

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed SSE events from the global ``/event`` stream.

        The underlying connection is long-lived; callers must iterate this
        generator inside a task and handle ``OpenCodeClientError`` to
        reconnect.
        """
        client = await self._client()
        async with client.stream("GET", "/event") as response:
            if response.status_code >= 400:
                await response.aread()
                raise OpenCodeClientError(
                    f"opencode /event failed: {response.status_code}",
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
                if isinstance(event, dict):
                    yield event
