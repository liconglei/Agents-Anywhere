from __future__ import annotations

from typing import Any

from connector.runtime_protocol import RuntimeInvalidRequestError

DEFAULT_SERVER_URL = "http://127.0.0.1:4096"
DEFAULT_REQUEST_TIMEOUT_S = 60.0
DEFAULT_STREAM_TIMEOUT_S = 300.0


def opencode_config_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "serverUrl": {
                "type": "string",
                "minLength": 1,
                "pattern": "^https?://",
                "title": "OpenCode serve URL",
                "description": "Base URL of a local `opencode serve` instance (REST + SSE).",
                "default": DEFAULT_SERVER_URL,
            },
            "apiKey": {
                "type": "string",
                "title": "Server password",
                "description": "Optional; sent as Authorization Bearer when opencode serve runs with OPENCODE_SERVER_PASSWORD.",
            },
            "requestTimeoutSeconds": {
                "type": "number",
                "minimum": 1,
                "maximum": 600,
                "default": DEFAULT_REQUEST_TIMEOUT_S,
                "title": "Request timeout (s)",
            },
        },
        "additionalProperties": False,
    }


def default_config_values() -> dict[str, Any]:
    return {
        "serverUrl": DEFAULT_SERVER_URL,
        "requestTimeoutSeconds": DEFAULT_REQUEST_TIMEOUT_S,
    }


def normalized_config_values(raw: dict[str, Any]) -> dict[str, Any]:
    values = {**default_config_values(), **raw}
    url = values.get("serverUrl")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise RuntimeInvalidRequestError("serverUrl must be an http(s) URL")
    values["serverUrl"] = url.rstrip("/")
    timeout = values.get("requestTimeoutSeconds")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 1 <= timeout <= 600:
        raise RuntimeInvalidRequestError("requestTimeoutSeconds must be a number between 1 and 600")
    values["requestTimeoutSeconds"] = float(timeout)
    key = values.get("apiKey")
    if key is not None and not isinstance(key, str):
        raise RuntimeInvalidRequestError("apiKey must be a string")
    if not key:
        values.pop("apiKey", None)
    return values


def opencode_capabilities() -> dict[str, bool]:
    return {
        "modelCatalog": True,
        "permissionCatalog": False,
        "sessionDiscovery": True,
        "sessionSnapshot": True,
        "sessionState": True,
        "sessionNotices": False,
        "createAndStartSession": True,
        "startTurn": True,
        "steerTurn": False,
        "interruptTurn": True,
        "commands": False,
        "interactions": False,
        "attachments": False,
        "ipc": True,
    }
