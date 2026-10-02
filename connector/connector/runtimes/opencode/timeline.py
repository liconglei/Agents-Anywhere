"""Map opencode session messages (info + parts) to platform timeline items.

OpenCode stores a session as an ordered list of messages; each message has
``info`` (role, ids, tokens, timing) and ``parts`` (text / reasoning / tool /
step-start / step-finish). The platform timeline is a flat, ordered list of
typed items, so we expand parts in message order and drop internal
``step-start``/``step-finish`` markers.
"""

from __future__ import annotations

from typing import Any

from connector.runtime_protocol.timeline import (
    MarkdownMessageContent,
    MessageTimelineItem,
    PlatformTimelineItem,
    ReasoningSystemContent,
    SystemTimelineItem,
    TextMessageContent,
    TimelineSource,
    ToolCallContent,
    ToolTimelineItem,
)

RUNTIME = "opencode"

_TOOL_STATUS_MAP = {
    "completed": "done",
    "running": "running",
    "error": "failed",
}


def _source(
    external_session_id: str | None,
    native_item_id: str,
    turn_id: str | None = None,
) -> TimelineSource:
    return TimelineSource(
        runtime=RUNTIME,
        external_session_id=external_session_id,
        native_item_id=native_item_id,
        turn_id=turn_id,
    )


def _tool_status(state: dict[str, Any] | None) -> str:
    status = (state or {}).get("status")
    return _TOOL_STATUS_MAP.get(status, "pending")


def map_messages_to_timeline(
    external_session_id: str | None,
    messages: list[dict[str, Any]],
) -> tuple[PlatformTimelineItem, ...]:
    """Expand opencode messages into an ordered tuple of platform items."""

    items: list[PlatformTimelineItem] = []

    for message in messages:
        info = message.get("info") or {}
        parts = message.get("parts") or []
        role = info.get("role")
        message_id = str(info.get("id") or "")
        if not message_id:
            continue

        # Turn id: an assistant message points back to the user message via
        # parentID; a user message is itself the turn anchor.
        turn_id = info.get("parentID") if role == "assistant" else message_id
        turn_id = str(turn_id) if turn_id else None

        if role == "user":
            text = "\n".join(
                str(part.get("text") or "")
                for part in parts
                if part.get("type") == "text"
            ).strip()
            if not text:
                continue
            items.append(
                MessageTimelineItem(
                    id=message_id,
                    type="message",
                    status="done",
                    role="user",
                    turn_id=turn_id,
                    content=TextMessageContent(text=text),
                    source=_source(external_session_id, message_id, turn_id),
                )
            )
            continue

        if role != "assistant":
            continue

        for part in parts:
            part_type = part.get("type")
            part_id = str(part.get("id") or f"{message_id}:{part_type}:{len(items)}")

            if part_type == "text":
                text = str(part.get("text") or "").strip()
                if not text:
                    continue
                items.append(
                    MessageTimelineItem(
                        id=part_id,
                        type="message",
                        status="done",
                        role="assistant",
                        turn_id=turn_id,
                        content=MarkdownMessageContent(text=text),
                        source=_source(external_session_id, part_id, turn_id),
                    )
                )

            elif part_type == "reasoning":
                text = str(part.get("text") or "").strip()
                if not text:
                    continue
                items.append(
                    SystemTimelineItem(
                        id=part_id,
                        type="system",
                        status="done",
                        role="system",
                        turn_id=turn_id,
                        content=ReasoningSystemContent(text=text),
                        source=_source(external_session_id, part_id, turn_id),
                    )
                )

            elif part_type == "tool":
                state = part.get("state") or {}
                tool_name = str(part.get("tool") or "tool")
                items.append(
                    ToolTimelineItem(
                        id=part_id,
                        type="tool",
                        status=_tool_status(state),
                        role="tool",
                        turn_id=turn_id,
                        content=ToolCallContent(
                            title=tool_name,
                            input=state.get("input"),
                            output=state.get("output"),
                        ),
                        source=_source(external_session_id, part_id, turn_id),
                        metadata={"tool": tool_name, "callID": part.get("callID")},
                    )
                )

            # step-start / step-finish and other internal parts are dropped.

    return tuple(items)


def derive_session_status(messages: list[dict[str, Any]]) -> str:
    """Derive a runtime status from the tail of the message list.

    - last assistant message has no ``completed`` time  -> running
    - last message is a user message (turn started)     -> running
    - otherwise                                          -> idle
    """

    if not messages:
        return "idle"
    last = messages[-1]
    info = last.get("info") or {}
    role = info.get("role")
    if role == "assistant":
        completed = (info.get("time") or {}).get("completed")
        return "idle" if completed else "running"
    if role == "user":
        return "running"
    return "idle"
