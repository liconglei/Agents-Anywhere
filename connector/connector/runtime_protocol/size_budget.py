"""Keep one timeline notification inside the backend frame budget.

A single ``timeline.itemUpsert`` must fit in ``CONNECTOR_WS_MAX_NOTIFICATION_BYTES``
(900 KiB). A tool that dumps a large file (``cat`` a 4 MiB yml, ``git diff`` of
a vendored tree) puts its whole output in ``ToolCallContent.output``, so the
frame is refused and the item never reaches the platform -- which used to end
the turn with no visible cause.

Trimming happens here rather than in every runtime adapter: the oversize item
is the same shape no matter which adapter produced it, and adapters should not
each invent their own limit. Oversize strings are kept head-first, because the
beginning of a tool result carries the part a reader actually acts on.
"""

from __future__ import annotations

import json
from typing import Any

# Leaves room for the rest of the item envelope (ids, source, metadata) plus the
# JSON escaping of the kept text.
MAX_TIMELINE_TEXT_BYTES = 256 * 1024
TRUNCATION_NOTICE = "\n\n[内容过大，已截断；完整输出未在时间线中传输]"


def _utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def _truncate_text(value: str, max_bytes: int) -> str:
    if _utf8_len(value) <= max_bytes:
        return value
    budget = max(max_bytes - _utf8_len(TRUNCATION_NOTICE), 0)
    if budget <= 0:
        return TRUNCATION_NOTICE.strip()
    encoded = value.encode("utf-8")[:budget]
    # A multi-byte character split by the cut is not valid UTF-8 on its own.
    cut = encoded.decode("utf-8", errors="ignore")
    # JSON escaping grows the payload again (quotes, backslashes, control
    # characters cost up to 6 bytes each), so measure the encoded form and keep
    # trimming until the serialized result really fits.
    while cut and len(json.dumps(cut + TRUNCATION_NOTICE, ensure_ascii=False).encode("utf-8")) > max_bytes:
        cut = cut[: max(len(cut) - 4096, len(cut) // 2)]
    return cut + TRUNCATION_NOTICE


def _shrink(value: Any, max_bytes: int) -> Any:
    """Recursively bound ``value``'s serialized size, preserving its shape."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value, max_bytes)
    if isinstance(value, dict):
        # Spread the budget over the keys, so no single oversized entry wins.
        share = max(max_bytes // max(len(value), 1), 1024)
        return {key: _shrink(item, share) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        share = max(max_bytes // max(len(value), 1), 1024)
        return [_shrink(item, share) for item in value]
    return _shrink(str(value), max_bytes)


def bound_content_field(field: Any, max_bytes: int = MAX_TIMELINE_TEXT_BYTES) -> Any:
    """Return ``field`` unchanged when small enough, else a bounded copy.

    ``max_bytes`` bounds the *serialized* size, because that is what the frame
    budget measures; JSON escaping can inflate a string well past its own
    length.
    """

    if field is None:
        return None
    if _serialized_size(field) <= max_bytes:
        return field
    bounded = _shrink(field, max_bytes)
    # Structure shrinking spreads the budget; a wide container can still land a
    # little over, so finish with one flat pass.
    if _serialized_size(bounded) > max_bytes:
        bounded = _truncate_text(_serialized(bounded), max_bytes)
    return bounded


def _serialized(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _serialized_size(value: Any) -> int:
    return _utf8_len(_serialized(value))