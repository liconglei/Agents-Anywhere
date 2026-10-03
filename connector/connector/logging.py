from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from loguru import logger

RpcLogNotifier = Callable[[str, Any], Awaitable[None]]
DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_connector_logging(*, debug: bool = False) -> None:
    level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level, format=DEFAULT_LOG_FORMAT)


class RpcLogSink:
    def __init__(self, notifier: RpcLogNotifier) -> None:
        self.notifier = notifier
        self._tasks: set[asyncio.Task[None]] = set()
        self._sink_id: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def install(
        self,
        *,
        level: str = "INFO",
        remove_default_sink: bool = False,
    ) -> RpcLogSink:
        if remove_default_sink:
            logger.remove()
        # Log lines can also arrive from asyncio.to_thread worker threads;
        # remember the loop the sink was attached to so those lines are still
        # delivered instead of dying on create_task ("no running event loop").
        with suppress(RuntimeError):
            self._loop = asyncio.get_running_loop()
        self._sink_id = logger.add(self._write, level=level, format="{message}")
        return self

    async def close(self) -> None:
        if self._sink_id is not None:
            logger.remove(self._sink_id)
            self._sink_id = None
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _write(self, message: Any) -> None:
        record = message.record
        payload = {
            "time": record["time"].isoformat(),
            "level": record["level"].name,
            "name": record["name"],
            "message": record["message"],
        }
        exception = record.get("exception")
        if exception is not None:
            payload["exception"] = str(exception)

        task = None

        def _emit() -> None:
            nonlocal task
            task = asyncio.create_task(self.notifier("connector/log", payload))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running is self._loop:
            _emit()
            return
        if self._loop is not None and not self._loop.is_closed():
            with suppress(RuntimeError):
                self._loop.call_soon_threadsafe(_emit)


def install_rpc_log_sink(
    notifier: RpcLogNotifier,
    *,
    level: str = "INFO",
    remove_default_sink: bool = False,
) -> RpcLogSink:
    return RpcLogSink(notifier).install(
        level=level,
        remove_default_sink=remove_default_sink,
    )
