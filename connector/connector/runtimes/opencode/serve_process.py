"""Ownership of an auto-started ``opencode serve`` child process.

Responsibilities:
- resolve a usable ``opencode`` executable (Windows-safe PATH lookup,
  ``--version`` validation, login-shell PATH recovery for GUI-launched
  connectors, ``npx`` fallback), mirroring the codex adapter's binary rules;
- spawn the child with its output pipes continuously drained (an undrained
  PIPE fills and blocks the serve process);
- monitor unexpected child exit and notify the runtime so it can restart
  with backoff.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from connector.launch import launch_target
from connector.logging import logger

LOGIN_SHELL_PATH_MARKER = "__AGENTS_ANYWHERE_PATH__"
VERSION_TIMEOUT_S = 15.0
VERSION_DRAIN_TIMEOUT_S = 2.0
HEALTH_POLL_INTERVAL_S = 1.0
DRAIN_FINISH_TIMEOUT_S = 1.0
STOP_WAIT_TIMEOUT_S = 5.0
_OUTPUT_TAIL_LINES = 80

PortInUseMarkers = ("EADDRINUSE", "address already in use", "already listening")

# opencode keeps its own log file open without file sharing, which makes the
# log the single-instance lock: a second instance dies during boot, sometimes
# with a native fastfail and zero output. Surfacing that beats a bare crash
# code, and the message doubles as the user-facing hint.
LOCK_HOLD_HINT = (
    "另一个 opencode 实例正在运行并持有单实例日志锁 "
    "(<data>/opencode/log/opencode.log)，连接器无法再启动第二个 serve。"
    "请关闭其它 opencode TUI/serve 会话（包括 CherryStudio 等内置实例），"
    "或将 serverUrl 指向已运行实例的地址。"
)

_FATAL_WINDOWS_CODES = {
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN (Bun 启动期 fastfail，常见于另一 opencode 实例持有日志锁)",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000005: "STATUS_ACCESS_VIOLATION",
}


def describe_exit_code(code: int | None) -> str:
    """Human-readable exit code, decoding fatal NTSTATUS values on Windows."""

    if code is None:
        return "code=unknown"
    name = _FATAL_WINDOWS_CODES.get(code)
    if code >= 0x80000000:
        decoded = f"0x{code:08X}"
        if name:
            decoded = f"{decoded} {name}"
        return f"code={code} ({decoded})"
    return f"code={code}"


def opencode_log_path(environment: dict[str, str]) -> str:
    xdg = str(environment.get("XDG_DATA_HOME") or "").strip()
    if xdg:
        return os.path.join(xdg, "opencode", "log", "opencode.log")
    home = str(environment.get("USERPROFILE") or "").strip() or os.path.expanduser("~")
    return os.path.join(home, ".local", "share", "opencode", "log", "opencode.log")


def is_serve_log_lock_held(environment: dict[str, str]) -> bool:
    """True when a live process holds opencode's single-instance log lock.

    Blocking (one CreateFileW probe); event-loop callers must run this in a
    thread. Only Windows exposes share-mode conflicts, so other platforms
    report False and rely on the spawn result itself.
    """

    if sys.platform != "win32":
        return False
    path = opencode_log_path(environment)
    if not os.path.isfile(path):
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    generic_read = 0x80000000
    open_existing = 3
    handle = kernel32.CreateFileW(path, generic_read, 0, None, open_existing, 0, None)
    invalid = ctypes.c_void_p(-1).value
    if handle is None or handle in (-1, invalid):
        error = ctypes.get_last_error()
        sharing_violation, lock_violation = 32, 33
        return error in (sharing_violation, lock_violation)
    with suppress(Exception):
        kernel32.CloseHandle(handle)
    return False


def find_executable_on_path(name: str, path_value: str | None) -> str | None:
    """Locate an executable the same way the child process would resolve it.

    On Windows, npm installs an extensionless POSIX shim plus ``.cmd``/``.ps1``
    wrappers; ``shutil.which`` can return the broken shim, so probe the real
    suffixes explicitly.
    """

    if path_value is None:
        return None
    if sys.platform == "win32":
        # npm installs an extensionless POSIX /bin/sh shim next to the real
        # .cmd wrapper; probing "" first would pick the shim, which the
        # version check can never run (slow npx fallback every boot).
        if os.path.splitext(name)[1]:
            suffixes: tuple[str, ...] = ("",)
        else:
            suffixes = (".exe", ".com", ".cmd", ".bat", ".ps1", "")
        for directory in path_value.split(os.pathsep):
            if not directory:
                continue
            for suffix in suffixes:
                candidate = os.path.join(os.path.expanduser(directory.strip('"')), name + suffix)
                if os.path.isfile(candidate):
                    return candidate
        return None
    from connector.launch import path_exists_for_launch

    for directory in path_value.split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(os.path.expanduser(directory), name)
        if path_exists_for_launch(candidate):
            return candidate
    return None


def read_login_shell_path(shell: str | None = None) -> str | None:
    """Return PATH as the user's login shell sees it, or None.

    A GUI-launched connector inherits a minimal PATH; reading the login
    shell's environment recovers toolchains installed via shell rc files.
    """

    if sys.platform == "win32":
        return None
    selected_shell = shell or os.environ.get("SHELL")
    if not selected_shell:
        import pwd

        try:
            selected_shell = pwd.getpwuid(os.getuid()).pw_shell
        except KeyError:
            return None
    command = f'printf "{LOGIN_SHELL_PATH_MARKER}%s\\n" "$PATH"'
    try:
        completed = subprocess.run(
            [selected_shell, "-lic", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("opencode login shell PATH read failed shell={} error={}", shell, exc)
        return None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(LOGIN_SHELL_PATH_MARKER):
            return line.removeprefix(LOGIN_SHELL_PATH_MARKER) or None
    return None


def serve_environment() -> dict[str, str]:
    env = dict(os.environ)
    shell_path = read_login_shell_path()
    if shell_path:
        env["PATH"] = shell_path
    return env


def resolve_serve_command(
    port: int,
    environment: dict[str, str] | None = None,
    *,
    hostname: str = "127.0.0.1",
) -> list[str]:
    """Build the ``opencode serve`` argv, validating candidates with --version.

    Falls back to ``npx`` when the system binary is missing or fails the
    version check (same semantics as the codex binary selection).
    """

    env = environment or os.environ
    path_value = env.get("PATH")
    reason: str | None = None
    candidate = find_executable_on_path("opencode", path_value)
    if candidate is not None:
        probe_started = time.monotonic()
        error = check_version_output(candidate, env)
        probe_elapsed = time.monotonic() - probe_started
        if error is None:
            logger.info(
                "opencode resolve: version probe ok in {:.2f}s candidate={}",
                probe_elapsed,
                candidate,
            )
            return [candidate, "serve", "--hostname", hostname, "--port", str(port)]
        reason = f"system opencode failed version check: {error}"
        logger.warning(
            "{} after {:.2f}s; falling back to npx path={}",
            reason,
            probe_elapsed,
            candidate,
        )
    else:
        reason = "opencode executable was not found on PATH"
        logger.info("opencode resolve: no system opencode on PATH; trying npx")

    npx = find_executable_on_path("npx", path_value)
    if npx is not None:
        return [npx, "opencode", "serve", "--hostname", hostname, "--port", str(port)]
    raise RuntimeError(f"opencode executable not found on PATH (and no npx fallback): {reason}")


def check_version_output(candidate: str, environment: dict[str, str]) -> str | None:
    """Return None when `<candidate> --version` looks like a real opencode CLI.

    Blocking; callers on the event loop must run this in a thread. On Windows a
    plain timeout only kills the direct child (powershell/cmd), while a
    grandchild holding the output pipes can hang ``communicate`` indefinitely —
    so the whole process tree is killed on timeout.
    """

    target = launch_target("opencode", candidate)
    command = target.command(["--version"])
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.Popen(
            command,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc) or exc.__class__.__name__
    try:
        stdout, _ = proc.communicate(timeout=VERSION_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            stdout, _ = proc.communicate(timeout=VERSION_DRAIN_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - timed-out child, best-effort drain
            stdout = ""
        return f"timed out after {int(VERSION_TIMEOUT_S)}s"
    if proc.returncode != 0:
        return f"exited with code {proc.returncode}"
    if re.search(r"(?m)^\D*\d+\.\d+(\.\d+)?\S*\s*$", stdout or "") is None:
        return "missing or invalid opencode version output"
    return None


def _kill_process_tree(proc: subprocess.Popen[str] | asyncio.subprocess.Process) -> None:
    if sys.platform == "win32":
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=5.0,
            )
        return
    with suppress(OSError):
        proc.kill()


def is_port_in_use(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in PortInUseMarkers)


class ServeProcess:
    """One supervised ``opencode serve`` child with drained output pipes.

    ``on_exit`` fires when the child terminates while not intentionally
    stopped. It receives the exit code and an output-tail string.
    """

    def __init__(
        self,
        *,
        on_exit: Callable[[int | None, str], Awaitable[None]] | None = None,
    ) -> None:
        self._on_exit = on_exit
        self.process: asyncio.subprocess.Process | None = None
        self._tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
        self._drain_tasks: tuple[asyncio.Task[None], ...] = ()
        self._monitor_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._exit_delivered = False
        self._spawned_at: float | None = None

    @property
    def tail(self) -> str:
        return "\n".join(self._tail)

    def boot_grace_expired(self, grace_s: float) -> bool:
        """True once the child has been alive longer than the boot grace."""

        if self.process is None or self._spawned_at is None:
            return False
        if self.process.returncode is not None:
            return True
        return asyncio.get_running_loop().time() - self._spawned_at > grace_s

    async def spawn(self, command: list[str], environment: dict[str, str] | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._stopping = False
        self.process = await asyncio.create_subprocess_exec(
            *command,
            # Never let the child inherit the connector's stdin: this process
            # speaks JSON-RPC over stdio with the desktop supervisor.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            **kwargs,
        )
        self._spawned_at = asyncio.get_running_loop().time()
        self._drain_tasks = tuple(
            t
            for t in (
                asyncio.create_task(self._drain(self.process.stdout), name="opencode-serve-stdout"),
                asyncio.create_task(self._drain(self.process.stderr), name="opencode-serve-stderr"),
            )
            if t is not None
        )
        self._monitor_task = asyncio.create_task(self._monitor(), name="opencode-serve-monitor")
        logger.info("opencode serve spawned pid={} command={}", self.process.pid, " ".join(command))

    async def _drain(self, reader: asyncio.StreamReader | None) -> None:
        if reader is None:
            return
        while True:
            try:
                line = await reader.readline()
            except (asyncio.LimitOverrunError, ValueError) as exc:  # pragma: no cover - oversized line
                logger.debug("opencode serve output read error: {}", exc)
                continue
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self._tail.append(text)
                logger.debug("opencode serve: {}", text)

    async def _finish_drain(self) -> None:
        """Let the output tasks consume buffered pipe data (EOF after exit)."""

        for task in self._drain_tasks:
            with suppress(Exception):
                await asyncio.wait_for(task, timeout=DRAIN_FINISH_TIMEOUT_S)
        self._drain_tasks = ()

    async def _monitor(self) -> None:
        proc = self.process
        if proc is None:
            return
        try:
            code = await proc.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - pragma: no cover - defensive
            logger.warning("opencode serve monitor failed: {}", exc)
            return
        await self._finish_drain()
        if self._stopping:
            return
        logger.warning(
            "opencode serve exited unexpectedly {} tail={}",
            describe_exit_code(code),
            self.tail[-400:],
        )
        await self._deliver_exit(code)

    async def _deliver_exit(self, code: int | None) -> None:
        """Fire ``on_exit`` at most once, from whichever path sees the death first."""

        if self._exit_delivered or self._stopping or self._on_exit is None:
            return
        self._exit_delivered = True
        with suppress(Exception):
            await self._on_exit(code, self.tail)

    async def wait_healthy(
        self,
        health_check: Callable[[], Awaitable[Any]],
        unhealthy: type[BaseException],
        timeout_s: float,
    ) -> None:
        """Poll ``health_check`` until it passes; raise on child exit/timeout."""

        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            proc = self.process
            if proc is not None and proc.returncode is not None:
                await self._finish_drain()
                if proc.returncode != 0:
                    logger.warning(
                        "opencode serve exited early {} tail={}",
                        describe_exit_code(proc.returncode),
                        self.tail[-400:],
                    )
                    await self._deliver_exit(proc.returncode)
                raise RuntimeError(
                    f"opencode serve exited early ({describe_exit_code(proc.returncode)}): "
                    f"{self.tail[-400:]}"
                )
            try:
                await health_check()
                return
            except unhealthy:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"opencode serve did not become healthy within {int(timeout_s)}s"
                )
            await asyncio.sleep(HEALTH_POLL_INTERVAL_S)

    async def stop(self) -> None:
        self._stopping = True
        tasks = [self._monitor_task, *self._drain_tasks]
        self._monitor_task = None
        self._drain_tasks = ()
        for task in tasks:
            if task is not None:
                task.cancel()
        for task in tasks:
            if task is not None:
                with suppress(BaseException):
                    await task
        proc = self.process
        self.process = None
        if proc is None or proc.returncode is not None:
            return
        logger.info("opencode stop: terminating auto-started serve (pid={})", proc.pid)
        if sys.platform == "win32":
            # ``terminate()`` would only kill the direct child (cmd/npx wrapper);
            # the real opencode.exe grandchild would survive and keep holding
            # the serve port. Kill the whole tree instead.
            with suppress(Exception):
                await asyncio.to_thread(_kill_process_tree, proc)
        else:
            with suppress(Exception):
                proc.terminate()
        with suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=STOP_WAIT_TIMEOUT_S)
        if proc.returncode is None:
            with suppress(Exception):
                proc.kill()
