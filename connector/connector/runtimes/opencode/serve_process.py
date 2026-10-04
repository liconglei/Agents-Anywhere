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
from pathlib import Path
from typing import Any

from connector.launch import launch_target
from connector.logging import logger

LOGIN_SHELL_PATH_MARKER = "__AGENTS_ANYWHERE_PATH__"
# Budget for one `<candidate> --version` probe. Kept short on purpose: the probe
# only picks a binary, and every candidate it rejects is paid for again on the
# next candidate and again on the next connector restart.
VERSION_TIMEOUT_S = 5.0
VERSION_DRAIN_TIMEOUT_S = 2.0
HEALTH_POLL_INTERVAL_S = 1.0
DRAIN_FINISH_TIMEOUT_S = 1.0
STOP_WAIT_TIMEOUT_S = 5.0
_OUTPUT_TAIL_LINES = 80

PortInUseMarkers = ("EADDRINUSE", "address already in use", "already listening")

# opencode logs to stderr instead of taking <data>/opencode/log/opencode.log
# exclusively, which is what used to make a second instance impossible.
# ``ServeProcess`` drains stderr into its ring buffer. It reduces the collision
# with another opencode but is not a guarantee on 1.18.34 -- see
# ``is_serve_log_lock_failure``.
SERVE_LOG_FLAGS: tuple[str, ...] = ("--print-logs", "--log-level", "INFO")

# opencode keeps its own log file open while any instance runs (TUI, CLI or
# serve). It used to be treated as a single-instance lock that made a second
# ``serve`` die during boot, sometimes with a native fastfail and zero output.
# On 1.18.34 that is not the case any more — a second ``opencode serve`` starts
# and answers health while another instance holds the log — so this is now only
# a diagnostic: the runtime probes for a serve to adopt, spawns anyway, and
# attaches the hint below to a spawn that actually died. Surfacing it beats a
# bare crash code, and it doubles as the user-facing hint.
LOCK_HOLD_HINT = (
    "另一个 opencode 实例正在运行并持有 opencode 日志文件 "
    "(<data>/opencode/log/opencode.log)，刚启动的 serve 因此退出。"
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
    """True when any live opencode process holds its log file open.

    Advisory only (see ``LOCK_HOLD_HINT``): on serve 1.18.34 this file is not
    an exclusive single-instance lock, so the probe must never be used to skip
    spawning — only to explain a spawn that failed. Blocking (one CreateFileW
    probe); event-loop callers must run this in a thread. Only Windows exposes
    share-mode conflicts, so other platforms report False.
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


_LAUNCHER_SUFFIXES = (".cmd", ".bat", ".ps1")
# A token in a launcher script that mentions opencode: quoted or bare, with the
# path separators npm shims use.
_SHIM_TARGET = re.compile(r"""[^\s"'()&|<>^]*opencode[^\s"'()&|<>^]*""", re.IGNORECASE)


def forwarded_executable(candidate: str) -> str | None:
    """The real executable a launcher script forwards to, or ``None``.

    npm installs ``opencode.cmd``/``opencode.ps1`` shims whose only job is to
    exec the package's native ``bin/opencode.exe``. Paying for the wrapper is
    expensive on Windows: ``opencode.cmd --version`` timed out after 15s on a
    real install while the binary it wraps answered in 0.5s, so every connector
    start burned the full timeout and then fell back to the even slower npx path.
    The shim is read for its target instead of assuming a package layout, and a
    shim whose target cannot be resolved is simply left alone.
    """

    if os.path.splitext(candidate)[1].lower() not in _LAUNCHER_SUFFIXES:
        return None
    try:
        text = Path(candidate).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    base = Path(candidate).parent
    for match in _SHIM_TARGET.finditer(text):
        resolved = _resolve_shim_path(match.group(0), base)
        if resolved is not None and os.path.normcase(resolved) != os.path.normcase(candidate):
            return resolved
    return None


def _resolve_shim_path(token: str, base: Path) -> str | None:
    text = token.strip().strip("\"'")
    # npm shims spell the script directory ``%dp0%`` (cmd) or ``$PSScriptRoot`` /
    # ``$basedir`` (PowerShell).
    for marker in ("%~dp0", "%dp0%", "$PSScriptRoot", "$basedir"):
        text = text.replace(marker, str(base))
    if not text or "%" in text or "$" in text:
        # An unexpanded variable means we would be guessing at the layout.
        return None
    path = Path(os.path.expandvars(text))
    if not path.is_absolute():
        path = base / path
    return str(path) if path.is_file() else None


def _serve_candidates(path_value: str | None) -> list[str]:
    """opencode executables to version-check, best first.

    The PATH hit is usually an npm shim, so the binary it forwards to is tried
    before it; the shim stays in the list as a fallback for installs whose
    target cannot be resolved.
    """

    found = find_executable_on_path("opencode", path_value)
    if found is None:
        return []
    forwarded = forwarded_executable(found)
    if forwarded is None:
        return [found]
    logger.info(
        "opencode resolve: {} forwards to {}; probing the binary first", found, forwarded
    )
    return [forwarded, found]


def is_serve_log_lock_failure(text: str) -> bool:
    """True when a spawn died because another opencode holds its log file.

    ``--print-logs`` routes the child's logs to stderr instead of its log file,
    which reduces the collision with another opencode instance but does **not**
    remove it: on 1.18.34 a spawn next to a running TUI still died with ``Unknown:
    FileSystem.open (.../opencode.log)``, and two further attempts took Bun's own
    crash paths (``STATUS_STACK_BUFFER_OVERRUN``, ``Illegal instruction``) for
    what is the same lock. Recognising it matters twice over: retrying cannot help
    because the holder is not going away on its own, and the raw message
    ("Unexpected error") hides the actual cause.
    """

    lowered = text.lower()
    if "filesystem.open" in lowered and "opencode.log" in lowered:
        return True
    # Bun's crash paths for the same failure: no usable message, just a fastfail.
    return "bun has crashed" in lowered or "Bun 启动 fastfail" in text


def resolve_serve_command(
    port: int,
    environment: dict[str, str] | None = None,
    *,
    hostname: str = "127.0.0.1",
) -> list[str]:
    """Build the ``opencode serve`` argv, validating candidates with --version.

    Falls back to ``npx`` when the system binary is missing or fails the
    version check (same semantics as the codex binary selection).

    ``--print-logs`` routes the child's logs to stderr instead of its log file,
    which ``ServeProcess`` drains into a ring buffer. It reduces the collision
    with another opencode instance but does not eliminate it (see
    ``is_serve_log_lock_failure``), so the spawn still has to happen while the log
    is held -- there is nothing else to fall back on when no serve answers -- and
    its failure has to be recognised rather than retried as "Unexpected error".
    """

    env = environment or os.environ
    path_value = env.get("PATH")
    reason: str | None = None
    for candidate in _serve_candidates(path_value):
        probe_started = time.monotonic()
        error = check_version_output(candidate, env)
        probe_elapsed = time.monotonic() - probe_started
        if error is None:
            logger.info(
                "opencode resolve: version probe ok in {:.2f}s candidate={}",
                probe_elapsed,
                candidate,
            )
            return [
                candidate,
                "serve",
                *SERVE_LOG_FLAGS,
                "--hostname",
                hostname,
                "--port",
                str(port),
            ]
        reason = f"{candidate} failed version check: {error}"
        logger.warning(
            "{} after {:.2f}s; trying the next candidate",
            reason,
            probe_elapsed,
        )
    if reason is None:
        reason = "opencode executable was not found on PATH"
        logger.info("opencode resolve: no system opencode on PATH; trying npx")

    npx = find_executable_on_path("npx", path_value)
    if npx is not None:
        return [
            npx,
            "opencode",
            "serve",
            *SERVE_LOG_FLAGS,
            "--hostname",
            hostname,
            "--port",
            str(port),
        ]
    raise RuntimeError(f"opencode executable not found on PATH (and no npx fallback): {reason}")


def check_version_output(candidate: str, environment: dict[str, Any]) -> str | None:
    """Return None when `<candidate> --version` looks like a real opencode CLI.

    Blocking; callers on the event loop must run this in a thread. On Windows a
    plain timeout only kills the direct child (powershell/cmd), while a
    grandchild holding the output pipes can hang ``communicate`` indefinitely —
    so the whole process tree is killed on timeout.

    The budget is deliberately short. This only picks a binary; it does not
    decide whether the runtime works, and the ``npx`` fallback that a failed
    probe falls back to was measured coming up in 2.1s while two 15s probes in a
    row left the client looking at ``status='starting'`` for the whole time. A
    candidate that cannot report its version within ``VERSION_TIMEOUT_S`` is not
    going to serve the turn any sooner.
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
            # Same as ServeProcess.spawn: a connector is spawned by the desktop
            # app, where stdin is not a console. Never let a probe inherit it.
            stdin=subprocess.DEVNULL,
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
        # Close the child's pipe transports explicitly. Cancelling the drain
        # readers and then walking away leaves the proactor transports to the
        # garbage collector, and on Windows their ``__del__`` asks for a file
        # descriptor that is already gone -- asyncio then reports ``ValueError:
        # I/O operation on closed pipe`` from a ``proactor_events`` repr during
        # interpreter teardown, which surfaced in the connector log as an ERROR on
        # an otherwise clean shutdown. It has to happen *after* the kill: closing
        # a subprocess transport while the child is alive kills the child.
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            with suppress(Exception):
                transport.close()
