from __future__ import annotations

import asyncio
import inspect
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence


LineCallback = Callable[[str, str], Awaitable[None] | None]


@dataclass(slots=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class ProcessError(RuntimeError):
    def __init__(self, result: ProcessResult):
        super().__init__(f"command failed ({result.returncode}): {' '.join(result.args)}\n{result.stderr[-2000:]}")
        self.result = result


class CommandRunner:
    """Async external process runner with incremental, test-friendly line handling."""

    async def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        on_line: LineCallback | None = None,
        check: bool = True,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Run ``args``; ``env`` adds variables to the inherited environment."""
        normalized = tuple(os.fspath(arg) for arg in args)
        process = await asyncio.create_subprocess_exec(
            *normalized,
            cwd=cwd,
            env={**os.environ, **env} if env else None,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        callback_failure: asyncio.Future[BaseException] = (
            asyncio.get_running_loop().create_future()
        )
        callback_enabled = on_line is not None

        async def consume(reader: asyncio.StreamReader, name: str) -> str:
            nonlocal callback_enabled
            chunks: list[bytes] = []
            pending = bytearray()

            async def deliver(raw_line: bytes) -> None:
                nonlocal callback_enabled
                if not on_line or not callback_enabled:
                    return
                line = raw_line.decode(errors="replace").rstrip("\r\n")
                try:
                    maybe = on_line(name, line)
                    if inspect.isawaitable(maybe):
                        await maybe
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    # Stop invoking a broken sink but keep draining both pipes
                    # so process termination can reach EOF.
                    callback_enabled = False
                    if not callback_failure.done():
                        callback_failure.set_result(exc)

            while True:
                # StreamReader.readline() has a 64 KiB line limit. Agent JSON is
                # commonly emitted as one much longer line, so consume bounded
                # byte chunks and perform newline framing ourselves.
                raw = await reader.read(64 * 1024)
                if not raw:
                    break
                chunks.append(raw)
                if on_line and callback_enabled:
                    pending.extend(raw)
                    while (newline := pending.find(b"\n")) >= 0:
                        line = bytes(pending[:newline + 1])
                        del pending[:newline + 1]
                        await deliver(line)
            if pending and callback_enabled:
                await deliver(bytes(pending))
            return b"".join(chunks).decode(errors="replace")

        out_task = asyncio.create_task(consume(process.stdout, "stdout"))
        err_task = asyncio.create_task(consume(process.stderr, "stderr"))
        wait_task = asyncio.create_task(process.wait())
        try:
            if stdin is not None and process.stdin is not None:
                try:
                    process.stdin.write(stdin)
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    # The child rejected the input and exited. Keep draining so
                    # its own diagnostic on stderr survives as a ProcessError
                    # instead of being replaced by a bare pipe error.
                    pass
                finally:
                    # Close either way: the child still waiting on EOF must get
                    # it, and the write transport must not outlive this call.
                    process.stdin.close()
            # A callback failure stops its pipe consumer. Notice that failure
            # while the child is still alive, before the undrained pipe can
            # fill and deadlock the child.
            active: set[asyncio.Future[object]] = {
                wait_task,
                out_task,
                err_task,
                callback_failure,
            }
            while not wait_task.done():
                done, _pending = await asyncio.wait(
                    active,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if callback_failure in done:
                    raise callback_failure.result()
                for reader_task in (out_task, err_task):
                    if reader_task not in done:
                        continue
                    error = reader_task.exception()
                    if error is not None:
                        raise error
                    active.remove(reader_task)
            returncode = await wait_task
            stdout, stderr = await asyncio.gather(out_task, err_task)
            if callback_failure.done():
                raise callback_failure.result()
        except BaseException:
            await terminate_process(process)
            out_task.cancel()
            err_task.cancel()
            wait_task.cancel()
            await asyncio.gather(
                out_task, err_task, wait_task, return_exceptions=True
            )
            raise
        result = ProcessResult(normalized, returncode, stdout, stderr)
        if check and returncode:
            raise ProcessError(result)
        return result

    async def run_interactive(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Run a command with the caller's terminal attached.

        The regular runner deliberately pipes all streams so structured agent
        calls can be inspected and parsed.  An interactive agent needs the
        opposite arrangement: its prompt, colors, and follow-up questions must
        be rendered by the agent itself and its stdin must remain connected to
        the user.  Keeping this as a separate method prevents accidentally
        turning a non-interactive pipeline invocation into a terminal session.
        """
        normalized = tuple(os.fspath(arg) for arg in args)
        process = await asyncio.create_subprocess_exec(
            *normalized,
            cwd=cwd,
            env={**os.environ, **env} if env else None,
            stdin=None,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )
        try:
            returncode = await process.wait()
        except BaseException:
            await terminate_process(process)
            raise
        result = ProcessResult(normalized, returncode, "", "")
        if check and returncode:
            raise ProcessError(result)
        return result


async def terminate_process(process: asyncio.subprocess.Process, timeout: float = 8.0) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), 3)
        except TimeoutError:
            process.kill()
            await process.wait()
