import asyncio
import sys

import pytest

import yakiflow.process as process_module
from yakiflow.process import CommandRunner, terminate_process


class MarkingAwaitable:
    def __init__(self, awaited: list[bool]) -> None:
        self.awaited = awaited

    def __await__(self):
        self.awaited.append(True)
        if False:
            yield
        return None


def test_command_runner_accepts_lines_longer_than_asyncio_limit() -> None:
    lines: list[tuple[str, str]] = []

    async def on_line(stream: str, line: str) -> None:
        lines.append((stream, line))

    result = asyncio.run(
        CommandRunner().run(
            [sys.executable, "-c", "print('x' * 200_000)"],
            on_line=on_line,
        )
    )

    assert len(result.stdout) == 200_001
    assert lines == [("stdout", "x" * 200_000)]


def test_command_runner_awaits_custom_line_callback_awaitable() -> None:
    awaited: list[bool] = []

    def on_line(_stream: str, _line: str) -> MarkingAwaitable:
        return MarkingAwaitable(awaited)

    asyncio.run(
        CommandRunner().run(
            [sys.executable, "-c", "print('line')"],
            on_line=on_line,
        )
    )

    assert awaited == [True]


def test_command_runner_env_adds_to_the_inherited_environment(monkeypatch) -> None:
    monkeypatch.setenv("YAKIFLOW_INHERITED", "kept")
    result = asyncio.run(
        CommandRunner().run(
            [
                sys.executable, "-c",
                "import os; print(os.environ['YAKIFLOW_INHERITED'], os.environ['YAKIFLOW_ADDED'])",
            ],
            env={"YAKIFLOW_ADDED": "added"},
        )
    )

    assert result.stdout == "kept added\n"


def test_command_runner_terminates_child_when_line_callback_fails() -> None:
    async def on_line(_stream: str, _line: str) -> None:
        raise RuntimeError("log sink failed")

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="log sink failed"):
            await asyncio.wait_for(
                CommandRunner().run(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "while True: print('x' * 65536)",
                    ],
                    on_line=on_line,
                ),
                timeout=3,
            )

    asyncio.run(exercise())


def test_terminate_process_uses_portable_windows_fallback(monkeypatch) -> None:
    class FakeProcess:
        returncode = None
        pid = 123

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    process = FakeProcess()
    monkeypatch.setattr(process_module.os, "name", "nt")
    monkeypatch.setattr(
        process_module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("killpg called")),
    )

    asyncio.run(terminate_process(process))

    assert process.terminated
    assert not process.killed
