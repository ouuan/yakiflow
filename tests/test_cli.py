from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import yakiflow.cli as cli
import yakiflow.tui as tui
from conftest import make_settings
from yakiflow.config import Settings
from yakiflow.doctor import Check
from yakiflow.job import YakiFlowJob
from yakiflow.process import ProcessResult
from yakiflow.translation import AgentBackend


class TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_c_overrides_parse_toml_literals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    args = cli._parser().parse_args([
        "run",
        "input.mp4",
        "-c", "transcription.backend=elevenlabs",
        "-c", "elevenlabs.diarize=false",
        "-c", "agent.draft.workers=7",
        "-c", "agent.draft.extra-options=[\"--search\", \"-v\"]",
        "-c", "target-language=zh-CN",
    ])

    settings = cli._settings(args)

    assert settings.transcription.backend == "elevenlabs"
    assert settings.elevenlabs.diarize is False
    assert settings.agent.draft.workers == 7
    assert settings.agent.draft.extra_options == ("--search", "-v")
    assert settings.target_language == "zh-CN"


def test_c_override_values_fall_back_to_strings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    args = cli._parser().parse_args([
        "run", "input.mp4",
        # An unquoted string is not a TOML literal but is obviously a string.
        "-c", "review.open-command=editor {file}",
    ])
    settings = cli._settings(args)
    assert settings.review.open_command == "editor {file}"


def test_unknown_c_override_is_reported_as_command_line(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    args = cli._parser().parse_args([
        "run", "input.mp4", "-c", "transcriptoin.backend=elevenlabs",
    ])
    with pytest.raises(ValueError, match=r"command line \(-c\)"):
        cli._settings(args)


def test_malformed_c_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli.parse_config_items(["transcription.backend"])


def test_dedicated_flags_outrank_c_overrides_and_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config = tmp_path / "config.toml"
    config.write_text(
        'target-language = "fr"\n'
        '[profiles.stream]\nstream = true\ntarget-language = "ja"\n'
    )
    args = cli._parser().parse_args([
        "run", "input.mp4",
        "--config-file", str(config),
        "--profile", "stream",
        "-c", "target-language=ko",
        "-t", "zh-CN",
        "--no-stream",
    ])

    settings = cli._settings(args)

    assert settings.target_language == "zh-CN"
    assert settings.stream is False


def test_missing_config_file_is_an_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    args = cli._parser().parse_args([
        "run", "input.mp4", "--config-file", str(tmp_path / "absent.toml"),
    ])
    with pytest.raises(ValueError, match="configuration file does not exist"):
        cli._settings(args)


def test_transcription_backend_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    args = cli._parser().parse_args([
        "run", "input.mp4", "--transcription-backend", "whisper-server",
    ])
    settings = cli._settings(args)
    assert settings.transcription.backend == "whisper-server"


def test_context_file_cli_option_is_repeatable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    args = cli._parser().parse_args([
        "run",
        "input.mp4",
        "--context-file",
        "chat.xml",
        "--context-file",
        "notes.txt",
    ])

    settings = cli._settings(args)

    assert settings.context_files == (
        tmp_path / "chat.xml",
        tmp_path / "notes.txt",
    )


def test_profile_option_is_available_on_config_loading_commands(monkeypatch) -> None:
    selected: list[str | None] = []

    def fake_load_settings(_values=None, **kwargs):
        selected.append(kwargs.get("profile"))
        return Settings()

    monkeypatch.setattr(cli, "load_settings", fake_load_settings)
    parser = cli._parser()
    for argv in (
        ["run", "input.mp4", "--profile", "stream"],
        ["doctor", "--profile", "stream"],
        ["models", "fetch", "--profile", "stream"],
    ):
        cli._settings(parser.parse_args(argv))

    assert selected == ["stream", "stream", "stream"]


def _run_argv(*extra: str) -> list[str]:
    return [
        "run", "https://example.com/video", "-s", "en", "-t", "zh-CN",
        "-c", "agent.backend=codex", *extra,
    ]


def test_run_stops_before_starting_on_fatal_health_check_failures(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    work_dir = tmp_path / "work"
    seen: list[bool | None] = []

    def fake_doctor(_settings, *, source_is_url):
        seen.append(source_is_url)
        return [
            Check("ffmpeg", False, "not found"),
            Check("tmux", False, "not found", False),
            Check("codex auth", False, "not logged in"),
        ]

    monkeypatch.setattr(cli, "run_doctor", fake_doctor)

    assert cli.main(_run_argv("--work-dir", str(work_dir))) == 2

    err = capsys.readouterr().err
    assert seen == [True]
    # Every fatal failure is named at once, and the advisory one only warns.
    assert "health check failed" in err
    assert "ffmpeg: not found" in err
    assert "codex auth: not logged in" in err
    assert "warning: tmux: not found" in err
    assert not work_dir.exists()


def test_run_starts_despite_advisory_health_check_failures(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    def started(*args, **kwargs):
        raise RuntimeError("job started")

    monkeypatch.setattr(
        cli, "run_doctor", lambda *args, **kwargs: [Check("tmux", False, "gone", False)]
    )
    monkeypatch.setattr(cli, "YakiFlowJob", started)

    assert cli.main(_run_argv()) == 2

    err = capsys.readouterr().err
    assert "warning: tmux: gone" in err
    assert "job started" in err


def test_skip_health_check_starts_without_checking(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    def unexpected(*args, **kwargs):
        raise AssertionError("health check should have been skipped")

    def started(*args, **kwargs):
        raise RuntimeError("job started")

    monkeypatch.setattr(cli, "run_doctor", unexpected)
    monkeypatch.setattr(cli, "YakiFlowJob", started)

    assert cli.main(_run_argv("--skip-health-check")) == 2
    assert "job started" in capsys.readouterr().err


class FakeKeyring:
    class errors:
        class PasswordDeleteError(Exception):
            pass

    def __init__(self) -> None:
        self.stored: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, entry: str, value: str) -> None:
        self.stored[(service, entry)] = value

    def delete_password(self, service: str, entry: str) -> None:
        if (service, entry) not in self.stored:
            raise self.errors.PasswordDeleteError()
        del self.stored[(service, entry)]


def test_secret_set_and_unset_use_the_keyring(monkeypatch, capsys) -> None:
    fake = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.setitem(sys.modules, "keyring.errors", fake.errors)
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: "sk-secret")

    assert cli.main(["secret", "set", "elevenlabs"]) == 0
    assert fake.stored == {("yakiflow", "elevenlabs"): "sk-secret"}

    assert cli.main(["secret", "unset", "elevenlabs"]) == 0
    assert fake.stored == {}

    # Unsetting an absent key reports rather than fails.
    assert cli.main(["secret", "unset", "elevenlabs"]) == 0
    capsys.readouterr()


def test_secret_set_rejects_empty_key(monkeypatch, capsys) -> None:
    fake = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.setitem(sys.modules, "keyring.errors", fake.errors)
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: "")

    assert cli.main(["secret", "set", "elevenlabs"]) == 2
    assert fake.stored == {}
    assert "no API key" in capsys.readouterr().err


def test_tui_exit_prints_resume_guide(tmp_path: Path, monkeypatch, capsys) -> None:
    work_dir = tmp_path / "preserved job"

    class FakeJob:
        def __init__(self, *, is_finished: bool = False) -> None:
            self.work_dir = work_dir
            self.is_finished = is_finished
            self.input_value = "video.mp4"
            self.settings = make_settings(
                source_language="en",
                target_language="zh",
                agent={"backend": "codex"},
            ).resolved()

    monkeypatch.setattr(cli, "run_doctor", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        cli.YakiFlowJob,
        "from_workdir",
        lambda *args, **kwargs: FakeJob(),
    )
    monkeypatch.setattr(tui, "run", lambda job: False)
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(stdin=TTY(), stdout=TTY(), stderr=sys.stderr),
    )

    assert cli.main(["resume", str(work_dir)]) == 2
    assert capsys.readouterr().err == (
        f"Job preserved. Resume with: yakiflow resume '{work_dir}'\n"
    )

    monkeypatch.setattr(
        cli.YakiFlowJob,
        "from_workdir",
        lambda *args, **kwargs: FakeJob(is_finished=True),
    )

    # A finished job refuses to run again, so repeating its resume command
    # could only repeat the same failure.
    assert cli.main(["resume", str(work_dir)]) == 2
    assert capsys.readouterr().err == ""


def test_memory_conflict_agent_retries_when_destination_changes_again(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "memory.md"
    work_dir = tmp_path / "work"
    destination.write_text("# Memory\n\n- original\n", encoding="utf-8")

    class UnusedBackend(AgentBackend):
        name = "unused"

        async def invoke_with_trace(
            self, prompt, *, system="", model, effort, schema, on_event=None
        ):
            raise AssertionError("structured agent should not run")

    class ConflictRunner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_interactive(self, args, *, cwd=None, check=True):
            assert cwd == work_dir.resolve()
            prompt = str(args[-1])
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                assert "+- destination edit 1" in prompt
                (cwd / "memory.md").write_text(
                    "# Memory\n\n- original\n- work edit\n- destination edit 1\n",
                    encoding="utf-8",
                )
                destination.write_text(
                    "# Memory\n\n- original\n- destination edit 1\n- destination edit 2\n",
                    encoding="utf-8",
                )
            else:
                assert "- destination edit 1" in prompt
                assert "+- destination edit 2" in prompt
                (cwd / "memory.md").write_text(
                    "# Memory\n\n- original\n- work edit\n"
                    "- destination edit 1\n- destination edit 2\n",
                    encoding="utf-8",
                )
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    runner = ConflictRunner()
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language="en",
            target_language="zh-CN",
            agent={"backend": "codex"},
            memory=str(destination),
            work_dir=str(work_dir),
        ),
        runner=runner,
        backend=UnusedBackend(),
    )
    job.memory_path.write_text(
        "# Memory\n\n- original\n- work edit\n", encoding="utf-8"
    )
    destination.write_text(
        "# Memory\n\n- original\n- destination edit 1\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO()),
    )

    assert asyncio.run(cli._complete_job(job, [])) == 0

    assert len(runner.prompts) == 2
    assert destination.read_text(encoding="utf-8") == (
        "# Memory\n\n- original\n- work edit\n"
        "- destination edit 1\n- destination edit 2\n"
    )


def test_non_tty_run_continues_when_signal_handlers_are_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    ran = False

    class FakeJob:
        work_dir = tmp_path / "work"

        async def run(self):
            nonlocal ran
            ran = True
            return []

    async def complete(_job, outputs):
        assert outputs == []
        return 0

    async def exercise() -> int:
        loop = asyncio.get_running_loop()
        removed = False

        def unsupported(*_args) -> None:
            raise NotImplementedError

        def remove(*_args) -> None:
            nonlocal removed
            removed = True

        # Restore the live loop before asyncio.run() closes it. Keeping bound
        # loop methods in pytest's outer monkeypatch fixture can retain a
        # closed loop and its executor across later asyncio.run() calls.
        with monkeypatch.context() as context:
            context.setattr(loop, "add_signal_handler", unsupported)
            context.setattr(loop, "remove_signal_handler", remove)
            result = await cli._run_job(FakeJob())
        assert not removed
        return result

    monkeypatch.setattr(cli, "_complete_job", complete)
    assert asyncio.run(exercise()) == 0
    assert ran


def test_non_tty_run_prints_only_finalized_output_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    staged = tmp_path / "work" / "movie.srt"
    finalized = tmp_path / "out" / "movie.srt"

    class FakeJob:
        work_dir = tmp_path / "work"

        async def run(self):
            return [staged]

    async def complete(_job, outputs):
        assert outputs == [staged]
        assert capsys.readouterr().out == ""
        print(finalized)
        return 0

    monkeypatch.setattr(cli, "_complete_job", complete)

    assert asyncio.run(cli._run_job(FakeJob())) == 0
    assert capsys.readouterr().out == f"{finalized}\n"
