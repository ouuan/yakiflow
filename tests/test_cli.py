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
        review_only = False

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

        async def run_interactive(self, args, *, cwd=None, check=True, env=None):
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


def _review_config(tmp_path: Path) -> Path:
    config = tmp_path / "review.toml"
    config.write_text(
        'output-mode = "translated"\n'
        'source-language = "ja"\n'
        'target-language = "zh-CN"\n'
        "[agent]\n"
        'backend = "codex"\n'
    )
    return config


def _review_settings(tmp_path: Path, *argv: str) -> Settings:
    return cli._review_settings(cli._parser().parse_args([
        "review", *argv, "--config-file", str(_review_config(tmp_path)),
    ]))


def test_review_layout_prefers_flags_then_file_names_then_config(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)

    unnamed = _review_settings(tmp_path, "movie.ass")
    named = _review_settings(tmp_path, "movie.source.ass")
    flagged = _review_settings(
        tmp_path, "movie.source.ass", "--output-mode", "bilingual"
    )
    assert (unnamed.output_mode, named.output_mode, flagged.output_mode) == (
        "translated", "source", "bilingual",
    )

    pair = _review_settings(tmp_path, "movie.en-fr.ass")
    assert (pair.output_mode, pair.source_language, pair.target_language) == (
        "bilingual", "en", "fr",
    )
    overridden = _review_settings(tmp_path, "movie.en-fr.ass", "-t", "de")
    assert (overridden.source_language, overridden.target_language) == ("en", "de")
    # The configured target language picks the split of an ambiguous pair.
    ambiguous = _review_settings(tmp_path, "movie.ko-zh-cn.ass")
    assert (ambiguous.source_language, ambiguous.target_language) == ("ko", "zh-CN")


def test_review_takes_the_three_files_of_an_all_output_together(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    files = ("movie.source.ass", "movie.translated.ass", "movie.ko-zh-cn.ass")

    settings = _review_settings(tmp_path, *files)

    assert (settings.output_mode, settings.source_language) == ("all", "ko")
    assert _review_settings(tmp_path, *files, "--output-mode", "all").output_mode == "all"
    for argv, error in (
        (("movie.source.ass", "--output-mode", "all"), "reviewed as one set"),
        (files[:2], "only as one 'all' output"),
        ((*files[:2], "other.ko-zh-cn.ass"), "only as one 'all' output"),
        ((*files[:2], "movie.ass"), "only as one 'all' output"),
        ((*files[:2], "movie.source.ass"), "only as one 'all' output"),
        ((*files, "--output-mode", "bilingual"), "reviews a single file"),
    ):
        with pytest.raises(ValueError, match=error):
            _review_settings(tmp_path, *argv)


def _review_subtitle(tmp_path: Path) -> Path:
    original = tmp_path / "subs" / "movie.ja-zh-cn.ass"
    original.parent.mkdir()
    original.write_text(
        "[Script Info]\nTitle: kept\n\n[Events]\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}你好\\Nこんにちは\n",
        encoding="utf-8",
    )
    return original


def test_review_refuses_to_start_without_a_terminal(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    original = _review_subtitle(tmp_path)
    work_dir = tmp_path / "work"

    def unexpected(*args, **kwargs):
        raise AssertionError("review should not have started")

    monkeypatch.setattr(cli, "run_review_doctor", unexpected)
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(stdin=io.StringIO(), stdout=TTY(), stderr=sys.stderr),
    )

    assert cli.main([
        "review", str(original), "-c", "agent.backend=codex",
        "--work-dir", str(work_dir),
    ]) == 2
    assert "review needs an interactive terminal" in capsys.readouterr().err
    assert not work_dir.exists()


class _FakeDisplay:
    async def close(self) -> None:
        return None


def _fake_review_agent(monkeypatch, edit: tuple[str, str]) -> list[dict]:
    """Stand in for the interactive Agent: record the call, edit the file."""
    calls: list[dict] = []

    async def start_display(settings, work_dir, outputs):
        return _FakeDisplay()

    async def review(settings, work_dir, outputs, **kwargs):
        calls.append({"settings": settings, "outputs": list(outputs), **kwargs})
        text = outputs[0].read_text(encoding="utf-8")
        outputs[0].write_text(text.replace(*edit), encoding="utf-8")
        return ProcessResult(("agent",), 0, "", "")

    monkeypatch.setattr(cli, "start_agent_file_display", start_display)
    monkeypatch.setattr(cli, "run_interactive_agent", review)
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(stdin=TTY(), stdout=TTY(), stderr=sys.stderr),
    )
    return calls


def test_review_runs_the_review_only_agent_and_writes_back(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    original = _review_subtitle(tmp_path)
    media = tmp_path / "movie.mkv"
    media.write_bytes(b"media")
    expected = original.read_text(encoding="utf-8").replace("你好", "您好")
    doctored: list[Settings] = []

    def unexpected(*args, **kwargs):
        raise AssertionError("review does not need the transcription checks")

    monkeypatch.setattr(cli, "run_doctor", unexpected)
    monkeypatch.setattr(
        cli, "run_review_doctor", lambda settings: doctored.append(settings) or []
    )
    calls = _fake_review_agent(monkeypatch, ("你好", "您好"))

    assert cli.main([
        "review", str(original), "--media", str(media),
        "-c", "agent.final.backend=codex", "-t", "zh-CN",
        "--work-dir", str(tmp_path / "work"),
    ]) == 0

    assert len(doctored) == 1
    [call] = calls
    assert call["review_only"] is True
    assert call["settings"].output_mode == "bilingual"
    assert (call["settings"].source_language, call["settings"].target_language) == (
        "ja", "zh-CN",
    )
    assert original.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out.startswith(f"{original}\n")


def test_resuming_a_review_only_job_skips_the_tui_and_the_pipeline_checks(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    original = _review_subtitle(tmp_path)
    expected = original.read_text(encoding="utf-8").replace("你好", "您好")
    job = YakiFlowJob.for_review(
        [original],
        make_settings(
            target_language="zh-CN",
            agent={"final": {"backend": "codex"}},
            memory=tmp_path / "memory.md",
            work_dir=tmp_path / "work",
        ),
    )
    job.close()
    doctored: list[Settings] = []

    def unexpected(*args, **kwargs):
        raise AssertionError("a review-only job has no pipeline to check or show")

    monkeypatch.setattr(cli, "run_doctor", unexpected)
    monkeypatch.setattr(tui, "run", unexpected)
    monkeypatch.setattr(
        cli, "run_review_doctor", lambda settings: doctored.append(settings) or []
    )
    calls = _fake_review_agent(monkeypatch, ("你好", "您好"))

    assert cli.main(["resume", str(tmp_path / "work")]) == 0

    assert len(doctored) == 1
    assert [call["review_only"] for call in calls] == [True]
    assert original.read_text(encoding="utf-8") == expected
