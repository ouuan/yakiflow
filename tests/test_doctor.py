from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yakiflow.doctor as doctor_module
from conftest import make_settings
from yakiflow.doctor import run_doctor


def _stub_dependencies(tmp_path: Path, monkeypatch) -> Path:
    """Create a Whisper model and stub out CLI discovery and auth checks."""
    model = tmp_path / "whisper.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    return model


def test_doctor_checks_whisperx_package_and_explicit_cuda(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    checks = run_doctor(
        make_settings(
            whisper={"model": model},
            agent={"backend": "codex"},
            alignment={"backend": "whisperx", "device": "cuda"},
        ).resolved()
    )
    mapped = {check.name: check for check in checks}

    assert mapped["WhisperX"].ok
    assert not mapped["WhisperX CUDA"].ok
    assert mapped["WhisperX CUDA"].detail == "CUDA unavailable"


def test_doctor_does_not_require_whisperx_for_default_vad(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)

    checks = run_doctor(
        make_settings(whisper={"model": model}, agent={"backend": "codex"}).resolved()
    )

    assert all(not check.name.startswith("WhisperX") for check in checks)


def test_doctor_names_both_whisperx_install_variants(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    checks = run_doctor(
        make_settings(
            whisper={"model": model},
            agent={"backend": "codex"},
            alignment={"backend": "whisperx"},
        ).resolved()
    )
    whisperx = next(check for check in checks if check.name == "WhisperX")

    assert not whisperx.ok
    assert "yakiflow[whisperx-cpu]" in whisperx.detail
    assert "yakiflow[whisperx-cuda]" in whisperx.detail


def test_yt_dlp_is_fatal_only_for_an_input_that_needs_downloading(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda command: None if command == "yt-dlp" else f"/bin/{command}",
    )
    settings = make_settings(
        whisper={"model": model}, agent={"backend": "codex"}
    ).resolved()

    def yt_dlp(source_is_url: bool | None) -> doctor_module.Check:
        checks = run_doctor(settings, source_is_url=source_is_url)
        return next(check for check in checks if check.name == "yt-dlp")

    assert not yt_dlp(False).ok and not yt_dlp(False).fatal
    assert yt_dlp(True).fatal
    # Without an input to judge — the `doctor` command — it stays fatal.
    assert yt_dlp(None).fatal


def test_missing_ffprobe_is_advisory(tmp_path: Path, monkeypatch) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda command: None if command == "ffprobe" else f"/bin/{command}",
    )

    checks = run_doctor(
        make_settings(whisper={"model": model}, agent={"backend": "codex"}).resolved()
    )
    ffprobe = next(check for check in checks if check.name == "ffprobe")

    # The subtitles are still written, only at the fallback resolution.
    assert not ffprobe.ok and not ffprobe.fatal
    assert "1920x1080" in ffprobe.detail


def test_missing_default_whisper_model_is_advisory(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    checks = run_doctor(make_settings(agent={"backend": "codex"}).resolved())
    model = next(check for check in checks if check.name == "whisper model")

    assert not model.ok
    assert not model.fatal
    assert "downloads it" in model.detail


def test_missing_configured_whisper_model_is_fatal(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_dependencies(tmp_path, monkeypatch)

    checks = run_doctor(
        make_settings(
            whisper={"model": tmp_path / "absent.bin"}, agent={"backend": "codex"}
        ).resolved()
    )
    model = next(check for check in checks if check.name == "whisper model")

    assert not model.ok
    assert model.fatal


def test_external_whisper_server_is_not_checked_for_a_local_model(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "_probe_server_url",
        lambda url, timeout=5.0: doctor_module.Check("whisper-server URL", True, url),
    )

    checks = run_doctor(
        make_settings(
            transcription={"backend": "whisper-server"},
            whisper={"model": tmp_path / "absent.bin", "server_url": "http://host:8080"},
            agent={"backend": "codex"},
        ).resolved()
    )

    # The remote server owns its model, exactly as the run itself assumes.
    assert all(check.name != "whisper model" for check in checks)


def test_gemini_cli_is_found_without_a_sign_in_command(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(doctor_module.subprocess, "run", run)

    checks = run_doctor(
        make_settings(
            whisper={"model": model}, agent={"backend": "gemini"}
        ).resolved()
    )
    mapped = {check.name: check for check in checks}

    assert mapped["gemini CLI"].ok
    assert mapped["gemini CLI"].detail == "/bin/gemini (sign-in not checked)"
    assert "gemini auth" not in mapped
    assert not any(command[0] == "gemini" for command in commands)


def test_doctor_without_an_agent_backend_checks_every_agent_cli(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)

    checks = run_doctor(make_settings(whisper={"model": model}).resolved())
    names = {check.name for check in checks}

    assert {"codex auth", "claude auth", "gemini CLI"} <= names


def test_agent_clis_are_advisory_until_a_backend_is_chosen(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda command: None if command == "gemini" else f"/bin/{command}",
    )

    unchosen = {
        check.name: check
        for check in run_doctor(make_settings(whisper={"model": model}).resolved())
    }
    chosen = {
        check.name: check
        for check in run_doctor(
            make_settings(whisper={"model": model}, agent={"backend": "gemini"}).resolved()
        )
    }

    assert not unchosen["gemini CLI"].ok and not unchosen["gemini CLI"].fatal
    assert not chosen["gemini CLI"].ok and chosen["gemini CLI"].fatal


def test_review_display_dependencies_are_advisory(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda command: None if command in {"tmux", "editor"} else f"/bin/{command}",
    )

    checks = run_doctor(
        make_settings(
            whisper={"model": model},
            agent={"backend": "codex"},
            review={"display_mode": "both", "open_command": "editor {srt}"},
        ).resolved()
    )
    mapped = {check.name: check for check in checks}

    assert not mapped["tmux"].ok and not mapped["tmux"].fatal
    assert (
        not mapped["review open command"].ok
        and not mapped["review open command"].fatal
    )


def test_doctor_checks_both_review_display_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)

    checks = run_doctor(
        make_settings(
            whisper={"model": model},
            agent={"backend": "codex"},
            review={"display_mode": "both", "open_command": "editor {srt}"},
        ).resolved()
    )
    mapped = {check.name: check for check in checks}

    assert mapped["tmux"].ok
    assert mapped["review open command"].ok
