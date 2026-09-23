from __future__ import annotations

import shutil
import shlex
import subprocess
import importlib.util
import json
import sys
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urlparse

from .config import Settings, default_model_path
from .subtitles import DEFAULT_PLAY_RES
from .transcription import needs_local_whisper


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    # Whether a failed check stops a run. An advisory check only degrades what
    # surrounds the pipeline — a review preview, a downloader this input does
    # not need — so a run may proceed without it.
    fatal: bool = True


def _which(
    checks: list[Check], label: str, command: str, *, fatal: bool = True
) -> None:
    path = shutil.which(command)
    checks.append(Check(label, bool(path), path or "not found", fatal))


def _probe_server_url(url: str, timeout: float = 5.0) -> Check:
    import http.client

    parsed = urlparse(url)
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    try:
        connection = connection_class(parsed.netloc, timeout=timeout)
        try:
            connection.request("OPTIONS", parsed.path or "/")
            response = connection.getresponse()
            response.read()
        finally:
            connection.close()
    except OSError as exc:
        return Check("whisper-server URL", False, f"{url}: {exc}")
    return Check(
        "whisper-server URL", True, f"{url} responded with HTTP {response.status}"
    )


def _probe_cuda() -> tuple[bool, str]:
    """Probe torch in an isolated interpreter so the CLI stays torch-free."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yakiflow.whisperx_worker", "--probe"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "CUDA probe timed out after 30 seconds"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"CUDA probe failed: {exc}"
    lines = (result.stdout or "").splitlines()
    try:
        event = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        event = {}
    if isinstance(event, dict) and event.get("kind") == "probe":
        return bool(event.get("cuda_available")), str(event.get("detail", "CUDA unavailable"))
    if result.returncode == 0:
        return False, "CUDA unavailable"
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    return False, detail[-1] if detail else f"CUDA probe exited with {result.returncode}"


def run_doctor(
    settings: Settings, *, source_is_url: bool | None = None
) -> list[Check]:
    """Check everything this configuration needs, fatal parts marked as such.

    ``source_is_url`` tells the downloader check what kind of input the run
    was given; left unset, as it is for the ``doctor`` command, every check
    that some input could need stays fatal.
    """
    checks: list[Check] = []
    _which(checks, "ffmpeg", settings.commands.ffmpeg)
    # Advisory: without it the subtitles are still written, only against the
    # default script resolution instead of the video's own.
    ffprobe = shutil.which(settings.commands.ffprobe)
    checks.append(Check(
        "ffprobe",
        bool(ffprobe),
        ffprobe or (
            "not found (subtitles fall back to a "
            f"{DEFAULT_PLAY_RES[0]}x{DEFAULT_PLAY_RES[1]} script resolution)"
        ),
        False,
    ))
    _which(checks, "yt-dlp", settings.commands.yt_dlp, fatal=source_is_url is not False)
    backend = settings.transcription.backend
    if backend == "whisper-cli":
        _which(checks, "whisper-cli", settings.whisper.cli)
    elif backend == "whisper-server":
        if settings.whisper.server_url:
            checks.append(_probe_server_url(settings.whisper.server_url))
        else:
            _which(checks, "whisper-server", settings.whisper.server)
    if needs_local_whisper(settings):
        model = settings.whisper.model
        present = bool(model and model.is_file())
        # The run downloads the default model itself, so only a configured
        # path — which it never replaces — stops it when the file is missing.
        downloadable = (
            not present
            and model is not None
            and model.expanduser().resolve()
            == default_model_path().expanduser().resolve()
        )
        checks.append(
            Check(
                "whisper model",
                present,
                f"{model}; the run downloads it" if downloadable else str(model),
                not downloadable,
            )
        )
        if settings.whisper.vad_model:
            checks.append(
                Check(
                    "whisper VAD model",
                    settings.whisper.vad_model.is_file(),
                    str(settings.whisper.vad_model),
                )
            )
    elif backend.startswith("elevenlabs"):
        sdk_installed = importlib.util.find_spec("elevenlabs") is not None
        checks.append(
            Check(
                "elevenlabs SDK",
                sdk_installed,
                "installed" if sdk_installed else
                "not installed; install yakiflow[elevenlabs]",
            )
        )
        from .elevenlabs import elevenlabs_api_key_source

        # Only where the key would come from is reported, never the key.
        source = elevenlabs_api_key_source(settings)
        checks.append(Check("elevenlabs API key", source != "not set", source))
    if settings.alignment.backend == "whisperx":
        whisperx_installed = importlib.util.find_spec("whisperx") is not None
        checks.append(
            Check(
                "WhisperX",
                whisperx_installed,
                (
                    "installed"
                    if whisperx_installed
                    else "not installed; install yakiflow[whisperx-cpu] or yakiflow[whisperx-cuda]"
                ),
            )
        )
        cuda_available, torch_detail = _probe_cuda()
        if settings.alignment.device == "cuda":
            checks.append(Check("WhisperX CUDA", cuda_available, torch_detail))
        else:
            selected = "cuda" if settings.alignment.device == "auto" and cuda_available else "cpu"
            checks.append(Check("WhisperX device", True, f"{selected} ({torch_detail})"))
    agent_backends = {
        stage.backend
        for stage in (settings.agent.draft, settings.agent.final)
        if stage.backend
    }
    for name in sorted(agent_backends) or ["codex", "claude"]:
        executable = shutil.which(name)
        if not executable:
            checks.append(Check(f"{name} CLI", False, "not found"))
            continue
        auth_command = [name, "login", "status"] if name == "codex" else [name, "auth", "status"]
        try:
            result = subprocess.run(
                auth_command,
                capture_output=True,
                text=True,
                # The locale encoding would raise on a non-ASCII account name
                # under LC_ALL=C and abort every remaining check.
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            detail = (result.stdout or result.stderr).strip().splitlines()
            checks.append(Check(f"{name} auth", result.returncode == 0, detail[-1] if detail else f"exit {result.returncode}"))
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(Check(f"{name} auth", False, str(exc)))
    # Both review checks are advisory: without them the subtitles still get
    # written, only the review has no preview to show them in.
    if settings.review.display_mode in {"split", "both"}:
        tmux = shutil.which("tmux")
        checks.append(Check("tmux", bool(tmux), tmux or "not found (split review will have no preview)", False))
    if settings.review.display_mode in {"open", "both"}:
        command = settings.review.open_command or ""
        try:
            executable = shlex.split(command)[0]
        except IndexError:
            executable = ""
        path = shutil.which(executable) if executable else None
        checks.append(Check("review open command", bool(path), path or "not found", False))
    return checks


def fatal_failures(checks: Sequence[Check]) -> list[Check]:
    return [check for check in checks if not check.ok and check.fatal]


def advisory_failures(checks: Sequence[Check]) -> list[Check]:
    return [check for check in checks if not check.ok and not check.fatal]
