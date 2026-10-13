"""Shared cloud credentials and durable transcription metadata."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .config import Settings
from .database import JobDatabase
from .media import pcm_audio_duration
from .subtitles import write_text_atomic

_KEYRING_CACHE: dict[str, str] = {}


def keyring_key(provider: str) -> str | None:
    if provider in _KEYRING_CACHE:
        return _KEYRING_CACHE[provider]
    try:
        import keyring

        key = keyring.get_password("yakiflow", provider)
    except Exception:
        return None
    if key:
        _KEYRING_CACHE[provider] = key
    return key


def api_key_source(settings: Settings, provider: str, env: str) -> str:
    config = getattr(settings, provider.replace("-", "_"))
    slot = next(
        (
            name
            for name in ("api_key", "api_key_file", "api_key_command")
            if getattr(config, name) is not None
        ),
        None,
    )
    if slot and config.api_key_from_cli:
        return slot
    if os.environ.get(env, "").strip():
        return "env"
    return slot or ("keyring" if keyring_key(provider) else "not set")


def api_key(settings: Settings, provider: str, env: str) -> str:
    config = getattr(settings, provider.replace("-", "_"))
    source = api_key_source(settings, provider, env)
    if source == "env":
        return os.environ[env].strip()
    if source == "keyring":
        return keyring_key(provider)
    if source == "api_key":
        key = config.api_key.strip()
    elif source == "api_key_file":
        try:
            key = config.api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"cannot read {provider}.api-key-file: {exc}") from exc
    elif source == "api_key_command":
        result = subprocess.run(
            config.api_key_command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                f"{provider}.api-key-command exited with status {result.returncode}"
            )
        key = result.stdout.strip()
    else:
        raise ValueError(
            f"no {provider} API key; set {env}, {provider}.api-key / api-key-file / api-key-command, or yakiflow secret set {provider}"
        )
    if not key:
        raise ValueError(
            f"{provider}.{source.replace('_', '-')} produced an empty API key"
        )
    return key


def save_raw(work_dir: Path, provider: str, name: str, data: object) -> Path:
    path = work_dir / provider / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))
    return path


def effective_diarization(
    db: JobDatabase, provider: str, audio: Path, minutes: int
) -> bool:
    saved = db.get_checkpoint("diarization_state")
    if saved is not None:
        if saved["backend"] != provider:
            raise ValueError(
                "stored diarization belongs to a different transcription backend"
            )
        return bool(saved["enabled"])
    duration = pcm_audio_duration(audio)
    if duration is None:
        raise ValueError("cannot measure complete recording duration for diarization")
    enabled = minutes > 0 and duration < minutes * 60
    db.checkpoint(
        "diarization_state",
        {"backend": provider, "enabled": enabled, "duration": duration},
    )
    return enabled


def http_client(extra: str):
    try:
        import httpx
    except ImportError:
        raise RuntimeError(
            f"install yakiflow[{extra}] for HTTP transcription"
        ) from None
    return httpx.Client(timeout=600 if extra == "mai" else 120, follow_redirects=True)
