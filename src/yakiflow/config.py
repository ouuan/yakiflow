from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from platformdirs import user_cache_path, user_config_path

from .media_player import DEFAULT_VIDEO_OPEN_COMMAND


DEFAULT_MODEL_NAME = "ggml-large-v3-turbo-q5_0.bin"

# Stamped into every work directory's stored configuration. A work directory
# whose stamp differs was created by an incompatible yakiflow and must be
# finished by the version that created it.
SETTINGS_SCHEMA_VERSION = 3

TRANSCRIPTION_BACKENDS = (
    "whisper-cli",
    "whisper-server",
    "elevenlabs",
    "elevenlabs-stream",
    "fun-asr",
    "fun-asr-stream",
    "mai",
    "mai-stream",
)
STREAM_BACKENDS = (
    "whisper-cli",
    "whisper-server",
    "elevenlabs-stream",
    "fun-asr-stream",
    "mai-stream",
)
_STREAM_DEFAULTS = {
    backend: (
        "whisper-server"
        if backend.startswith("whisper")
        else backend.removesuffix("-stream") + "-stream"
    )
    for backend in TRANSCRIPTION_BACKENDS
}
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh")
AGENT_BACKENDS = ("codex", "claude", "gemini")

# The three spellings of the single "where does the ElevenLabs key come from"
# slot. A configuration layer that sets any one of them replaces the whole
# slot, exactly as it would replace a single-valued setting.
_API_KEY_SLOT = ("api_key", "api_key_file", "api_key_command")


def default_model_path() -> Path:
    return user_cache_path("yakiflow") / "models" / DEFAULT_MODEL_NAME


def _path(default: Path | None = None) -> Any:
    return field(default=default, metadata={"kind": "path"})


def _paths() -> Any:
    return field(default=(), metadata={"kind": "paths"})


def _argv(default: tuple[str, ...] | None = ()) -> Any:
    return field(default=default, metadata={"kind": "argv"})


def _grp(cls: type) -> Any:
    # ``from __future__ import annotations`` leaves ``f.type`` a string, so the
    # group class itself travels in the field metadata.
    return field(default_factory=cls, metadata={"group": cls})


@dataclass(frozen=True, slots=True)
class TranscriptionSettings:
    backend: str = "whisper-cli"
    stream_backend: str | None = None


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    model: Path | None = _path()
    vad: bool = True
    vad_model: Path | None = _path()
    cli: str = "whisper-cli"
    server: str = "whisper-server"
    server_url: str | None = None
    stream_chunk_seconds: int = 15
    stream_context_seconds: int = 5


@dataclass(frozen=True, slots=True)
class ElevenLabsSettings:
    api_key: str | None = None
    api_key_file: Path | None = _path()
    api_key_command: str | None = None
    model: str = "scribe_v2"
    realtime_model: str = "scribe_v2_realtime"
    diarize: bool = True
    num_speakers: int | None = None
    use_speaker_library: bool = True
    max_minutes: float = 600
    chunk_overlap_seconds: float = 30
    # Set by ``load_settings`` when the command line won the API-key slot, so
    # key resolution can rank that ahead of the environment variable. Internal:
    # a configuration file cannot set it, but a stored job round-trips it.
    api_key_from_cli: bool = field(default=False, metadata={"internal": True})


@dataclass(frozen=True, slots=True)
class FunASRSettings:
    api_key: str | None = None
    api_key_file: Path | None = _path()
    api_key_command: str | None = None
    api_key_from_cli: bool = field(default=False, metadata={"internal": True})
    model: str = "fun-asr"
    realtime_model: str = "fun-asr-realtime"
    api_url: str = "https://dashscope.aliyuncs.com/api/v1"
    realtime_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    diarize_max_minutes: int = 120
    diarize_over_limit: str = "disable"
    max_minutes: float = 720
    chunk_overlap_seconds: float = 30


@dataclass(frozen=True, slots=True)
class MAISettings:
    api_key: str | None = None
    api_key_file: Path | None = _path()
    api_key_command: str | None = None
    api_key_from_cli: bool = field(default=False, metadata={"internal": True})
    region: str | None = None
    endpoint: str | None = None
    model: str = "MAI-Transcribe-2"
    realtime_model: str = "MAI-Transcribe-2-Streaming"
    diarize_max_minutes: int = 15
    diarize_over_limit: str = "disable"
    max_minutes: float = 120
    chunk_overlap_seconds: float = 30


@dataclass(frozen=True, slots=True)
class AlignmentSettings:
    # ``None`` resolves per transcription backend: word-level ElevenLabs
    # backends default to ``none``, Whisper backends to ``vad``.
    backend: str | None = None
    device: str = "auto"
    model: str | None = None


@dataclass(frozen=True, slots=True)
class SubtitleSettings:
    max_cue_seconds: float = 8.0
    max_cue_chars: int = 84


@dataclass(frozen=True, slots=True)
class DraftAgentSettings:
    # The shared keys default to "unset" so the [agent] table can fill them.
    backend: str | None = None
    model: str | None = None
    effort: str | None = None
    extra_options: tuple[str, ...] | None = _argv(None)
    # Everything below only means something for drafting, so writing it
    # into [agent] or [agent.final] is an unknown-key error by construction.
    workers: int = 4
    batch_size: int = 20
    preceding_context: int = 10
    following_context: int = 5
    word_batch_size: int = 200
    word_following_context: int = 40
    timeout_seconds: float = 600
    max_attempts: int = 3
    retry_delay_seconds: float = 1


@dataclass(frozen=True, slots=True)
class StreamAgentSettings:
    backend: str | None = None
    model: str | None = None
    effort: str | None = None
    extra_options: tuple[str, ...] | None = _argv(None)
    # Capture uses its own latency-oriented defaults, independent of draft.
    workers: int = 4
    batch_size: int = 3
    preceding_context: int = 6
    following_context: int = 0
    word_batch_size: int = 80
    word_following_context: int = 0
    timeout_seconds: float = 120
    max_attempts: int = 2
    retry_delay_seconds: float = 1


@dataclass(frozen=True, slots=True)
class FinalAgentSettings:
    backend: str | None = None
    model: str | None = None
    effort: str | None = None
    extra_options: tuple[str, ...] | None = _argv(None)


@dataclass(frozen=True, slots=True)
class AgentSettings:
    backend: str | None = None
    model: str | None = None
    effort: str | None = None
    extra_options: tuple[str, ...] | None = _argv(None)
    draft: DraftAgentSettings = _grp(DraftAgentSettings)
    final: FinalAgentSettings = _grp(FinalAgentSettings)
    stream: StreamAgentSettings = _grp(StreamAgentSettings)


@dataclass(frozen=True, slots=True)
class ReviewSettings:
    display_mode: str = "split"
    open_command: str | None = None
    auto_open_video: bool = False
    video_open_command: str = DEFAULT_VIDEO_OPEN_COMMAND


@dataclass(frozen=True, slots=True)
class CommandsSettings:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    yt_dlp: str = "yt-dlp"
    yt_dlp_options: tuple[str, ...] = _argv()


_STAGE_MODEL_DEFAULTS = {
    "draft": {"codex": "gpt-6-luna", "claude": "sonnet", "gemini": "flash"},
    "stream": {"codex": "gpt-6-luna", "claude": "sonnet", "gemini": "flash"},
    "final": {"codex": "gpt-6.1-sol", "claude": "opus", "gemini": "pro"},
}
_STAGE_EFFORT_DEFAULTS = {"draft": "medium", "stream": "low", "final": "medium"}


@dataclass(frozen=True, slots=True)
class Settings:
    actual_diarization: bool | None = field(default=None, metadata={"internal": True})
    source_language: str | None = None
    target_language: str | None = None
    output_mode: str = "bilingual"
    download_dir: Path | None = _path()
    output_dir: Path | None = _path()
    memory: Path | None = _path()
    context_files: tuple[Path, ...] = _paths()
    work_dir: Path | None = _path()
    keep_workdir: bool = False
    transcription: TranscriptionSettings = _grp(TranscriptionSettings)
    whisper: WhisperSettings = _grp(WhisperSettings)
    elevenlabs: ElevenLabsSettings = _grp(ElevenLabsSettings)
    fun_asr: FunASRSettings = _grp(FunASRSettings)
    mai: MAISettings = _grp(MAISettings)
    alignment: AlignmentSettings = _grp(AlignmentSettings)
    stream: bool = False
    subtitles: SubtitleSettings = _grp(SubtitleSettings)
    agent: AgentSettings = _grp(AgentSettings)
    review: ReviewSettings = _grp(ReviewSettings)
    commands: CommandsSettings = _grp(CommandsSettings)

    @property
    def diarization_enabled(self) -> bool:
        """Whether cues can carry speakers.

        Batch cloud backends can diarize. A job's duration-based decision
        overrides this configured capability once the complete audio exists.
        """
        if self.actual_diarization is not None:
            return self.actual_diarization
        return (
            (self.transcription.backend == "elevenlabs" and self.elevenlabs.diarize)
            or (
                self.transcription.backend == "fun-asr"
                and self.fun_asr.diarize_max_minutes > 0
            )
            or (
                self.transcription.backend == "mai" and self.mai.diarize_max_minutes > 0
            )
        )

    def resolved(self) -> Settings:
        """Fill derived defaults; safe to apply to already-resolved settings."""

        def absolute(path: Path | None) -> Path | None:
            return path.expanduser().resolve() if path is not None else None

        whisper = replace(
            self.whisper,
            model=absolute(self.whisper.model or default_model_path()),
            vad_model=absolute(self.whisper.vad_model) if self.whisper.vad else None,
        )
        elevenlabs = replace(
            self.elevenlabs,
            api_key_file=absolute(self.elevenlabs.api_key_file),
        )
        alignment = replace(
            self.alignment,
            backend=self.alignment.backend
            or (
                "none"
                if not self.transcription.backend.startswith("whisper")
                else "vad"
            ),
        )
        agent = replace(
            self.agent,
            draft=self._resolved_stage(self.agent.draft, "draft"),
            stream=self._resolved_stage(self.agent.stream, "stream"),
            final=self._resolved_stage(self.agent.final, "final"),
        )
        return replace(
            self,
            memory=absolute(self.memory or user_config_path("yakiflow") / "memory.md"),
            download_dir=absolute(self.download_dir),
            output_dir=absolute(self.output_dir),
            context_files=tuple(
                path.expanduser().resolve() for path in self.context_files
            ),
            work_dir=absolute(self.work_dir),
            transcription=replace(
                self.transcription,
                stream_backend=self.transcription.stream_backend
                or _STREAM_DEFAULTS.get(self.transcription.backend),
            ),
            whisper=whisper,
            elevenlabs=elevenlabs,
            fun_asr=replace(
                self.fun_asr, api_key_file=absolute(self.fun_asr.api_key_file)
            ),
            mai=replace(
                self.mai,
                api_key_file=absolute(self.mai.api_key_file),
                region=self.mai.region or os.environ.get("SPEECH_REGION") or None,
            ),
            alignment=alignment,
            agent=agent,
        )

    def for_stream(self) -> Settings:
        """Select the capture backend and agent without changing saved batch settings."""
        return replace(
            self,
            transcription=replace(
                self.transcription, backend=self.transcription.stream_backend
            ),
            agent=replace(
                self.agent,
                draft=DraftAgentSettings(
                    **{
                        f.name: getattr(self.agent.stream, f.name)
                        for f in fields(DraftAgentSettings)
                    }
                ),
            ),
            alignment=replace(self.alignment, backend="none"),
        )

    def _resolved_stage(
        self,
        stage: DraftAgentSettings | StreamAgentSettings | FinalAgentSettings,
        name: str,
    ) -> DraftAgentSettings | StreamAgentSettings | FinalAgentSettings:
        """Resolve one stage: stage value, then [agent] shared, then default."""
        backend = stage.backend or self.agent.backend
        model = stage.model or self.agent.model
        if model is None and backend is not None:
            model = _STAGE_MODEL_DEFAULTS[name].get(backend)
        effort = stage.effort or self.agent.effort or _STAGE_EFFORT_DEFAULTS[name]
        extra_options = stage.extra_options
        if extra_options is None:
            extra_options = self.agent.extra_options
        if extra_options is None:
            extra_options = ()
        return replace(
            stage,
            backend=backend,
            model=model,
            effort=effort,
            extra_options=extra_options,
        )


# Old flat settings and where each one moved. Only consulted to make the
# unknown-key error explain the regrouped schema instead of looking like a typo.
_RELOCATED = {
    "whisper_model": "whisper.model",
    "vad": "whisper.vad",
    "vad_model": "whisper.vad-model",
    "whisper_cli": "whisper.cli",
    "whisper_server": "whisper.server",
    "alignment_backend": "alignment.backend",
    "alignment_device": "alignment.device",
    "alignment_model": "alignment.model",
    "translation_backend": "agent.backend",
    "draft_model": "agent.draft.model",
    "draft_effort": "agent.draft.effort",
    "final_model": "agent.final.model",
    "final_effort": "agent.final.effort",
    "draft_codex_options": "agent.draft.extra-options",
    "final_codex_options": "agent.final.extra-options",
    "draft_claude_options": "agent.draft.extra-options",
    "final_claude_options": "agent.final.extra-options",
    "agent_workers": "agent.draft.workers",
    "translation_batch_size": "agent.draft.batch-size",
    "translation_context": "agent.draft.preceding-context",
    "translation_following_context": "agent.draft.following-context",
    "draft_agent_timeout_seconds": "agent.draft.timeout-seconds",
    "agent_max_attempts": "agent.draft.max-attempts",
    "agent_retry_delay_seconds": "agent.draft.retry-delay-seconds",
    "review_display_mode": "review.display-mode",
    "review_open_command": "review.open-command",
    "auto_open_video": "review.auto-open-video",
    "video_open_command": "review.video-open-command",
    "stream_chunk_seconds": "whisper.stream-chunk-seconds",
    "stream_context_seconds": "whisper.stream-context-seconds",
    "ffmpeg": "commands.ffmpeg",
    "yt_dlp": "commands.yt-dlp",
    "yt_dlp_options": "commands.yt-dlp-options",
}


def _read_toml(
    path: Path | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}, {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    raw_profiles = data.pop("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError(f"profiles must be a table in {path}")
    profiles: dict[str, dict[str, Any]] = {}
    for name, values in raw_profiles.items():
        if not isinstance(values, dict):
            raise ValueError(f"profile {name!r} must be a table in {path}")
        for key in values:
            if key.replace("-", "_") in {
                "profile",
                "profiles",
                "inherit",
                "inherits",
                "extends",
            }:
                raise ValueError(
                    f"profile {name!r} cannot inherit from another profile in {path}"
                )
        profiles[str(name)] = values
    return data, profiles


def _clean(
    values: Mapping[str, Any],
    *,
    source: str | None = None,
    cls: type = Settings,
    prefix: str = "",
) -> dict[str, Any]:
    """Normalize one settings mapping into a nested dict of leaf values.

    ``source`` names a configuration layer. Naming one makes unknown keys an
    error, so a misspelled setting is reported instead of silently leaving the
    default in place; mappings rebuilt from a saved job stay permissive so a
    same-schema work directory with fewer fields can still be resumed.
    """
    specs = {f.name: f for f in fields(cls)}
    result: dict[str, Any] = {}
    for raw_key, value in values.items():
        key = str(raw_key).replace("-", "_")
        dotted = f"{prefix}{key.replace('_', '-')}"
        spec = specs.get(key)
        if spec is None or (spec.metadata.get("internal") and source is not None):
            if source is None:
                continue
            relocated = _RELOCATED.get(key) if not prefix else None
            if relocated is not None:
                raise ValueError(
                    f"setting {dotted!r} in {source} moved to {relocated!r}: "
                    "the configuration schema is grouped now; see the README"
                )
            raise ValueError(f"unknown setting {dotted!r} in {source}")
        if value is None:
            continue
        group = spec.metadata.get("group")
        if group is not None:
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{dotted!r} is a table of settings, not a single value"
                    + (f" in {source}" if source else "")
                )
            result[key] = _clean(value, source=source, cls=group, prefix=f"{dotted}.")
            continue
        if isinstance(value, Mapping):
            raise ValueError(
                f"{dotted!r} is a single setting, not a table"
                + (f" in {source}" if source else "")
            )
        kind = spec.metadata.get("kind")
        if kind == "path":
            result[key] = Path(value).expanduser()
        elif kind == "paths":
            items = (
                value
                if isinstance(value, Sequence) and not isinstance(value, str)
                else (value,)
            )
            result[key] = tuple(Path(item).expanduser() for item in items)
        elif kind == "argv":
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"{dotted} must be an array of strings")
            result[key] = tuple(value)
        else:
            result[key] = value
    if cls in (ElevenLabsSettings, FunASRSettings, MAISettings) and source is not None:
        slots = [name for name in _API_KEY_SLOT if name in result]
        if len(slots) > 1:
            listed = ", ".join(name.replace("_", "-") for name in slots)
            raise ValueError(
                f"{prefix}{listed} are "
                f"mutually exclusive in {source}: they are three spellings of "
                "the same API-key setting"
            )
    return result


def _merge_layer(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    """Deep-merge one cleaned configuration layer into ``base``, leaf by leaf.

    Each provider's API-key slot merges as a unit: a layer that sets any of its
    three spellings replaces whichever spelling a lower layer had chosen.
    """
    for provider in ("elevenlabs", "fun_asr", "mai"):
        incoming = overlay.get(provider)
        if isinstance(incoming, Mapping) and any(
            name in incoming for name in _API_KEY_SLOT
        ):
            existing = base.get(provider)
            if isinstance(existing, dict):
                for name in _API_KEY_SLOT:
                    existing.pop(name, None)
    stream = overlay.get("agent", {}).get("stream", {})
    previous = base.get("agent", {}).get("stream", {})
    if (
        "backend" in stream
        and stream["backend"] != previous.get("backend")
        and "model" not in stream
    ):
        previous.pop("model", None)
    _deep_merge(base, overlay)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            if isinstance(node, dict):
                _deep_merge(node, value)
            else:
                base[key] = dict(value)
        else:
            base[key] = value


def _build(mapping: Mapping[str, Any], cls: type = Settings) -> Any:
    kwargs: dict[str, Any] = {}
    for spec in fields(cls):
        if spec.name not in mapping:
            continue
        group = spec.metadata.get("group")
        value = mapping[spec.name]
        kwargs[spec.name] = _build(value, group) if group is not None else value
    return cls(**kwargs)


def build_settings(values: Mapping[str, Any]) -> Settings:
    """Build unresolved ``Settings`` from one nested, permissive mapping."""
    return _build(_clean(values, source=None))


def load_settings(
    cli: Mapping[str, Any] | None = None,
    *,
    project_file: Path | None = None,
    user_file: Path | None = None,
    profile: str | None = None,
    cli_config: Mapping[str, Any] | None = None,
    stored: Mapping[str, Any] | None = None,
) -> Settings:
    """Load base/profile settings and overlay the command line.

    ``cli_config`` carries the nested ``-c key=value`` overrides and is checked
    as strictly as a configuration file; ``cli`` carries the values of the
    dedicated flags, which argparse already validated. ``stored`` is a saved
    job's configuration and forms the lowest layer when resuming.
    """
    user_file = user_file or user_config_path("yakiflow") / "config.toml"
    if project_file is None:
        candidate = Path.cwd() / "yakiflow.toml"
        project_file = candidate if candidate.exists() else None
    user_base, user_profiles = _read_toml(user_file)
    project_base, project_profiles = _read_toml(project_file)
    if (
        profile is not None
        and profile not in user_profiles
        and profile not in project_profiles
    ):
        raise ValueError(f"unknown profile {profile!r}")
    # Precedence, lowest first: stored job settings, user base, project base,
    # user profile, project profile, -c overrides, dedicated flags.
    merged: dict[str, Any] = {}
    if stored is not None:
        _merge_layer(merged, _clean(stored, source=None))
    _merge_layer(merged, _clean(user_base, source=str(user_file)))
    _merge_layer(merged, _clean(project_base, source=str(project_file)))
    if profile is not None:
        _merge_layer(
            merged,
            _clean(
                user_profiles.get(profile, {}),
                source=f"profile {profile!r} in {user_file}",
            ),
        )
        _merge_layer(
            merged,
            _clean(
                project_profiles.get(profile, {}),
                source=f"profile {profile!r} in {project_file}",
            ),
        )
    config_layer = _clean(cli_config or {}, source="command line (-c)")
    _merge_layer(merged, config_layer)
    flag_layer = _clean(cli or {}, source=None)
    _merge_layer(merged, flag_layer)
    for layer in (config_layer, flag_layer):
        for provider in ("elevenlabs", "fun_asr", "mai"):
            group = layer.get(provider, {})
            if any(name in group for name in _API_KEY_SLOT):
                merged.setdefault(provider, {})["api_key_from_cli"] = True
    return _build(merged).resolved()


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def validate_batch_settings(config: Any, provider: str) -> None:
    """Also used at the request boundary by direct transcriber callers."""
    for field_name, positive in (
        ("max_minutes", True),
        ("chunk_overlap_seconds", False),
    ):
        value = getattr(config, field_name)
        if (
            type(value) not in (int, float)
            or (type(value) is float and not math.isfinite(value))
            or (value <= 0 if positive else value < 0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(
                f"{provider}.{field_name.replace('_', '-')} must be a finite {qualifier} number"
            )
    if hasattr(config, "diarize_max_minutes"):
        if (
            type(config.diarize_max_minutes) is not int
            or config.diarize_max_minutes < 0
        ):
            raise ValueError(
                f"{provider}.diarize-max-minutes must be a non-negative integer"
            )
        if config.diarize_over_limit not in ("disable", "split"):
            raise ValueError(f"{provider}.diarize-over-limit must be disable or split")


def validate_run_settings(
    settings: Settings, *, check_credentials: bool = True
) -> None:
    for provider in ("elevenlabs", "fun-asr", "mai"):
        validate_batch_settings(getattr(settings, provider.replace("-", "_")), provider)
    if not isinstance(settings.stream, bool):
        raise ValueError(
            "stream must be a boolean; the old [stream] table is unsupported"
        )
    if settings.transcription.backend not in TRANSCRIPTION_BACKENDS:
        raise ValueError(
            "transcription.backend must be one of " + ", ".join(TRANSCRIPTION_BACKENDS)
        )
    if settings.transcription.stream_backend not in STREAM_BACKENDS:
        raise ValueError(
            "transcription.stream-backend must be Whisper or a realtime backend"
        )
    if settings.stream:
        settings = settings.for_stream()
    if not settings.target_language:
        raise ValueError("target-language is required (-t/--target-language)")
    if not settings.source_language:
        raise ValueError(
            "source-language is required (-s/--source-language; "
            "use 'auto' for automatic detection)"
        )
    if settings.output_mode not in {"source", "translated", "bilingual", "all"}:
        raise ValueError(
            "output-mode must be source, translated, bilingual, or all (--output-mode)"
        )
    cloud_backend = not settings.transcription.backend.startswith("whisper")
    if settings.whisper.server_url is not None:
        parsed = urlparse(settings.whisper.server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "whisper.server-url must be an http(s) URL, got "
                f"{settings.whisper.server_url!r}"
            )
    if settings.alignment.backend not in {"vad", "whisperx", "none"}:
        raise ValueError("alignment.backend must be vad, whisperx, or none")
    if cloud_backend and settings.alignment.backend == "vad":
        raise ValueError(
            "alignment.backend 'vad' needs Whisper's VAD intervals; use "
            "'none' (or 'whisperx') with a cloud transcription backend"
        )
    if settings.alignment.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("alignment.device must be auto, cpu, or cuda")
    if check_credentials and settings.transcription.backend.startswith("fun-asr"):
        from .cloud import api_key

        api_key(settings, "fun-asr", "DASHSCOPE_API_KEY")
    if check_credentials and settings.transcription.backend.startswith("elevenlabs"):
        from .elevenlabs import elevenlabs_api_key

        elevenlabs_api_key(settings)
    for name, url, schemes in (
        ("api-url", settings.fun_asr.api_url, {"http", "https"}),
        ("realtime-url", settings.fun_asr.realtime_url, {"ws", "wss"}),
    ):
        parsed = urlparse(url)
        if parsed.scheme not in schemes or not parsed.netloc:
            raise ValueError(f"fun-asr.{name} must be a valid URL")
    if (
        not settings.fun_asr.model.startswith("fun-asr")
        or "flash" in settings.fun_asr.model
        or "realtime" in settings.fun_asr.model
    ):
        raise ValueError(
            "fun-asr.model must select a batch Fun-ASR model; Flash is unsupported"
        )
    if not settings.fun_asr.realtime_model.startswith("fun-asr-realtime"):
        raise ValueError("fun-asr.realtime-model must select Fun-ASR realtime")
    if settings.mai.endpoint:
        parsed = urlparse(settings.mai.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("mai.endpoint must be an http(s) URL")
    if settings.transcription.backend.startswith("mai"):
        from .cloud import api_key

        if check_credentials:
            api_key(settings, "mai", "SPEECH_KEY")
        if not settings.mai.region and (
            settings.transcription.backend == "mai-stream" or not settings.mai.endpoint
        ):
            raise ValueError(
                "mai.region or SPEECH_REGION is required (or mai.endpoint for REST)"
            )
    if settings.mai.region and not all(
        c.isalnum() or c == "-" for c in settings.mai.region
    ):
        raise ValueError("mai.region must be an Azure region identifier")
    num_speakers = settings.elevenlabs.num_speakers
    if num_speakers is not None and not 1 <= num_speakers <= 32:
        raise ValueError(
            "elevenlabs.num-speakers must be between 1 and 32, the most "
            "speakers diarization separates"
        )
    for stage_name in ("draft", "final"):
        stage = getattr(settings.agent, stage_name)
        if stage.backend is None:
            raise ValueError(
                "agent.backend is required (" + ", ".join(AGENT_BACKENDS) + ")"
            )
        if stage.backend not in AGENT_BACKENDS:
            raise ValueError(
                f"agent.{stage_name}.backend must be one of "
                + ", ".join(AGENT_BACKENDS)
                + f", got {stage.backend!r}"
            )
        if stage.effort not in EFFORT_LEVELS:
            raise ValueError(
                f"agent.{stage_name}.effort must be one of " + ", ".join(EFFORT_LEVELS)
            )
    draft = settings.agent.draft
    _require_positive(draft.workers, "agent.draft.workers")
    _require_positive(draft.batch_size, "agent.draft.batch-size")
    _require_non_negative(draft.preceding_context, "agent.draft.preceding-context")
    _require_non_negative(draft.following_context, "agent.draft.following-context")
    if draft.word_batch_size < 50:
        raise ValueError("agent.draft.word-batch-size must be at least 50")
    _require_non_negative(
        draft.word_following_context, "agent.draft.word-following-context"
    )
    _require_positive(draft.timeout_seconds, "agent.draft.timeout-seconds")
    _require_positive(draft.max_attempts, "agent.draft.max-attempts")
    _require_non_negative(draft.retry_delay_seconds, "agent.draft.retry-delay-seconds")
    _require_positive(settings.subtitles.max_cue_seconds, "subtitles.max-cue-seconds")
    if settings.subtitles.max_cue_chars < 8:
        raise ValueError("subtitles.max-cue-chars must be at least 8")
    if settings.review.display_mode not in {"split", "open", "both"}:
        raise ValueError("review.display-mode must be split, open, or both")
    if (
        settings.review.display_mode in {"open", "both"}
        and not settings.review.open_command
    ):
        raise ValueError(
            "review.open-command is required when review.display-mode is open or both"
        )
    _require_positive(
        settings.whisper.stream_chunk_seconds, "whisper.stream-chunk-seconds"
    )
    _require_non_negative(
        settings.whisper.stream_context_seconds, "whisper.stream-context-seconds"
    )
