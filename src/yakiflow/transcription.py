from __future__ import annotations

import asyncio
import json
import re
import socket
import uuid
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right, insort
from collections.abc import Awaitable, Callable
from math import inf
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar, Sequence
from urllib.parse import urlparse

from .config import Settings
from .database import JobDatabase
from .models import Cue, TranscriptEvent, cue_id
from .process import CommandRunner, ProcessError, terminate_process


EventCallback = Callable[[TranscriptEvent], Awaitable[None]]
ProgressCallback = Callable[[float], Awaitable[None]]
WarningCallback = Callable[[str], Awaitable[None]]
LINE_RE = re.compile(
    r"\[\s*(?P<start>\d\d:\d\d:\d\d[.,]\d{3})\s*-->\s*(?P<end>\d\d:\d\d:\d\d[.,]\d{3})\s*\]\s*(?P<text>.*)"
)
PROGRESS_RE = re.compile(r"progress\s*=\s*(?P<percent>\d{1,3})%")
VAD_SEGMENT_RE = re.compile(
    r"VAD segment \d+: start = (?P<start>\d+(?:\.\d+)?), "
    r"end = (?P<end>\d+(?:\.\d+)?)"
)
_PIPE_CHUNK_BYTES = 64 * 1024
# How far before the last durable cue a crashed run restarts, and how far apart
# two renderings of the same speech may start before they stop looking equal.
RECOVERY_OVERLAP_SECONDS = 5.0
_DUPLICATE_START_TOLERANCE = 2.5
_VAD_CHECKPOINT_INTERVAL_SECONDS = 2.0


def normalize_source_language(language: object) -> str | None:
    """Normalize Whisper locale labels to WhisperX's language codes."""
    if not isinstance(language, str):
        return None
    normalized = language.strip().replace("_", "-").casefold()
    if not normalized or normalized == "auto":
        return None
    return normalized.split("-", 1)[0]


def detected_source_language(data: dict[str, Any]) -> str | None:
    result = data.get("result")
    language = result.get("language") if isinstance(result, dict) else None
    if language is None:
        language = data.get("language")
    return normalize_source_language(language)


def parse_timestamp(
    value: str | int | float, *, integer_milliseconds: bool = False
) -> float:
    if isinstance(value, (int, float)):
        return (
            float(value) / 1000
            if integer_milliseconds and isinstance(value, int)
            else float(value)
        )
    parts = value.replace(",", ".").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(value)


def parse_vad_segment(value: str, *, offset: float = 0.0) -> tuple[float, float] | None:
    """Parse one original-timeline VAD interval from whisper.cpp stderr."""
    match = VAD_SEGMENT_RE.search(value)
    if match is None:
        return None
    start = float(match.group("start")) + offset
    end = float(match.group("end")) + offset
    return (round(start, 3), round(end, 3)) if end >= start else None


INGEST_VAD_MERGE_TOLERANCE = 0.01


def merge_vad_intervals(
    intervals: Sequence[tuple[float, float]],
    tolerance: float = INGEST_VAD_MERGE_TOLERANCE,
) -> list[tuple[float, float]]:
    """Coalesce speech spans that are separated by at most ``tolerance``."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end < start:
            continue
        if merged and start <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def append_vad_interval(
    merged: list[tuple[float, float]],
    interval: tuple[float, float],
    tolerance: float = INGEST_VAD_MERGE_TOLERANCE,
) -> None:
    """Insert one interval into an already-merged list, in place.

    whisper.cpp reports its speech spans in order, so the common case costs a
    comparison instead of re-sorting and re-merging everything seen so far.
    """
    start, end = interval
    if end < start:
        return
    if not merged or start > merged[-1][1] + tolerance:
        merged.append(interval)
    elif start >= merged[-1][0]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        # Out of order: fall back to the full merge.
        merged[:] = merge_vad_intervals([*merged, interval], tolerance)


def parse_json_full(data: dict[str, Any], *, offset: float = 0.0, ordinal_offset: int = 0) -> list[Cue]:
    entries = data.get("transcription") or data.get("segments") or []
    cues: list[Cue] = []
    for entry in entries:
        timestamps = entry.get("timestamps", {})
        offsets = entry.get("offsets", {})

        def entry_time(timestamp_key: str, segment_key: str, default: float) -> float:
            if timestamp_key in timestamps:
                return parse_timestamp(timestamps[timestamp_key])
            if segment_key in entry:
                # The segments JSON format measures start/end in seconds,
                # including when those values happen to be integral.
                return parse_timestamp(entry[segment_key])
            if timestamp_key in offsets:
                # whisper.cpp's integer offset fields are milliseconds.
                return parse_timestamp(
                    offsets[timestamp_key], integer_milliseconds=True
                )
            return default

        relative_start = entry_time("from", "start", 0.0)
        relative_end = entry_time("to", "end", relative_start)
        start = relative_start + offset
        end = relative_end + offset
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        ordinal = ordinal_offset + len(cues)
        cues.append(Cue(cue_id(ordinal), start, max(start, end), text, metadata={"whisper": entry}))
    return cues


def _start_order(cue: Cue) -> tuple[float, float]:
    return (cue.start, cue.end)


def merge_overlap(existing: list[Cue], incoming: list[Cue]) -> list[Cue]:
    """Merge recovery/stream results without duplicating an overlap window."""
    result = sorted(existing, key=_start_order)
    for cue in incoming:
        normalized = " ".join(cue.source.casefold().split())
        # Compare against every cue whose start is close in time, rather than
        # against a fixed number of trailing cues: dense speech packs far more
        # cues into the re-transcribed overlap than any fixed count covers, and
        # a duplicate outside that count would be appended a second time.
        first = bisect_left(
            result, (cue.start - _DUPLICATE_START_TOLERANCE, -inf), key=_start_order
        )
        last = bisect_right(
            result, (cue.start + _DUPLICATE_START_TOLERANCE, inf), key=_start_order
        )
        duplicate = any(
            " ".join(old.source.casefold().split()) == normalized
            for old in result[first:last]
        )
        if duplicate:
            continue
        merged = Cue(
            cue_id(len(result)), cue.start, cue.end, cue.source,
            metadata=cue.metadata, speaker=cue.speaker,
        )
        # Keep the list ordered so the next lookup can bisect it.
        insort(result, merged, key=_start_order)
    return result


def whisper_language_and_vad_args(settings: Settings) -> list[str]:
    """Return the language/VAD flags shared by whisper-cli and whisper-server."""
    args: list[str] = []
    if settings.source_language:
        args += ["--language", settings.source_language]
    if settings.whisper.vad_model:
        args += ["--vad", "--vad-model", str(settings.whisper.vad_model)]
    return args


def needs_local_whisper(settings: Settings) -> bool:
    """Whether this configuration runs a Whisper model on this machine.

    Gates the default-model download and the model-file checks: an external
    whisper-server owns its own model, and the ElevenLabs backends have none.
    """
    backend = settings.transcription.backend
    if backend == "whisper-cli":
        return True
    if backend == "whisper-server":
        return settings.whisper.server_url is None
    return False


async def extract_tail(
    runner: CommandRunner,
    ffmpeg: str,
    audio: Path,
    start: float,
    destination: Path,
) -> None:
    """Write 16 kHz mono PCM covering ``audio`` from ``start`` onwards."""
    await runner.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", audio,
        "-ac", "1", "-ar", "16000", destination,
    ])


class ChunkTimeline:
    """The accumulated preview timeline built from overlapping stream chunks."""

    def __init__(self) -> None:
        self.cues: list[Cue] = []

    def add(self, incoming: Sequence[Cue]) -> list[Cue]:
        """Merge one chunk's cues in and return only the newly added ones."""
        known = {cue.id for cue in self.cues}
        self.cues = merge_overlap(self.cues, list(incoming))
        return [cue for cue in self.cues if cue.id not in known]


class ElapsedProgressTicker:
    """Synthetic progress for backends that report none of their own.

    Emits ``min(0.95, elapsed / estimate)`` on an interval; a real progress
    figure reported by the backend replaces the synthetic curve from then on.
    """

    def __init__(
        self,
        estimate_seconds: float,
        on_progress: ProgressCallback | None,
        interval: float = 2.0,
    ):
        self.estimate_seconds = max(1.0, estimate_seconds)
        self.on_progress = on_progress
        self.interval = interval
        self._real: float | None = None
        self._task: asyncio.Task[None] | None = None

    def report_real(self, fraction: float) -> None:
        current = self._real if self._real is not None else 0.0
        self._real = min(1.0, max(current, fraction))

    async def _run(self) -> None:
        started = monotonic()
        while True:
            await asyncio.sleep(self.interval)
            if self.on_progress is None:
                continue
            if self._real is not None:
                await self.on_progress(self._real)
            else:
                elapsed = monotonic() - started
                await self.on_progress(min(0.95, elapsed / self.estimate_seconds))

    async def __aenter__(self) -> ElapsedProgressTicker:
        if self.on_progress is not None:
            self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class Transcriber(ABC):
    """One speech-to-text backend, usable for chunk previews and full audio.

    The preview and the authoritative pass each build their own instance:
    provisional chunk state is discarded anyway, and per-instance process or
    socket lifetime keeps the two passes from sharing half-open resources.
    """

    name: ClassVar[str]
    # A continuous-feed backend paces its own transport (small realtime
    # frames), so the streaming acquirer hands over every new second of audio
    # as it appears instead of cutting fixed chunks with overlap context.
    continuous_feed: ClassVar[bool] = False

    def __init__(self) -> None:
        self._chunk_timeline = ChunkTimeline()
        self.on_warning: WarningCallback | None = None

    async def _warn(self, message: str) -> None:
        if self.on_warning is not None:
            await self.on_warning(message)

    @property
    def chunk_cues(self) -> list[Cue]:
        return self._chunk_timeline.cues

    @abstractmethod
    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]: ...

    async def start_chunks(self) -> None:
        """Prepare for streaming chunk previews; default is nothing to do."""

    @abstractmethod
    async def submit_chunk(self, wav: Path, offset: float) -> list[Cue]:
        """Transcribe one stream excerpt and return the newly added cues."""

    async def close_chunks(self) -> None:
        """Release chunk-preview resources; default is nothing to hold."""


class WhisperCliTranscriber(Transcriber):
    name = "whisper-cli"

    def __init__(self, settings: Settings, work_dir: Path, db: JobDatabase, runner: CommandRunner | None = None):
        super().__init__()
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner or CommandRunner()

    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]:
        preserved = list(resume_from)
        input_audio = audio
        input_offset = 0.0
        prefix = self.work_dir / "whisper-final"
        if preserved:
            # Whisper's stdout cues are complete segments. Restart at the end
            # of the last durable segment so their IDs, text, and translations
            # remain attached to the same timeline prefix.
            input_offset = max(cue.end for cue in preserved)
            input_audio = self.work_dir / "whisper-resume.wav"
            await self._extract_tail(audio, input_offset, input_audio)
            prefix = self.work_dir / "whisper-resume"
        incremental: list[Cue] = []
        stored_intervals = self.db.get_checkpoint("whisper_vad_intervals", [])
        vad_intervals = (
            [(float(start), float(end)) for start, end in stored_intervals]
            if preserved and self.settings.whisper.vad_model
            else []
        )
        self.db.checkpoint("whisper_vad_intervals", vad_intervals)

        last_vad_checkpoint = monotonic()

        def record_vad(value: str, *, offset: float) -> None:
            nonlocal last_vad_checkpoint
            interval = parse_vad_segment(value, offset=offset)
            if interval is None:
                return
            append_vad_interval(vad_intervals, interval)
            # VAD is computed before decoding. Persist it incrementally so an
            # interrupted Whisper run can resume without losing the timeline,
            # but a committed write per line means one commit per speech span;
            # a short interval bounds that, and ``transcribe`` flushes the final
            # list either way.
            now = monotonic()
            if now - last_vad_checkpoint < _VAD_CHECKPOINT_INTERVAL_SECONDS:
                return
            last_vad_checkpoint = now
            self.db.checkpoint("whisper_vad_intervals", vad_intervals)

        async def line(stream: str, value: str) -> None:
            self.db.log("whisper-cli", stream, value)
            if stream == "stderr" and self.settings.whisper.vad_model:
                record_vad(value, offset=input_offset)
            progress = PROGRESS_RE.search(value)
            if progress and on_progress:
                await on_progress(min(100, int(progress.group("percent"))) / 100)
            match = LINE_RE.search(value)
            if stream != "stdout" or not match:
                return
            start = parse_timestamp(match.group("start")) + input_offset
            end = parse_timestamp(match.group("end")) + input_offset
            cue = Cue(
                cue_id(len(preserved) + len(incremental)),
                start,
                # whisper.cpp can print an inverted span when VAD remaps a
                # segment onto the original timeline. Clamp it exactly as the
                # JSON path does: a callback exception aborts the whole run.
                max(start, end),
                match.group("text").strip(),
            )
            incremental.append(cue)
            if on_event:
                await on_event(TranscriptEvent(cue, final=False))

        try:
            await self.runner.run(self._cli_args(input_audio, prefix), on_line=line)
        except ProcessError:
            # Resume from five seconds before the last stable cue and merge the
            # overlap. This retry remains a single full-model invocation.
            if not incremental:
                raise
            durable = preserved + incremental
            resume_at = max(
                0.0, max(cue.end for cue in durable) - RECOVERY_OVERLAP_SECONDS
            )
            recovered = self.work_dir / "recovery.wav"
            await self._extract_tail(audio, resume_at, recovered)
            recovery_prefix = self.work_dir / "whisper-recovery"

            async def recovery_line(stream: str, value: str) -> None:
                self.db.log("whisper-cli-recovery", stream, value)
                if stream == "stderr" and self.settings.whisper.vad_model:
                    record_vad(value, offset=resume_at)

            await self.runner.run(
                self._cli_args(recovered, recovery_prefix),
                on_line=recovery_line,
            )
            recovered_cues = self._load_json(
                recovery_prefix,
                offset=resume_at,
                ordinal_offset=len(durable),
            )
            cues = merge_overlap(durable, recovered_cues)
        else:
            completed = (
                self._load_json(
                    prefix,
                    offset=input_offset,
                    ordinal_offset=len(preserved),
                )
                if self._json_path(prefix).exists()
                else incremental
            )
            cues = preserved + completed
        finally:
            # whisper.cpp prints its whole VAD pass in one burst before it
            # starts decoding, so every throttled write above can fall inside a
            # single interval. Flush unconditionally: an interrupt during the
            # much longer decode is exactly what the incremental persist is for.
            self.db.checkpoint("whisper_vad_intervals", vad_intervals)
        for cue in cues:
            if on_event:
                await on_event(TranscriptEvent(cue, final=True))
        return cues

    async def submit_chunk(self, wav: Path, offset: float) -> list[Cue]:
        """Run one full whisper-cli pass over a stream excerpt.

        Each chunk reloads the model, which makes this the slowest chunk
        backend — but a slow chunk only delays the preview: the stream loop
        re-cuts its tail from the last emitted second, so a backlog is
        absorbed rather than accumulated.
        """
        prefix = self.work_dir / "whisper-chunk"

        async def line(stream: str, value: str) -> None:
            self.db.log("whisper-cli-chunk", stream, value)

        await self.runner.run(self._cli_args(wav, prefix), on_line=line)
        incoming = self._load_json(
            prefix, offset=offset, ordinal_offset=len(self.chunk_cues)
        )
        return self._chunk_timeline.add(incoming)

    def _cli_args(self, input_audio: Path, prefix: Path) -> list[str | Path]:
        """Build one whisper-cli invocation for an input file and output prefix."""
        args: list[str | Path] = [
            self.settings.whisper.cli, "-m", self.settings.whisper.model,
            "-f", input_audio, "-mc", "0", "--print-progress",
            "--output-json-full", "--output-file", prefix,
        ]
        args += whisper_language_and_vad_args(self.settings)
        return args

    async def _extract_tail(self, audio: Path, start: float, destination: Path) -> None:
        await extract_tail(
            self.runner, self.settings.commands.ffmpeg, audio, start, destination
        )

    @staticmethod
    def _json_path(prefix: Path) -> Path:
        return prefix.with_suffix(".json")

    def _load_json(self, prefix: Path, *, offset: float = 0.0, ordinal_offset: int = 0) -> list[Cue]:
        # whisper.cpp writes byte-level token text verbatim, so a multi-byte
        # character split across tokens can leave invalid UTF-8 in the file.
        # Every other read of external-tool output here is equally lenient.
        with self._json_path(prefix).open(encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        language = detected_source_language(data)
        if language:
            self.db.checkpoint("detected_source_language", language)
        return parse_json_full(data, offset=offset, ordinal_offset=ordinal_offset)


def _allocate_loopback_port() -> int:
    """Reserve an unused loopback port long enough to select its number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return int(reservation.getsockname()[1])


class WhisperServerTranscriber(Transcriber):
    """whisper-server client for stream chunks and whole-file transcription.

    With ``whisper.server-url`` configured this talks to an already-running
    external instance — nothing is spawned, probed, or terminated. Otherwise
    a private local server is launched for the lifetime of the pass.
    """

    name = "whisper-server"

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        runner: CommandRunner | None = None,
        port: int | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner or CommandRunner()
        self.external_url = settings.whisper.server_url
        if self.external_url:
            parsed = urlparse(self.external_url)
            self.tls = parsed.scheme == "https"
            self.host = parsed.hostname or "127.0.0.1"
            self.port: int | None = parsed.port or (443 if self.tls else 80)
            self.request_path = (parsed.path or "").rstrip("/")
        else:
            self.tls = False
            self.host = "127.0.0.1"
            self.port = port
            self.request_path = f"/yakiflow-{uuid.uuid4().hex}"
        self.process: asyncio.subprocess.Process | None = None
        self._log_tasks: list[asyncio.Task[None]] = []
        self._stderr_hook: Callable[[str], None] | None = None

    async def start_chunks(self) -> None:
        if not self.external_url:
            await self._spawn()

    async def submit_chunk(self, wav: Path, offset: float) -> list[Cue]:
        payload = await asyncio.to_thread(self._post_audio, wav, 180.0)
        language = detected_source_language(payload)
        if language:
            self.db.checkpoint("detected_source_language", language)
        incoming = parse_json_full(
            payload, offset=offset, ordinal_offset=len(self.chunk_cues)
        )
        return self._chunk_timeline.add(incoming)

    async def close_chunks(self) -> None:
        await self._shutdown()

    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]:
        from .media import pcm_audio_duration

        preserved = list(resume_from)
        input_audio = audio
        input_offset = 0.0
        if preserved:
            input_offset = max(cue.end for cue in preserved)
            input_audio = self.work_dir / "whisper-resume.wav"
            await extract_tail(
                self.runner,
                self.settings.commands.ffmpeg,
                audio,
                input_offset,
                input_audio,
            )
        duration = pcm_audio_duration(input_audio) or 60.0
        # whisper-server reports no request progress of its own; large models
        # run near real time, so elapsed time over duration is the estimate,
        # replaced by any real `progress = NN%` line the local server prints.
        ticker = ElapsedProgressTicker(max(30.0, duration), on_progress)
        vad_intervals: list[tuple[float, float]] = []

        def scrape(line: str) -> None:
            interval = parse_vad_segment(line, offset=input_offset)
            if interval is not None:
                append_vad_interval(vad_intervals, interval)
            progress = PROGRESS_RE.search(line)
            if progress is not None:
                ticker.report_real(min(100, int(progress.group("percent"))) / 100)

        spawned = not self.external_url
        if spawned:
            self._stderr_hook = scrape
            await self._spawn()
        try:
            async with ticker:
                try:
                    payload = await asyncio.to_thread(
                        self._post_audio, input_audio, None
                    )
                except OSError as exc:
                    if not spawned:
                        raise RuntimeError(
                            f"whisper-server at {self.external_url} did not "
                            f"accept the request: {exc}"
                        ) from exc
                    # A local server that dropped the connection gets exactly
                    # one restart; a second failure is a real problem.
                    await self._shutdown()
                    await self._spawn()
                    payload = await asyncio.to_thread(
                        self._post_audio, input_audio, None
                    )
        finally:
            if spawned:
                await self._shutdown()
        language = detected_source_language(payload)
        if language:
            self.db.checkpoint("detected_source_language", language)
        if vad_intervals:
            # Not every whisper-server build prints its VAD spans; when this
            # one did not, the checkpoint stays as-is and alignment falls back
            # to its PCM energy analysis.
            stored = [
                (float(start), float(end))
                for start, end in self.db.get_checkpoint(
                    "whisper_vad_intervals", []
                )
            ]
            self.db.checkpoint(
                "whisper_vad_intervals",
                merge_vad_intervals(stored + vad_intervals),
            )
        cues = preserved + parse_json_full(
            payload, offset=input_offset, ordinal_offset=len(preserved)
        )
        for cue in cues:
            if on_event:
                await on_event(TranscriptEvent(cue, final=True))
        return cues

    async def _spawn(self) -> None:
        if self.port is None:
            self.port = _allocate_loopback_port()
        args: list[str] = [
            self.settings.whisper.server, "-m", str(self.settings.whisper.model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--request-path", self.request_path,
        ]
        args += whisper_language_and_vad_args(self.settings)
        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def drain(reader: asyncio.StreamReader, stream: str) -> None:
            # StreamReader.readline() raises once a line exceeds its 64 KiB
            # limit, which would silently kill this drainer and let the
            # undrained pipe block whisper-server. Frame the lines ourselves.
            def record(line: str) -> None:
                self.db.log("whisper-server", stream, line)
                if stream == "stderr" and self._stderr_hook is not None:
                    self._stderr_hook(line)

            pending = bytearray()
            while raw := await reader.read(_PIPE_CHUNK_BYTES):
                pending.extend(raw)
                while (newline := pending.find(b"\n")) >= 0:
                    line = bytes(pending[:newline])
                    del pending[:newline + 1]
                    record(line.decode(errors="replace"))
            if pending:
                record(bytes(pending).decode(errors="replace"))
        self._log_tasks = [
            asyncio.create_task(drain(self.process.stdout, "stdout")),
            asyncio.create_task(drain(self.process.stderr, "stderr")),
        ]
        try:
            await self._wait_until_ready()
        except BaseException:
            await self._shutdown()
            raise

    async def _wait_until_ready(self, timeout: float = 180.0) -> None:
        """Wait until whisper-server's HTTP listener accepts connections."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("whisper-server exited during startup")
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", self.port),
                    timeout=0.5,
                )
            except (OSError, asyncio.TimeoutError):
                if loop.time() >= deadline:
                    raise RuntimeError(
                        f"whisper-server was not ready after {timeout:g} seconds"
                    )
                await asyncio.sleep(0.1)
                continue
            try:
                writer.write(
                    (
                        f"OPTIONS {self.request_path}/inference HTTP/1.1\r\n"
                        "Host: 127.0.0.1\r\nConnection: close\r\n\r\n"
                    ).encode()
                )
                await writer.drain()
                status_line = await asyncio.wait_for(reader.readline(), 0.5)
            finally:
                writer.close()
                await writer.wait_closed()
            if not status_line.startswith(b"HTTP/1.1 200"):
                raise RuntimeError(
                    "whisper-server port is owned by an unexpected listener"
                )
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("whisper-server exited during startup")
            return

    def _post_audio(self, wav_path: Path, timeout: float | None) -> dict[str, Any]:
        """POST one WAV to /inference, streaming it instead of loading it.

        A stream chunk is small, but the authoritative pass posts the whole
        reference audio — hours of PCM — so the request body is generated from
        the file in pieces, with the exact Content-Length computed up front.
        ``timeout=None`` likewise exists for that pass: transcribing a long
        file takes as long as it takes.
        """
        import http.client
        boundary = "----yakiflow-whisper-boundary"
        prologue = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n"
            # Repeat `auto` for each independent chunk; never replace it with
            # the language detected from the first chunk.
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{self.settings.source_language}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"chunk.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        ).encode()
        epilogue = f"\r\n--{boundary}--\r\n".encode()
        size = wav_path.stat().st_size

        def body():
            yield prologue
            with wav_path.open("rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    yield chunk
            yield epilogue

        connection_class = (
            http.client.HTTPSConnection if self.tls else http.client.HTTPConnection
        )
        connection = connection_class(self.host, self.port, timeout=timeout)
        try:
            connection.request(
                "POST",
                f"{self.request_path}/inference",
                body=body(),
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(prologue) + size + len(epilogue)),
                },
            )
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        if response.status >= 400:
            raise RuntimeError(
                f"whisper-server HTTP {response.status}: {payload[:500]!r}"
            )
        return json.loads(payload)

    async def _shutdown(self) -> None:
        if self.process is not None:
            # Share the project's single escalation policy so any helper
            # whisper-server spawned dies with it and stops holding the port.
            await terminate_process(self.process)
            self.process = None
        if self._log_tasks:
            await asyncio.gather(*self._log_tasks, return_exceptions=True)
            self._log_tasks = []


def word_level_backend(backend: str) -> bool:
    return backend in {"elevenlabs", "elevenlabs-stream", "fun-asr", "mai"}


def make_transcriber(
    settings: Settings,
    work_dir: Path,
    db: JobDatabase,
    runner: CommandRunner | None = None,
) -> Transcriber:
    backend = settings.transcription.backend
    if backend == "whisper-cli":
        return WhisperCliTranscriber(settings, work_dir, db, runner)
    if backend == "whisper-server":
        return WhisperServerTranscriber(settings, work_dir, db, runner)
    if backend in {"elevenlabs", "elevenlabs-stream"}:
        from . import elevenlabs

        cls = (
            elevenlabs.ElevenLabsTranscriber
            if backend == "elevenlabs"
            else elevenlabs.ElevenLabsRealtimeTranscriber
        )
        return cls(settings, work_dir, db, runner)
    if backend in {"fun-asr", "fun-asr-stream"}:
        from .fun_asr import FunASRTranscriber, FunASRRealtimeTranscriber
        cls = FunASRTranscriber if backend == "fun-asr" else FunASRRealtimeTranscriber
        return cls(settings, work_dir, db, runner)
    if backend in {"mai", "mai-stream"}:
        from .mai import MAITranscriber, MAIRealtimeTranscriber
        cls = MAITranscriber if backend == "mai" else MAIRealtimeTranscriber
        return cls(settings, work_dir, db, runner)
    raise ValueError(f"unsupported transcription backend: {backend}")
