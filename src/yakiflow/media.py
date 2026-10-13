from __future__ import annotations

import asyncio
import json
import re
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from time import monotonic, time
from typing import Awaitable, Callable, Iterable
from urllib.parse import urlparse

from .config import Settings
from .database import JobDatabase
from .process import CommandRunner


@dataclass(frozen=True, slots=True)
class MediaSource:
    value: str
    is_url: bool

    @classmethod
    def parse(cls, value: str) -> MediaSource:
        parsed = urlparse(value)
        is_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        return cls(
            value if is_url else str(Path(value).expanduser().resolve()),
            is_url,
        )


@dataclass(slots=True)
class MediaArtifact:
    source: MediaSource
    audio_path: Path
    media_path: Path | None = None
    persistent: bool = False


# Receives one ready-to-transcribe excerpt and the original-timeline second it
# starts at.
ChunkCallback = Callable[[Path, float], Awaitable[None]]
ProgressCallback = Callable[[float], Awaitable[None]]
WarningCallback = Callable[[str], Awaitable[None]]
# Receives the still-growing media file of a stream once it holds decodable
# audio, and again whenever the download switches files (rename, remux).
MediaFileCallback = Callable[[Path], Awaitable[None]]
DOWNLOAD_PROGRESS_RE = re.compile(r"\[download\]\s+(?:~\s*)?(?P<percent>\d+(?:\.\d+)?)%")
# How far a decoded tail may fall short of the audio already processed before it
# counts as a broken extraction rather than rounding.
TAIL_SHORTFALL_TOLERANCE = 0.25
STREAM_POLL_SECONDS = 1.0
# Continuous feeding (transcribers with ``continuous_feed``): every pass hands
# over all newly decoded audio once at least this much is new, so the realtime
# transport never sits idle a whole chunk interval. The context absorbs any
# seek imprecision at the decode boundary; the byte-exact trim against the fed
# end in the transcriber keeps the delivered PCM gapless. The chunk-seconds
# and context-seconds stream settings do not apply in this mode.
CONTINUOUS_FEED_SECONDS = 1.0
CONTINUOUS_CONTEXT_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class StreamTail:
    """One decoded PCM window of a growing download, placed on the timeline."""

    path: Path
    start: float
    end: float


def _sizes(paths: Iterable[Path]) -> list[tuple[Path, int]]:
    """Pair each existing non-empty file with its size, tolerating races.

    yt-dlp renames and removes its fragment files while the directory is being
    polled, so a path can vanish between the glob and the ``stat``.
    """
    sized: list[tuple[Path, int]] = []
    for path in paths:
        try:
            info = path.stat()
        except OSError:
            continue
        if info.st_size > 0 and S_ISREG(info.st_mode):
            sized.append((path, info.st_size))
    return sized


def pcm_audio_duration(audio: Path) -> float | None:
    """Return the playable duration of a PCM WAV file, or ``None`` when unusable.

    Streaming snapshots and interrupted extractions can leave a truncated or
    zero-rate header behind, so every caller wants the guarded answer.
    """
    try:
        with wave.open(str(audio), "rb") as source:
            rate = source.getframerate()
            duration = source.getnframes() / rate if rate > 0 else 0.0
    except (EOFError, OSError, wave.Error):
        return None
    return duration if duration > 0 else None


async def probe_video_resolution(
    runner: CommandRunner, ffprobe: str, media_path: Path
) -> tuple[int, int] | None:
    """Return the picture size of ``media_path``, or ``None`` when it has none.

    Cover art is a video stream to ffprobe, so an audio download carrying an
    embedded thumbnail would otherwise report the artwork's size as the
    video's. Raises whatever ffprobe failing raises; every caller has a
    resolution to fall back to.
    """
    result = await runner.run([
        ffprobe, "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=width,height:stream_disposition=attached_pic",
        "-of", "json", media_path,
    ])
    streams = json.loads(result.stdout).get("streams") or []
    for stream in streams:
        if (stream.get("disposition") or {}).get("attached_pic"):
            continue
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width > 0 and height > 0:
            return width, height
    return None


def slice_pcm_wav(source: Path, destination: Path, start: float, end: float) -> None:
    """Copy the ``[start, end)`` second window of a PCM WAV into a new one."""
    with wave.open(str(source), "rb") as reader:
        parameters = reader.getparams()
        first = max(0, round(start * parameters.framerate))
        last = min(parameters.nframes, round(end * parameters.framerate))
        reader.setpos(first)
        frames = reader.readframes(max(0, last - first))
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(parameters.nchannels)
        writer.setsampwidth(parameters.sampwidth)
        writer.setframerate(parameters.framerate)
        writer.writeframes(frames)


async def _drain_stream_chunks(
    tail: StreamTail,
    emitted_duration: float,
    chunk_seconds: float,
    context_seconds: float,
    excerpt: Path,
    on_chunk: ChunkCallback,
) -> float:
    """Cut every complete, not-yet-processed chunk out of one decoded tail.

    Cutting the excerpts here rather than in the callback keeps them free: the
    tail already holds decoded PCM, so a window is a byte range instead of a
    second ffmpeg pass over the growing download.
    """
    while tail.end >= emitted_duration + chunk_seconds:
        start = max(0.0, emitted_duration - context_seconds)
        end = emitted_duration + chunk_seconds
        await asyncio.to_thread(
            slice_pcm_wav, tail.path, excerpt, start - tail.start, end - tail.start
        )
        await on_chunk(excerpt, start)
        emitted_duration = end
    return emitted_duration


async def _drain_stream_available(
    tail: StreamTail,
    emitted_duration: float,
    min_seconds: float,
    context_seconds: float,
    excerpt: Path,
    on_chunk: ChunkCallback,
) -> float:
    """Hand over all newly decoded audio in one excerpt, however much it is.

    The continuous-feed counterpart of ``_drain_stream_chunks``: one slice per
    pass keeps the per-second cost at a single byte-range copy while the
    transcriber cuts its own transport frames out of it.
    """
    if tail.end < emitted_duration + min_seconds:
        return emitted_duration
    start = max(0.0, emitted_duration - context_seconds)
    await asyncio.to_thread(
        slice_pcm_wav, tail.path, excerpt, start - tail.start, tail.end - tail.start
    )
    await on_chunk(excerpt, start)
    return tail.end


class MediaAcquirer:
    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        runner: CommandRunner | None = None,
        on_progress: ProgressCallback | None = None,
        on_warning: WarningCallback | None = None,
        on_media_file: MediaFileCallback | None = None,
    ):
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner or CommandRunner()
        self.on_progress = on_progress
        self.on_warning = on_warning
        self.on_media_file = on_media_file

    async def _progress(self, fraction: float) -> None:
        if self.on_progress:
            await self.on_progress(max(0.0, min(1.0, fraction)))

    async def _download_log(self, stream: str, line: str) -> None:
        self.db.log("yt-dlp", stream, line)
        match = DOWNLOAD_PROGRESS_RE.search(line)
        if match:
            # Reserve the tail for audio extraction/finalization.
            await self._progress(float(match.group("percent")) / 100 * 0.85)

    async def acquire(self, source: MediaSource) -> MediaArtifact:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if source.is_url:
            media_path = await self._download(source)
            persistent = self.settings.download_dir is not None
        else:
            media_path = Path(source.value).expanduser().resolve()
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            persistent = True
            await self._progress(0.2)
        audio_path = self.work_dir / "reference.wav"
        await self.extract_audio(media_path, audio_path)
        await self._progress(0.95)
        artifact = MediaArtifact(source, audio_path, media_path, persistent)
        self.db.add_artifact("reference_audio", audio_path)
        self.db.add_artifact("source_media", media_path)
        return artifact

    async def _download(self, source: MediaSource) -> Path:
        target_dir = (self.settings.download_dir or self.work_dir / "download").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        args = [self.settings.commands.yt_dlp, "--newline", "--no-playlist", "--print", "after_move:filepath", "-P", target_dir]
        if self.settings.download_dir is None:
            args += ["-f", "bestaudio/best", "-o", "source.%(ext)s"]
        else:
            # yt-dlp's normal best format and merge logic preserves the preferred
            # container and falls back to mkv when needed.
            args += ["--merge-output-format", "mkv"]
        args.extend(self.settings.commands.yt_dlp_options)
        args.append(source.value)

        started = time()
        result = await self.runner.run(args, on_line=self._download_log)
        candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        for candidate in reversed(candidates):
            if candidate.is_file():
                return candidate.resolve()
        # A configured download_dir is shared with the user's own downloads, so
        # only files this run wrote may stand in for the printed path. Stat once
        # and tolerate a vanishing entry: the directory is not ours alone.
        files: list[tuple[Path, float]] = []
        for path in target_dir.iterdir():
            if path.name.endswith((".part", ".ytdl")):
                continue
            try:
                info = path.stat()
            except OSError:
                continue
            if S_ISREG(info.st_mode) and info.st_mtime >= started - 1:
                files.append((path, info.st_mtime))
        if not files:
            raise RuntimeError("yt-dlp completed without producing a media file")
        return max(files, key=lambda item: item[1])[0].resolve()

    async def extract_audio(
        self, media_path: Path, output: Path, *, start: float = 0.0
    ) -> None:
        args = [self.settings.commands.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if start > 0:
            # Input seeking: ffmpeg resumes at the keyframe before ``start`` and
            # drops the samples ahead of it, so the window lands sample-accurate
            # even on the partially written containers a download leaves behind.
            args += ["-ss", str(start)]
        args += ["-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output]
        await self.runner.run(args, on_line=lambda stream, line: self.db.log("ffmpeg", stream, line))

    async def acquire_stream(
        self,
        source: MediaSource,
        on_chunk: ChunkCallback,
        *,
        continuous: bool = False,
    ) -> MediaArtifact:
        """Download once, feeding the growing audio to a streaming transcriber.

        Each pass decodes only the audio it has not handed over yet and cuts the
        ready chunks out of it. Extraction is opportunistic: formats that ffmpeg
        cannot read while growing simply become available at a later pass or at
        finalization. With ``continuous`` every pass hands over all newly
        available audio instead of fixed chunks (the stream chunk settings do
        not apply); the transcriber paces its own transport.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if not source.is_url:
            return await self.acquire(source)
        target_dir = (self.settings.download_dir or self.work_dir / "download").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        download_stem = f"source-{uuid.uuid4().hex}"
        template = target_dir / f"{download_stem}.%(ext)s"
        args = [
            self.settings.commands.yt_dlp, "--newline", "--no-playlist",
            "--print", "after_move:filepath", "-o", template,
        ]
        if self.settings.download_dir is None:
            args += ["-f", "bestaudio/best"]
        else:
            args += ["--merge-output-format", "mkv"]
        # A live service can expose a single fragmented MP4 format (``fmp4``).
        # ``--merge-output-format`` is ignored when no merge is needed, and
        # recent yt-dlp versions reject that unusual extension for safety.
        # Remuxing gives yt-dlp a safe, stable final extension in both cases.
        args += ["--remux-video", "mkv"]
        args.extend(self.settings.commands.yt_dlp_options)
        args.append(source.value)
        task = asyncio.create_task(self.runner.run(args, on_line=self._download_log))
        result = None
        interrupted = False
        emitted_duration = 0.0
        reported_media: Path | None = None
        if continuous:
            chunk_seconds: float = CONTINUOUS_FEED_SECONDS
            context_seconds: float = CONTINUOUS_CONTEXT_SECONDS
        else:
            chunk_seconds = self.settings.whisper.stream_chunk_seconds
            context_seconds = self.settings.whisper.stream_context_seconds
        tail_path = self.work_dir / "stream-tail.wav"
        excerpt = self.work_dir / "stream-chunk.wav"
        # A wall-clock deadline, not a poll count: decoding and transcribing a
        # chunk takes far longer than a poll, so counting polls would add a
        # whole chunk interval on top of each chunk instead of absorbing it.
        next_pass = monotonic()
        try:
            while not task.done():
                # Sleep right up to the deadline instead of in fixed poll
                # ticks: a tick landing just before the deadline would stretch
                # every interval by most of a poll. The floor keeps a passed
                # deadline from turning the loop hot, and the poll cap keeps
                # the download's completion checked regularly.
                delay = min(
                    STREAM_POLL_SECONDS, max(0.0, next_pass - monotonic())
                )
                await asyncio.sleep(max(delay, STREAM_POLL_SECONDS / 20))
                if monotonic() < next_pass:
                    continue
                pass_started = monotonic()
                sized = _sizes(target_dir.glob(f"{download_stem}.*"))
                if not sized:
                    next_pass = pass_started + STREAM_POLL_SECONDS
                    continue
                growing = max(sized, key=lambda item: item[1])[0]
                # Decode only what is still unprocessed. Re-decoding the whole
                # download every pass costs time proportional to how long the
                # stream has run, which eventually outgrows the interval itself.
                tail_start = max(0.0, emitted_duration - context_seconds)
                available_end = emitted_duration
                try:
                    await self.extract_audio(growing, tail_path, start=tail_start)
                    duration = await asyncio.to_thread(pcm_audio_duration, tail_path)
                    available_end = tail_start + (duration or 0.0)
                    if not duration and emitted_duration <= 0:
                        # Route this through the handler below rather than
                        # skipping quietly: a download whose audio never becomes
                        # readable produces no chunks at all.
                        raise ValueError(
                            f"{tail_path.name} has no readable PCM duration yet"
                        )
                    # The download already held every second up to
                    # ``emitted_duration``, so a tail that stops short of that
                    # was decoded wrong however the failure is spelled.
                    if available_end + TAIL_SHORTFALL_TOLERANCE < emitted_duration:
                        raise ValueError(
                            f"{tail_path.name} decoded only {available_end:.2f}s "
                            f"of audio already processed to {emitted_duration:.2f}s"
                        )
                    drain = (
                        _drain_stream_available if continuous
                        else _drain_stream_chunks
                    )
                    emitted_duration = await drain(
                        StreamTail(tail_path, tail_start, available_end),
                        emitted_duration,
                        chunk_seconds,
                        context_seconds,
                        excerpt,
                        on_chunk,
                    )
                except Exception as exc:
                    # A tail that cannot be read yet is expected, but a failing
                    # transcriber or Agent would otherwise repeat silently for
                    # the whole download.
                    self.db.log("stream-extract", "stderr", str(exc))
                    if self.on_warning:
                        await self.on_warning(f"stream chunk skipped: {exc}")
                    # Retry a failed pass at poll pace even when a backlog
                    # would otherwise ask for an immediate one.
                    next_pass = pass_started + STREAM_POLL_SECONDS
                    continue
                if self.on_media_file is not None and growing != reported_media:
                    # Only after a successful decode: a player pointed at the
                    # file is then guaranteed to find playable content.
                    reported_media = growing
                    await self.on_media_file(growing)
                # Wait only for the audio the next chunk is still missing,
                # counted from when this pass measured the download: the audio
                # that arrived while the pass ran already counts toward the
                # next chunk. A backlog leaves this in the past and drains on
                # the next poll.
                next_pass = pass_started + max(
                    0.0, emitted_duration + chunk_seconds - available_end
                )
            result = await task
        except asyncio.CancelledError:
            interrupted = True
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if result is not None:
            printed = [
                Path(line.strip()) for line in result.stdout.splitlines()
                if line.strip()
            ]
            files = [
                path for path in printed
                if path.is_file() and path.parent.resolve() == target_dir
                and path.name.startswith(f"{download_stem}.")
            ]
        else:
            files = []
        if not files:
            files = [
                p for p in target_dir.glob(f"{download_stem}.*")
                if p.is_file() and not p.name.endswith((".part", ".ytdl"))
            ]
        sized = _sizes(files) if files else []
        if not sized:
            # A graceful first interrupt may leave a useful partial file.
            sized = _sizes(target_dir.glob(f"{download_stem}.*"))
        if not sized:
            if interrupted:
                # Nothing had been written when the interrupt arrived. Report
                # the interruption so the job is preserved as resumable rather
                # than recorded as a hard failure.
                raise asyncio.CancelledError
            raise RuntimeError("stream ended without usable media")
        media_path = max(sized, key=lambda item: item[1])[0]
        audio_path = self.work_dir / "reference.wav"
        await self.extract_audio(media_path, audio_path)
        # Feed the final remainder after extraction, including audio received
        # while the last preview request or downloader shutdown was running.
        duration = pcm_audio_duration(audio_path) or 0.0
        if duration > emitted_duration:
            start = max(0.0, emitted_duration - context_seconds)
            slice_pcm_wav(audio_path, excerpt, start, duration)
            await on_chunk(excerpt, start)
        self.db.add_artifact("reference_audio", audio_path)
        self.db.add_artifact("source_media", media_path)
        return MediaArtifact(source, audio_path, media_path, self.settings.download_dir is not None)


def default_output_base(artifact: MediaArtifact, settings: Settings, work_dir: Path) -> Path:
    if settings.output_dir:
        return settings.output_dir / (artifact.media_path.stem if artifact.media_path else "subtitle")
    if artifact.persistent and artifact.media_path:
        return artifact.media_path.parent / artifact.media_path.stem
    return Path.cwd() / (artifact.media_path.stem if artifact.media_path else "subtitle")
