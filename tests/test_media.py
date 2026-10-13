import asyncio
import json
import wave
from pathlib import Path

from conftest import make_settings
from yakiflow.database import JobDatabase
from yakiflow import media
from yakiflow.media import (
    MediaAcquirer,
    MediaSource,
    _drain_stream_chunks,
    probe_video_resolution,
)
from yakiflow.process import ProcessResult


class FakeRunner:
    def __init__(self, media: Path):
        self.media = media
        self.calls = []

    async def run(self, args, **kwargs):
        self.calls.append([str(value) for value in args])
        if args[0] == "yt-dlp":
            on_line = kwargs.get("on_line")
            if on_line:
                result = on_line("stderr", "[download]  50.0% of 10.00MiB")
                if asyncio.iscoroutine(result):
                    await result
            self.media.parent.mkdir(parents=True, exist_ok=True)
            self.media.write_bytes(b"media")
            return ProcessResult(tuple(map(str, args)), 0, str(self.media) + "\n", "")
        Path(args[-1]).write_bytes(b"wav")
        return ProcessResult(tuple(map(str, args)), 0, "", "")


def test_media_source_preserves_urls_and_canonicalizes_local_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert MediaSource.parse("clip.mp4") == MediaSource(
        str(tmp_path / "clip.mp4"), False
    )
    assert MediaSource.parse("https://example.test/live") == MediaSource(
        "https://example.test/live", True
    )


def test_persistent_url_downloads_once_then_extracts(tmp_path: Path) -> None:
    media = tmp_path / "downloads" / "video.mkv"
    runner = FakeRunner(media)
    settings = make_settings(
        download_dir=media.parent,
        commands={
            "ffmpeg": "ffmpeg",
            "yt_dlp": "yt-dlp",
            "yt_dlp_options": ["--cookies-from-browser", "chrome"],
        },
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    progress: list[float] = []

    async def on_progress(fraction: float) -> None:
        progress.append(fraction)

    artifact = asyncio.run(
        MediaAcquirer(
            settings, tmp_path / "work", db, runner, on_progress=on_progress
        ).acquire(MediaSource.parse("https://example.test/v"))
    )
    assert artifact.media_path == media.resolve()
    assert sum(call[0] == "yt-dlp" for call in runner.calls) == 1
    yt_dlp_call = next(call for call in runner.calls if call[0] == "yt-dlp")
    assert yt_dlp_call[-3:-1] == ["--cookies-from-browser", "chrome"]
    assert sum(call[0] == "ffmpeg" for call in runner.calls) == 1
    assert progress == [0.425, 0.95]
    db.close()


def _write_tone(path: Path, seconds: float, rate: int = 16000) -> None:
    """Write a mono PCM ramp whose samples encode their own position."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(
            b"".join(
                (index % 30000).to_bytes(2, "little")
                for index in range(round(seconds * rate))
            )
        )


def _first_sample(path: Path) -> int:
    with wave.open(str(path), "rb") as reader:
        return int.from_bytes(reader.readframes(1), "little")


def test_stream_drain_cuts_each_chunk_with_its_leading_context(tmp_path: Path) -> None:
    tail = tmp_path / "tail.wav"
    _write_tone(tail, 47)
    excerpt = tmp_path / "chunk.wav"
    cuts: list[tuple[float, float | None, int]] = []

    async def on_chunk(path: Path, start: float) -> None:
        cuts.append((start, media.pcm_audio_duration(path), _first_sample(path)))

    emitted = asyncio.run(
        _drain_stream_chunks(
            media.StreamTail(tail, 0.0, 47.0), 0.0, 15, 5, excerpt, on_chunk
        )
    )

    # The first chunk has no earlier audio to prepend; the rest carry five
    # seconds of it, and every cut starts where the samples say it should.
    assert cuts == [(0, 15, 0), (10, 20, 10 * 16000 % 30000), (25, 20, 25 * 16000 % 30000)]
    assert emitted == 45


def test_continuous_drain_hands_over_all_available_audio(tmp_path: Path) -> None:
    tail = tmp_path / "tail.wav"
    # A tail decoded from 29s onwards: one second of context before the 30
    # seconds already emitted, plus 17 fresh seconds.
    _write_tone(tail, 18)
    excerpt = tmp_path / "chunk.wav"
    cuts: list[tuple[float, float | None, int]] = []

    async def on_chunk(path: Path, start: float) -> None:
        cuts.append((start, media.pcm_audio_duration(path), _first_sample(path)))

    emitted = asyncio.run(
        media._drain_stream_available(
            media.StreamTail(tail, 29.0, 47.0), 30.0, 1.0, 1.0, excerpt, on_chunk
        )
    )

    # Everything new goes out in a single excerpt with its one-second context.
    assert cuts == [(29.0, 18, 0)]
    assert emitted == 47.0


def test_continuous_drain_waits_for_the_minimum_new_audio(tmp_path: Path) -> None:
    tail = tmp_path / "tail.wav"
    _write_tone(tail, 1.5)
    excerpt = tmp_path / "chunk.wav"
    handed: list[float] = []

    async def on_chunk(path: Path, start: float) -> None:
        handed.append(start)

    emitted = asyncio.run(
        media._drain_stream_available(
            media.StreamTail(tail, 29.0, 30.5), 30.0, 1.0, 1.0, excerpt, on_chunk
        )
    )

    assert handed == []
    assert emitted == 30.0


def test_stream_drain_indexes_chunks_against_the_tail_offset(tmp_path: Path) -> None:
    tail = tmp_path / "tail.wav"
    # A tail decoded from 25s onwards, as the loop requests once it has emitted
    # 30 seconds with five seconds of context.
    _write_tone(tail, 22)
    excerpt = tmp_path / "chunk.wav"
    starts: list[float] = []
    firsts: list[int] = []

    async def on_chunk(path: Path, start: float) -> None:
        starts.append(start)
        firsts.append(_first_sample(path))

    emitted = asyncio.run(
        _drain_stream_chunks(
            media.StreamTail(tail, 25.0, 47.0), 30.0, 15, 5, excerpt, on_chunk
        )
    )

    assert starts == [25]
    # The chunk begins at the very start of the tail, not 25 seconds into it.
    assert firsts == [0]
    assert emitted == 45


def test_stream_decodes_only_the_audio_it_has_not_processed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(media, "STREAM_POLL_SECONDS", 0.01)

    class GrowingRunner:
        """A download that gains 30 seconds of audio between decoding passes."""

        def __init__(self) -> None:
            self.tail_seeks: list[float] = []
            self.downloaded = 14.99
            self.finished = asyncio.Event()

        async def run(self, args, **kwargs):
            values = [str(value) for value in args]
            output = Path(values[-1])
            if values[0] == "yt-dlp":
                template = values[values.index("-o") + 1]
                path = Path(template.replace("%(ext)s", "mkv"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"media")
                await self.finished.wait()
                return ProcessResult(tuple(values), 0, f"{path}\n", "")
            start = float(values[values.index("-ss") + 1]) if "-ss" in values else 0.0
            if output.name == "stream-tail.wav":
                self.tail_seeks.append(start)
                self.downloaded = min(74.99, self.downloaded + 30)
            _write_tone(output, self.downloaded - start, rate=200)
            return ProcessResult(tuple(values), 0, "", "")

    runner = GrowingRunner()
    starts: list[float] = []
    media_files: list[Path] = []

    async def on_chunk(_path: Path, start: float) -> None:
        starts.append(start)
        if len(starts) == 4:
            runner.finished.set()

    async def on_media_file(path: Path) -> None:
        media_files.append(path)

    db = JobDatabase(tmp_path / "db.sqlite3")
    asyncio.run(
        MediaAcquirer(
            make_settings(download_dir=tmp_path / "downloads"),
            tmp_path / "work",
            db,
            runner,
            on_media_file=on_media_file,
        ).acquire_stream(MediaSource.parse("https://example.test/live"), on_chunk)
    )

    assert starts == [0, 10, 25, 40, 55]  # includes the final 14.99 seconds
    # Each pass resumes five seconds before the audio it still owes instead of
    # decoding the whole download over again.
    assert runner.tail_seeks[:2] == [0.0, 25.0]
    assert all(seek > 0 for seek in runner.tail_seeks[1:])
    # The growing download is reported once it decodes, and only once while the
    # path stays the same across passes.
    assert [path.name.startswith("source-") for path in media_files] == [True]
    assert media_files[0].suffix == ".mkv"
    db.close()


def test_stream_download_ignores_files_from_previous_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    stale = download_dir / "source.webm"
    stale.write_bytes(b"old" * 10_000)

    class StreamRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.downloaded: Path | None = None

        async def run(self, args, **kwargs):
            values = [str(value) for value in args]
            self.calls.append(values)
            if values[0] == "yt-dlp":
                template = values[values.index("-o") + 1]
                self.downloaded = Path(template.replace("%(ext)s", "mkv"))
                self.downloaded.write_bytes(b"current")
                return ProcessResult(tuple(values), 0, f"{self.downloaded}\n", "")
            Path(values[-1]).write_bytes(b"wav")
            return ProcessResult(tuple(values), 0, "", "")

    def no_complete_chunk(_audio: Path) -> float:
        return 0.0

    monkeypatch.setattr(media, "pcm_audio_duration", no_complete_chunk)
    runner = StreamRunner()
    db = JobDatabase(tmp_path / "db.sqlite3")
    artifact = asyncio.run(
        MediaAcquirer(
            make_settings(
                download_dir=download_dir,
                commands={
                    "ffmpeg": "ffmpeg",
                    "yt_dlp": "yt-dlp",
                    "yt_dlp_options": ["--cookies-from-browser", "chrome"],
                },
            ),
            tmp_path / "work",
            db,
            runner,
        ).acquire_stream(MediaSource.parse("https://example.test/current"), lambda *_: None)
    )

    assert artifact.media_path == runner.downloaded
    assert artifact.media_path != stale
    yt_dlp_call = next(call for call in runner.calls if call[0] == "yt-dlp")
    assert Path(yt_dlp_call[yt_dlp_call.index("-o") + 1]).name.startswith(
        "source-"
    )
    assert yt_dlp_call[yt_dlp_call.index("--remux-video") + 1] == "mkv"
    assert yt_dlp_call[-3:-1] == ["--cookies-from-browser", "chrome"]
    db.close()


class ProbeRunner:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[list[str]] = []

    async def run(self, args, **kwargs):
        args = [str(value) for value in args]
        self.calls.append(args)
        return ProcessResult(tuple(args), 0, self.payload, "")


def test_probe_video_resolution_reads_the_picture_size(tmp_path: Path) -> None:
    runner = ProbeRunner(json.dumps({
        "streams": [
            {"width": 1080, "height": 1920, "disposition": {"attached_pic": 0}}
        ]
    }))

    resolution = asyncio.run(
        probe_video_resolution(runner, "ffprobe", tmp_path / "clip.mkv")
    )

    assert resolution == (1080, 1920)
    assert runner.calls[0][0] == "ffprobe"
    assert str(tmp_path / "clip.mkv") in runner.calls[0]


def test_probe_video_resolution_ignores_cover_art(tmp_path: Path) -> None:
    # An audio download with an embedded thumbnail has a video stream whose
    # size says nothing about a picture the subtitles would be shown over.
    runner = ProbeRunner(json.dumps({
        "streams": [
            {"width": 640, "height": 640, "disposition": {"attached_pic": 1}}
        ]
    }))

    assert asyncio.run(
        probe_video_resolution(runner, "ffprobe", tmp_path / "audio.mp3")
    ) is None


def test_probe_video_resolution_returns_none_without_a_video_stream(
    tmp_path: Path,
) -> None:
    runner = ProbeRunner(json.dumps({"streams": []}))

    assert asyncio.run(
        probe_video_resolution(runner, "ffprobe", tmp_path / "audio.opus")
    ) is None
