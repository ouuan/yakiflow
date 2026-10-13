"""Durable, frame-exact planning and word merging for file STT providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import unicodedata
import uuid
import wave
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import validate_batch_settings
from .media import slice_pcm_wav
from .models import Cue, TranscriptEvent, Word
from .subtitles import write_text_atomic
from .transcription import EventCallback, ProgressCallback, Transcriber
from .word_processing import cues_from_words


@dataclass(frozen=True)
class BatchChunk:
    index: int
    start_frame: int
    end_frame: int
    rate: int

    @property
    def start(self) -> float:
        return self.start_frame / self.rate

    @property
    def end(self) -> float:
        return self.end_frame / self.rate


@dataclass
class BatchResult:
    words: list[Word]
    language: str | None = None


def plan_chunks(
    *,
    frames: int,
    rate: int,
    bytes_per_frame: int,
    max_seconds: float,
    overlap_seconds: float,
    max_bytes: int,
) -> list[BatchChunk]:
    """Each request, including its overlap, stays strictly below both caps.

    ceil(limit * rate) - 1 is the greatest integral frame count strictly
    below the duration limit. WAV slices have a 44-byte PCM header.
    """
    size_capacity = (max_bytes - 45) // bytes_per_frame
    capacity = math.ceil(min(max_seconds * rate, size_capacity + 1)) - 1
    # Multiplication/division can round on opposite sides of a fractional
    # boundary. Check the actual duration that will be sent as well.
    if capacity / rate >= max_seconds:
        capacity -= 1
    if frames <= 0 or capacity <= 0:
        raise ValueError(
            "batch transcription requires nonempty PCM audio and a limit allowing at least one frame"
        )
    overlap_frames = overlap_seconds * rate
    if overlap_frames >= capacity or round(overlap_frames) >= capacity:
        raise ValueError(
            "chunk-overlap-seconds must be smaller than the available chunk length after duration and file size limits"
        )
    overlap = round(overlap_frames)
    chunks = []
    start = 0
    while start < frames:
        end = min(frames, start + capacity)
        chunks.append(BatchChunk(len(chunks), start, end, rate))
        if end == frames:
            break
        start = end - overlap
    return chunks


def _normalized(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def _matches(left: list[Word], right: list[Word], start: float, end: float):
    """Reciprocal unique nearest matches; ties never collapse simultaneous speech."""
    by_text = defaultdict(list)
    for i, word in enumerate(left):
        token = _normalized(word.text)
        if token and word.end >= start and word.start <= end:
            by_text[token].append((word.start, i))
    for entries in by_text.values():
        entries.sort()
    candidates = defaultdict(list)
    reverse = defaultdict(list)
    tolerance = 0.6
    for j, word in enumerate(right):
        if word.end < start or word.start > end:
            continue
        entries = by_text.get(_normalized(word.text), [])
        lo = bisect_left(entries, (word.start - tolerance, -1))
        hi = bisect_right(entries, (word.start + tolerance, math.inf))
        for _, i in entries[lo:hi]:
            other = left[i]
            if abs(other.end - word.end) <= tolerance:
                distance = abs(other.start - word.start) + abs(other.end - word.end)
                candidates[j].append((distance, i))
                reverse[i].append((distance, j))

    def unique_best(entries):
        entries = sorted(entries)
        if len(entries) > 1 and math.isclose(
            entries[0][0], entries[1][0], abs_tol=1e-6
        ):
            return None
        return entries[0][1]

    best_left = {j: unique_best(entries) for j, entries in candidates.items()}
    best_right = {i: unique_best(entries) for i, entries in reverse.items()}
    return [
        (i, j) for j, i in best_left.items() if i is not None and best_right[i] == j
    ]


def merge_chunk_words(
    chunks: Sequence[BatchChunk], results: Sequence[BatchResult]
) -> list[Word]:
    """Map speakers conservatively, then choose one rendering of each utterance.

    Matching uses the full adjacent responses, before midpoint ownership. A
    union of matches also handles overlaps spanning more than two chunks.
    """
    if len(chunks) != len(results):
        raise ValueError("one batch result is required per planned chunk")
    if len(chunks) == 1:
        return [
            replace(
                w,
                ordinal=i,
                start=w.start + chunks[0].start,
                end=w.end + chunks[0].start,
            )
            for i, w in enumerate(
                sorted(results[0].words, key=lambda w: (w.start, w.end))
            )
        ]
    tracks: list[list[Word]] = []
    parents = {}
    next_speaker = 0

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for k, (chunk, result) in enumerate(zip(chunks, results)):
        words = [
            replace(w, start=w.start + chunk.start, end=w.end + chunk.start)
            for w in result.words
        ]
        for j in range(len(words)):
            parents[k, j] = (k, j)
        pairs = []
        if k and chunk.start < chunks[k - 1].end:
            pairs = _matches(tracks[-1], words, chunk.start, chunks[k - 1].end)
        votes = Counter(
            (words[j].speaker, tracks[-1][i].speaker)
            for i, j in pairs
            if words[j].speaker is not None and tracks[-1][i].speaker is not None
        )
        forward = defaultdict(set)
        reverse = defaultdict(set)
        for local, global_id in votes:
            forward[local].add(global_id)
            reverse[global_id].add(local)
        mapping = {}
        for word in words:
            local = word.speaker
            if local is None or local in mapping:
                continue
            choices = forward[local]
            global_id = next(iter(choices)) if len(choices) == 1 else None
            if (
                global_id is not None
                and votes[local, global_id] >= 2
                and len(reverse[global_id]) == 1
            ):
                mapping[local] = global_id
            else:
                mapping[local] = str(next_speaker)
                next_speaker += 1
        for word in words:
            if word.speaker is not None:
                word.speaker = mapping[word.speaker]
        tracks.append(words)
        for i, j in pairs:
            parents[root((k, j))] = root((k - 1, i))

    groups = defaultdict(list)
    for k, words in enumerate(tracks):
        left_cut = (chunks[k - 1].end + chunks[k].start) / 2 if k else -math.inf
        right_cut = (
            (chunks[k].end + chunks[k + 1].start) / 2
            if k + 1 < len(chunks)
            else math.inf
        )
        for j, word in enumerate(words):
            owned = left_cut <= (word.start + word.end) / 2 < right_cut
            groups[root((k, j))].append((owned, k, j, word))
    merged = []
    for variants in groups.values():
        owned = [entry for entry in variants if entry[0]]
        # Unmatched words follow midpoint ownership. Matched words survive
        # even when timestamp drift puts every rendering on the other side.
        if owned or len(variants) > 1:
            merged.append((owned or variants)[0][3])
    merged.sort(key=lambda word: (word.start, word.end))
    return [replace(word, ordinal=i) for i, word in enumerate(merged)]


def _audio_identity(audio: Path) -> dict:
    with wave.open(str(audio), "rb") as reader:
        params = reader.getparams()
        if params.comptype != "NONE" or params.framerate <= 0 or params.nframes <= 0:
            raise ValueError("batch transcription requires nonempty PCM WAV audio")
        digest = hashlib.sha256()
        size = 0
        while data := reader.readframes(65536):
            digest.update(data)
            size += len(data)
        if size != params.nframes * params.nchannels * params.sampwidth:
            raise ValueError(
                "batch transcription requires a complete PCM WAV recording"
            )
    return {
        "frames": params.nframes,
        "rate": params.framerate,
        "bytes_per_frame": params.nchannels * params.sampwidth,
        "sha256": digest.hexdigest(),
    }


class BatchTranscriber(Transcriber):
    """Providers implement only size preparation, one request and word parsing."""

    max_file_bytes: int

    @property
    def batch_config(self):
        return getattr(self.settings, self.name.replace("-", "_"))

    def request_config(self) -> dict:
        config = asdict(self.batch_config)
        # Secrets can rotate on resume; realtime settings do not affect file requests.
        for key in list(config):
            if key.startswith(("api_key", "realtime_")):
                del config[key]
        return {
            "backend": self.name,
            "source_language": self.settings.source_language,
            **config,
        }

    async def prepare_size_limit(self, audio: Path) -> int:
        return self.max_file_bytes

    def legacy_result_exists(self) -> bool:
        return (self.work_dir / self.name / "result.json").exists()

    def legacy_task_exists(self) -> bool:
        return False

    async def transcribe_one(
        self,
        audio: Path,
        *,
        chunk: BatchChunk,
        cache_dir: Path,
        namespace: str,
        diarize: bool,
        on_progress: ProgressCallback | None,
    ) -> BatchResult:
        raise NotImplementedError

    def finish_words(self, words: list[Word], duration: float) -> None:
        pass

    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]:
        config = self.batch_config
        validate_batch_settings(config, self.name)
        identity = await asyncio.to_thread(_audio_identity, audio)
        duration = identity["frames"] / identity["rate"]
        request = self.request_config()
        plan_key = f"batch_plan_{self.name}"
        plan = self.db.get_checkpoint(plan_key)
        if plan is not None:
            if plan["request"] != request:
                raise ValueError(
                    f"{self.name} saved batch plan uses different request settings; restore the original configuration or start a new job"
                )
            if plan["audio"] != identity:
                raise ValueError(
                    f"{self.name} saved batch plan belongs to different audio"
                )
        else:
            if not hasattr(config, "diarize_max_minutes"):
                diarize = config.diarize
            else:
                from .cloud import effective_diarization

                diarize = effective_diarization(
                    self.db,
                    self.name,
                    audio,
                    config.diarize_max_minutes,
                    over_limit=config.diarize_over_limit,
                    persist=False,
                )
            max_seconds = config.max_minutes * 60
            if diarize and getattr(config, "diarize_over_limit", None) == "split":
                max_seconds = min(max_seconds, config.diarize_max_minutes * 60)
            size_limit = await self.prepare_size_limit(audio)
            chunks = plan_chunks(
                **{k: identity[k] for k in ("frames", "rate", "bytes_per_frame")},
                max_seconds=max_seconds,
                overlap_seconds=config.chunk_overlap_seconds,
                max_bytes=size_limit,
            )
            if len(chunks) > 1 and (
                self.legacy_result_exists() or self.legacy_task_exists()
            ):
                raise ValueError(
                    f"{self.name} saved whole-file result or task cannot be reused with a split plan; restore its settings or start a new job"
                )
            plan = {
                "version": 1,
                "id": uuid.uuid4().hex,
                "request": request,
                "audio": identity,
                "diarize": diarize,
                "max_bytes": size_limit,
                "chunks": [asdict(chunk) for chunk in chunks],
            }
            self.db.checkpoint(plan_key, plan)
        if hasattr(config, "diarize_max_minutes"):
            self.db.checkpoint(
                "diarization_state",
                {
                    "backend": self.name,
                    "enabled": plan["diarize"],
                    "duration": duration,
                },
            )
        chunks = [BatchChunk(**entry) for entry in plan["chunks"]]
        directory = self.work_dir / self.name / "batch" / plan["id"]
        directory.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            directory / "plan.json", json.dumps(plan, ensure_ascii=False, indent=2)
        )
        results = []
        for chunk in chunks:
            cache = directory / f"chunk-{chunk.index:06d}"
            cache.mkdir(parents=True, exist_ok=True)
            normalized = cache / "words.json"

            async def progress(value, index=chunk.index):
                if on_progress:
                    await on_progress((index + min(1, max(0, value))) / len(chunks))

            if normalized.exists():
                data = json.loads(normalized.read_text(encoding="utf-8"))
                result = BatchResult(
                    [Word(**word) for word in data["words"]], data["language"]
                )
            else:
                single = len(chunks) == 1
                excerpt = audio if single else cache / "audio.wav"
                if not single:
                    await asyncio.to_thread(
                        slice_pcm_wav, audio, excerpt, chunk.start, chunk.end
                    )
                if excerpt.stat().st_size >= plan["max_bytes"]:
                    raise ValueError(
                        f"{self.name} chunk exceeds its file size limit ({plan['max_bytes']} bytes)"
                    )
                namespace = (
                    self.name.replace("-", "_")
                    if single
                    else f"{self.name.replace('-', '_')}_{plan['id']}_chunk_{chunk.index:06d}"
                )
                result = await self.transcribe_one(
                    excerpt,
                    chunk=chunk,
                    cache_dir=self.work_dir if single else cache,
                    namespace=namespace,
                    diarize=plan["diarize"],
                    on_progress=progress,
                )
                write_text_atomic(
                    normalized,
                    json.dumps(
                        {
                            "words": [asdict(word) for word in result.words],
                            "language": result.language,
                        },
                        ensure_ascii=False,
                    ),
                )
                if not single:
                    excerpt.unlink(missing_ok=True)
            results.append(result)
            await progress(1)
        words = merge_chunk_words(chunks, results)
        self.finish_words(words, duration)
        language = next(
            (result.language for result in results if result.language), None
        )
        if language:
            self.db.checkpoint("detected_source_language", language)
        self.db.replace_transcript_words(words)
        preview = cues_from_words(
            words,
            max_cue_seconds=self.settings.subtitles.max_cue_seconds,
            max_cue_chars=self.settings.subtitles.max_cue_chars,
        )
        if on_event:
            for cue in preview:
                await on_event(TranscriptEvent(cue, final=True))
        return preview
