"""ElevenLabs speech-to-text support.

This module hosts everything specific to the ElevenLabs backends: API-key
resolution, word-stream normalization, the mechanical preview segmentation,
word-batch splitting for the draft agent, and the transcribers themselves.

The authoritative output of these backends is word-level. Final cues are cut
by the draft agent (see ``TranslationPipeline.segment_and_translate``); the
mechanical ``cues_from_words`` segmentation exists only so previews have
something to show before the agent has spoken.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from itertools import islice
from math import ceil
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

from .batch_transcription import BatchResult, BatchTranscriber
from .config import Settings
from .database import JobDatabase
from .models import Cue, TranscriptEvent, Word, cue_text_weight, preview_cue_id
from .process import CommandRunner
from .transcription import (
    ElapsedProgressTicker,
    EventCallback,
    ProgressCallback,
    Transcriber,
    normalize_source_language,
)


KEYRING_SERVICE = "yakiflow"
KEYRING_ENTRY = "elevenlabs"
API_KEY_ENV_VAR = "ELEVENLABS_API_KEY"

# A cut between two cues (mechanical or agent-made) is natural where the whole
# conversation is silent at least this long. Shared by the preview segmenter,
# the batch splitter, and the junction-repair heuristic so all three agree on
# what a legitimate boundary looks like.
PAUSE_SPLIT_SECONDS = 0.8
SENTENCE_END_CHARS = ".?!。？！…"
_CLOSING_QUOTES = "\"'”’»›』」）)]"

# The documented ceilings are 10 hours of audio in a file of at most 3 GB.
# Duration is the binding one here: 10 hours of 16 kHz mono PCM is ~1.2 GB.
MAX_UPLOAD_BYTES = 3_000_000_000

_NO_KEY_ERROR = (
    "no ElevenLabs API key is configured; provide one via "
    "-c elevenlabs.api-key=…, the ELEVENLABS_API_KEY environment variable, "
    "elevenlabs.api-key / api-key-file / api-key-command in a configuration "
    "file, or 'yakiflow secret set elevenlabs' (needs yakiflow[keyring])"
)


def _configured_slot(settings: Settings) -> str | None:
    """Name which spelling of the config API-key slot is set, if any."""
    elevenlabs = settings.elevenlabs
    if elevenlabs.api_key is not None:
        return "api_key"
    if elevenlabs.api_key_file is not None:
        return "api_key_file"
    if elevenlabs.api_key_command is not None:
        return "api_key_command"
    return None


def _slot_key(settings: Settings) -> str | None:
    """Resolve the configured slot to its key.

    A configured source that fails to produce a key is an error, never a
    silent fall-through to a lower-priority source: the key that would be used
    instead is not the one the user pointed at.
    """
    elevenlabs = settings.elevenlabs
    slot = _configured_slot(settings)
    if slot is None:
        return None
    if slot == "api_key":
        key = (elevenlabs.api_key or "").strip()
        if not key:
            raise ValueError("elevenlabs.api-key is empty")
        return key
    if slot == "api_key_file":
        assert elevenlabs.api_key_file is not None
        try:
            text = elevenlabs.api_key_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"cannot read elevenlabs.api-key-file {elevenlabs.api_key_file}: {exc}"
            ) from exc
        key = text.rstrip()
        if not key:
            raise ValueError(
                f"elevenlabs.api-key-file {elevenlabs.api_key_file} is empty"
            )
        return key
    assert elevenlabs.api_key_command is not None
    result = subprocess.run(
        elevenlabs.api_key_command,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(
            "elevenlabs.api-key-command exited with status "
            f"{result.returncode}{suffix}"
        )
    key = result.stdout.strip()
    if not key:
        raise ValueError("elevenlabs.api-key-command printed no API key")
    return key


_keyring_cached_key: str | None = None


def _keyring_key() -> str | None:
    """Fetch the keyring key at most once per process.

    A key that resolved when the job started must keep resolving for the rest
    of the run: a Secret Service hiccup hours in would otherwise resurface as
    "no API key is configured" and fail work the key already authorized.
    """
    global _keyring_cached_key
    if _keyring_cached_key is not None:
        return _keyring_cached_key
    try:
        import keyring
    except ImportError:
        return None
    try:
        key = keyring.get_password(KEYRING_SERVICE, KEYRING_ENTRY) or None
    except Exception:
        # An unavailable Secret Service backend means this layer has no key,
        # the same as when the optional dependency is not installed at all.
        key = None
    _keyring_cached_key = key
    return key


def elevenlabs_api_key(settings: Settings) -> str:
    """Resolve the API key at time of use; never stored in ``Settings``.

    Priority, highest first: the config slot when the command line set it, the
    environment variable, the config slot from configuration files, the OS
    keyring.
    """
    from_cli = settings.elevenlabs.api_key_from_cli
    if from_cli:
        key = _slot_key(settings)
        if key:
            return key
    env = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if env:
        return env
    if not from_cli:
        key = _slot_key(settings)
        if key:
            return key
    key = _keyring_key()
    if key:
        return key
    raise ValueError(_NO_KEY_ERROR)


def elevenlabs_api_key_source(settings: Settings) -> str:
    """Name where the key would come from, without touching the key itself."""
    slot_labels = {
        "api_key": "config",
        "api_key_file": "file",
        "api_key_command": "command",
    }
    slot = _configured_slot(settings)
    if slot is not None and settings.elevenlabs.api_key_from_cli:
        return slot_labels[slot]
    if os.environ.get(API_KEY_ENV_VAR, "").strip():
        return "env"
    if slot is not None:
        return slot_labels[slot]
    if _keyring_key():
        return "keyring"
    return "not set"


def normalize_speaker(speaker_id: object) -> str | None:
    """Reduce the API's ``speaker_N`` labels to their bare number.

    The numbering carries no meaning beyond identity, so the shortest stable
    spelling is used everywhere: database, ``Cue.speaker``, the ASS Name
    field, and the draft agent's input.
    """
    if speaker_id is None:
        return None
    label = str(speaker_id).strip()
    if not label:
        return None
    if label.startswith("speaker_"):
        return label[len("speaker_"):]
    return label


def _get(item: Any, key: str, default: Any = None) -> Any:
    """Read one field off an SDK typed object or a plain response dict."""
    if isinstance(item, dict):
        return item.get(key, default)
    value = getattr(item, key, default)
    return default if value is None else value


def _jsonable(value: Any) -> Any:
    """Best plain-data rendering of an SDK typed object, for ``json.dumps``."""
    for method in ("model_dump", "dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                break
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def save_raw_response(
    work_dir: Path, name: str, payload: Any, *, append: bool = False
) -> None:
    """Keep a complete API response on disk under ``stt-responses/``.

    Transcription consumes only a few fields of each response and discards
    the rest, so the verbatim payload is what settles questions like "did the
    API really return those zero-width words". Best-effort by design: a
    serialization or filesystem hiccup must never disturb transcription.
    """
    import json

    try:
        directory = work_dir / "stt-responses"
        directory.mkdir(parents=True, exist_ok=True)
        if append:
            line = json.dumps(payload, ensure_ascii=False, default=_jsonable)
            with (directory / name).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        else:
            text = json.dumps(
                payload, ensure_ascii=False, indent=2, default=_jsonable
            )
            (directory / name).write_text(text + "\n", encoding="utf-8")
    except Exception:
        pass


def words_from_response(items: Sequence[Any], *, offset: float = 0.0) -> list[Word]:
    """Normalize an API word list: fold spacing, drop audio events.

    Spacing entries become part of the preceding word's text, so joining the
    word texts reconstructs the transcript verbatim; a leading spacing with no
    word before it is dropped.
    """
    words: list[Word] = []
    for item in items:
        kind = str(_get(item, "type", "word"))
        if kind == "audio_event":
            continue
        text = str(_get(item, "text", ""))
        if kind == "spacing":
            if words:
                words[-1].text += text
            continue
        start = float(_get(item, "start", 0.0)) + offset
        end = max(start, float(_get(item, "end", start)) + offset)
        logprob = _get(item, "logprob")
        words.append(Word(
            ordinal=len(words),
            start=start,
            end=end,
            text=text,
            speaker=normalize_speaker(_get(item, "speaker_id")),
            logprob=float(logprob) if logprob is not None else None,
        ))
    return words


from .word_processing import (  # re-export the original public helpers
    SQUEEZED_WORD_SECONDS, MAX_REPAIRED_RUN_SECONDS, WordBatch, cues_from_words, dispatchable_word_count, qualified_silences,
    repair_squeezed_words, split_word_batches,
)


def _error_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return int(status) if isinstance(status, int) else None


# --- batch request wall-clock estimate ---

# ElevenLabs splits a batch request into internally parallel segments:
# ``min(4, ceil(duration / 480))``, so wall clock stops shrinking once the
# audio is long enough to saturate all four.
_SEGMENT_SECONDS = 480.0
_MAX_SEGMENTS = 4

# Published measurements put Scribe v2 near 55x real time on a ten-minute
# file, which the split above runs as two segments; that is ~27x per segment,
# rounded down here for headroom against a busy queue.
_SEGMENT_SPEED = 25.0

# The upload is the other half of the wall clock: the audio handed to this
# backend is 16 kHz mono PCM, so an hour of it is ~115 MB. Assume a modest
# uplink rather than the speed of a well-connected server.
_UPLOAD_BYTES_PER_SECOND = 2_000_000.0

# Request setup plus the queue wait before a segment starts.
_REQUEST_OVERHEAD_SECONDS = 2.0


def estimate_convert_seconds(duration: float | None, size_bytes: int) -> float:
    """Guess how long one whole-file request takes, for synthetic progress.

    The endpoint reports nothing until it answers, so the progress bar rides
    this estimate; it only needs the right order of magnitude.
    """
    audio_seconds = duration if duration else 600.0
    segments = min(
        _MAX_SEGMENTS, max(1, ceil(audio_seconds / _SEGMENT_SECONDS))
    )
    transcribe_seconds = audio_seconds / (_SEGMENT_SPEED * segments)
    upload_seconds = max(0, size_bytes) / _UPLOAD_BYTES_PER_SECOND
    return max(
        15.0, _REQUEST_OVERHEAD_SECONDS + upload_seconds + transcribe_seconds
    )


class ElevenLabsTranscriber(BatchTranscriber):
    """Batch Scribe transcription through the official SDK."""

    name = "elevenlabs"
    max_file_bytes = MAX_UPLOAD_BYTES
    _RETRY_ATTEMPTS = 3

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        runner: CommandRunner | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner
        self._client_factory = client_factory
        self._client: Any = None

    def _make_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                try:
                    from elevenlabs.client import AsyncElevenLabs
                except ImportError as exc:
                    raise RuntimeError(
                        "the elevenlabs SDK is not installed; install "
                        "yakiflow[elevenlabs] to use the ElevenLabs backends"
                    ) from exc
                self._client = AsyncElevenLabs(
                    api_key=elevenlabs_api_key(self.settings)
                )
        return self._client

    def _convert_kwargs(self, *, diarize: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model_id": self.settings.elevenlabs.model,
            "diarize": diarize,
            "tag_audio_events": False,
        }
        language = normalize_source_language(self.settings.source_language)
        if language is not None:
            kwargs["language_code"] = self.settings.source_language
        if diarize and self.settings.elevenlabs.num_speakers is not None:
            kwargs["num_speakers"] = self.settings.elevenlabs.num_speakers
        if diarize and self.settings.elevenlabs.use_speaker_library:
            kwargs["use_speaker_library"] = True
        return kwargs

    async def _convert(self, audio: Path, *, diarize: bool) -> Any:
        client = self._make_client()
        delay = 5.0
        for attempt in range(1, self._RETRY_ATTEMPTS + 1):
            try:
                with audio.open("rb") as fh:
                    return await client.speech_to_text.convert(
                        file=fh, **self._convert_kwargs(diarize=diarize)
                    )
            except Exception as exc:
                status = _error_status(exc)
                if status in {401, 403}:
                    raise RuntimeError(
                        "ElevenLabs rejected the API key (HTTP "
                        f"{status}); check the configured key"
                    ) from exc
                if status == 413:
                    raise RuntimeError(
                        "ElevenLabs rejected the upload as too large "
                        "(HTTP 413)"
                    ) from exc
                retryable = status == 429 or (status is not None and status >= 500)
                if not retryable or attempt == self._RETRY_ATTEMPTS:
                    if status is not None:
                        body = getattr(exc, "body", None)
                        summary = f": {str(body)[:300]}" if body else ""
                        raise RuntimeError(
                            f"ElevenLabs request failed with HTTP {status}{summary}"
                        ) from exc
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable ElevenLabs retry state")

    def legacy_result_exists(self) -> bool:
        return (self.work_dir / "stt-responses" / "transcribe.json").exists()

    async def transcribe_one(
        self, audio, *, chunk, cache_dir, namespace, diarize, on_progress
    ):
        import json

        raw = cache_dir / "stt-responses" / "transcribe.json"
        if raw.exists():
            response = json.loads(raw.read_text(encoding="utf-8"))
        else:
            estimate = estimate_convert_seconds(
                chunk.end - chunk.start, audio.stat().st_size
            )
            async with ElapsedProgressTicker(estimate, on_progress):
                response = await self._convert(audio, diarize=diarize)
            save_raw_response(cache_dir, "transcribe.json", response)
        return BatchResult(
            words_from_response(_get(response, "words", []) or []),
            normalize_source_language(_get(response, "language_code")),
        )

    def finish_words(self, words, duration):
        repair_squeezed_words(words, duration=duration)

    async def submit_chunk(self, wav: Path, offset: float) -> list[Cue]:
        # Diarization is forced off for chunks: speaker numbering is not
        # stable between independent requests, and the preview is provisional
        # anyway.
        response = await self._convert(wav, diarize=False)
        save_raw_response(self.work_dir, f"chunk-{offset:.2f}s.json", response)
        words = words_from_response(
            _get(response, "words", []) or [], offset=offset
        )
        incoming = cues_from_words(
            words,
            max_cue_seconds=self.settings.subtitles.max_cue_seconds,
            max_cue_chars=self.settings.subtitles.max_cue_chars,
        )
        return self._chunk_timeline.add(incoming)


# --- realtime (Scribe v2 Realtime WebSocket) ---

# How far before the last durable word a broken realtime session resumes, and
# how far apart two hearings of the same word may start and still be one word.
RECONNECT_OVERLAP_SECONDS = 5.0
_WORD_MERGE_TOLERANCE = 0.6
_MAX_RECONNECTS = 3
# How much fed-but-uncommitted audio may be in flight before the feeder waits
# for the server to catch up.
_FLOW_CONTROL_WINDOW_SECONDS = 30.0
_EVENT_QUIET_TIMEOUT_SECONDS = 30.0
# The server wants realtime frames of 0.1–1 s; one oversized message risks the
# chunk-size cap, and long silent stretches between sends starve the session.
_REALTIME_FRAME_SECONDS = 1.0
# The preview keeps this much sent-but-uncommitted PCM so a server-side session
# close can re-feed it to the next session instead of losing the words.
_PREVIEW_BUFFER_MAX_SECONDS = 60.0
# Background music can keep the server's VAD from ever committing; after this
# much uncommitted audio the preview asks for a manual commit so subtitles keep
# flowing and the re-feed buffer stays bounded.
_FALLBACK_COMMIT_SECONDS = 25.0


def merge_streamed_words(
    existing: Sequence[Word],
    incoming: Sequence[Word],
    tolerance: float = _WORD_MERGE_TOLERANCE,
) -> list[Word]:
    """Merge words from a re-fed overlap without duplicating any of them.

    A reconnect re-feeds a few seconds the previous session already
    transcribed; the same speech renders as words with nearly the same start
    and the same text, and those duplicates are dropped. The merge is
    strictly append-only: a differing hearing that would land *between*
    already-durable words loses to them, because word ordinals are the
    currency of dispatched agent batches and finished cue coverage — a
    renumbering would silently corrupt both.
    """
    result = list(existing)
    last_start = result[-1].start if result else float("-inf")
    for word in incoming:
        normalized = word.text.strip().casefold()
        duplicate = any(
            abs(candidate.start - word.start) <= tolerance
            and candidate.text.strip().casefold() == normalized
            for candidate in result[-80:]
        )
        if duplicate or word.start < last_start:
            continue
        word.ordinal = len(result)
        result.append(word)
        last_start = word.start
    return result


class _SessionEnded(Exception):
    """The realtime session ended before the audio did (limit or disconnect)."""


# Event kinds that mean the session is over but a fresh one may continue the
# work, as opposed to the fatal ``error`` kind (auth, quota, terms, …).
_SESSION_END_KINDS = {
    "session_limit", "closed", "insufficient_audio_activity", "commit_throttled",
}


def _describe_event(kind: str, data: Any) -> str:
    if data is None:
        return kind
    return f"{kind}: {str(data)[:300]}"


class _RealtimeSession:
    """Thin transport adapter over the SDK's realtime connection.

    Owns only open/send/commit/close and a queue of received events; every
    piece of timeline math lives in the transcriber, so an SDK interface
    change touches exactly this class. Should the pinned SDK lose its
    realtime client, an equivalent hand-written ``websockets`` client can
    stand in behind this same seam.
    """

    _ERROR_EVENTS = (
        "error", "auth_error", "quota_exceeded", "transcriber_error",
        "input_error", "invalid_request", "queue_overflow",
        "resource_exhausted", "chunk_size_exceeded", "rate_limited",
        "unaccepted_terms",
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self._connection: Any = None
        self._events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._first_send = True
        self._previous_text = ""

    async def open(self, previous_text: str = "") -> None:
        try:
            from elevenlabs.client import AsyncElevenLabs
            from elevenlabs.realtime import AudioFormat, CommitStrategy
        except ImportError as exc:
            raise RuntimeError(
                "the elevenlabs SDK is not installed; install "
                "yakiflow[elevenlabs] to use the ElevenLabs backends"
            ) from exc
        client = AsyncElevenLabs(api_key=elevenlabs_api_key(self.settings))
        options: dict[str, Any] = {
            "model_id": self.settings.elevenlabs.realtime_model,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": 16000,
            "commit_strategy": CommitStrategy.VAD,
            "include_timestamps": True,
        }
        if normalize_source_language(self.settings.source_language) is not None:
            options["language_code"] = self.settings.source_language
        self._connection = await client.speech_to_text.realtime.connect(options)
        self._previous_text = previous_text
        self._first_send = True

        def enqueue(kind: str):
            def handler(*args: Any) -> None:
                self._events.put_nowait((kind, args[0] if args else None))
            return handler

        self._connection.on(
            "committed_transcript_with_timestamps", enqueue("words")
        )
        self._connection.on(
            "session_time_limit_exceeded", enqueue("session_limit")
        )
        self._connection.on("close", enqueue("closed"))
        # These end the session but are the server's fault or pacing feedback,
        # not a broken configuration: a fresh session may simply continue.
        self._connection.on(
            "insufficient_audio_activity", enqueue("insufficient_audio_activity")
        )
        self._connection.on("commit_throttled", enqueue("commit_throttled"))
        for event in self._ERROR_EVENTS:
            self._connection.on(event, enqueue("error"))

    async def send_pcm(self, pcm: bytes) -> None:
        import base64

        from websockets.exceptions import ConnectionClosed

        payload: dict[str, Any] = {
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
        }
        if self._first_send and self._previous_text:
            payload["previous_text"] = self._previous_text
        self._first_send = False
        try:
            await self._connection.send(payload)
        except ConnectionClosed as exc:
            raise _SessionEnded(f"connection closed: {exc}") from exc

    async def commit(self) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            await self._connection.commit()
        except ConnectionClosed as exc:
            raise _SessionEnded(f"connection closed: {exc}") from exc

    def pending_events(self) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except asyncio.QueueEmpty:
                return events

    async def next_event(self, timeout: float) -> tuple[str, Any] | None:
        try:
            return await asyncio.wait_for(self._events.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                pass
            self._connection = None


class ElevenLabsRealtimeTranscriber(Transcriber):
    """Realtime Scribe transcription over one (or few) WebSocket sessions.

    The only remote backend with true incremental persistence: committed
    words land in ``transcript_words`` as they arrive, so an interrupted
    authoritative pass resumes from the last durable word instead of zero.
    """

    name = "elevenlabs-stream"

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        runner: CommandRunner | None = None,
        session_factory: Callable[[], Any] | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner
        self._session_factory = session_factory or (
            lambda: _RealtimeSession(settings)
        )
        self._session: Any = None
        self._words: list[Word] = []
        self._persist = False
        self._session_base = 0.0
        self._fed_end: float | None = None
        self._emitted_preview_ids: set[str] = set()
        # Preview re-feed buffer: PCM already ingested from chunks but not yet
        # safely committed by the server, so a closed session can be replayed
        # into its replacement instead of losing those words.
        self._buffer = bytearray()
        self._buffer_start = 0.0
        self._pending_index = 0
        self._commit_floor = 0.0
        self._frame_rate = 16000
        self._bytes_per_frame = 2

    # --- shared event handling ---

    def _log_raw_event(self, kind: str, data: Any) -> None:
        """Append one received event to the raw session log.

        The full session log, not just words: close/error events are exactly
        what a post-mortem of a broken stream needs. Times in word events are
        session-relative; session_base recovers the media timeline across
        reconnects.
        """
        save_raw_response(
            self.work_dir,
            "realtime-events.jsonl",
            {"kind": kind, "session_base": self._session_base, "event": data},
            append=True,
        )

    def _handle_events(self, events: Sequence[tuple[str, Any]]) -> bool:
        """Fold received events into the word stream; True when words changed."""
        changed = False
        for kind, data in events:
            self._log_raw_event(kind, data)
            if kind == "words":
                incoming = words_from_response(
                    _get(data, "words", []) or [], offset=self._session_base
                )
                if not incoming:
                    continue
                merged = merge_streamed_words(self._words, incoming)
                if len(merged) != len(self._words):
                    fresh = merged[len(self._words):]
                    self._words = merged
                    changed = True
                    if self._persist:
                        # Append-only merges make the delta exactly the tail.
                        self.db.append_transcript_words(fresh)
                    durable_end = self._words[-1].end
                    self._commit_floor = max(self._commit_floor, durable_end)
                    self._trim_buffer(durable_end - RECONNECT_OVERLAP_SECONDS)
            elif kind in _SESSION_END_KINDS:
                raise _SessionEnded(_describe_event(kind, data))
            elif kind == "error":
                raise RuntimeError(f"ElevenLabs realtime error: {data}")
        return changed

    def _preview(self) -> list[Cue]:
        return cues_from_words(
            self._words,
            max_cue_seconds=self.settings.subtitles.max_cue_seconds,
            max_cue_chars=self.settings.subtitles.max_cue_chars,
        )

    def _trim_buffer(self, cut: float) -> None:
        """Drop buffered PCM before ``cut`` seconds of the media timeline."""
        if not self._buffer:
            return
        frames = round((cut - self._buffer_start) * self._frame_rate)
        drop = min(frames * self._bytes_per_frame, len(self._buffer))
        if drop <= 0:
            return
        del self._buffer[:drop]
        self._pending_index = max(0, self._pending_index - drop)
        self._buffer_start += drop / (self._frame_rate * self._bytes_per_frame)

    async def _absorb_session_end(self, reason: str) -> None:
        """Close the ended session, but read what the server said first.

        The close often trails an explicit event naming the cause; folding the
        drained events into the reason (or, for a fatal error, into the
        RuntimeError the caller sees) keeps it from being lost as a bare
        disconnect the way it would be otherwise.
        """
        session, self._session = self._session, None
        events = session.pending_events()
        await session.close()
        for kind, data in events:
            if kind in _SESSION_END_KINDS:
                self._log_raw_event(kind, data)
        details = [reason] + [
            _describe_event(kind, data)
            for kind, data in events
            if kind in _SESSION_END_KINDS and data is not None
        ]
        message = f"ElevenLabs realtime session ended ({'; '.join(details)})"
        self.db.log("elevenlabs-stream", "stderr", message)
        await self._warn(message)
        self._handle_events(
            [event for event in events if event[0] not in _SESSION_END_KINDS]
        )

    # --- chunk previews: buffered continuous feed in realtime frames ---

    continuous_feed = True

    async def start_chunks(self) -> None:
        self._persist = True
        # The server closes a session that sits idle while the download spins
        # up, so the socket opens lazily with the first chunk instead.
        return

    async def submit_chunk(self, wav: Path, offset: float) -> list[Cue]:
        import wave

        with wave.open(str(wav), "rb") as reader:
            parameters = reader.getparams()
            frames = reader.readframes(parameters.nframes)
        self._frame_rate = parameters.framerate
        self._bytes_per_frame = parameters.sampwidth * parameters.nchannels
        duration = parameters.nframes / parameters.framerate
        if self._fed_end is None:
            self._fed_end = offset
            self._buffer_start = offset
            self._commit_floor = offset
        # The excerpt re-carries a short overlap already ingested; only the
        # samples past the fed end are new.
        skip_frames = min(
            parameters.nframes,
            max(0, round((self._fed_end - offset) * parameters.framerate)),
        )
        fresh = frames[skip_frames * self._bytes_per_frame:]
        if fresh:
            if not self._buffer:
                self._buffer_start = self._fed_end
                self._pending_index = 0
            self._buffer.extend(fresh)
            self._fed_end = max(self._fed_end, offset + duration)
            # A server that stops committing would grow the buffer without
            # bound; beyond the cap the oldest audio is surrendered to the
            # authoritative pass, exactly like before the buffer existed.
            self._trim_buffer(self._fed_end - _PREVIEW_BUFFER_MAX_SECONDS)
        try:
            await self._pump()
        except _SessionEnded as exc:
            # The next chunk opens a fresh session and re-feeds the buffered
            # audio; the authoritative pass owns durable delivery.
            await self._absorb_session_end(str(exc))
        preview = self._preview()
        self._chunk_timeline.cues = preview
        fresh_cues = [
            cue for cue in preview if cue.id not in self._emitted_preview_ids
        ]
        self._emitted_preview_ids.update(cue.id for cue in fresh_cues)
        return fresh_cues

    async def _pump(self) -> None:
        """Deliver un-sent buffered PCM to the session in realtime frames."""
        if self._session is None:
            if not self._buffer:
                # Nothing to say yet; an idle session would only be closed as
                # starved by the server.
                return
            self._session = self._session_factory()
            self._session_base = self._buffer_start
            self._pending_index = 0
            self._commit_floor = self._buffer_start
            previous_text = "".join(
                word.text for word in self._words[-60:]
            ).strip()[-500:]
            await self._session.open(previous_text)
        frame_bytes = round(
            self._frame_rate * _REALTIME_FRAME_SECONDS
        ) * self._bytes_per_frame
        while self._pending_index < len(self._buffer):
            frame = bytes(
                self._buffer[self._pending_index:self._pending_index + frame_bytes]
            )
            await self._session.send_pcm(frame)
            self._pending_index += len(frame)
        self._handle_events(self._session.pending_events())
        sent_end = self._buffer_start + self._pending_index / (
            self._frame_rate * self._bytes_per_frame
        )
        if sent_end - self._commit_floor >= _FALLBACK_COMMIT_SECONDS:
            await self._session.commit()
            self._commit_floor = sent_end

    async def close_chunks(self) -> None:
        if self._session is None:
            return
        try:
            await self._session.commit()
            # Give the final commit a moment to come back as words so the last
            # preview cues are not cut off mid-sentence.
            while True:
                event = await self._session.next_event(2.0)
                if event is None:
                    break
                self._handle_events([event])
        except Exception:
            pass
        await self._session.close()
        self._session = None
        self._chunk_timeline.cues = self._preview()

    # --- the authoritative pass ---

    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]:
        from .media import pcm_audio_duration

        self._persist = True
        duration = pcm_audio_duration(audio) or 0.0
        stored = self.db.list_transcript_words()
        if stored:
            self._words = stored
        reconnects = 0
        while True:
            if self._words:
                start_at = max(
                    0.0, self._words[-1].end - RECONNECT_OVERLAP_SECONDS
                )
                previous_text = "".join(
                    word.text for word in self._words[-60:]
                ).strip()[-500:]
            else:
                start_at = 0.0
                previous_text = ""
            self._session = self._session_factory()
            self._session_base = start_at
            await self._session.open(previous_text)
            try:
                await self._feed(audio, start_at, duration, on_event, on_progress)
                break
            except _SessionEnded as exc:
                await self._absorb_session_end(str(exc))
                reconnects += 1
                if reconnects > _MAX_RECONNECTS:
                    raise RuntimeError(
                        "the ElevenLabs realtime session ended "
                        f"{reconnects} times before the audio did; the words "
                        "delivered so far are preserved, resume to continue"
                    ) from None
            finally:
                if self._session is not None:
                    await self._session.close()
                    self._session = None
        preview = self._preview()
        for cue in preview:
            if on_event:
                await on_event(TranscriptEvent(cue, final=True))
        return preview

    async def _feed(
        self,
        audio: Path,
        start_at: float,
        duration: float,
        on_event: EventCallback | None,
        on_progress: ProgressCallback | None,
    ) -> None:
        import wave

        async def note_progress(fed: float) -> None:
            if on_progress and duration:
                await on_progress(min(0.95, fed / duration))

        async def emit_new_preview() -> None:
            if on_event is None:
                return
            preview = self._preview()
            fresh = [
                cue
                for cue in preview
                if cue.id not in self._emitted_preview_ids
            ]
            self._emitted_preview_ids.update(cue.id for cue in fresh)
            for cue in fresh:
                await on_event(TranscriptEvent(cue, final=False))

        with wave.open(str(audio), "rb") as reader:
            parameters = reader.getparams()
            bytes_per_second = (
                parameters.framerate * parameters.sampwidth * parameters.nchannels
            )
            reader.setpos(
                min(parameters.nframes, round(start_at * parameters.framerate))
            )
            fed = start_at
            acked_floor = start_at
            while True:
                frames = reader.readframes(parameters.framerate)
                if not frames:
                    break
                await self._session.send_pcm(frames)
                fed += len(frames) / bytes_per_second
                self._fed_end = fed
                if self._handle_events(self._session.pending_events()):
                    await emit_new_preview()
                # Flow control: never run more than the window ahead of what
                # the server has demonstrably consumed. Committed words are
                # the only explicit acknowledgement, but silence commits
                # nothing, so waited wall clock also counts as progress: a
                # realtime server consumes at least in real time.
                while (
                    fed
                    - max(
                        self._words[-1].end if self._words else start_at,
                        acked_floor,
                    )
                    > _FLOW_CONTROL_WINDOW_SECONDS
                ):
                    event = await self._session.next_event(10.0)
                    if event is None:
                        acked_floor += 10.0
                        continue
                    if self._handle_events([event]):
                        await emit_new_preview()
                await note_progress(fed)
        await self._session.commit()
        while True:
            event = await self._session.next_event(_EVENT_QUIET_TIMEOUT_SECONDS)
            if event is None:
                break
            if event[0] in _SESSION_END_KINDS:
                # The audio was already fully fed; whatever was committed is
                # in, so the session ending now is completion, not failure.
                self._log_raw_event(*event)
                break
            if self._handle_events([event]):
                await emit_new_preview()
        await note_progress(duration or self._fed_end or 0.0)
