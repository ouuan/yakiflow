"""Subtitle timing adjustment based on Whisper's speech timeline.

When whisper.cpp VAD is enabled, its stderr reports speech spans on the
original audio timeline. This module consumes those spans directly. If no VAD
model was configured, it falls back to a small dependency-free PCM detector.
"""

from __future__ import annotations

import inspect
import asyncio
import gc
import json
import math
import sys
import wave
from array import array
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

from .models import Cue

_ORIGINAL_IMPORT_MODULE = import_module
from .transcription import merge_vad_intervals


WarningListener = Callable[[str], Awaitable[None] | None]
AlignmentProgressListener = Callable[[int, int], Awaitable[None] | None]
AlignmentModelFailureDecision = Literal["retry", "fallback"]
AlignmentModelFailureListener = Callable[
    [str],
    Awaitable[AlignmentModelFailureDecision] | AlignmentModelFailureDecision,
]

_SHORT_CUE_GAP = 0.7
_LONG_CUE_GAP = 1.0
_END_EXTENSION = 0.5
_MIDPOINT_WEIGHT = 0.5


async def _warn(listener: WarningListener | None, message: str) -> None:
    if listener is None:
        return
    maybe = listener(message)
    if inspect.isawaitable(maybe):
        await maybe


def _pcm_window_levels(
    audio: Path,
    window_seconds: float,
    requirement: str,
) -> tuple[list[float], int, int]:
    """Return per-window RMS levels of a mono 16-bit PCM file.

    The window size in frames and the declared sample rate are returned
    alongside the levels so that only callers needing a wall-clock step divide
    by the rate; a header declaring a zero rate stays readable otherwise.

    The alignment stage scans the same reference audio at the same window size
    from both the VAD backend and the volume refiner, so the last result is
    kept: the scan costs a full read of the file plus an RMS per 20 ms.
    """
    stat = audio.stat()
    key = (str(audio), stat.st_mtime_ns, stat.st_size, window_seconds)
    cached = _PCM_LEVEL_CACHE.get(key)
    if cached is not None:
        return cached
    levels_and_shape = _read_pcm_window_levels(audio, window_seconds, requirement)
    _PCM_LEVEL_CACHE.clear()
    _PCM_LEVEL_CACHE[key] = levels_and_shape
    return levels_and_shape


_PCM_LEVEL_CACHE: dict[
    tuple[str, int, int, float], tuple[list[float], int, int]
] = {}


def clear_pcm_level_cache() -> None:
    """Release the cached RMS envelope once the alignment stage is done.

    One entry is several megabytes for a feature-length file, and nothing after
    alignment reads it again.
    """
    _PCM_LEVEL_CACHE.clear()


def _read_pcm_window_levels(
    audio: Path,
    window_seconds: float,
    requirement: str,
) -> tuple[list[float], int, int]:
    with wave.open(str(audio), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(requirement)
        rate = wav.getframerate()
        window = max(1, int(rate * window_seconds))
        levels: list[float] = []
        while frames := wav.readframes(window):
            samples = array("h")
            samples.frombytes(frames)
            levels.append(
                math.sqrt(math.sumprod(samples, samples) / len(samples))
                if samples
                else 0.0
            )
    return levels, window, rate


def crosstalk_cue_ids(cues: Sequence[Cue]) -> set[str]:
    """IDs of the cues that overlap a cue of a different *named* speaker.

    This is the criterion the subtitle module states, applied to timing: only
    two different named speakers overlapping can be two people talking at
    once. Any other overlap — same speaker, or either side unnamed — cannot
    be, and stays fair game for the timing passes; end extension in
    particular repairs exactly those by pulling an end back to the next start.

    Every pass recomputes this for the cues it is about to touch rather than
    receiving it: forced alignment splits and renumbers cues, so a set decided
    outside would no longer name the same cues inside.
    """
    ordered = sorted(
        (cue for cue in cues if cue.speaker), key=lambda cue: (cue.start, cue.end)
    )
    crosstalk: set[str] = set()
    # Cues still running at the current start. Touching endpoints are not an
    # overlap, so a cue leaves as soon as its end reaches the next start.
    active: list[Cue] = []
    for cue in ordered:
        active = [other for other in active if other.end > cue.start]
        for other in active:
            if other.speaker != cue.speaker:
                crosstalk.add(cue.id)
                crosstalk.add(other.id)
        active.append(cue)
    return crosstalk


@dataclass(slots=True)
class AlignmentResult:
    cues: list[Cue]
    backend: str
    warning: str | None = None
    low_confidence_ids: list[str] = field(default_factory=list)


class AlignmentBackend(ABC):
    @abstractmethod
    async def align(
        self,
        audio: Path,
        cues: Sequence[Cue],
        *,
        vad_intervals: Sequence[tuple[float, float]] = (),
        on_warning: WarningListener | None = None,
        on_progress: AlignmentProgressListener | None = None,
        on_model_failure: AlignmentModelFailureListener | None = None,
    ) -> AlignmentResult: ...


class WhisperVadAlignmentBackend(AlignmentBackend):
    """Align cue boundaries to Whisper's speech timeline."""

    _TIMESTAMP_TOLERANCE = 0.04
    _MAX_START_LOOKAHEAD = 2.0
    _MOVED_START_THRESHOLD = 0.02
    _VAD_OVERLAP = 0.1
    _PROCESSED_SILENCE = 0.1
    _SINGLE_CHARACTER_LONG_SILENCE = 0.0
    _MULTI_CHARACTER_LONG_SILENCE = 0.32
    _SINGLE_CHARACTER_TOKEN_LENGTH = 1
    _MIN_REMAINING_VAD_SPEECH = 0.8
    _END_SEARCH_RADIUS = 1.0
    _MOVED_START_CONFIDENCE = 0.9
    _UNCHANGED_START_CONFIDENCE = 0.75
    _MISSING_START_CONFIDENCE = 0.35
    _MILLISECONDS_PER_SECOND = 1000
    _SECONDS_PER_MINUTE = 60
    _MINUTES_PER_HOUR = 60
    _TIMESTAMP_PARTS = 3
    _PCM_WINDOW_SECONDS = 0.02
    _PCM_NOISE_FLOOR_QUANTILE = 0.15
    _PCM_NOISE_FLOOR_MULTIPLIER = 3.0
    _PCM_PEAK_MULTIPLIER = 0.035
    _PCM_MIN_LEVEL = 8.0
    _PCM_MIN_ACTIVE_WINDOWS = 3
    _LEADING_TOKEN_COUNT = 2
    _EPSILON = 1e-9

    async def align(
        self,
        audio: Path,
        cues: Sequence[Cue],
        *,
        vad_intervals: Sequence[tuple[float, float]] = (),
        on_warning: WarningListener | None = None,
        on_progress: AlignmentProgressListener | None = None,
        on_model_failure: AlignmentModelFailureListener | None = None,
    ) -> AlignmentResult:
        original = list(cues)
        if not original:
            return AlignmentResult([], "whisper-vad")
        try:
            # The implementation only scans a PCM file; avoiding a worker
            # thread keeps alignment reliable on Python builds where spawning
            # threads from an async subprocess callback is restricted.
            return self._align(audio, original, list(vad_intervals))
        except Exception as exc:
            # A timing pass must never make a successful transcription fail.
            await _warn(on_warning, f"Whisper VAD timing adjustment skipped: {exc}")
            return AlignmentResult(
                original,
                "whisper-vad-unaligned",
                f"timing adjustment skipped: {exc}",
                [cue.id for cue in original],
            )

    @classmethod
    def _align(
        cls,
        audio: Path,
        cues: list[Cue],
        vad_intervals: list[tuple[float, float]],
    ) -> AlignmentResult:
        intervals = cls._merge_intervals(vad_intervals)
        backend = "whisper-vad"
        has_whisper_vad = bool(intervals)
        if not intervals:
            intervals = cls._pcm_intervals(audio)
            backend = "pcm-vad"
        if not intervals:
            raise ValueError("audio contains no speech activity")

        # Crosstalk keeps the timing the ASR gave it, which is the best account
        # of who spoke when; the pass only moves the cues around it, for which
        # a frozen cue is an ordinary boundary.
        frozen = crosstalk_cue_ids(cues)
        starts: list[float] = []
        confidences: list[float | None] = []
        low: list[str] = []
        previous_end = -math.inf
        pcm_confirmation_intervals: list[tuple[float, float]] | None = None
        pcm_confirmation_attempted = False
        for cue in cues:
            if cue.id in frozen:
                starts.append(cue.start)
                confidences.append(cue.timing_confidence)
                previous_end = cue.end
                continue
            start: float | None = None
            if has_whisper_vad:
                long_silence_candidates = cls._long_silence_start_candidates(
                    cue, previous_end, intervals
                )
                for candidate, token_range in reversed(long_silence_candidates):
                    if token_range is None:
                        start = candidate
                        break
                    if not pcm_confirmation_attempted:
                        pcm_confirmation_attempted = True
                        try:
                            pcm_confirmation_intervals = cls._pcm_intervals(audio)
                        except (EOFError, OSError, ValueError, wave.Error):
                            # This PCM pass only confirms an ambiguous token.
                            # If it is unavailable, retain the normal alignment.
                            pcm_confirmation_intervals = None
                    if pcm_confirmation_intervals is not None and not cls._has_pcm_speech(
                        pcm_confirmation_intervals, token_range
                    ):
                        start = candidate
                        break
            # Pick the first voiced span belonging to this cue.  A small
            # tolerance handles Whisper's millisecond rounding.
            if start is None and cls._has_sufficient_remaining_vad_speech(
                cue.start, intervals
            ):
                start = cue.start
            if start is None:
                candidates = [
                    left
                    for left, right in intervals
                    if right >= cue.start - cls._TIMESTAMP_TOLERANCE
                    and left <= min(cue.end, cue.start + cls._MAX_START_LOOKAHEAD)
                    and left >= previous_end + cls._TIMESTAMP_TOLERANCE
                ]
                if candidates:
                    start = max(cue.start, candidates[0])
            if start is not None:
                confidence = (
                    cls._MOVED_START_CONFIDENCE
                    if start > cue.start + cls._MOVED_START_THRESHOLD
                    else cls._UNCHANGED_START_CONFIDENCE
                )
            else:
                start = cue.start
                confidence = cls._MISSING_START_CONFIDENCE
                low.append(cue.id)
            # Never move a start backwards or beyond its original segment.
            start = min(max(cue.start, start), cue.end)
            starts.append(start)
            confidences.append(confidence)
            previous_end = cue.end

        refined_ends = [
            cue.end
            if cue.id in frozen
            else cls._refined_end(
                intervals,
                cue.end,
                starts[index],
                starts[index + 1] if index + 1 < len(cues) else None,
            )
            for index, cue in enumerate(cues)
        ]
        output = [
            cue
            if cue.id in frozen
            else cue.with_timing(
                starts[index],
                max(starts[index], refined_ends[index]),
                confidences[index],
            )
            for index, cue in enumerate(cues)
        ]
        return AlignmentResult(output, backend, low_confidence_ids=low)

    @classmethod
    def _has_sufficient_remaining_vad_speech(
        cls,
        cue_start: float,
        intervals: Sequence[tuple[float, float]],
    ) -> bool:
        return any(
            left <= cue_start + cls._TIMESTAMP_TOLERANCE
            and right >= cue_start - cls._TIMESTAMP_TOLERANCE
            and right - cue_start > cls._MIN_REMAINING_VAD_SPEECH
            for left, right in intervals
        )

    @classmethod
    def _long_silence_start_candidates(
        cls,
        cue: Cue,
        previous_end: float,
        intervals: Sequence[tuple[float, float]],
    ) -> list[tuple[float, tuple[float, float] | None]]:
        token_ranges = cls._leading_token_ranges(cue)
        if not token_ranges:
            return []
        first_token_text = cls._first_token_text(cue)
        if first_token_text is None:
            return []
        long_silence_threshold = (
            cls._SINGLE_CHARACTER_LONG_SILENCE
            if len(first_token_text.strip()) == cls._SINGLE_CHARACTER_TOKEN_LENGTH
            else cls._MULTI_CHARACTER_LONG_SILENCE
        )
        token_start, token_end = token_ranges[0]
        next_token_range = (
            token_ranges[1] if len(token_ranges) == cls._LEADING_TOKEN_COUNT else None
        )
        processed_end = 0.0
        candidates: list[tuple[float, tuple[float, float] | None]] = []
        for (left, right), (next_left, _) in zip(intervals, intervals[1:]):
            processed_start = processed_end
            processed_end += right - left + cls._VAD_OVERLAP
            silence_start = processed_end
            silence_end = silence_start + cls._PROCESSED_SILENCE
            crosses_silence = (
                token_start <= silence_start + cls._EPSILON
                and token_end + cls._EPSILON >= silence_end
            )
            valid_gap = (
                next_left - right + cls._EPSILON >= long_silence_threshold
                and next_left >= previous_end + cls._TIMESTAMP_TOLERANCE
                and next_left <= cue.end
            )
            if valid_gap and crosses_silence:
                candidates.append((next_left, None))
            elif valid_gap and next_token_range is not None:
                next_token_start, next_token_end = next_token_range
                token_is_near_silence = (
                    token_end <= silence_start + cls._EPSILON
                    and silence_start - token_end
                    <= cls._PROCESSED_SILENCE + cls._EPSILON
                )
                next_token_crosses_silence = (
                    next_token_start <= silence_start + cls._EPSILON
                    and next_token_end + cls._EPSILON >= silence_end
                )
                token_maps_to_current_interval = (
                    token_start >= processed_start - cls._EPSILON
                    and token_end <= silence_start + cls._EPSILON
                )
                if (
                    token_is_near_silence
                    and next_token_crosses_silence
                    and token_maps_to_current_interval
                ):
                    original_token_range = (
                        left + token_start - processed_start,
                        left + token_end - processed_start,
                    )
                    candidates.append((next_left, original_token_range))
            processed_end = silence_end
        return candidates

    @classmethod
    def _first_token_text(cls, cue: Cue) -> str | None:
        whisper = cue.metadata.get("whisper")
        if not isinstance(whisper, dict):
            return None
        tokens = whisper.get("tokens")
        if not isinstance(tokens, list):
            return None
        for token in tokens:
            if not isinstance(token, dict):
                continue
            text = token.get("text")
            if isinstance(text, str) and text and not cls._is_special_token(text):
                return text
        return None

    @classmethod
    def _leading_token_ranges(cls, cue: Cue) -> list[tuple[float, float]]:
        whisper = cue.metadata.get("whisper")
        if not isinstance(whisper, dict):
            return []
        tokens = whisper.get("tokens")
        if not isinstance(tokens, list):
            return []
        result: list[tuple[float, float]] = []
        for token in tokens:
            if not isinstance(token, dict):
                continue
            text = token.get("text")
            if not isinstance(text, str) or not text or cls._is_special_token(text):
                continue
            token_range = cls._token_range(token)
            if token_range is None:
                return result
            result.append(token_range)
            if len(result) == cls._LEADING_TOKEN_COUNT:
                break
        return result

    @classmethod
    def _has_pcm_speech(
        cls,
        intervals: Sequence[tuple[float, float]],
        token_range: tuple[float, float],
    ) -> bool:
        token_start, token_end = token_range
        return any(
            right >= token_start - cls._TIMESTAMP_TOLERANCE
            and left <= token_end + cls._TIMESTAMP_TOLERANCE
            for left, right in intervals
        )

    @staticmethod
    def _is_special_token(text: str) -> bool:
        return (text.startswith("[_") and text.endswith("]")) or (
            text.startswith("<|") and text.endswith("|>")
        )

    @classmethod
    def _token_range(cls, token: dict[str, Any]) -> tuple[float, float] | None:
        for key in ("offsets", "timestamps"):
            values = token.get(key)
            if not isinstance(values, dict):
                continue
            start = cls._token_time(values.get("from"))
            end = cls._token_time(values.get("to"))
            if start is not None and end is not None and end >= start:
                return start, end
        return None

    @classmethod
    def _token_time(cls, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value / cls._MILLISECONDS_PER_SECOND
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if not isinstance(value, str):
            return None
        try:
            parts = value.replace(",", ".").split(":")
            if len(parts) == cls._TIMESTAMP_PARTS:
                hours, minutes, seconds = parts
                return (
                    int(hours)
                    * cls._MINUTES_PER_HOUR
                    * cls._SECONDS_PER_MINUTE
                    + int(minutes) * cls._SECONDS_PER_MINUTE
                    + float(seconds)
                )
            result = float(value)
            return result if math.isfinite(result) else None
        except ValueError:
            return None

    @classmethod
    def _refined_end(
        cls,
        intervals: Sequence[tuple[float, float]],
        original_end: float,
        start: float,
        next_start: float | None,
    ) -> float:
        candidates = [
            right
            for left, right in intervals
            if abs(right - original_end) <= cls._END_SEARCH_RADIUS + cls._EPSILON
            and left <= original_end + cls._TIMESTAMP_TOLERANCE
            and right >= start
            and (next_start is None or right <= next_start)
        ]
        if not candidates:
            return original_end
        return min(candidates, key=lambda right: (abs(right - original_end), -right))

    @classmethod
    def _merge_intervals(
        cls,
        intervals: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        # Alignment deliberately coalesces at the wider timestamp tolerance:
        # Whisper's millisecond rounding must not look like a speech gap here.
        return merge_vad_intervals(intervals, cls._TIMESTAMP_TOLERANCE)

    @classmethod
    def _pcm_intervals(cls, audio: Path) -> list[tuple[float, float]]:
        # Intervals are expressed in whole windows of ``_PCM_WINDOW_SECONDS``,
        # so the sample rate is never divided by here.
        levels, _, _ = _pcm_window_levels(
            audio,
            cls._PCM_WINDOW_SECONDS,
            "Whisper VAD fallback requires mono 16-bit PCM audio",
        )
        if not levels or max(levels) <= 0:
            return []
        ordered = sorted(levels)
        floor_index = int(len(ordered) * cls._PCM_NOISE_FLOOR_QUANTILE)
        floor = ordered[min(len(ordered) - 1, floor_index)]
        threshold = max(
            floor * cls._PCM_NOISE_FLOOR_MULTIPLIER,
            max(levels) * cls._PCM_PEAK_MULTIPLIER,
            cls._PCM_MIN_LEVEL,
        )
        active = [level >= threshold for level in levels]
        result: list[tuple[float, float]] = []
        begin: int | None = None
        for index, is_active in enumerate(active + [False]):
            if is_active and begin is None:
                begin = index
            elif not is_active and begin is not None:
                if index - begin >= cls._PCM_MIN_ACTIVE_WINDOWS:
                    window_seconds = cls._PCM_WINDOW_SECONDS
                    result.append((begin * window_seconds, index * window_seconds))
                begin = None
        return cls._merge_intervals(result)


def extend_cue_ends(
    cues: Sequence[Cue],
    *,
    duration: float | None = None,
) -> list[Cue]:
    """Apply the shared subtitle end-extension policy to a final timeline.

    Crosstalk cues pass through untouched — not extended, and not even clamped
    to the audio duration: media length and ASR word times can disagree by a
    hair, and clamping would pin a crosstalk pair to one shared end and erase
    the overlap. Everything else is extended as usual, which is also how an
    illegitimate overlap gets repaired here, by pulling the earlier end back
    to the next start.
    """
    if not cues:
        return []
    frozen = crosstalk_cue_ids(cues)
    ordered = sorted(cues, key=lambda cue: (cue.start, cue.end))
    maximum = math.inf if duration is None else max(0.0, duration)
    output: list[Cue] = []
    for index, cue in enumerate(ordered):
        if cue.id in frozen:
            output.append(cue)
            continue
        start = min(maximum, cue.start)
        base_end = min(maximum, max(start, cue.end))
        if index + 1 == len(ordered):
            end = min(maximum, base_end + _END_EXTENSION)
        else:
            next_start = min(maximum, max(0.0, ordered[index + 1].start))
            gap = next_start - base_end
            if gap <= _SHORT_CUE_GAP:
                end = next_start
            elif gap < _LONG_CUE_GAP:
                end = (base_end + next_start) * _MIDPOINT_WEIGHT
            else:
                end = min(maximum, base_end + _END_EXTENSION)
        output.append(
            cue.with_timing(start, max(start, end), cue.timing_confidence)
        )
    return output


_PRE_ALIGNMENT_LONG_SILENCE_SECONDS = 6.0
_PRE_ALIGNMENT_SILENCE_PRE_ROLL_SECONDS = 3.0


def adjust_cue_starts_for_long_vad_silences(
    cues: Sequence[Cue],
    vad_intervals: Sequence[tuple[float, float]],
) -> list[Cue]:
    """Move starts near the end of the last long VAD silence in each cue.

    Crosstalk cues keep the start the ASR gave them.
    """
    intervals = WhisperVadAlignmentBackend._merge_intervals(vad_intervals)
    if not intervals:
        return list(cues)

    frozen = crosstalk_cue_ids(cues)
    adjusted: list[Cue] = []
    for cue in cues:
        if cue.id in frozen:
            adjusted.append(cue)
            continue
        previous_speech_end = cue.start
        last_long_silence_end: float | None = None
        for speech_start, speech_end in intervals:
            if speech_end <= cue.start:
                continue
            if speech_start >= cue.end:
                break
            silence_start = max(cue.start, previous_speech_end)
            if (
                speech_start - silence_start
                > _PRE_ALIGNMENT_LONG_SILENCE_SECONDS
            ):
                last_long_silence_end = speech_start
            previous_speech_end = max(previous_speech_end, speech_end)

        if last_long_silence_end is None:
            adjusted.append(cue)
            continue
        start = max(
            cue.start,
            last_long_silence_end - _PRE_ALIGNMENT_SILENCE_PRE_ROLL_SECONDS,
        )
        adjusted.append(cue.with_timing(start, cue.end, cue.timing_confidence))
    return adjusted


class PcmVolumeStartRefiner:
    """Move cue starts past leading silence using a low-SNR-aware PCM envelope."""

    _WINDOW_SECONDS = 0.02
    _NOISE_QUANTILE = 0.2
    _LOOKBACK_SECONDS = 0.4
    _LOCAL_NOISE_CAP_MULTIPLIER = 1.5
    _LOCAL_NOISE_CAP_MARGIN = 3.0
    _WEAK_NOISE_MULTIPLIER = 1.3
    _STRONG_NOISE_MULTIPLIER = 2.0
    _WEAK_PEAK_MULTIPLIER = 0.0005
    _STRONG_PEAK_MULTIPLIER = 0.003
    _MIN_WEAK_LEVEL = 4.0
    _MIN_STRONG_LEVEL = 8.0
    _WEAK_RUN_WINDOWS = 4
    _STRONG_CONFIRM_WINDOWS = 3
    _STRONG_MIN_ACTIVE_WINDOWS = 2
    _PRE_ROLL_SECONDS = 0.06
    _MIN_SHIFT_SECONDS = 0.005

    async def refine(
        self,
        audio: Path,
        cues: Sequence[Cue],
        *,
        on_warning: WarningListener | None = None,
    ) -> list[Cue]:
        original = list(cues)
        if not original:
            return []
        try:
            return await asyncio.to_thread(self._refine, audio, original)
        except Exception as exc:
            await _warn(on_warning, f"PCM volume start refinement skipped: {exc}")
            return original

    @classmethod
    def _refine(cls, audio: Path, cues: list[Cue]) -> list[Cue]:
        levels, window, rate = _pcm_window_levels(
            audio,
            cls._WINDOW_SECONDS,
            "volume refinement requires mono 16-bit PCM audio",
        )
        step = window / rate
        if not levels or max(levels) <= 0:
            return cues
        global_floor = cls._quantile(levels, cls._NOISE_QUANTILE)
        # Two voices at once is exactly the case an envelope cannot read: the
        # other speaker's onset would move this cue's start.
        frozen = crosstalk_cue_ids(cues)
        output: list[Cue] = []
        for cue in cues:
            if cue.id in frozen:
                output.append(cue)
                continue
            detected = cls._detect_onset(levels, step, cue, global_floor)
            if detected is None:
                output.append(cue)
                continue
            onset, noise_floor, weak_threshold, strong_threshold = detected
            refined_start = max(cue.start, onset - cls._PRE_ROLL_SECONDS)
            if refined_start <= cue.start + cls._MIN_SHIFT_SECONDS:
                output.append(cue)
                continue
            metadata = dict(cue.metadata)
            metadata["volume_start"] = {
                "original_start": cue.start,
                "detected_start": onset,
                "refined_start": refined_start,
                "noise_floor": noise_floor,
                "weak_threshold": weak_threshold,
                "strong_threshold": strong_threshold,
            }
            output.append(
                Cue(
                    cue.id,
                    refined_start,
                    cue.end,
                    cue.source,
                    cue.translated,
                    cue.timing_confidence,
                    metadata,
                    cue.speaker,
                )
            )
        return output

    @classmethod
    def _detect_onset(
        cls,
        levels: Sequence[float],
        step: float,
        cue: Cue,
        global_floor: float,
    ) -> tuple[float, float, float, float] | None:
        start_index = max(0, int(math.floor(cue.start / step)))
        end_index = min(len(levels), int(math.ceil(cue.end / step)))
        if end_index <= start_index:
            return None
        search = levels[start_index:end_index]
        lookback_windows = max(1, round(cls._LOOKBACK_SECONDS / step))
        lookback = levels[max(0, start_index - lookback_windows):start_index]
        local_floor = (
            cls._quantile(lookback, cls._NOISE_QUANTILE)
            if lookback
            else global_floor
        )
        # A preceding cue may contain loud speech throughout the lookback.
        # Cap its influence so it cannot mask a quiet speaker in this cue.
        local_floor_cap = (
            global_floor * cls._LOCAL_NOISE_CAP_MULTIPLIER
            + cls._LOCAL_NOISE_CAP_MARGIN
        )
        noise_floor = max(global_floor, min(local_floor, local_floor_cap))
        peak = max(search)
        weak_threshold = max(
            noise_floor * cls._WEAK_NOISE_MULTIPLIER,
            peak * cls._WEAK_PEAK_MULTIPLIER,
            cls._MIN_WEAK_LEVEL,
        )
        strong_threshold = max(
            noise_floor * cls._STRONG_NOISE_MULTIPLIER,
            peak * cls._STRONG_PEAK_MULTIPLIER,
            cls._MIN_STRONG_LEVEL,
        )

        weak_run = 0
        for offset, level in enumerate(search):
            if level > weak_threshold:
                weak_run += 1
            else:
                weak_run = 0
            if level > strong_threshold:
                confirmation = search[
                    offset:offset + cls._STRONG_CONFIRM_WINDOWS
                ]
                if sum(value > weak_threshold for value in confirmation) >= (
                    cls._STRONG_MIN_ACTIVE_WINDOWS
                ):
                    onset = max(cue.start, (start_index + offset) * step)
                    return onset, noise_floor, weak_threshold, strong_threshold
            if weak_run >= cls._WEAK_RUN_WINDOWS:
                first_offset = offset - cls._WEAK_RUN_WINDOWS + 1
                onset = max(cue.start, (start_index + first_offset) * step)
                return onset, noise_floor, weak_threshold, strong_threshold
        return None

    @staticmethod
    def _quantile(values: Sequence[float], quantile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int(len(ordered) * quantile))
        return ordered[index]


class AlignmentModelDecisionRequired(RuntimeError):
    """Raised when model loading fails without an interactive decision hook."""


class _AlignmentFallbackRequested(RuntimeError):
    pass


class _WhisperXZeroFirstTokenScore(ValueError):
    def __init__(self, words: list[dict[str, Any]]) -> None:
        super().__init__("WhisperX first token score was 0")
        self.words = words


class _WhisperXInProcessBackend(AlignmentBackend):
    """Forced alignment around each original Whisper cue.

    WhisperX is intentionally imported lazily and is only used for its audio
    loader and alignment model. Transcription remains exclusively owned by
    whisper.cpp.
    """

    SAMPLE_RATE = 16_000
    INITIAL_BEFORE = 0.2
    INITIAL_AFTER = 0.5
    RETRY_PADDING = 2.0
    BOUNDARY_TOLERANCE = 0.05
    NLTK_DOWNLOAD_TIMEOUT = 60.0

    def __init__(
        self,
        *,
        language: str | None,
        device: str = "auto",
        model_name: str | None = None,
        vad_backend: WhisperVadAlignmentBackend | None = None,
        fallback_on_error: bool = True,
    ) -> None:
        self.language = language
        self.device = device
        self.model_name = model_name
        self.vad_backend = vad_backend or WhisperVadAlignmentBackend()
        self.fallback_on_error = fallback_on_error

    async def align(
        self,
        audio: Path,
        cues: Sequence[Cue],
        *,
        vad_intervals: Sequence[tuple[float, float]] = (),
        on_warning: WarningListener | None = None,
        on_progress: AlignmentProgressListener | None = None,
        on_model_failure: AlignmentModelFailureListener | None = None,
    ) -> AlignmentResult:
        original = list(cues)
        if not original:
            return AlignmentResult([], "whisperx")
        module: Any = None
        audio_data: Any = None
        align_model: Any = None
        model_metadata: Any = None
        selected_device = "cpu"
        try:
            module = await asyncio.to_thread(import_module, "whisperx")
            selected_device = await asyncio.to_thread(
                self._resolve_device, self.device
            )
            language = self.language
            if not language:
                raise RuntimeError(
                    "WhisperX alignment needs a detected or configured source language"
                )
            await self._ensure_nltk_punkt(language, on_model_failure)
            audio_data = await asyncio.to_thread(module.load_audio, str(audio))
            duration = len(audio_data) / self.SAMPLE_RATE
            if duration <= 0:
                raise RuntimeError("WhisperX loaded empty audio")
            load_kwargs = {"model_name": self.model_name} if self.model_name else {}
            while True:
                try:
                    align_model, model_metadata = await asyncio.to_thread(
                        module.load_align_model,
                        language_code=language,
                        device=selected_device,
                        **load_kwargs,
                    )
                except Exception as exc:
                    message = (
                        "WhisperX alignment model download or load failed: "
                        f"{exc}"
                    )
                    if on_model_failure is None:
                        raise AlignmentModelDecisionRequired(
                            f"{message}; retry/fallback choice requires the interactive UI"
                        ) from exc
                    decision = await self._request_model_failure_decision(
                        on_model_failure, message
                    )
                    if decision == "retry":
                        continue
                    raise _AlignmentFallbackRequested(message) from exc
                break
        except (asyncio.CancelledError, AlignmentModelDecisionRequired):
            align_model = None
            model_metadata = None
            audio_data = None
            await asyncio.to_thread(self._release_model_cache, selected_device)
            raise
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            warning = f"WhisperX unavailable; using Whisper VAD alignment: {exc}"
            await _warn(on_warning, warning)
            try:
                fallback = await self.vad_backend.align(
                    audio,
                    original,
                    vad_intervals=vad_intervals,
                    on_warning=on_warning,
                )
                fallback_cues: list[Cue] = []
                for cue in fallback.cues:
                    metadata = dict(cue.metadata)
                    metadata.setdefault("parent_id", cue.id)
                    metadata["alignment_backend"] = "vad-fallback"
                    fallback_cues.append(
                        Cue(
                            cue.id,
                            cue.start,
                            cue.end,
                            cue.source,
                            cue.translated,
                            cue.timing_confidence,
                            metadata,
                            cue.speaker,
                        )
                    )
                numbered = self._renumber(fallback_cues)
                return AlignmentResult(
                    numbered,
                    "whisperx-vad-fallback",
                    warning,
                    [cue.id for cue in numbered],
                )
            finally:
                align_model = None
                model_metadata = None
                audio_data = None
                await asyncio.to_thread(self._release_model_cache, selected_device)

        try:
            aligned_by_parent: dict[str, list[Cue]] = {}
            attempts_by_parent: dict[str, list[dict[str, Any]]] = {}
            failed_ids: list[str] = []
            zero_first_token_ids: list[str] = []
            zero_score_words_by_parent: dict[str, list[dict[str, Any]]] = {}
            frozen = crosstalk_cue_ids(original)
            total_cues = len(original)
            for completed_cues, cue in enumerate(original, start=1):
                if cue.id in frozen:
                    # Forced alignment assumes one voice at a time, so a
                    # crosstalk cue keeps its ASR timing. Not an alignment
                    # failure — recording it as one would drag the whole
                    # timeline into the VAD fallback — but progress still
                    # counts it, or the bar would stop short of its total.
                    await self._notify_progress(
                        on_progress, completed_cues, total_cues
                    )
                    continue
                result: list[Cue] | None = None
                attempts: list[dict[str, Any]] = []
                zero_score_words: list[dict[str, Any]] | None = None
                for before, after in (
                    (self.INITIAL_BEFORE, self.INITIAL_AFTER),
                    (self.RETRY_PADDING, self.RETRY_PADDING),
                ):
                    window_start = max(0.0, cue.start - before)
                    window_end = min(duration, cue.end + after)
                    attempt: dict[str, Any] = {
                        "window_start": window_start,
                        "window_end": window_end,
                    }
                    attempts.append(attempt)
                    try:
                        result, touches_boundary = await asyncio.to_thread(
                            self._align_cue,
                            module,
                            align_model,
                            model_metadata,
                            audio_data,
                            cue,
                            window_start,
                            window_end,
                            selected_device,
                            duration,
                        )
                    except _WhisperXZeroFirstTokenScore as exc:
                        result = None
                        touches_boundary = False
                        attempt["outcome"] = "zero-first-token-score"
                        zero_score_words = exc.words
                    except Exception:
                        result = None
                        touches_boundary = False
                        attempt["outcome"] = "failed"
                    else:
                        attempt["outcome"] = (
                            "boundary-clipped" if touches_boundary else "success"
                        )
                    if result and not touches_boundary:
                        break
                    result = None
                attempts_by_parent[cue.id] = attempts
                if result:
                    selected = attempts[-1]
                    for aligned_cue in result:
                        whisperx_metadata = dict(
                            aligned_cue.metadata.get("whisperx", {})
                        )
                        whisperx_metadata.update(
                            {
                                "window_start": selected["window_start"],
                                "window_end": selected["window_end"],
                                "attempts": attempts,
                            }
                        )
                        aligned_cue.metadata["whisperx"] = whisperx_metadata
                    aligned_by_parent[cue.id] = result
                else:
                    failed_ids.append(cue.id)
                    if zero_score_words is not None:
                        zero_first_token_ids.append(cue.id)
                        zero_score_words_by_parent[cue.id] = zero_score_words
                await self._notify_progress(
                    on_progress,
                    completed_cues,
                    total_cues,
                )

            fallback_by_id: dict[str, Cue] = {}
            if failed_ids:
                fallback = await self.vad_backend.align(
                    audio,
                    original,
                    vad_intervals=vad_intervals,
                    on_warning=on_warning,
                )
                fallback_by_id = {cue.id: cue for cue in fallback.cues}
                warning_parts: list[str] = []
                if zero_first_token_ids:
                    warning_parts.append(
                        "WhisperX first token score was 0 for cue IDs "
                        f"{', '.join(zero_first_token_ids)}"
                    )
                other_failed_ids = [
                    cue_id
                    for cue_id in failed_ids
                    if cue_id not in zero_first_token_ids
                ]
                if other_failed_ids:
                    warning_parts.append(
                        "WhisperX could not align cue IDs "
                        f"{', '.join(other_failed_ids)}"
                    )
                await _warn(
                    on_warning,
                    "; ".join(warning_parts) + "; used Whisper VAD fallback",
                )

            combined: list[Cue] = []
            for parent in original:
                forced = aligned_by_parent.get(parent.id)
                if forced:
                    combined.extend(forced)
                    continue
                if parent.id in frozen:
                    combined.append(parent)
                    continue
                fallback_cue = fallback_by_id.get(parent.id, parent)
                metadata = dict(fallback_cue.metadata)
                metadata["parent_id"] = parent.id
                metadata["alignment_backend"] = "vad-fallback"
                whisperx_metadata = dict(metadata.get("whisperx", {}))
                whisperx_metadata["attempts"] = attempts_by_parent.get(parent.id, [])
                if parent.id in zero_score_words_by_parent:
                    whisperx_metadata["words"] = zero_score_words_by_parent[parent.id]
                metadata["whisperx"] = whisperx_metadata
                combined.append(
                    Cue(
                        fallback_cue.id,
                        fallback_cue.start,
                        fallback_cue.end,
                        fallback_cue.source,
                        fallback_cue.translated,
                        fallback_cue.timing_confidence,
                        metadata,
                        fallback_cue.speaker,
                    )
                )

            combined = self._normalize_timeline(combined, duration)
            numbered = self._renumber(combined)
            low = [cue.id for cue in numbered if cue.metadata.get("parent_id") in failed_ids]
            return AlignmentResult(numbered, "whisperx", low_confidence_ids=low)
        finally:
            # Models can retain a sizeable CUDA cache. Release all task-local
            # references on success, fallback, cancellation, and exceptions.
            align_model = None
            model_metadata = None
            audio_data = None
            await asyncio.to_thread(self._release_model_cache, selected_device)

    @staticmethod
    async def _notify_progress(
        listener: AlignmentProgressListener | None,
        completed: int,
        total: int,
    ) -> None:
        if listener is None:
            return
        maybe = listener(completed, total)
        if inspect.isawaitable(maybe):
            await maybe

    @staticmethod
    async def _request_model_failure_decision(
        listener: AlignmentModelFailureListener,
        message: str,
    ) -> AlignmentModelFailureDecision:
        try:
            maybe = listener(message)
            decision = await maybe if inspect.isawaitable(maybe) else maybe
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise AlignmentModelDecisionRequired(
                "could not obtain the WhisperX retry/fallback decision"
            ) from exc
        if decision not in {"retry", "fallback"}:
            raise AlignmentModelDecisionRequired(
                f"invalid WhisperX model failure decision: {decision!r}"
            )
        return decision

    @staticmethod
    def _release_model_cache(device: str) -> None:
        gc.collect()
        if device != "cuda":
            return
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        try:
            import torch

            available = bool(torch.cuda.is_available())
        except (ImportError, RuntimeError):
            available = False
        if requested == "cuda" and not available:
            raise RuntimeError("CUDA was requested for WhisperX but is unavailable")
        return "cuda" if available else "cpu"

    async def _ensure_nltk_punkt(
        self,
        language: str,
        on_failure: AlignmentModelFailureListener | None,
    ) -> None:
        """Download WhisperX's sentence tokenizer before loading its model."""
        whisperx_alignment = await asyncio.to_thread(
            import_module, "whisperx.alignment"
        )
        loader = getattr(whisperx_alignment, "nltk_load", None)
        if loader is None:
            raise RuntimeError("WhisperX does not expose its NLTK resource loader")
        punkt_languages = getattr(whisperx_alignment, "PUNKT_LANGUAGES", {})
        punkt_language = (
            punkt_languages.get(language, "english")
            if isinstance(punkt_languages, dict)
            else "english"
        )
        resource = f"tokenizers/punkt_tab/{punkt_language}.pickle"

        try:
            await asyncio.to_thread(loader, resource)
            return
        except LookupError:
            pass

        while True:
            download_dir: Path | None = None
            try:
                download_dir = await self._download_nltk_punkt()
                await asyncio.to_thread(loader, resource)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                destination = f" to {download_dir}" if download_dir else ""
                message = (
                    f"NLTK punkt_tab download{destination} failed: {exc}"
                )
            if on_failure is None:
                raise AlignmentModelDecisionRequired(
                    f"{message}; retry/fallback choice requires the interactive UI"
                )
            decision = await self._request_model_failure_decision(
                on_failure, message
            )
            if decision == "retry":
                continue
            raise _AlignmentFallbackRequested(message)

    @classmethod
    async def _download_nltk_punkt(cls) -> Path:
        nltk = await asyncio.to_thread(import_module, "nltk")
        downloader_module = await asyncio.to_thread(import_module, "nltk.downloader")
        download_dir = Path(
            downloader_module.Downloader().default_download_dir()
        ).expanduser()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "nltk.downloader",
            "--quiet",
            "--exit-on-error",
            "--dir",
            str(download_dir),
            "punkt_tab",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), cls.NLTK_DOWNLOAD_TIMEOUT
            )
        except TimeoutError as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            raise TimeoutError(
                f"download to {download_dir} timed out after "
                f"{cls.NLTK_DOWNLOAD_TIMEOUT:g} seconds"
            ) from exc
        except asyncio.CancelledError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        if process.returncode:
            detail = (stderr or stdout).decode(errors="replace").strip()
            reason = detail or f"downloader exited with {process.returncode}"
            raise RuntimeError(f"download to {download_dir}: {reason}")
        if download_dir not in map(Path, getattr(nltk.data, "path", ())):
            nltk.data.path.insert(0, str(download_dir))
        return download_dir

    @classmethod
    def _align_cue(
        cls,
        module: Any,
        model: Any,
        metadata: Any,
        audio_data: Any,
        cue: Cue,
        window_start: float,
        window_end: float,
        device: str,
        duration: float,
    ) -> tuple[list[Cue], bool]:
        if window_end <= window_start:
            raise ValueError("empty WhisperX search window")
        first_sample = max(0, int(window_start * cls.SAMPLE_RATE))
        last_sample = min(len(audio_data), int(math.ceil(window_end * cls.SAMPLE_RATE)))
        excerpt = audio_data[first_sample:last_sample]
        relative_start = max(0.0, cue.start - window_start)
        relative_end = min(window_end - window_start, cue.end - window_start)
        payload = [{"start": relative_start, "end": relative_end, "text": cue.source}]
        raw = module.align(
            payload,
            model,
            metadata,
            excerpt,
            device,
            return_char_alignments=False,
        )
        if not isinstance(raw, dict):
            raise ValueError("WhisperX returned a non-object result")
        segments = raw.get("segments")
        if not isinstance(segments, list) or not segments:
            word_segments = raw.get("word_segments")
            if isinstance(word_segments, list) and word_segments:
                segments = [{"text": cue.source, "words": word_segments}]
            else:
                raise ValueError("WhisperX returned no aligned segments")

        output: list[Cue] = []
        all_words: list[dict[str, Any]] = []
        first_token_seen = False
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("WhisperX returned an invalid aligned segment")
            raw_words = segment.get("words")
            if not isinstance(raw_words, list):
                raise ValueError("WhisperX segment has no word list")
            if not first_token_seen:
                first_token = next(
                    (word for word in raw_words if isinstance(word, dict)),
                    None,
                )
                if first_token is not None:
                    first_token_seen = True
                    first_score = first_token.get("score")
                    if (
                        isinstance(first_score, (int, float))
                        and not isinstance(first_score, bool)
                        and float(first_score) == 0.0
                    ):
                        raise _WhisperXZeroFirstTokenScore(
                            cls._fallback_word_metadata(
                                segments, window_start, duration
                            )
                        )
            words = cls._valid_words(raw_words, window_start, duration)
            if not words:
                raise ValueError("WhisperX segment has no valid timed words")
            all_words.extend(words)
            scores = [word["score"] for word in words if word.get("score") is not None]
            score = sum(scores) / len(scores) if scores else None
            source = str(segment.get("text") or "").strip() or cue.source
            word_metadata = [
                {
                    key: word[key]
                    for key in ("word", "start", "end", "score")
                    if key in word and word[key] is not None
                }
                for word in words
            ]
            cue_metadata = dict(cue.metadata)
            cue_metadata.update(
                {
                    "parent_id": cue.id,
                    "alignment_backend": "whisperx",
                    "whisperx": {"words": word_metadata, "score": score},
                }
            )
            output.append(
                Cue(
                    cue.id,
                    words[0]["start"],
                    words[-1]["end"],
                    source,
                    None,
                    score,
                    cue_metadata,
                    # A sentence split into several cues is still the same
                    # utterance by the same person.
                    cue.speaker,
                )
            )
        if not output or not all_words:
            raise ValueError("WhisperX returned no valid timed words")
        output.sort(key=lambda item: (item.start, item.end))
        all_words.sort(key=lambda word: (word["start"], word["end"]))
        split = len(output) > 1
        if not split:
            # WhisperX may normalize whitespace in its segment. The corrected
            # Agent source and its translation remain authoritative.
            output[0].source = cue.source
            output[0].translated = cue.translated
        # Speech reaching a window edge means the window was cut too tight and
        # a wider retry is worth trying — but the file's own start and end are
        # not edges that widening can move, so a cue there would otherwise fail
        # both attempts and always fall back to VAD timings.
        touches = (
            window_start > cls.BOUNDARY_TOLERANCE
            and all_words[0]["start"] <= window_start + cls.BOUNDARY_TOLERANCE
        ) or (
            window_end < duration - cls.BOUNDARY_TOLERANCE
            and all_words[-1]["end"] >= window_end - cls.BOUNDARY_TOLERANCE
        )
        return output, touches

    @staticmethod
    def _fallback_word_metadata(
        segments: Sequence[Any],
        window_start: float,
        duration: float,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            words = segment.get("words")
            if not isinstance(words, list):
                continue
            for word in words:
                if not isinstance(word, dict):
                    continue
                item: dict[str, Any] = {"word": str(word.get("word", ""))}
                score = word.get("score")
                if (
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                ):
                    item["score"] = float(score)
                for key in ("start", "end"):
                    value = word.get(key)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    ):
                        absolute = float(value) + window_start
                        if 0 <= absolute <= duration + 1e-6:
                            item[key] = absolute
                output.append(item)
        return output

    @staticmethod
    def _valid_words(
        words: Sequence[Any], window_start: float, duration: float
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            start = word.get("start")
            end = word.get("end")
            if start is None and end is None:
                continue
            if isinstance(start, bool) or isinstance(end, bool):
                raise ValueError("WhisperX returned an invalid word time")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError("WhisperX returned an invalid word time")
            absolute_start = float(start) + window_start
            absolute_end = float(end) + window_start
            if (
                not math.isfinite(absolute_start)
                or not math.isfinite(absolute_end)
                or absolute_end < absolute_start
                or absolute_start < 0
                or absolute_end > duration + 1e-6
            ):
                raise ValueError("WhisperX returned an illegal word time range")
            score = word.get("score")
            valid_score = (
                float(score)
                if isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                else None
            )
            output.append(
                {
                    "word": str(word.get("word", "")),
                    "start": absolute_start,
                    "end": absolute_end,
                    "score": valid_score,
                }
            )
        output.sort(key=lambda value: (value["start"], value["end"]))
        return output

    @staticmethod
    def _normalize_timeline(
        cues: list[Cue],
        duration: float,
    ) -> list[Cue]:
        if not cues:
            return []
        frozen = crosstalk_cue_ids(cues)
        # Same ordering as the word pipeline uses, speaker included, so a
        # crosstalk pair that starts and ends together keeps one stable order.
        ordered = sorted(cues, key=lambda cue: (cue.start, cue.end, cue.speaker or ""))
        result: list[Cue] = []
        for cue in ordered:
            if cue.id in frozen:
                # Clamping a crosstalk pair against a duration that disagrees
                # with the ASR by a hair would pin both to one end.
                result.append(cue)
                continue
            start = min(duration, max(0.0, cue.start))
            end = min(duration, max(start, cue.end))
            result.append(cue.with_timing(start, max(start, end), cue.timing_confidence))
        return result

    @staticmethod
    def _renumber(cues: Sequence[Cue]) -> list[Cue]:
        output: list[Cue] = []
        for index, cue in enumerate(cues, start=1):
            metadata = dict(cue.metadata)
            metadata.setdefault("parent_id", cue.id)
            output.append(
                Cue(
                    str(index),
                    cue.start,
                    cue.end,
                    cue.source,
                    cue.translated,
                    cue.timing_confidence,
                    metadata,
                    cue.speaker,
                )
            )
        return output


class WhisperXAlignmentBackend(AlignmentBackend):
    """Run the WhisperX implementation in a short-lived child process.

    The parent process only handles JSON and the resulting cues.  In
    particular, importing this module never imports torch or WhisperX.
    """

    def __init__(
        self,
        *,
        language: str | None,
        device: str = "auto",
        model_name: str | None = None,
        vad_backend: WhisperVadAlignmentBackend | None = None,
    ) -> None:
        self.language = language
        self.device = device
        self.model_name = model_name
        self.vad_backend = vad_backend or WhisperVadAlignmentBackend()

    async def align(
        self,
        audio: Path,
        cues: Sequence[Cue],
        *,
        vad_intervals: Sequence[tuple[float, float]] = (),
        on_warning: WarningListener | None = None,
        on_progress: AlignmentProgressListener | None = None,
        on_model_failure: AlignmentModelFailureListener | None = None,
    ) -> AlignmentResult:
        original = list(cues)
        if not original:
            return AlignmentResult([], "whisperx")
        from .process import CommandRunner

        if import_module is not _ORIGINAL_IMPORT_MODULE:
            return await _WhisperXInProcessBackend(
                language=self.language,
                device=self.device,
                model_name=self.model_name,
                vad_backend=self.vad_backend,
            ).align(
                audio, original, vad_intervals=vad_intervals,
                on_warning=on_warning, on_progress=on_progress,
                on_model_failure=on_model_failure,
            )

        request = {
            "audio": str(audio),
            "language": self.language,
            "device": self.device,
            "model_name": self.model_name,
            "vad_intervals": [list(item) for item in vad_intervals],
            "cues": [_cue_to_json(cue) for cue in original],
        }
        while True:
            events: list[dict[str, Any]] = []

            async def on_line(stream: str, line: str) -> None:
                if stream != "stdout":
                    return
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"WhisperX worker sent invalid JSON: {exc}") from exc
                if not isinstance(event, dict):
                    raise RuntimeError("WhisperX worker sent a non-object event")
                events.append(event)
                kind = event.get("kind")
                if kind == "progress":
                    await _WhisperXInProcessBackend._notify_progress(
                        on_progress, int(event.get("completed", 0)), int(event.get("total", 0))
                    )
                elif kind == "warning":
                    await _warn(on_warning, str(event.get("message", "WhisperX worker warning")))

            result = await CommandRunner().run(
                [sys.executable, "-m", "yakiflow.whisperx_worker"],
                on_line=on_line,
                check=False,
                stdin=(json.dumps(request, ensure_ascii=False) + "\n").encode(),
            )
            error = next((event for event in events if event.get("kind") == "error"), None)
            if error is not None and error.get("category") == "model_failure":
                message = str(error.get("message", "WhisperX model initialization failed"))
                if on_model_failure is None:
                    raise AlignmentModelDecisionRequired(
                        f"{message}; retry/fallback choice requires the interactive UI"
                    )
                decision = await _WhisperXInProcessBackend._request_model_failure_decision(
                    on_model_failure, message
                )
                if decision == "retry":
                    continue
                warning = f"WhisperX unavailable; using Whisper VAD alignment: {message}"
                await _warn(on_warning, warning)
                return await self._fallback(
                    audio, original, vad_intervals, on_warning, warning
                )
            if error is not None and error.get("category") == "fallback":
                message = str(error.get("message", "WhisperX initialization failed"))
                warning = f"WhisperX unavailable; using Whisper VAD alignment: {message}"
                await _warn(on_warning, warning)
                return await self._fallback(
                    audio, original, vad_intervals, on_warning, warning
                )
            result_event = next((event for event in events if event.get("kind") == "result"), None)
            if result_event is None:
                detail = result.stderr.strip()[-2000:] or f"worker exited with status {result.returncode}"
                raise RuntimeError(f"WhisperX worker failed without a result: {detail}")
            return AlignmentResult(
                [_cue_from_json(item) for item in result_event.get("cues", [])],
                str(result_event.get("backend", "whisperx")),
                result_event.get("warning"),
                list(result_event.get("low_confidence_ids", [])),
            )

    async def _fallback(
        self,
        audio: Path,
        cues: list[Cue],
        vad_intervals: Sequence[tuple[float, float]],
        on_warning: WarningListener | None,
        warning: str,
    ) -> AlignmentResult:
        fallback = await self.vad_backend.align(
            audio, cues, vad_intervals=vad_intervals, on_warning=on_warning
        )
        output: list[Cue] = []
        for cue in fallback.cues:
            metadata = dict(cue.metadata)
            metadata.setdefault("parent_id", cue.id)
            metadata["alignment_backend"] = "vad-fallback"
            output.append(Cue(cue.id, cue.start, cue.end, cue.source, cue.translated,
                              cue.timing_confidence, metadata, cue.speaker))
        numbered = _WhisperXInProcessBackend._renumber(output)
        return AlignmentResult(numbered, "whisperx-vad-fallback", warning,
                               [cue.id for cue in numbered])


def _cue_to_json(cue: Cue) -> dict[str, Any]:
    return {"id": cue.id, "start": cue.start, "end": cue.end, "source": cue.source,
            "translated": cue.translated, "timing_confidence": cue.timing_confidence,
            "metadata": cue.metadata, "speaker": cue.speaker}


def _cue_from_json(value: Mapping[str, Any]) -> Cue:
    return Cue(str(value["id"]), float(value["start"]), float(value["end"]),
               str(value.get("source", "")), value.get("translated"),
               value.get("timing_confidence"), dict(value.get("metadata", {})),
               value.get("speaker"))


def make_alignment_backend(
    backend: str = "vad",
    *,
    language: str | None = None,
    device: str = "auto",
    model_name: str | None = None,
) -> AlignmentBackend:
    if backend == "whisperx":
        return WhisperXAlignmentBackend(
            language=language,
            device=device,
            model_name=model_name,
        )
    return WhisperVadAlignmentBackend()
