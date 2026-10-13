"""Shared word timing repair, preview segmentation and agent batching."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from math import ceil
from statistics import median
from typing import Sequence

from .models import Cue, Word, cue_text_weight, preview_cue_id

SQUEEZED_WORD_SECONDS = 0.05
MAX_REPAIRED_RUN_SECONDS = 5.0
_MIN_RATE_SAMPLES = 3
_FALLBACK_SECONDS_PER_WEIGHT = 0.06
PAUSE_SPLIT_SECONDS = 0.8
SENTENCE_END_CHARS = ".?!。？！…"
_CLOSING_QUOTES = "\"'”’»›』」）)]"

def _word_weight(word: Word) -> int:
    """Display-width weight of one word, the measure cue length also uses."""
    return max(1, cue_text_weight(word.text.strip()))


def _speech_rates(
    words: Sequence[Word],
) -> tuple[dict[str | None, float], float]:
    """Per-speaker and overall median seconds per weight unit.

    The median, not the mean: a word that swallowed a pause would drag an
    average up, and there is always one. Words below the squeeze threshold are
    left out of the measurement entirely — whatever produced them, their
    duration says nothing about how fast anybody speaks.
    """
    samples: dict[str | None, list[float]] = {}
    for word in words:
        duration = word.end - word.start
        if duration < SQUEEZED_WORD_SECONDS:
            continue
        samples.setdefault(word.speaker, []).append(duration / _word_weight(word))
    everyone = [value for values in samples.values() for value in values]
    overall = (
        median(everyone)
        if len(everyone) >= _MIN_RATE_SAMPLES
        else _FALLBACK_SECONDS_PER_WEIGHT
    )
    rates = {
        speaker: median(values) if len(values) >= _MIN_RATE_SAMPLES else overall
        for speaker, values in samples.items()
    }
    return rates, overall


def repair_squeezed_words(
    words: Sequence[Word], *, duration: float | None = None
) -> None:
    """Give back a plausible duration to words squeezed by a speaker change.

    The batch API keeps one monotonic timeline for the whole conversation, so
    it has no way to say two people spoke at once. When B cuts in, the rest of
    A's words are crushed into the sliver before B starts, each about 0.01 s
    long. Cue times are derived mechanically from word times — by the preview
    segmenter and by the draft agent's segmentation alike — so A's subtitle
    would flash past while A is still audibly speaking.

    The speaker change is the anchor, because the squeeze always ends at the
    moment the other speaker starts. Walking back along the interrupted
    speaker's own track from the last word before the change, every
    consecutive word shorter than ``SQUEEZED_WORD_SECONDS`` joins the run,
    which stops at the first word of ordinary length: a clean hand-off is left
    alone, and a short word in the middle of somebody's turn never touches a
    change point in the first place.

    The run is re-spread over the duration its text deserves at that speaker's
    measured pace, capped by ``MAX_REPAIRED_RUN_SECONDS``, by the speaker's own
    next word — nobody overlaps themselves — and by the audio duration. Both
    ends of every word in the run move: extending only the last word would let
    words inside one run overlap each other, and a cue cut mid-run would then
    publish two overlapping subtitles of one speaker, which this project
    treats as a defect.

    Ordinals never change, and the stream stops being sorted by start time.
    That is the repair, not a side effect: A really did speak past B's start.
    """
    if len({word.speaker for word in words}) < 2:
        return
    rates, overall = _speech_rates(words)
    for index in range(len(words) - 1):
        speaker = words[index].speaker
        if speaker is None or words[index + 1].speaker == speaker:
            continue
        run: list[int] = []
        for candidate in range(index, -1, -1):
            if words[candidate].speaker != speaker:
                continue
            if words[candidate].end - words[candidate].start >= SQUEEZED_WORD_SECONDS:
                break
            run.append(candidate)
        run.reverse()
        if not run:
            continue
        weights = [_word_weight(words[position]) for position in run]
        total = sum(weights)
        span_start = words[run[0]].start
        end = span_start + total * rates.get(speaker, overall)
        end = min(end, span_start + MAX_REPAIRED_RUN_SECONDS)
        own_next = next(
            (
                other.start
                for other in islice(words, index + 1, None)
                if other.speaker == speaker
            ),
            None,
        )
        if own_next is not None:
            end = min(end, own_next)
        if duration is not None:
            end = min(end, duration)
        if end <= max(span_start, words[run[-1]].end):
            continue
        covered = 0
        for position, weight in zip(run, weights):
            words[position].start = span_start + (end - span_start) * covered / total
            covered += weight
            words[position].end = span_start + (end - span_start) * covered / total


def qualified_silences(words: Sequence[Word]) -> list[tuple[int, float]]:
    """``(index, gap)`` pairs where a cut after ``words[index]`` is legal.

    A silence qualifies only when the whole conversation is quiet: the gap is
    measured from the furthest end of any word so far, so a speaker whose word
    spans the pause disqualifies it even when the next word starts late.
    """
    silences: list[tuple[int, float]] = []
    max_end = float("-inf")
    for index in range(len(words) - 1):
        max_end = max(max_end, words[index].end)
        gap = words[index + 1].start - max_end
        if gap >= PAUSE_SPLIT_SECONDS:
            silences.append((index, gap))
    return silences


@dataclass(frozen=True, slots=True)
class WordBatch:
    """One contiguous slice of the time-ordered word stream for one agent call."""

    words: tuple[Word, ...]
    # A forced end means no qualified silence existed anywhere in the search
    # window, so the boundary cuts through continuous speech and must be
    # revisited by junction repair.
    forced_end: bool = False


def split_word_batches(
    words: Sequence[Word], target_size: int, hard_limit: int | None = None
) -> list[WordBatch]:
    """Slice the full-track word stream at qualified silences.

    Words of everyone speaking at once must share a batch — the agent cannot
    resolve half of a conversation — so cuts land only on silences no word
    spans. The cut nearest ``target_size`` with the largest gap wins; when the
    window up to the target holds none, the search widens to ``hard_limit``
    (double the target) and takes the first qualified silence; only truly
    continuous speech forces a cut, marked for junction repair.
    """
    hard_limit = hard_limit or 2 * target_size
    silences = dict(qualified_silences(words))
    batches: list[WordBatch] = []
    start = 0
    total = len(words)
    while start < total:
        if total - start <= hard_limit:
            batches.append(WordBatch(tuple(words[start:total])))
            break
        window_low = start + max(0, target_size // 2 - 1)
        window_high = start + target_size - 1
        best: int | None = None
        for index in range(window_low, window_high + 1):
            if index in silences and (best is None or silences[index] > silences[best]):
                best = index
        if best is None:
            for index in range(window_high + 1, start + hard_limit - 1):
                if index in silences:
                    best = index
                    break
        if best is not None:
            batches.append(WordBatch(tuple(words[start:best + 1])))
            start = best + 1
        else:
            cut = start + hard_limit
            batches.append(WordBatch(tuple(words[start:cut]), forced_end=True))
            start = cut
    return batches


def dispatchable_word_count(words: Sequence[Word]) -> int:
    """How many leading words may be dispatched while the stream still runs.

    Everything after the last qualified silence is held back so a dispatched
    batch always ends on a real pause; the held-back tail joins a batch once
    more audio arrives or the stream ends.
    """
    silences = qualified_silences(words)
    return silences[-1][0] + 1 if silences else 0


def cues_from_words(
    words: Sequence[Word],
    *,
    max_cue_seconds: float,
    max_cue_chars: int,
) -> list[Cue]:
    """Mechanically segment words into provisional preview cues.

    Preview only: these cues stand in until the draft agent has segmented and
    translated the words, and never reach the final output. Each speaker is
    segmented on their own track, so the result legitimately contains
    overlapping cues when people talk over each other.

    Text length is width-weighted (CJK counts double), and either limit may
    run over while the other measure is still under half of its own limit —
    sparse slow speech keeps its long pause-free cue, and a dense quick
    remark keeps its text together.
    """
    tracks: dict[str | None, list[Word]] = {}
    for word in words:
        tracks.setdefault(word.speaker, []).append(word)
    collected: list[tuple[float, float, str | None, str, list[int]]] = []
    for speaker, track in tracks.items():
        current: list[Word] = []

        def flush() -> None:
            if not current:
                return
            source = "".join(word.text for word in current).strip()
            if source:
                collected.append((
                    current[0].start,
                    max(current[0].start, max(word.end for word in current)),
                    speaker,
                    source,
                    [current[0].ordinal, current[-1].ordinal],
                ))
            current.clear()

        for word in track:
            if current:
                gap = word.start - current[-1].end
                previous = current[-1].text.rstrip().rstrip(_CLOSING_QUOTES)
                combined = "".join(w.text for w in current) + word.text
                duration = word.end - current[0].start
                weight = cue_text_weight(combined.strip())
                if (
                    gap >= PAUSE_SPLIT_SECONDS
                    or (previous and previous[-1] in SENTENCE_END_CHARS)
                    or (duration > max_cue_seconds and 2 * weight >= max_cue_chars)
                    or (weight > max_cue_chars and 2 * duration >= max_cue_seconds)
                ):
                    flush()
            current.append(word)
        flush()
    collected.sort(key=lambda item: (item[0], item[1], item[2] or ""))
    return [
        Cue(
            preview_cue_id(position),
            start,
            end,
            source,
            metadata={"word_range": word_range},
            speaker=speaker,
        )
        for position, (start, end, speaker, source, word_range) in enumerate(
            collected, 1
        )
    ]


