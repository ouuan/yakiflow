import asyncio
import sys
from pathlib import Path

import pytest

from conftest import make_settings
from yakiflow.database import JobDatabase
from yakiflow.elevenlabs import (
    SQUEEZED_WORD_SECONDS,
    ElevenLabsTranscriber,
    cues_from_words,
    dispatchable_word_count,
    estimate_convert_seconds,
    normalize_speaker,
    qualified_silences,
    repair_squeezed_words,
    split_word_batches,
    words_from_response,
)
from yakiflow.models import Word


def _word(ordinal: int, start: float, end: float, text: str, speaker=None) -> Word:
    return Word(ordinal=ordinal, start=start, end=end, text=text, speaker=speaker)


def _settings(**overrides):
    values = {
        "source_language": "en",
        "target_language": "zh-CN",
        "transcription": {"backend": "elevenlabs"},
        "agent": {"backend": "codex"},
    }
    values.update(overrides)
    return make_settings(values).resolved()


# --- speaker normalization and word normalization ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("speaker_0", "0"), ("speaker_12", "12"), ("narrator", "narrator"), (None, None), ("", None)],
)
def test_normalize_speaker(raw, expected) -> None:
    assert normalize_speaker(raw) == expected


def test_words_from_response_folds_spacing_and_drops_audio_events() -> None:
    items = [
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4, "speaker_id": "speaker_1"},
        {"type": "spacing", "text": " ", "start": 0.4, "end": 0.5},
        {"type": "audio_event", "text": "(laughter)", "start": 0.5, "end": 1.0},
        {"type": "word", "text": "world", "start": 1.0, "end": 1.4, "speaker_id": "speaker_1", "logprob": -0.1},
    ]
    words = words_from_response(items)
    assert [word.text for word in words] == ["Hello ", "world"]
    assert [word.ordinal for word in words] == [0, 1]
    assert words[0].speaker == "1"
    assert words[1].logprob == -0.1
    assert "".join(word.text for word in words) == "Hello world"


def test_words_from_response_applies_the_chunk_offset() -> None:
    words = words_from_response(
        [{"type": "word", "text": "hi", "start": 1.0, "end": 1.5}], offset=100.0
    )
    assert (words[0].start, words[0].end) == (101.0, 101.5)


# --- squeezed-word repair ---


def _interrupted() -> list[Word]:
    """Speaker 1 at 0.1 s per weight unit, cut off mid-sentence by speaker 2.

    The three opening words are what the pace is measured from; the last four
    are the API's squeeze, each 0.01 s long, crushed into the sliver before
    the other speaker starts.
    """
    return [
        _word(0, 10.80, 11.14, "我觉得", "1"),
        _word(1, 11.14, 11.40, "这件", "1"),
        _word(2, 11.40, 11.64, "东西", "1"),
        _word(3, 11.64, 11.65, "事", "1"),
        _word(4, 11.65, 11.66, "情", "1"),
        _word(5, 11.66, 11.67, "很", "1"),
        _word(6, 11.67, 11.68, "重要", "1"),
        _word(7, 11.75, 12.30, "那个", "2"),
    ]


def test_repair_respreads_the_squeezed_run_at_the_speaker_pace() -> None:
    words = _interrupted()
    repair_squeezed_words(words)

    # Weight 10 ("事情很重要") at the measured 0.06 s per unit is 0.6 s.
    assert words[3].start == 11.64, "the run keeps the start it was given"
    assert words[6].end == pytest.approx(12.24)
    run = words[3:7]
    assert [word.end - word.start for word in run] == pytest.approx(
        [0.12, 0.12, 0.12, 0.24]
    )
    # Redistributed, not merely stretched at the end: the words inside the run
    # stay in order and never overlap each other.
    assert [word.start for word in run] == pytest.approx(
        [word.end for word in words[2:6]]
    )
    # The repair is the whole point: speaker 1 now runs past speaker 2's start.
    assert words[6].end > words[7].start
    assert [word.ordinal for word in words] == list(range(8))


def test_repair_never_reaches_the_speakers_own_next_word() -> None:
    words = _interrupted()
    words.append(_word(8, 12.00, 12.40, "所以", "1"))
    repair_squeezed_words(words)

    # 0.6 s would land at 12.24; nobody overlaps themselves.
    assert words[6].end == pytest.approx(12.00)
    assert [word.end - word.start for word in words[3:7]] == pytest.approx(
        [0.072, 0.072, 0.072, 0.144]
    )


def test_repair_is_capped_by_the_audio_duration() -> None:
    words = _interrupted()
    repair_squeezed_words(words, duration=12.0)

    assert words[6].end == pytest.approx(12.0)


def test_repair_leaves_a_single_speaker_transcript_alone() -> None:
    for speaker in ("1", None):
        words = [
            _word(0, 0.0, 0.40, "我觉得", speaker),
            _word(1, 0.40, 0.66, "这件", speaker),
            _word(2, 0.66, 0.90, "东西", speaker),
            _word(3, 0.90, 0.91, "很", speaker),
            _word(4, 0.91, 0.92, "重要", speaker),
        ]
        before = [(word.start, word.end) for word in words]
        repair_squeezed_words(words)
        assert [(word.start, word.end) for word in words] == before


def test_repair_skips_a_clean_hand_off() -> None:
    words = _interrupted()
    # The word before the change has an ordinary duration, so the hand-off was
    # clean and the short words earlier in the turn are none of this pass's
    # business.
    words[6] = _word(6, 11.67, 11.95, "重要", "1")
    before = [(word.start, word.end) for word in words]
    repair_squeezed_words(words)

    assert [(word.start, word.end) for word in words] == before


def test_repair_leaves_a_short_word_inside_a_turn_alone() -> None:
    words = [
        _word(0, 0.00, 0.34, "我觉得", "1"),
        _word(1, 0.34, 0.36, "这", "1"),  # a clipped word mid-turn
        _word(2, 0.40, 0.66, "件事", "1"),
        _word(3, 0.66, 0.90, "东西", "1"),
        _word(4, 1.00, 1.55, "那个", "2"),
    ]
    before = [(word.start, word.end) for word in words]
    repair_squeezed_words(words)

    assert [(word.start, word.end) for word in words] == before


def test_repair_threshold_is_exclusive() -> None:
    words = _interrupted()
    words[6] = _word(6, 11.67, 11.67 + SQUEEZED_WORD_SECONDS, "重要", "1")
    before = [(word.start, word.end) for word in words]
    repair_squeezed_words(words)

    # Exactly at the threshold still counts as a real duration, so the run
    # ends before it starts and nothing moves.
    assert [(word.start, word.end) for word in words] == before


def test_repair_gives_each_speaker_their_own_pace() -> None:
    words = [
        # Speaker 1 speaks at 0.1 s per weight unit, speaker 2 at 0.03.
        _word(0, 0.00, 0.60, "我觉得", "1"),
        _word(1, 0.60, 1.00, "这件", "1"),
        _word(2, 1.00, 1.40, "东西", "1"),
        _word(3, 1.40, 1.41, "很重要", "1"),
        _word(4, 1.50, 1.62, "不对", "2"),
        _word(5, 1.62, 1.74, "其实", "2"),
        _word(6, 1.74, 1.86, "这样", "2"),
        _word(7, 1.86, 1.87, "很重要", "2"),
        _word(8, 2.50, 2.70, "好吧", "1"),
    ]
    repair_squeezed_words(words)

    # The same six weight units of text, restored to each speaker's own pace.
    assert words[3].end - words[3].start == pytest.approx(0.6)
    assert words[7].end - words[7].start == pytest.approx(0.18)


def test_repair_opens_no_batch_cut_inside_the_crosstalk() -> None:
    words = _interrupted()
    words[7] = _word(7, 12.60, 13.10, "那个", "2")
    # Before the repair the squeeze leaves the whole conversation looking
    # silent from 11.68 to 12.60, which is a legal place to end an agent batch
    # — right in the middle of speaker 1's sentence.
    assert 6 in dict(qualified_silences(words))

    repair_squeezed_words(words)

    assert not any(3 <= index <= 6 for index, _gap in qualified_silences(words))


# --- mechanical preview segmentation ---


def test_cues_from_words_splits_on_pause_and_sentence_end() -> None:
    words = [
        _word(0, 0.0, 0.4, "First "),
        _word(1, 0.5, 0.9, "part. "),  # sentence end
        _word(2, 1.0, 1.4, "Second "),
        _word(3, 2.5, 2.9, "after"),  # 1.1 s pause: 2.5 - 1.4 >= 0.8
    ]
    cues = cues_from_words(words, max_cue_seconds=8.0, max_cue_chars=84)
    assert [cue.source for cue in cues] == ["First part.", "Second", "after"]
    assert cues[0].metadata["word_range"] == [0, 1]
    assert cues[0].id == "preview-1"


def test_cues_from_words_strips_closing_quotes_before_sentence_check() -> None:
    words = [
        _word(0, 0.0, 0.4, "He said "),
        _word(1, 0.5, 0.9, "“done.” "),
        _word(2, 1.0, 1.4, "Then"),
    ]
    cues = cues_from_words(words, max_cue_seconds=8.0, max_cue_chars=84)
    assert [cue.source for cue in cues] == ["He said “done.”", "Then"]


def test_cues_from_words_splits_on_duration_and_length_limits() -> None:
    long_words = [
        _word(index, index * 1.0, index * 1.0 + 0.9, f"w{index} ")
        for index in range(12)
    ]
    by_duration = cues_from_words(long_words, max_cue_seconds=5.0, max_cue_chars=30)
    assert len(by_duration) > 1
    assert all(cue.end - cue.start <= 5.0 for cue in by_duration)

    by_length = cues_from_words(long_words, max_cue_seconds=7.0, max_cue_chars=12)
    assert len(by_length) > 1
    assert all(len(cue.source) <= 12 for cue in by_length)


def test_cues_from_words_counts_wide_characters_double() -> None:
    words = [
        _word(index, index * 0.5, index * 0.5 + 0.4, "字") for index in range(8)
    ]
    # len("字" * 8) fits within 8, but the weighted width does not.
    cues = cues_from_words(words, max_cue_seconds=4.0, max_cue_chars=8)
    assert [cue.source for cue in cues] == ["字字字字", "字字字字"]


def test_cues_from_words_relaxes_one_limit_while_the_other_is_under_half() -> None:
    sparse = [
        _word(index, index * 2.0, index * 2.0 + 1.5, f"w{index} ")
        for index in range(3)
    ]
    # 5.5 s of slow speech, but the text is under half the character limit.
    slow = cues_from_words(sparse, max_cue_seconds=2.0, max_cue_chars=84)
    assert [cue.source for cue in slow] == ["w0 w1 w2"]

    dense = [
        _word(index, index * 0.2, index * 0.2 + 0.15, f"w{index} ")
        for index in range(6)
    ]
    # 17 characters of quick speech, but under half the duration limit.
    quick = cues_from_words(dense, max_cue_seconds=8.0, max_cue_chars=12)
    assert [cue.source for cue in quick] == ["w0 w1 w2 w3 w4 w5"]


def test_cues_from_words_keeps_overlapping_speaker_tracks() -> None:
    words = [
        _word(0, 0.0, 0.5, "one ", "1"),
        _word(1, 0.4, 0.9, "uh ", "2"),
        _word(2, 0.6, 1.1, "two", "1"),
        _word(3, 1.0, 1.5, "huh", "2"),
    ]
    cues = cues_from_words(words, max_cue_seconds=8.0, max_cue_chars=84)
    assert [(cue.speaker, cue.source) for cue in cues] == [
        ("1", "one two"),
        ("2", "uh huh"),
    ]
    # Sorted by start, and the overlap survives.
    assert cues[0].start <= cues[1].start
    assert cues[1].start < cues[0].end
    assert cues[0].metadata["word_range"] == [0, 2]
    assert cues[1].metadata["word_range"] == [1, 3]


# --- batch splitting ---


def test_qualified_silence_requires_the_whole_conversation_to_pause() -> None:
    words = [
        _word(0, 0.0, 5.0, "long ", "1"),  # spans the apparent gap below
        _word(1, 1.0, 1.5, "short", "2"),
        _word(2, 3.0, 3.5, "next", "2"),  # gap to previous *word* is 1.5 s
        _word(3, 6.5, 7.0, "after", "1"),  # 1.5 s after every track went quiet
    ]
    assert [index for index, _gap in qualified_silences(words)] == [2]


def test_split_word_batches_prefers_the_largest_silence_near_the_target() -> None:
    words = []
    time = 0.0
    for index in range(30):
        words.append(_word(index, time, time + 0.3, f"w{index} "))
        # Silences after words 9 (1.0 s) and 12 (2.0 s), inside the window
        # around a target of 14.
        time += 0.4 + (1.0 if index == 9 else 2.0 if index == 12 else 0.0)
    batches = split_word_batches(words, 14, hard_limit=28)
    assert batches[0].words[-1].ordinal == 12
    assert not batches[0].forced_end
    assert batches[0].words[0].ordinal == 0
    # No word is dropped or duplicated.
    seen = [word.ordinal for batch in batches for word in batch.words]
    assert seen == list(range(30))


def test_split_word_batches_widens_the_window_before_forcing() -> None:
    words = []
    time = 0.0
    for index in range(40):
        words.append(_word(index, time, time + 0.3, f"w{index} "))
        time += 0.4 + (1.0 if index == 25 else 0.0)  # only silence: after 25
    batches = split_word_batches(words, 10, hard_limit=30)
    # Window up to 10 held no silence; the widened search found index 25.
    assert batches[0].words[-1].ordinal == 25
    assert not batches[0].forced_end


def test_split_word_batches_forces_a_cut_through_continuous_speech() -> None:
    words = [
        _word(index, index * 0.4, index * 0.4 + 0.35, f"w{index} ")
        for index in range(50)
    ]
    batches = split_word_batches(words, 10, hard_limit=20)
    assert batches[0].forced_end
    assert len(batches[0].words) == 20
    seen = [word.ordinal for batch in batches for word in batch.words]
    assert seen == list(range(50))


def test_dispatchable_word_count_holds_back_the_tail() -> None:
    words = [
        _word(0, 0.0, 0.4, "a "),
        _word(1, 0.5, 0.9, "b "),
        _word(2, 2.0, 2.4, "c "),  # qualified silence after word 1
        _word(3, 2.5, 2.9, "d"),
    ]
    assert dispatchable_word_count(words) == 2
    assert dispatchable_word_count(words[:2]) == 0


# --- the batch transcriber ---


class FakeConvertResponse:
    def __init__(self, words, language_code="en"):
        self.words = words
        self.language_code = language_code


class FakeSpeechToText:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def convert(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.speech_to_text = FakeSpeechToText(outcomes)


class FakeApiError(Exception):
    def __init__(self, status_code, body=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


def _write_wav(path: Path, seconds: float = 2.0) -> None:
    import wave

    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * int(16000 * seconds))


def test_convert_estimate_follows_the_parallel_segment_split() -> None:
    # The API transcribes a request as ``min(4, ceil(duration / 480))``
    # internally parallel segments, so between 8 and 32 minutes the extra
    # audio arrives with the extra workers and the estimate stays flat.
    assert estimate_convert_seconds(480.0, 0) == estimate_convert_seconds(1920.0, 0)
    # Past four segments there is no further speedup: twice the audio, twice
    # the transcription time, with the same 2 s request overhead.
    saturated = estimate_convert_seconds(3840.0, 0)
    assert estimate_convert_seconds(7680.0, 0) == pytest.approx(2 * saturated - 2.0)


def test_convert_estimate_counts_the_upload() -> None:
    hour = 3600.0
    uploaded = estimate_convert_seconds(hour, 32000 * int(hour))
    assert uploaded > estimate_convert_seconds(hour, 0)
    # Still well under real time: the bar should not crawl as if the request
    # ran at the speed of the audio.
    assert uploaded < hour / 10


def test_convert_estimate_holds_a_floor_and_survives_unknown_duration() -> None:
    assert estimate_convert_seconds(2.0, 64000) == 15.0
    # The 600 s fallback takes 12 s across two segments at 25x speed;
    # adding 2 s of overhead still leaves it below the floor.
    assert estimate_convert_seconds(None, 0) == 15.0


def test_transcribe_sends_the_documented_parameters(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio)
    response = FakeConvertResponse([
        {"type": "word", "text": "hi", "start": 0.1, "end": 0.4, "speaker_id": "speaker_0"},
    ], language_code="pt-BR")
    client = FakeClient([response])
    db = JobDatabase(tmp_path / "job.sqlite3")
    settings = _settings(
        elevenlabs={"num-speakers": 3, "use-speaker-library": True}
    )
    transcriber = ElevenLabsTranscriber(
        settings, tmp_path, db, client_factory=lambda: client
    )

    preview = asyncio.run(transcriber.transcribe(audio))

    call = client.speech_to_text.calls[0]
    assert call["model_id"] == "scribe_v2"
    assert call["language_code"] == "en"
    assert call["diarize"] is True
    assert call["num_speakers"] == 3
    assert call["use_speaker_library"] is True
    assert call["tag_audio_events"] is False
    assert "file" in call
    assert db.get_checkpoint("detected_source_language") == "pt"
    stored = db.list_transcript_words()
    assert [(word.text, word.speaker) for word in stored] == [("hi", "0")]
    assert [cue.speaker for cue in preview] == ["0"]
    db.close()


def test_transcribe_omits_auto_language_and_unrequested_options(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio)
    client = FakeClient([FakeConvertResponse([])])
    db = JobDatabase(tmp_path / "job.sqlite3")
    settings = _settings(
        source_language="auto", elevenlabs={"use-speaker-library": False}
    )
    transcriber = ElevenLabsTranscriber(
        settings, tmp_path, db, client_factory=lambda: client
    )

    asyncio.run(transcriber.transcribe(audio))

    call = client.speech_to_text.calls[0]
    assert "language_code" not in call
    assert "num_speakers" not in call
    assert "use_speaker_library" not in call
    db.close()


def test_chunks_force_diarization_off(tmp_path: Path) -> None:
    audio = tmp_path / "chunk.wav"
    _write_wav(audio, 0.5)
    client = FakeClient([FakeConvertResponse([
        {"type": "word", "text": "hi", "start": 0.1, "end": 0.4, "speaker_id": "speaker_0"},
    ])])
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(
        _settings(elevenlabs={"num-speakers": 3, "use-speaker-library": True}),
        tmp_path,
        db,
        client_factory=lambda: client,
    )

    incoming = asyncio.run(transcriber.submit_chunk(audio, 30.0))

    call = client.speech_to_text.calls[0]
    assert call["diarize"] is False
    assert "num_speakers" not in call
    assert "use_speaker_library" not in call
    assert [cue.start for cue in incoming] == [30.1]
    db.close()


def test_raw_responses_are_saved_verbatim(tmp_path: Path) -> None:
    import json

    audio = tmp_path / "reference.wav"
    _write_wav(audio)
    word = {"type": "word", "text": "hi", "start": 11.44, "end": 11.44}
    client = FakeClient([
        FakeConvertResponse([dict(word, speaker_id="speaker_1")]),
        FakeConvertResponse([word]),
    ])
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(
        _settings(), tmp_path, db, client_factory=lambda: client
    )

    asyncio.run(transcriber.transcribe(audio))
    asyncio.run(transcriber.submit_chunk(audio, 30.0))

    full = json.loads((tmp_path / "stt-responses" / "transcribe.json").read_text())
    assert full["words"][0]["speaker_id"] == "speaker_1"
    assert full["words"][0]["end"] == 11.44
    chunk = json.loads((tmp_path / "stt-responses" / "chunk-30.00s.json").read_text())
    # Verbatim: the chunk offset is not folded into the stored timestamps.
    assert chunk["words"][0]["start"] == 11.44
    db.close()


def test_transcribe_persists_repaired_words(tmp_path: Path) -> None:
    import json

    audio = tmp_path / "reference.wav"
    _write_wav(audio, 3.0)
    client = FakeClient([FakeConvertResponse([
        {"type": "word", "text": "我觉得", "start": 0.80, "end": 1.14,
         "speaker_id": "speaker_1"},
        {"type": "word", "text": "这件", "start": 1.14, "end": 1.40,
         "speaker_id": "speaker_1"},
        {"type": "word", "text": "东西", "start": 1.40, "end": 1.64,
         "speaker_id": "speaker_1"},
        {"type": "word", "text": "很重要", "start": 1.64, "end": 1.65,
         "speaker_id": "speaker_1"},
        {"type": "word", "text": "那个", "start": 1.75, "end": 2.30,
         "speaker_id": "speaker_2"},
    ])])
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(
        _settings(), tmp_path, db, client_factory=lambda: client
    )

    preview = asyncio.run(transcriber.transcribe(audio))

    stored = db.list_transcript_words()
    assert stored[3].end == pytest.approx(2.0)
    # The preview cues are cut from the repaired words, so the interrupted
    # speaker's line now overlaps the interrupting one.
    assert [(cue.speaker, cue.end) for cue in preview] == [
        ("1", pytest.approx(2.0)), ("2", 2.30),
    ]
    assert preview[0].end > preview[1].start
    # The response on disk is still what the API said, squeeze and all.
    raw = json.loads((tmp_path / "stt-responses" / "transcribe.json").read_text())
    assert raw["words"][3]["end"] == 1.65
    db.close()


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_report_the_key_without_retrying(
    tmp_path: Path, status: int
) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio)
    client = FakeClient([FakeApiError(status)])
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(
        _settings(), tmp_path, db, client_factory=lambda: client
    )

    with pytest.raises(RuntimeError, match="API key"):
        asyncio.run(transcriber.transcribe(audio))
    assert len(client.speech_to_text.calls) == 1
    db.close()


def test_payload_too_large_is_not_retried(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio)
    client = FakeClient([FakeApiError(413)])
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(
        _settings(), tmp_path, db, client_factory=lambda: client
    )

    with pytest.raises(RuntimeError, match="too large"):
        asyncio.run(transcriber.transcribe(audio))
    assert len(client.speech_to_text.calls) == 1
    db.close()


def test_rate_limits_are_retried_with_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    audio = tmp_path / "reference.wav"
    _write_wav(audio)
    client = FakeClient([
        FakeApiError(429),
        FakeApiError(503),
        FakeConvertResponse([]),
    ])
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(
        _settings(), tmp_path, db, client_factory=lambda: client
    )

    asyncio.run(transcriber.transcribe(audio))

    assert len(client.speech_to_text.calls) == 3
    assert delays[:2] == [5.0, 10.0]
    db.close()


def test_overlong_audio_is_rejected_before_upload(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio, seconds=1.0)
    # Rewrite the header to claim an absurd duration without paying for it.
    import wave

    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(1)
        writer.writeframes(b"\x00\x00" * 40000)
    client = FakeClient([])
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(
        _settings(), tmp_path, db, client_factory=lambda: client
    )

    with pytest.raises(ValueError, match="limit"):
        asyncio.run(transcriber.transcribe(audio))
    assert client.speech_to_text.calls == []
    db.close()


def test_missing_sdk_is_reported_with_the_extra_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "elevenlabs.client", None)
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsTranscriber(_settings(), tmp_path, db)
    with pytest.raises(RuntimeError, match=r"yakiflow\[elevenlabs\]"):
        transcriber._make_client()
    db.close()


def test_transcript_words_round_trip(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    words = [
        Word(ordinal=0, start=0.0, end=0.5, text="hi ", speaker="1", logprob=-0.2),
        Word(ordinal=1, start=0.6, end=0.9, text="there", speaker=None),
    ]
    db.replace_transcript_words(words)
    assert db.list_transcript_words() == words

    db.append_transcript_words([
        Word(ordinal=1, start=0.6, end=1.0, text="there!", speaker="2"),
        Word(ordinal=2, start=1.2, end=1.5, text="ok", speaker="2"),
    ])
    stored = db.list_transcript_words()
    assert [word.text for word in stored] == ["hi ", "there!", "ok"]
    assert stored[1].speaker == "2"
    db.close()
