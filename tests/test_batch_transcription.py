import asyncio
import json
import wave
from itertools import pairwise
from pathlib import Path

import pytest
from conftest import make_settings
from test_elevenlabs import FakeClient, FakeConvertResponse
from test_fun_asr import DONE, POLICY, Client, Response
from test_fun_asr import RESULT as FUN_RESULT
from test_mai import RESULT as MAI_RESULT

from yakiflow.batch_transcription import (
    BatchChunk,
    BatchResult,
    merge_chunk_words,
    plan_chunks,
)
from yakiflow.cloud import effective_diarization
from yakiflow.config import load_settings, validate_run_settings
from yakiflow.database import JobDatabase
from yakiflow.elevenlabs import ElevenLabsTranscriber
from yakiflow.fun_asr import FunASRTranscriber
from yakiflow.mai import MAITranscriber
from yakiflow.models import Word

PROVIDERS = {
    "elevenlabs": ElevenLabsTranscriber,
    "fun-asr": FunASRTranscriber,
    "mai": MAITranscriber,
}


def wav(path, seconds, rate=100):
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"\0\0" * round(seconds * rate))
    return path


def settings(provider, **config):
    return make_settings(
        source_language="en",
        target_language="zh",
        agent={"backend": "codex"},
        transcription={"backend": provider},
        **{
            provider: {"region": "centralus", **config} if provider == "mai" else config
        },
    ).resolved()


@pytest.mark.parametrize(
    "provider,minutes", [("elevenlabs", 600), ("fun-asr", 720), ("mai", 120)]
)
def test_defaults_and_toml_cli_loading(tmp_path, provider, minutes):
    config = getattr(settings(provider), provider.replace("-", "_"))
    assert config.max_minutes == minutes
    assert config.chunk_overlap_seconds == 30
    if provider != "elevenlabs":
        assert config.diarize_over_limit == "disable"
    path = tmp_path / "config.toml"
    path.write_text(f"[{provider}]\nmax-minutes = 2.5\nchunk-overlap-seconds = 0.5\n")
    loaded = load_settings(
        user_file=Path("/missing"),
        project_file=path,
        cli_config={provider: {"max-minutes": 1.25}},
    )
    config = getattr(loaded, provider.replace("-", "_"))
    assert (config.max_minutes, config.chunk_overlap_seconds) == (1.25, 0.5)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "name,value",
    [
        ("max-minutes", 0),
        ("max-minutes", -1),
        ("max-minutes", float("inf")),
        ("max-minutes", float("nan")),
        ("max-minutes", True),
        ("max-minutes", "1"),
        ("chunk-overlap-seconds", -1),
        ("chunk-overlap-seconds", float("inf")),
        ("chunk-overlap-seconds", float("nan")),
        ("chunk-overlap-seconds", False),
        ("chunk-overlap-seconds", "0"),
    ],
)
def test_invalid_batch_configuration(provider, name, value):
    with pytest.raises(ValueError, match=name):
        validate_run_settings(
            settings(provider, **{name: value}), check_credentials=False
        )


@pytest.mark.parametrize("provider", ["fun-asr", "mai"])
def test_invalid_diarization_policy(provider):
    with pytest.raises(ValueError, match="diarize-over-limit"):
        validate_run_settings(
            settings(provider, **{"diarize-over-limit": "other"}),
            check_credentials=False,
        )


@pytest.mark.parametrize("frames,count", [(5999, 1), (6000, 2), (6001, 2), (19000, 6)])
@pytest.mark.parametrize("overlap", [0, 30])
def test_frame_exact_limits_and_complete_coverage(frames, count, overlap):
    chunks = plan_chunks(
        frames=frames,
        rate=100,
        bytes_per_frame=2,
        max_seconds=60,
        overlap_seconds=overlap,
        max_bytes=1_000_000,
    )
    if frames != 19000 or overlap == 30:
        assert len(chunks) == count
    assert chunks[0].start_frame == 0
    assert chunks[-1].end_frame == frames
    for chunk in chunks:
        assert 0 < chunk.end - chunk.start < 60
    for left, right in pairwise(chunks):
        assert right.start_frame == left.end_frame - overlap * 100
        assert right.end_frame > left.end_frame


def test_file_size_budget_and_fractional_limit():
    chunks = plan_chunks(
        frames=1000,
        rate=100,
        bytes_per_frame=4,
        max_seconds=60,
        overlap_seconds=0.25,
        max_bytes=1044,
    )
    assert chunks[0].end_frame == 249
    assert all(44 + (c.end_frame - c.start_frame) * 4 < 1044 for c in chunks)
    chunks = plan_chunks(
        frames=10,
        rate=100,
        bytes_per_frame=2,
        max_seconds=0.015,
        overlap_seconds=0,
        max_bytes=1000,
    )
    assert all(c.end_frame - c.start_frame == 1 for c in chunks)


def test_finite_extreme_values_report_overlap_or_use_size_budget():
    chunks = plan_chunks(
        frames=1000,
        rate=100,
        bytes_per_frame=2,
        max_seconds=1e308,
        overlap_seconds=0,
        max_bytes=1044,
    )
    assert chunks[0].end_frame == 499
    with pytest.raises(ValueError, match="chunk-overlap-seconds"):
        plan_chunks(
            frames=1000,
            rate=100,
            bytes_per_frame=2,
            max_seconds=60,
            overlap_seconds=1e308,
            max_bytes=1044,
        )


@pytest.mark.parametrize("overlap", [0.99, 1, 30])
def test_overlap_must_allow_at_least_one_frame_of_progress(overlap):
    with pytest.raises(ValueError, match="chunk-overlap-seconds"):
        plan_chunks(
            frames=500,
            rate=100,
            bytes_per_frame=2,
            max_seconds=1,
            overlap_seconds=overlap,
            max_bytes=10000,
        )


def stub_requests(transcriber, calls, *, fail_at=None, failure=RuntimeError):
    async def request(audio, *, chunk, cache_dir, namespace, diarize, on_progress):
        with wave.open(str(audio), "rb") as reader:
            calls.append(
                (
                    chunk.index,
                    diarize,
                    reader.getnframes(),
                    reader.getframerate(),
                    namespace,
                )
            )
        if chunk.index == fail_at:
            raise failure("interrupted")
        if on_progress:
            await on_progress(0.5)
        start = min(5.1, (chunk.end - chunk.start) / 2)
        return BatchResult(
            [
                Word(
                    0,
                    start,
                    start + 0.2,
                    f"word{chunk.index}",
                    "0" if diarize else None,
                )
            ],
            "en",
        )

    transcriber.transcribe_one = request

    async def size_limit(audio):
        return transcriber.max_file_bytes

    transcriber.prepare_size_limit = size_limit


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("seconds,count", [(59.99, 1), (60, 2), (60.01, 2)])
def test_duration_boundary_uses_common_batch_flow(tmp_path, provider, seconds, count):
    db = JobDatabase(tmp_path / "db")
    transcriber = PROVIDERS[provider](
        settings(provider, **{"max-minutes": 1}), tmp_path, db
    )
    calls = []
    stub_requests(transcriber, calls)
    asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", seconds)))
    assert len(calls) == count
    assert all(frames / rate < 60 for _, _, frames, rate, _ in calls)
    assert db.get_checkpoint(f"batch_plan_{provider}")["chunks"][-1][
        "end_frame"
    ] == round(seconds * 100)
    db.close()


@pytest.mark.parametrize("provider", ["fun-asr", "mai"])
@pytest.mark.parametrize(
    "mode,threshold,max_minutes,seconds,diarize,cap",
    [
        ("disable", 1, 0.5, 60, False, 30),  # short chunks must not re-enable it
        ("disable", 1, 0.5, 61, False, 30),
        ("disable", 1, 0.5, 59, True, 30),
        ("split", 1, 5, 60, True, 60),  # only the diarization cap binds
        ("split", 1, 0.5, 61, True, 30),  # both caps apply in one plan
        ("split", 0, 0.5, 61, False, 30),
        ("disable", 0, 0.5, 61, False, 30),
    ],
)
def test_whole_recording_diarization_decision(
    tmp_path, provider, mode, threshold, max_minutes, seconds, diarize, cap
):
    db = JobDatabase(tmp_path / "db")
    config = settings(
        provider,
        **{
            "max-minutes": max_minutes,
            "chunk-overlap-seconds": 5,
            "diarize-max-minutes": threshold,
            "diarize-over-limit": mode,
        },
    )
    transcriber = PROVIDERS[provider](config, tmp_path, db)
    calls = []
    stub_requests(transcriber, calls)
    asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", seconds)))
    assert len(calls) > 1
    assert all(
        enabled is diarize and frames / rate < cap
        for _, enabled, frames, rate, _ in calls
    )
    assert db.get_checkpoint("diarization_state")["enabled"] is diarize
    db.close()


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_resume_skips_complete_chunks_and_merges_once(tmp_path, provider, failure):
    db = JobDatabase(tmp_path / "db")
    config = settings(provider, **{"max-minutes": 1, "chunk-overlap-seconds": 5})
    audio = wav(tmp_path / "audio.wav", 130)
    transcriber = PROVIDERS[provider](config, tmp_path, db)
    calls = []
    stub_requests(transcriber, calls, fail_at=1, failure=failure)
    with pytest.raises(failure):
        asyncio.run(transcriber.transcribe(audio))
    assert db.list_transcript_words() == []
    saved = db.get_checkpoint(f"batch_plan_{provider}")
    resumed = PROVIDERS[provider](config, tmp_path, db)
    calls = []
    stub_requests(resumed, calls)
    events, progress = [], []

    async def event(value):
        assert len(calls) == len(saved["chunks"]) - 1
        events.append(value)

    async def advance(value):
        progress.append(value)

    result = asyncio.run(resumed.transcribe(audio, event, advance))
    assert [c[0] for c in calls] == list(range(1, len(saved["chunks"])))
    assert all(e.final for e in events)
    assert progress == sorted(progress) and progress[-1] == 1
    assert db.get_checkpoint("detected_source_language") == "en"
    words = db.list_transcript_words()
    assert [w.ordinal for w in words] == list(range(len(words)))
    assert words[-1].start == pytest.approx(
        saved["chunks"][-1]["start_frame"] / 100 + 5.1
    )
    reference_dir = tmp_path / "uninterrupted"
    reference_dir.mkdir()
    reference_db = JobDatabase(reference_dir / "db")
    reference = PROVIDERS[provider](config, reference_dir, reference_db)
    stub_requests(reference, [])
    assert asyncio.run(reference.transcribe(audio)) == result
    assert reference_db.list_transcript_words() == words
    reference_db.close()
    calls.clear()
    assert asyncio.run(resumed.transcribe(audio)) == result
    assert calls == []
    assert db.list_transcript_words() == words
    db.close()


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "change", [{"max-minutes": 2}, {"chunk-overlap-seconds": 0}, {"model": "different"}]
)
def test_changed_request_settings_rejected_on_resume(tmp_path, provider, change):
    db = JobDatabase(tmp_path / "db")
    config = {"max-minutes": 1, "chunk-overlap-seconds": 5}
    first = PROVIDERS[provider](settings(provider, **config), tmp_path, db)
    stub_requests(first, [])
    audio = wav(tmp_path / "audio.wav", 130)
    asyncio.run(first.transcribe(audio))
    resumed = PROVIDERS[provider](
        settings(provider, **{**config, **change}), tmp_path, db
    )
    with pytest.raises(ValueError, match="different request settings"):
        asyncio.run(resumed.transcribe(audio))
    db.close()


def w(start, text, speaker="0", end=None):
    return Word(99, start, start + 0.3 if end is None else end, text, speaker)


def merge(left, right, overlap=4):
    chunks = [
        BatchChunk(0, 0, 1000, 100),
        BatchChunk(1, round((10 - overlap) * 100), 1400, 100),
    ]
    return merge_chunk_words(chunks, [BatchResult(left), BatchResult(right)])


def test_time_offsets_punctuation_dedup_and_swapped_speakers():
    words = merge(
        [
            w(1, "lead", "A"),
            w(6.2, "Hello!", "A"),
            w(6.7, "world", "A"),
            w(7.3, "YES", "B"),
            w(7.7, "indeed.", "B"),
        ],
        [
            w(0.2, "hello", "B"),
            w(0.7, "WORLD?", "B"),
            w(1.3, "yes", "A"),
            w(1.7, "indeed", "A"),
            w(5, "tail-a", "B"),
            w(6, "tail-b", "A"),
        ],
    )
    assert [word.text for word in words] == [
        "lead",
        "Hello!",
        "world",
        "YES",
        "indeed.",
        "tail-a",
        "tail-b",
    ]
    assert [word.speaker for word in words] == ["0", "0", "0", "1", "1", "0", "1"]
    assert [word.start for word in words][-2:] == [11, 12]
    assert [word.ordinal for word in words] == list(range(7))


def test_midpoint_ownership_keeps_complete_boundary_words():
    words = merge(
        [w(7.6, "left", end=8.1), w(8.2, "discard-left")],
        [w(1.2, "discard-right"), w(1.9, "right", end=2.5)],
    )
    assert [(word.text, word.start, word.end) for word in words] == [
        ("left", 7.6, 8.1),
        ("right", 7.9, 8.5),
    ]


@pytest.mark.parametrize(
    "left,right",
    [
        ([w(6.2, "one")], [w(0.2, "one"), w(5, "tail")]),  # insufficient evidence
        (
            [w(6.2, "one", "A"), w(6.8, "two", "B")],
            [w(0.2, "one"), w(0.8, "two"), w(5, "tail")],
        ),  # tie
        (
            [w(6.2, "one", "A"), w(6.8, "two", "A"), w(7.4, "three", "B")],
            [w(0.2, "one"), w(0.8, "two"), w(1.4, "three"), w(5, "tail")],
        ),  # conflict
        (
            [
                w(6.2, "one", "A"),
                w(6.8, "two", "A"),
                w(7.4, "three", "A"),
                w(7.8, "four", "A"),
            ],
            [
                w(0.2, "one", "A"),
                w(0.8, "two", "A"),
                w(1.4, "three", "B"),
                w(1.8, "four", "B"),
                w(5, "tail", "A"),
            ],
        ),
    ],
)
def test_insufficient_ambiguous_or_conflicting_speakers_get_new_ids(left, right):
    words = merge(left, right)
    assert words[-1].speaker not in {word.speaker for word in words[:-1]}


def test_zero_overlap_does_not_infer_speaker_continuity():
    words = merge([w(1, "same")], [w(1, "same")], overlap=0)
    assert [(word.start, word.speaker) for word in words] == [(1, "0"), (11, "1")]


def test_simultaneous_same_text_from_different_speakers_is_not_collapsed():
    words = merge(
        [w(7, "yes", "A"), w(7, "yes", "B"), w(9, "later", "A")],
        [w(1, "yes", "A"), w(1, "yes", "B"), w(3, "later", "A"), w(3, "also", "B")],
    )
    assert len([word for word in words if word.text == "yes"]) == 2
    assert {word.text for word in words if word.start == 9} == {"later", "also"}


def test_overlap_spanning_three_chunks_deduplicates_once():
    chunks = [BatchChunk(i, i * 200, 1000 + i * 200, 100) for i in range(3)]
    words = merge_chunk_words(
        chunks, [BatchResult([w(9 - i * 2, "same")]) for i in range(3)]
    )
    assert [(word.start, word.text) for word in words] == [(9, "same")]


def test_fun_upload_policy_shortens_first_plan_and_checkpoints_are_isolated(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    # At 16 kHz, a 1 MiB upload policy binds well before a minute.
    policy = {"data": {**POLICY["data"], "max_file_size_mb": 1}}
    responses = [Response(policy)]
    for i in range(2):
        if i:
            responses.append(Response(policy))
        responses += [
            Response(),
            Response({"output": {"task_id": f"task-{i}"}}),
            Response(DONE),
            Response(FUN_RESULT),
        ]
    client = Client(responses)
    config = settings("fun-asr", **{"max-minutes": 1, "chunk-overlap-seconds": 0})
    transcriber = FunASRTranscriber(
        config, tmp_path, db, client_factory=lambda: client, poll_seconds=0
    )
    asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", 40, rate=16000)))
    plan = db.get_checkpoint("batch_plan_fun-asr")
    assert len(plan["chunks"]) == 2
    assert plan["chunks"][0]["end_frame"] == (1024**2 - 45) // 2
    assert client.requests[0][1].endswith("/uploads")
    for i in range(2):
        namespace = f"fun_asr_{plan['id']}_chunk_{i:06d}"
        assert db.get_checkpoint(namespace + "_task")["id"] == f"task-{i}"
        assert db.get_checkpoint(namespace + "_upload")
    assert db.get_checkpoint("fun_asr_task") is None
    db.close()


def test_fun_resume_polls_failed_chunk_without_resubmitting(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    config = settings("fun-asr", **{"max-minutes": 1, "chunk-overlap-seconds": 0})
    client = Client(
        [
            Response(POLICY),
            Response(),
            Response({"output": {"task_id": "first"}}),
            Response(DONE),
            Response(FUN_RESULT),
            Response(POLICY),
            Response(),
            Response({"output": {"task_id": "second"}}),
            Response(status=503),
        ]
    )
    audio = wav(tmp_path / "audio.wav", 70)
    transcriber = FunASRTranscriber(config, tmp_path, db, client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        asyncio.run(transcriber.transcribe(audio))
    resumed_client = Client([Response(DONE), Response(FUN_RESULT)])
    resumed = FunASRTranscriber(
        config, tmp_path, db, client_factory=lambda: resumed_client
    )
    asyncio.run(resumed.transcribe(audio))
    assert len(resumed_client.requests) == 2
    assert resumed_client.requests[0][1].endswith("/tasks/second")
    assert all(method == "GET" for method, *_ in resumed_client.requests)
    assert len(db.list_transcript_words()) == 2
    db.close()


def test_mai_size_limit_splits_whole_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECH_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    client = Client([Response(MAI_RESULT)] * 3)
    transcriber = MAITranscriber(
        settings("mai", **{"chunk-overlap-seconds": 0}),
        tmp_path,
        db,
        client_factory=lambda: client,
    )
    # Exercise the real per-request guard with a small limit, without a 250 MB fixture.
    transcriber.max_file_bytes = 1044
    asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", 10)))
    plan = db.get_checkpoint("batch_plan_mai")
    assert len(plan["chunks"]) == 3
    assert len(client.requests) == 3
    assert all(
        44 + (c["end_frame"] - c["start_frame"]) * 2 < 1044 for c in plan["chunks"]
    )
    db.close()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_invalid_overlap_fails_before_any_chunk_upload(tmp_path, provider):
    db = JobDatabase(tmp_path / "db")
    transcriber = PROVIDERS[provider](
        settings(provider, **{"max-minutes": 0.5}), tmp_path, db
    )
    calls = []
    stub_requests(transcriber, calls)
    with pytest.raises(ValueError, match="chunk-overlap-seconds"):
        asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", 50)))
    assert calls == []
    assert db.get_checkpoint(f"batch_plan_{provider}") is None
    assert db.get_checkpoint("diarization_state") is None
    db.close()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_changed_audio_rejected_even_when_duration_matches(tmp_path, provider):
    db = JobDatabase(tmp_path / "db")
    transcriber = PROVIDERS[provider](settings(provider), tmp_path, db)
    stub_requests(transcriber, [])
    audio = wav(tmp_path / "audio.wav", 2)
    asyncio.run(transcriber.transcribe(audio))
    data = bytearray(audio.read_bytes())
    data[-1] = 1
    audio.write_bytes(data)
    with pytest.raises(ValueError, match="different audio"):
        asyncio.run(transcriber.transcribe(audio))
    db.close()


@pytest.mark.parametrize("provider", ["fun-asr", "mai"])
def test_legacy_whole_file_cache_is_imported_without_request(tmp_path, provider):
    directory = tmp_path / provider
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps(FUN_RESULT if provider == "fun-asr" else MAI_RESULT)
    )
    db = JobDatabase(tmp_path / "db")
    transcriber = PROVIDERS[provider](
        settings(provider),
        tmp_path,
        db,
        client_factory=lambda: pytest.fail("legacy cache must skip HTTP"),
    )
    asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", 2)))
    assert len(db.list_transcript_words()) == 1
    db.close()


def test_elevenlabs_legacy_cache_and_repair_after_merge(tmp_path):
    db = JobDatabase(tmp_path / "db")
    left = [
        {"text": text, "start": start, "end": start, "speaker_id": "speaker_0"}
        for text, start in [("one", 35), ("two", 40)]
    ]
    right = [
        {"text": text, "start": start, "end": start, "speaker_id": "speaker_1"}
        for text, start in [("one!", 5.01), ("two?", 10.01)]
    ]
    right.append({"text": "tail", "start": 35, "end": 35.2, "speaker_id": "speaker_1"})
    client = FakeClient([FakeConvertResponse(left), FakeConvertResponse(right)])
    transcriber = ElevenLabsTranscriber(
        settings("elevenlabs", **{"max-minutes": 1}),
        tmp_path,
        db,
        client_factory=lambda: client,
    )
    finish = transcriber.finish_words
    merged = []

    def check_before_repair(words, duration):
        merged.extend(words)
        assert len(words) == 3
        assert words[0].start == words[0].end == 35
        assert words[1].start == words[1].end == 40
        finish(words, duration)

    transcriber.finish_words = check_before_repair
    asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", 70)))
    assert len(merged) == 3
    for cache in (tmp_path / "elevenlabs" / "batch").glob("*/chunk-*/words.json"):
        words = json.loads(cache.read_text())["words"]
        assert words[0]["start"] == words[0]["end"]
    db.close()

    legacy_dir = tmp_path / "legacy"
    raw_dir = legacy_dir / "stt-responses"
    raw_dir.mkdir(parents=True)
    (raw_dir / "transcribe.json").write_text(
        json.dumps({"words": left, "language_code": "en"})
    )
    db = JobDatabase(legacy_dir / "db")
    legacy = ElevenLabsTranscriber(
        settings("elevenlabs"),
        legacy_dir,
        db,
        client_factory=lambda: pytest.fail("legacy cache must skip SDK"),
    )
    asyncio.run(legacy.transcribe(wav(legacy_dir / "audio.wav", 50)))
    assert len(db.list_transcript_words()) == 2
    db.close()


def test_zero_threshold_overrides_legacy_enabled_decision(tmp_path):
    db = JobDatabase(tmp_path / "db")
    db.checkpoint(
        "diarization_state", {"backend": "mai", "enabled": True, "duration": 1}
    )
    assert not effective_diarization(db, "mai", Path("unused"), 0)
    assert not db.get_checkpoint("diarization_state")["enabled"]
    db.close()


def test_fun_policy_can_reject_overlap_before_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    client = Client([Response({"data": {**POLICY["data"], "max_file_size_mb": 1}})])
    transcriber = FunASRTranscriber(
        settings("fun-asr"), tmp_path, db, client_factory=lambda: client
    )
    with pytest.raises(ValueError, match="chunk-overlap-seconds"):
        asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", 40, rate=32000)))
    assert len(client.requests) == 1 and client.requests[0][0] == "GET"
    assert db.get_checkpoint("batch_plan_fun-asr") is None
    assert db.get_checkpoint("diarization_state") is None
    db.close()


@pytest.mark.parametrize("provider", ["fun-asr", "mai"])
def test_legacy_result_cannot_be_mixed_into_split_plan(tmp_path, provider):
    directory = tmp_path / provider
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps(FUN_RESULT if provider == "fun-asr" else MAI_RESULT)
    )
    db = JobDatabase(tmp_path / "db")
    transcriber = PROVIDERS[provider](
        settings(provider, **{"max-minutes": 1}),
        tmp_path,
        db,
        client_factory=lambda: pytest.fail("legacy result must skip HTTP"),
    )
    with pytest.raises(ValueError, match="whole-file result or task"):
        asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav", 70)))
    assert db.get_checkpoint(f"batch_plan_{provider}") is None
    db.close()
