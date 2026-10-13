import asyncio
from pathlib import Path

import pytest

from conftest import make_settings
from yakiflow.job import StreamExit, YakiFlowJob
from yakiflow.media import MediaArtifact, MediaSource
from yakiflow.models import Cue, Word


def job_at(tmp_path):
    settings = make_settings(
        source_language="en", target_language="zh", stream=True,
        transcription={"backend": "elevenlabs"},
        agent={"backend": "codex", "stream": {"batch_size": 3}},
        work_dir=tmp_path / "work", memory=tmp_path / "memory.md",
    )
    job = YakiFlowJob("input.wav", settings)
    artifact = MediaArtifact(MediaSource.parse("input.wav"), tmp_path / "reference.wav")
    return job, artifact


def test_exit_leaves_choice_unanswered_and_resume_asks_again(tmp_path):
    job, artifact = job_at(tmp_path)
    job.db.upsert_cues([Cue("1", 1, 2, "hello")])
    async def choose(available):
        assert available
        return "exit"
    job.stream_completion_listener = choose
    with pytest.raises(StreamExit):
        asyncio.run(job._complete_stream(artifact))
    assert job.db.get_checkpoint("stream_completion_choice") is None
    job.close()
    resumed = YakiFlowJob.from_workdir(tmp_path / "work")
    async def review(available):
        assert available
        return "review"
    resumed.stream_completion_listener = review
    asyncio.run(resumed._complete_stream(artifact))
    assert resumed.settings.agent.draft.batch_size == 3
    assert resumed.db.get_checkpoint("transcribed")
    cues = resumed.db.list_cues()
    assert asyncio.run(resumed._alignment_stage(artifact, cues)) == cues
    resumed.close()


def test_no_transcript_cannot_enter_review(tmp_path):
    job, artifact = job_at(tmp_path)
    async def choose(available):
        assert not available
        return "review"
    job.stream_completion_listener = choose
    with pytest.raises(ValueError, match="completion choice"):
        asyncio.run(job._complete_stream(artifact))
    job.close()


def test_noninteractive_retranscription_checks_before_reset_and_resumes_once(tmp_path, monkeypatch):
    from yakiflow import doctor
    from yakiflow.doctor import Check
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    job, artifact = job_at(tmp_path)
    job.db.upsert_cues([Cue("1", 0, 1, "live", "translated")])
    job.db.replace_transcript_words([Word(0, 0, 1, "live")])
    job.db.checkpoint("word_coverage", [0])
    monkeypatch.setattr(doctor, "run_doctor", lambda *args, **kwargs: [Check("SDK", False, "missing")])
    with pytest.raises(RuntimeError, match="resource check"):
        asyncio.run(job._complete_stream(artifact))
    assert job.db.list_cues()[0].translated == "translated"
    assert job.db.get_checkpoint("stream_completion_choice") is None
    monkeypatch.setattr(doctor, "run_doctor", lambda *args, **kwargs: [])
    asyncio.run(job._complete_stream(artifact))
    assert job.db.list_cues() == []
    assert job.db.list_transcript_words() == []
    assert job.db.get_checkpoint("word_coverage") is None
    assert job.settings.transcription.backend == "elevenlabs"
    assert job.settings.stream is False
    job.db.upsert_cues([Cue("1", 0, 1, "batch")])
    job.close()
    resumed = YakiFlowJob.from_workdir(tmp_path / "work")
    asyncio.run(resumed._complete_stream(artifact))
    assert resumed.db.list_cues()[0].source == "batch"
    resumed.close()


def test_review_resume_uses_cached_audio_without_loading_capture_model(tmp_path, monkeypatch):
    job, artifact = job_at(tmp_path)
    artifact.audio_path.write_bytes(b"cached")
    job.db.add_artifact("reference_audio", artifact.audio_path)
    job.db.upsert_cues([Cue("1", 0, 1, "live", "translated")])
    job.db.checkpoint("stream_completion_choice", "review")
    job.close()
    resumed = YakiFlowJob.from_workdir(tmp_path / "work")
    async def no_model():
        pytest.fail("review of live results must not load the capture model")
    monkeypatch.setattr(resumed, "_ensure_model", no_model)
    async def transcription(_artifact): return resumed.db.list_cues()
    async def translation(_artifact, cues): return list(cues)
    monkeypatch.setattr(resumed, "_transcription_stage", transcription)
    monkeypatch.setattr(resumed, "_translation_stage", translation)
    monkeypatch.setattr(resumed, "_post_alignment_translation_stage", translation)
    monkeypatch.setattr(resumed, "_publish", lambda *_: [])
    asyncio.run(resumed.run())
    assert resumed.db.get_checkpoint("stream_keep_timing")
    resumed.close()


@pytest.mark.parametrize("stop_error", [False, True])
def test_live_sdk_partial_is_preview_only_and_final_tail_is_preserved(tmp_path, monkeypatch, stop_error):
    from dataclasses import replace
    from yakiflow import job as job_module
    job, artifact = job_at(tmp_path)
    job.settings = replace(job.settings, transcription=replace(job.settings.transcription, backend="mai-stream"))
    events = []
    async def listener(event): events.append(event)
    job.listener = listener
    class Transcriber:
        continuous_feed = True
        chunk_cues = []
        async def start_chunks(self): pass
        async def submit_chunk(self, wav, offset):
            return [Cue("sdk-partial", 0, .1, "temporary", metadata={"transient": True})]
        async def close_chunks(self):
            self.chunk_cues = [Cue("1", 0, .2, "final tail")]
            if stop_error: raise RuntimeError("service stopped")
    class Acquirer:
        def __init__(self, *args, **kwargs): pass
        async def acquire_stream(self, source, on_chunk, **kwargs):
            await on_chunk(artifact.audio_path, 0)
            assert job.db.list_cues() == []
            assert any(event.cue and event.cue.source == "temporary" for event in events)
            return artifact
    monkeypatch.setattr(job_module, "make_transcriber", lambda *args: Transcriber())
    monkeypatch.setattr(job_module, "MediaAcquirer", Acquirer)
    if stop_error:
        with pytest.raises(RuntimeError, match="service stopped"):
            asyncio.run(job._media_stage())
    else:
        asyncio.run(job._media_stage())
    assert [cue.source for cue in job.db.list_cues(stable_only=True)] == ["final tail"]
    assert "final tail" in (job.work_dir / "input.live.incomplete.ass").read_text()
    job.close()
