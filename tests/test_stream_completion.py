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
