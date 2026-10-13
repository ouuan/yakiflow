import asyncio
import json
import threading
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_settings
from yakiflow.cloud import api_key, api_key_source, effective_diarization
from yakiflow.config import load_settings, validate_run_settings
from yakiflow.database import JobDatabase
from yakiflow.fun_asr import FunASRTranscriber, FunASRRealtimeTranscriber
from yakiflow.models import Cue


def wav(path, seconds=2):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * int(seconds * 16000))
    return path


def settings(**groups):
    return make_settings(source_language="en", target_language="zh",
        transcription={"backend": "fun-asr"}, agent={"backend": "codex"}, **groups).resolved()


class Response:
    def __init__(self, data=None, status=200):
        self.data, self.status_code = data, status
    def json(self):
        return self.data


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False
    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return next(self.responses)
    def close(self):
        self.closed = True


POLICY = {"data": {"upload_dir": "directory", "upload_host": "https://upload.test",
    "max_file_size_mb": 100, "oss_access_key_id": "id", "policy": "policy", "signature": "signature",
    "x_oss_object_acl": "private", "x_oss_forbid_overwrite": "true"}}
RESULT = {"transcripts": [{"text": "Hello", "sentences": [{"speaker_id": 0,
    "words": [{"begin_time": 100, "end_time": 500, "text": "Hello", "punctuation": "."}]}]}]}
DONE = {"output": {"task_status": "SUCCEEDED", "results": [
    {"subtask_status": "SUCCEEDED", "transcription_url": "https://result.test"}]}}


def test_http_upload_submit_poll_and_cached_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    client = Client([Response(POLICY), Response(), Response({"output": {"task_id": "task"}}),
                     Response({"output": {"task_status": "RUNNING"}}), Response(DONE), Response(RESULT)])
    transcriber = FunASRTranscriber(settings(), tmp_path, db, client_factory=lambda: client, poll_seconds=0)
    preview = asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav")))
    assert preview[0].speaker == "0"
    assert db.list_transcript_words()[0].text == "Hello."
    submit = client.requests[2][2]
    assert submit["headers"]["X-DashScope-OssResourceResolve"] == "enable"
    assert submit["json"]["parameters"] == {"diarization_enabled": True, "language_hints": ["en"]}
    assert submit["json"]["input"]["file_urls"][0].startswith("oss://directory/")
    assert "headers" not in client.requests[-1][2]
    assert db.get_checkpoint("fun_asr_task")["id"] == "task"
    assert client.closed
    resumed = FunASRTranscriber(settings(), tmp_path, db, client_factory=lambda: pytest.fail("cached result"))
    assert asyncio.run(resumed.transcribe(tmp_path / "audio.wav")) == preview
    db.close()


def test_resume_polls_existing_task_without_uploading(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    db.checkpoint("fun_asr_task", {"id": "existing", "model": "fun-asr", "api_url": settings().fun_asr.api_url})
    client = Client([Response(DONE), Response(RESULT)])
    transcriber = FunASRTranscriber(settings(), tmp_path, db, client_factory=lambda: client)
    asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav")))
    assert client.requests[0][1].endswith("/tasks/existing")
    assert all(method == "GET" for method, *_ in client.requests)
    db.close()


@pytest.mark.parametrize("response", [Response({"output": {"task_status": "FAILED"}}),
    Response({"output": {"task_status": "SUCCEEDED", "results": [{"subtask_status": "FAILED"}]}}),
    Response(status=404)])
def test_task_subtask_and_expired_link_errors_keep_task(tmp_path, monkeypatch, response):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    db.checkpoint("fun_asr_task", {"id": "existing", "model": "fun-asr", "api_url": settings().fun_asr.api_url})
    transcriber = FunASRTranscriber(settings(), tmp_path, db, client_factory=lambda: Client([response]))
    with pytest.raises(RuntimeError):
        asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav")))
    assert db.get_checkpoint("fun_asr_task")["id"] == "existing"
    db.close()


@pytest.mark.parametrize("minutes,duration,enabled", [(0, 1, False), (1, 59, True), (1, 60, False),
    (1, 61, False), (120, 7199, True), (120, 7200, False), (120, 7201, False)])
def test_diarization_boundary_persists(tmp_path, monkeypatch, minutes, duration, enabled):
    from yakiflow import cloud
    monkeypatch.setattr(cloud, "pcm_audio_duration", lambda _: duration)
    db = JobDatabase(tmp_path / "db")
    assert effective_diarization(db, "fun-asr", Path("audio"), minutes) is enabled
    assert effective_diarization(db, "fun-asr", Path("audio"), 0 if enabled else 999) is enabled
    assert db.get_checkpoint("diarization_state")["duration"] == duration
    db.close()


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_threshold_rejected(value, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_run_settings(settings(fun_asr={"diarize_max_minutes": value}))


def test_config_mapping_credentials_and_key_slot_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env")
    assert settings().transcription.stream_backend == "fun-asr-stream"
    assert settings().alignment.backend == "none"
    assert api_key(settings(fun_asr={"api_key": "config"}), "fun-asr", "DASHSCOPE_API_KEY") == "env"
    config = tmp_path / "config.toml"
    config.write_text('[fun-asr]\napi-key-file="missing"\n')
    loaded = load_settings(user_file=Path('/missing'), project_file=config,
                           cli_config={"fun-asr": {"api-key": "cli"}})
    assert loaded.fun_asr.api_key_file is None
    assert api_key(loaded, "fun-asr", "DASHSCOPE_API_KEY") == "cli"
    assert api_key_source(loaded, "fun-asr", "DASHSCOPE_API_KEY") == "api_key"
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_settings(user_file=Path('/missing'), project_file=Path('/missing'),
            cli_config={"fun-asr": {"api-key": "a", "api-key-command": "b"}})


def test_sdk_callback_thread_partial_final_stop_and_duplicate_merge(tmp_path):
    db = JobDatabase(tmp_path / "db")
    class Session:
        def __init__(self, callback): self.callback = callback
        def start(self): pass
        def send(self, pcm):
            self.callback("partial", {"start": 0, "end": .1, "text": "preview"})
            self.callback("final", {"start": 0, "end": .1, "text": "hello"})
        def stop(self):
            self.callback("final", {"start": .1, "end": .2, "text": "tail"})
            self.callback("closed")
    transcriber = FunASRRealtimeTranscriber(settings(), tmp_path, db, session_factory=Session)
    async def run():
        await transcriber.start_chunks()
        await transcriber.submit_chunk(wav(tmp_path / "part.wav", .2), 0)
        await transcriber.close_chunks()
    asyncio.run(run())
    assert [cue.source for cue in transcriber.chunk_cues] == ["hello", "tail"]
    assert all(item["source"] != "preview" for item in db.get_checkpoint("fun-asr-stream_final"))
    assert db.get_checkpoint("diarization_state")["enabled"] is False
    assert list((tmp_path / "fun-asr-stream").glob("event-*.json"))
    db.close()


def test_sdk_disconnect_replays_uncommitted_pcm(tmp_path):
    db = JobDatabase(tmp_path / "db")
    sessions = []
    class Session:
        def __init__(self, callback):
            self.callback, self.frames = callback, []
            sessions.append(self)
        def start(self): pass
        def send(self, pcm):
            self.frames.append(pcm)
            if len(sessions) == 1: raise ConnectionError("disconnected")
            self.callback("final", {"start": 0, "end": .1, "text": "recovered"})
        def stop(self): self.callback("closed")
    transcriber = FunASRRealtimeTranscriber(settings(), tmp_path, db, session_factory=Session)
    async def run():
        await transcriber.start_chunks()
        await transcriber.submit_chunk(wav(tmp_path / "part.wav", .1), 0)
        await transcriber.close_chunks()
    asyncio.run(run())
    assert sessions[1].frames == sessions[0].frames
    assert [cue.source for cue in transcriber.chunk_cues] == ["recovered"]
    db.close()


def test_real_sdk_adapter_normalizes_callbacks(tmp_path, monkeypatch):
    import sys
    from yakiflow.fun_asr import _RecognitionSession
    instances = []
    class Recognition:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)
        def start(self): pass
        def send_audio_frame(self, pcm): self.pcm = pcm
        def stop(self): pass
    fake = SimpleNamespace(Recognition=Recognition, RecognitionCallback=object,
        RecognitionResult=SimpleNamespace(is_sentence_end=lambda sentence: sentence["sentence_end"]))
    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "dashscope.audio", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "dashscope.audio.asr", fake)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    events = []
    session = _RecognitionSession(settings(), lambda *event: events.append(event))
    callback = instances[0].kwargs["callback"]
    result = SimpleNamespace(get_sentence=lambda: {"begin_time": 100, "end_time": 300,
        "text": "hello", "sentence_end": True})
    thread = threading.Thread(target=lambda: callback.on_event(result))
    thread.start(); thread.join()
    assert events[0][0] == "final"
    assert events[0][1]["start"] == .1
    assert events[0][1]["end"] == .3
    assert events[0][1]["raw"]["sentence_end"]
    assert instances[0].kwargs["model"] == "fun-asr-realtime"
    assert instances[0].kwargs["language_hints"] == ["en"]


def test_file_and_command_credentials_do_not_fall_through(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    key = tmp_path / "key"
    key.write_text("file-key\n")
    assert api_key(settings(fun_asr={"api_key_file": key}), "fun-asr", "DASHSCOPE_API_KEY") == "file-key"
    monkeypatch.setattr("yakiflow.cloud.subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="command-key\n"))
    assert api_key(settings(fun_asr={"api_key_command": "secret-tool"}), "fun-asr", "DASHSCOPE_API_KEY") == "command-key"
    key.write_text("")
    with pytest.raises(ValueError, match="empty"):
        api_key(settings(fun_asr={"api_key_file": key}), "fun-asr", "DASHSCOPE_API_KEY")


def test_doctor_checks_only_selected_fun_dependency(monkeypatch):
    from yakiflow.doctor import run_doctor
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    monkeypatch.setattr("yakiflow.doctor.shutil.which", lambda _: None)
    batch = run_doctor(settings(), source_is_url=False)
    realtime = run_doctor(replace(settings(), stream=True), source_is_url=False)
    assert "fun-asr httpx" in {check.name for check in batch}
    assert "fun-asr dashscope" not in {check.name for check in batch}
    assert "fun-asr dashscope" in {check.name for check in realtime}
    assert "fun-asr httpx" not in {check.name for check in realtime}
