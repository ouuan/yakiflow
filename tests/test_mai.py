import asyncio
import json
import sys
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import make_settings
from test_fun_asr import Client, Response, wav

from yakiflow.config import validate_run_settings
from yakiflow.database import JobDatabase
from yakiflow.mai import MAIRealtimeTranscriber, MAITranscriber, _SpeechSession
from yakiflow.transcription import word_level_backend


def settings(**groups):
    return make_settings(
        source_language="auto",
        target_language="zh",
        transcription={"backend": "mai"},
        agent={"backend": "codex"},
        mai={"region": "swedencentral"},
        **groups,
    ).resolved()


RESULT = {
    "phrases": [
        {
            "text": "hello",
            "speaker": 2,
            "words": [
                {
                    "text": "hello",
                    "offsetMilliseconds": 100,
                    "durationMilliseconds": 300,
                }
            ],
        }
    ]
}


@pytest.mark.parametrize(
    "duration,threshold,enabled",
    [
        (1, 0, False),
        (899, 15, True),
        (900, 15, False),
        (901, 15, False),
        (900, 16, True),
    ],
)
def test_rest_word_timestamps_and_diarization_boundary(
    tmp_path, monkeypatch, duration, threshold, enabled
):
    from yakiflow import cloud, mai

    monkeypatch.setenv("SPEECH_KEY", "test")
    monkeypatch.setattr(cloud, "pcm_audio_duration", lambda _: duration)
    monkeypatch.setattr(mai, "pcm_audio_duration", lambda _: duration)
    db = JobDatabase(tmp_path / "db")
    configured = make_settings(
        source_language="en",
        mai={"region": "swedencentral", "diarize_max_minutes": threshold},
    ).resolved()
    client = Client([Response(RESULT)])
    transcriber = MAITranscriber(
        configured, tmp_path, db, client_factory=lambda: client
    )
    preview = asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav")))
    request = client.requests[0]
    definition = json.loads(request[2]["data"]["definition"])
    assert definition["diarization"]["enabled"] is enabled
    assert definition["enhancedMode"]["modelOptions"]["timestamps"] == "word"
    assert definition["enhancedMode"]["model"] == "MAI-Transcribe-2"
    assert definition["locales"] == ["en"]
    assert request[2]["headers"] == {"Ocp-Apim-Subscription-Key": "test"}
    assert "api-version=2025-10-15" in request[1]
    assert preview[0].speaker == ("2" if enabled else None)
    word = db.list_transcript_words()[0]
    assert (word.start, word.end) == (0.1, 0.4)
    assert db.get_checkpoint("diarization_state")["enabled"] is enabled
    resumed = MAITranscriber(
        configured,
        tmp_path,
        db,
        client_factory=lambda: pytest.fail("cached REST result"),
    )
    assert asyncio.run(resumed.transcribe(tmp_path / "audio.wav")) == preview
    assert client.closed
    db.close()


@pytest.mark.parametrize("duration", [7200, 7201, 0])
def test_rest_duration_limit_prevents_request(tmp_path, monkeypatch, duration):
    monkeypatch.setattr("yakiflow.mai.pcm_audio_duration", lambda _: duration)
    db = JobDatabase(tmp_path / "db")
    transcriber = MAITranscriber(
        settings(),
        tmp_path,
        db,
        client_factory=lambda: pytest.fail("must validate before HTTP"),
    )
    with pytest.raises(ValueError, match="shorter than 2 hours"):
        asyncio.run(transcriber.transcribe(wav(tmp_path / "audio.wav")))
    db.close()


def test_rest_file_size_limit_and_service_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECH_KEY", "test")
    db = JobDatabase(tmp_path / "db")
    transcriber = MAITranscriber(
        settings(), tmp_path, db, client_factory=lambda: Client([Response(status=503)])
    )
    audio = wav(tmp_path / "audio.wav")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        asyncio.run(transcriber.transcribe(audio))
    monkeypatch.setattr(Path, "stat", lambda _: SimpleNamespace(st_size=250 * 1024**2))
    with pytest.raises(ValueError, match="250 MB"):
        asyncio.run(transcriber.transcribe(audio))
    db.close()


def test_mai_config_mapping_region_environment_and_credentials(monkeypatch):
    monkeypatch.setenv("SPEECH_KEY", "test")
    monkeypatch.setenv("SPEECH_REGION", "centralus")
    configured = make_settings(
        source_language="auto",
        target_language="zh",
        transcription={"backend": "mai"},
        agent={"backend": "codex"},
    ).resolved()
    assert configured.mai.region == "centralus"
    assert configured.transcription.stream_backend == "mai-stream"
    assert configured.alignment.backend == "none"
    assert word_level_backend("mai")
    assert not word_level_backend("mai-stream")
    validate_run_settings(configured)
    monkeypatch.delenv("SPEECH_REGION")
    with pytest.raises(ValueError, match="SPEECH_REGION"):
        validate_run_settings(
            make_settings(
                source_language="auto",
                target_language="zh",
                transcription={"backend": "mai"},
                agent={"backend": "codex"},
            ).resolved()
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_run_settings(
            make_settings(
                source_language="auto",
                target_language="zh",
                mai={"diarize_max_minutes": -1},
                agent={"backend": "codex"},
            ).resolved()
        )


def fake_sdk(monkeypatch):
    class Signal:
        def connect(self, callback):
            self.callback = callback

        def emit(self, event):
            self.callback(event)

    class Recognizer:
        def __init__(self, **kwargs):
            self.config = kwargs["speech_config"]
            for name in ("recognizing", "recognized", "canceled", "session_stopped"):
                setattr(self, name, Signal())

        def start_continuous_recognition_async(self):
            return SimpleNamespace(get=lambda: None)

        def stop_continuous_recognition_async(self):
            return SimpleNamespace(get=lambda: None)

    class Input:
        def __init__(self, **kwargs):
            self.pcm = bytearray()
            self.on_close = lambda: None

        def write(self, pcm):
            self.pcm.extend(pcm)

        def close(self):
            self.on_close()

    sdk = SimpleNamespace(
        SpeechConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        SpeechRecognizer=Recognizer,
        ResultReason=SimpleNamespace(NoMatch="NoMatch", RecognizedSpeech="Speech"),
        CancellationReason=SimpleNamespace(Error="Error"),
        audio=SimpleNamespace(
            PushAudioInputStream=Input,
            AudioStreamFormat=lambda **kw: kw,
            AudioConfig=lambda **kw: kw,
        ),
    )
    cognitive = SimpleNamespace(speech=sdk)
    monkeypatch.setitem(
        sys.modules, "azure", SimpleNamespace(cognitiveservices=cognitive)
    )
    monkeypatch.setitem(sys.modules, "azure.cognitiveservices", cognitive)
    monkeypatch.setitem(sys.modules, "azure.cognitiveservices.speech", sdk)
    return sdk


def test_speech_sdk_thread_callbacks_final_tail_nomatch_and_close(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SPEECH_KEY", "test")
    fake_sdk(monkeypatch)
    db = JobDatabase(tmp_path / "db")
    sessions = []

    def factory(callback):
        session = _SpeechSession(settings(), callback)
        sessions.append(session)

        def on_close():
            event = SimpleNamespace(
                result=SimpleNamespace(
                    reason="Speech",
                    offset=1_000_000,
                    duration=timedelta(seconds=0.1),
                    text="final tail",
                    json='{"Text":"final tail"}',
                )
            )
            session.recognizer.recognized.emit(event)
            session.recognizer.session_stopped.emit(None)

        session.input.on_close = on_close
        return session

    transcriber = MAIRealtimeTranscriber(
        settings(), tmp_path, db, session_factory=factory
    )

    async def run():
        await transcriber.start_chunks()
        await transcriber.submit_chunk(wav(tmp_path / "audio.wav", 0.1), 0)
        session = sessions[0]
        partial = SimpleNamespace(
            result=SimpleNamespace(
                reason="Speech", offset=0, duration=500_000, text="preview", json="{}"
            )
        )

        def events():
            session.recognizer.recognizing.emit(partial)
            session.recognizer.recognized.emit(
                SimpleNamespace(result=SimpleNamespace(reason="NoMatch", json="{}"))
            )

        thread = threading.Thread(target=events)
        thread.start()
        thread.join()
        await asyncio.sleep(0)
        transcriber._drain()
        assert transcriber.chunk_cues == []
        assert transcriber._partial.source == "preview"
        await transcriber.close_chunks()

    asyncio.run(run())
    assert [(cue.start, cue.end, cue.source) for cue in transcriber.chunk_cues] == [
        (0.1, 0.2, "final tail")
    ]
    assert sessions[0].recognizer.config.model == "MAI-Transcribe-2-Streaming"
    assert db.get_checkpoint("diarization_state")["enabled"] is False
    assert len(sessions[0].input.pcm) == 3200
    assert db.get_checkpoint("mai-stream_final")[0]["source"] == "final tail"
    db.close()


def test_speech_sdk_cancellation_reports_service_error(monkeypatch):
    monkeypatch.setenv("SPEECH_KEY", "test")
    fake_sdk(monkeypatch)
    events = []
    session = _SpeechSession(settings(), lambda *event: events.append(event))
    session.recognizer.canceled.emit(
        SimpleNamespace(reason="Error", error_code="Denied", error_details="bad region")
    )
    with pytest.raises(RuntimeError, match="bad region"):
        session.stop()
    assert events == [("error", "Denied: bad region")]


def test_mai_doctor_checks_selected_sdk_and_region(monkeypatch):
    from yakiflow.doctor import run_doctor

    monkeypatch.setenv("SPEECH_KEY", "test")
    monkeypatch.setattr("yakiflow.doctor.shutil.which", lambda _: None)
    batch = run_doctor(settings(), source_is_url=False)
    stream = run_doctor(settings(stream=True), source_is_url=False)
    assert "mai httpx" in {check.name for check in batch}
    assert "mai azure.cognitiveservices.speech" in {check.name for check in stream}
    assert next(check for check in stream if check.name == "mai region/endpoint").ok


def test_word_normalization_preserves_phrase_spacing_and_punctuation():
    from yakiflow.mai import words_from_result

    for text, tokens in [
        ("Hello world!", ["hello", "world"]),
        ("你好世界！", ["你好", "世界"]),
        ("你好，world！", ["你好", "world"]),
    ]:
        response = {
            "phrases": [
                {
                    "text": text,
                    "words": [
                        {
                            "text": token,
                            "offsetMilliseconds": i * 100,
                            "durationMilliseconds": 100,
                        }
                        for i, token in enumerate(tokens)
                    ],
                }
            ]
        }
        assert (
            "".join(
                word.text for word in words_from_result(response, diarize=False)
            ).strip()
            == text
        )
