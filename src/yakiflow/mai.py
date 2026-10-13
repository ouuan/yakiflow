"""MAI file transcription REST API and Azure Speech realtime SDK."""

from __future__ import annotations

import asyncio
import json
import threading

from .cloud import api_key, effective_diarization, http_client, save_raw
from .media import pcm_audio_duration
from .models import TranscriptEvent, Word
from .sdk_stream import SDKStreamTranscriber
from .transcription import Transcriber, normalize_source_language
from .word_processing import cues_from_words


def words_from_result(data, *, diarize):
    words = []
    for phrase in data.get("phrases", []):
        speaker = phrase.get("speaker") if diarize else None
        text = phrase.get("text", "")
        cursor = 0
        phrase_start = len(words)
        for item in phrase.get("words", []):
            if not item.get("text", "").strip():
                continue
            token = item["text"]
            position = text.casefold().find(token.casefold(), cursor)
            if position >= 0:
                if len(words) > phrase_start:
                    words[-1].text += text[cursor:position]
                token = text[position : position + len(token)]
                cursor = position + len(token)
            elif (
                len(words) > phrase_start
                and token[0].isascii()
                and words[-1].text[-1].isascii()
            ):
                words[-1].text += " "
            start = item["offsetMilliseconds"] / 1000
            words.append(
                Word(
                    len(words),
                    start,
                    start + item["durationMilliseconds"] / 1000,
                    token,
                    str(speaker) if speaker is not None else None,
                )
            )
        if len(words) > phrase_start and cursor:
            words[-1].text += text[cursor:]
        if len(words) > phrase_start:
            words[-1].text += " "
    return words


class MAITranscriber(Transcriber):
    name = "mai"

    def __init__(self, settings, work_dir, db, runner=None, *, client_factory=None):
        super().__init__()
        self.settings, self.work_dir, self.db = settings, work_dir, db
        self.client_factory = client_factory or (lambda: http_client("mai"))

    async def transcribe(
        self, audio, on_event=None, on_progress=None, *, resume_from=()
    ):
        duration = pcm_audio_duration(audio)
        if duration is None or duration <= 0 or duration >= 7200:
            raise ValueError(
                "MAI REST requires a nonempty recording shorter than 2 hours; split manually"
            )
        if audio.stat().st_size >= 250_000_000:
            raise ValueError(
                "MAI REST requires files smaller than 250 MB; split manually"
            )
        diarize = effective_diarization(
            self.db, self.name, audio, self.settings.mai.diarize_max_minutes
        )
        raw = self.work_dir / self.name / "result.json"
        if raw.exists():
            result = json.loads(raw.read_text())
        else:
            endpoint = (
                self.settings.mai.endpoint
                or f"https://{self.settings.mai.region}.api.cognitive.microsoft.com"
            )
            if "/speechtotext/" not in endpoint:
                endpoint = (
                    endpoint.rstrip("/")
                    + "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
                )
            definition = {
                "enhancedMode": {
                    "enabled": True,
                    "model": self.settings.mai.model,
                    "modelOptions": {"timestamps": "word"},
                },
                "diarization": {"enabled": diarize},
            }
            language = normalize_source_language(self.settings.source_language)
            if language:
                definition["locales"] = [language]
            save_raw(self.work_dir, self.name, "request.json", definition)
            client = self.client_factory()
            try:
                with audio.open("rb") as file:
                    response = await asyncio.to_thread(
                        client.request,
                        "POST",
                        endpoint,
                        headers={
                            "Ocp-Apim-Subscription-Key": api_key(
                                self.settings, "mai", "SPEECH_KEY"
                            )
                        },
                        data={"definition": json.dumps(definition)},
                        files={"audio": (audio.name, file, "audio/wav")},
                    )
                if response.status_code >= 400:
                    raise RuntimeError(f"MAI REST HTTP {response.status_code}")
                result = response.json()
                save_raw(self.work_dir, self.name, "result.json", result)
            finally:
                await asyncio.to_thread(client.close)
        words = words_from_result(result, diarize=diarize)
        if not words and any(
            phrase.get("text", "").strip() for phrase in result.get("phrases", [])
        ):
            raise RuntimeError("MAI response has text but no word timestamps")
        self.db.replace_transcript_words(words)
        preview = cues_from_words(
            words,
            max_cue_seconds=self.settings.subtitles.max_cue_seconds,
            max_cue_chars=self.settings.subtitles.max_cue_chars,
        )
        if on_event:
            for cue in preview:
                await on_event(TranscriptEvent(cue, final=True))
        if on_progress:
            await on_progress(1)
        return preview

    async def submit_chunk(self, wav, offset):
        raise RuntimeError("MAI batch cannot capture live audio; use mai-stream")


def _seconds(value):
    # SDK offsets are 100 ns ticks; duration may be exposed as timedelta.
    return (
        value.total_seconds() if hasattr(value, "total_seconds") else value / 10_000_000
    )


class _SpeechSession:
    def __init__(self, settings, callback):
        try:
            import azure.cognitiveservices.speech as sdk
        except ImportError:
            raise RuntimeError(
                "install yakiflow[mai] for realtime transcription"
            ) from None
        config = sdk.SpeechConfig(
            subscription=api_key(settings, "mai", "SPEECH_KEY"),
            endpoint=f"wss://{settings.mai.region}.stt.speech.microsoft.com/speech/universal/v2",
        )
        config.model = settings.mai.realtime_model
        language = normalize_source_language(settings.source_language)
        if language:
            config.speech_recognition_language = language
        self.input = sdk.audio.PushAudioInputStream(
            stream_format=sdk.audio.AudioStreamFormat(
                samples_per_second=16000, bits_per_sample=16, channels=1
            )
        )
        self.recognizer = sdk.SpeechRecognizer(
            speech_config=config, audio_config=sdk.audio.AudioConfig(stream=self.input)
        )
        self.done = threading.Event()
        self.error = None

        def result(event, final):
            item = event.result
            if final and item.reason == sdk.ResultReason.NoMatch:
                callback("nomatch", getattr(item, "json", ""))
                return
            if not final or item.reason == sdk.ResultReason.RecognizedSpeech:
                start = _seconds(item.offset)
                callback(
                    "final" if final else "partial",
                    {
                        "start": start,
                        "end": start + _seconds(item.duration),
                        "text": item.text,
                        "raw": getattr(item, "json", ""),
                    },
                )

        def canceled(event):
            if event.reason == sdk.CancellationReason.Error:
                self.error = f"{event.error_code}: {event.error_details}"
                callback("error", self.error)
            else:
                callback("closed")
            self.done.set()

        def stopped(event):
            callback("closed")
            self.done.set()

        self.recognizer.recognizing.connect(lambda event: result(event, False))
        self.recognizer.recognized.connect(lambda event: result(event, True))
        self.recognizer.canceled.connect(canceled)
        self.recognizer.session_stopped.connect(stopped)

    def start(self):
        self.recognizer.start_continuous_recognition_async().get()

    def send(self, pcm):
        self.input.write(pcm)

    def stop(self):
        self.input.close()
        # Closing input lets the service finalize its last segment before stop.
        completed = self.done.wait(45)
        self.recognizer.stop_continuous_recognition_async().get()
        if self.error:
            raise RuntimeError(f"MAI realtime canceled: {self.error}")
        if not completed:
            raise TimeoutError("MAI realtime did not finalize after input closed")


class MAIRealtimeTranscriber(SDKStreamTranscriber):
    name = "mai-stream"

    def make_session(self, callback):
        return _SpeechSession(self.settings, callback)
