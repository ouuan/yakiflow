"""Fun-ASR file HTTP API and DashScope realtime SDK."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from .cloud import api_key, effective_diarization, http_client, save_raw
from .media import pcm_audio_duration
from .models import TranscriptEvent, Word
from .sdk_stream import SDKStreamTranscriber
from .transcription import Transcriber, normalize_source_language
from .word_processing import cues_from_words


def words_from_result(data, *, diarize):
    words = []
    for transcript in data.get("transcripts", []):
        for sentence in transcript.get("sentences", []):
            speaker = sentence.get("speaker_id") if diarize else None
            for item in sentence.get("words", []):
                text = item.get("text", "") + item.get("punctuation", "")
                if text.strip():
                    words.append(
                        Word(
                            len(words),
                            item["begin_time"] / 1000,
                            item["end_time"] / 1000,
                            text,
                            str(speaker) if speaker is not None else None,
                        )
                    )
    return words


class FunASRTranscriber(Transcriber):
    name = "fun-asr"

    def __init__(
        self,
        settings,
        work_dir,
        db,
        runner=None,
        *,
        client_factory=None,
        poll_seconds=2,
    ):
        super().__init__()
        self.settings, self.work_dir, self.db = settings, work_dir, db
        self.client_factory = client_factory or (lambda: http_client("fun-asr"))
        self.poll_seconds = poll_seconds

    async def _request(self, client, method, url, **kwargs):
        response = await asyncio.to_thread(client.request, method, url, **kwargs)
        if response.status_code >= 400:
            # Do not include URLs (signed credentials) or request headers in errors.
            hint = (
                " (upload or result link may have expired)"
                if response.status_code in {403, 404}
                else ""
            )
            raise RuntimeError(
                f"Fun-ASR HTTP {response.status_code}{hint}; saved task is retained"
            )
        return response

    async def transcribe(
        self, audio, on_event=None, on_progress=None, *, resume_from=()
    ):
        duration = pcm_audio_duration(audio)
        if duration is None or duration <= 0 or duration > 12 * 3600:
            raise ValueError(
                "Fun-ASR requires a nonempty recording no longer than 12 hours; split the input manually"
            )
        if audio.stat().st_size > 1024**3:
            raise ValueError(
                "Fun-ASR temporary upload limit is 1 GB; split the input manually"
            )
        diarize = effective_diarization(
            self.db, self.name, audio, self.settings.fun_asr.diarize_max_minutes
        )
        raw = self.work_dir / self.name / "result.json"
        if raw.exists():
            result = json.loads(raw.read_text())
        else:
            client = self.client_factory()
            try:
                result = await self._run_task(client, audio, diarize)
                save_raw(self.work_dir, self.name, "result.json", result)
            finally:
                await asyncio.to_thread(client.close)
        words = words_from_result(result, diarize=diarize)
        if not words and any(
            t.get("text", "").strip() for t in result.get("transcripts", [])
        ):
            raise RuntimeError("Fun-ASR response has text but no word timestamps")
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

    async def _run_task(self, client, audio, diarize):
        base = self.settings.fun_asr.api_url.rstrip("/")
        headers = {
            "Authorization": "Bearer "
            + api_key(self.settings, self.name, "DASHSCOPE_API_KEY")
        }
        task = self.db.get_checkpoint("fun_asr_task")
        if task and (
            task["model"] != self.settings.fun_asr.model or task["api_url"] != base
        ):
            raise ValueError("Fun-ASR saved task uses a different model or API URL")
        if not task:
            upload = self.db.get_checkpoint("fun_asr_upload")
            if (
                not upload
                or upload["expires_at"] <= time.time()
                or upload["model"] != self.settings.fun_asr.model
            ):
                response = await self._request(
                    client,
                    "GET",
                    base + "/uploads",
                    headers=headers,
                    params={
                        "action": "getPolicy",
                        "model": self.settings.fun_asr.model,
                    },
                )
                policy = response.json()["data"]
                if audio.stat().st_size > int(policy["max_file_size_mb"]) * 1024**2:
                    raise ValueError(
                        "recording exceeds the Fun-ASR upload policy size limit; split manually"
                    )
                key = policy["upload_dir"].rstrip("/") + "/" + uuid.uuid4().hex + ".wav"
                form = {
                    "OSSAccessKeyId": policy["oss_access_key_id"],
                    "policy": policy["policy"],
                    "Signature": policy["signature"],
                    "key": key,
                    "x-oss-object-acl": policy["x_oss_object_acl"],
                    "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
                    "success_action_status": "200",
                }
                with audio.open("rb") as file:
                    await self._request(
                        client,
                        "POST",
                        policy["upload_host"],
                        data=form,
                        files={"file": (audio.name, file, "audio/wav")},
                    )
                upload = {
                    "url": "oss://" + key,
                    "model": self.settings.fun_asr.model,
                    "expires_at": time.time() + 48 * 3600,
                }
                self.db.checkpoint("fun_asr_upload", upload)
                save_raw(self.work_dir, self.name, "upload.json", upload)
            parameters = {"diarization_enabled": diarize}
            language = normalize_source_language(self.settings.source_language)
            if language:
                parameters["language_hints"] = [language]
            response = await self._request(
                client,
                "POST",
                base + "/services/audio/asr/transcription",
                headers={
                    **headers,
                    "X-DashScope-Async": "enable",
                    "X-DashScope-OssResourceResolve": "enable",
                },
                json={
                    "model": self.settings.fun_asr.model,
                    "input": {"file_urls": [upload["url"]]},
                    "parameters": parameters,
                },
            )
            submitted = response.json()
            save_raw(self.work_dir, self.name, "submitted.json", submitted)
            if not submitted.get("output", {}).get("task_id"):
                raise RuntimeError("Fun-ASR submission did not return a task ID")
            task = {
                "id": submitted["output"]["task_id"],
                "model": self.settings.fun_asr.model,
                "api_url": base,
            }
            self.db.checkpoint("fun_asr_task", task)
        while True:
            response = await self._request(
                client, "GET", base + "/tasks/" + task["id"], headers=headers
            )
            status = response.json()
            save_raw(self.work_dir, self.name, "task.json", status)
            output = status.get("output", {})
            state = output.get("task_status")
            if state == "SUCCEEDED":
                results = output.get("results", [])
                if len(results) != 1 or results[0].get("subtask_status") != "SUCCEEDED":
                    raise RuntimeError("Fun-ASR subtask failed: " + str(results))
                response = await self._request(
                    client, "GET", results[0]["transcription_url"]
                )
                return response.json()
            if state not in {"PENDING", "RUNNING"}:
                raise RuntimeError("Fun-ASR task failed: " + str(output or status))
            await asyncio.sleep(self.poll_seconds)

    async def submit_chunk(self, wav, offset):
        raise RuntimeError(
            "Fun-ASR batch cannot capture live audio; use fun-asr-stream"
        )


class _RecognitionSession:
    def __init__(self, settings, callback):
        try:
            import dashscope
            from dashscope.audio.asr import (
                Recognition,
                RecognitionCallback,
                RecognitionResult,
            )
        except ImportError:
            raise RuntimeError(
                "install yakiflow[fun-asr] for realtime transcription"
            ) from None

        class Callback(RecognitionCallback):
            def on_event(self, result):
                sentence = result.get_sentence()
                final = RecognitionResult.is_sentence_end(sentence)
                start = sentence.get("begin_time", 0) / 1000
                end = (sentence.get("end_time") or sentence.get("begin_time", 0)) / 1000
                callback(
                    "final" if final else "partial",
                    {
                        "text": sentence.get("text", ""),
                        "start": start,
                        "end": max(start, end),
                        "raw": sentence,
                    },
                )

            def on_error(self, result):
                callback("error", str(result))

            def on_close(self):
                callback("closed")

            def on_complete(self):
                callback("complete")

        dashscope.base_websocket_api_url = settings.fun_asr.realtime_url
        kwargs = {}
        language = normalize_source_language(settings.source_language)
        if language:
            kwargs["language_hints"] = [language]
        self.recognition = Recognition(
            model=settings.fun_asr.realtime_model,
            format="pcm",
            sample_rate=16000,
            callback=Callback(),
            heartbeat=True,
            api_key=api_key(settings, "fun-asr", "DASHSCOPE_API_KEY"),
            **kwargs,
        )

    def start(self):
        self.recognition.start()

    def send(self, pcm):
        self.recognition.send_audio_frame(pcm)

    def stop(self):
        self.recognition.stop()  # Blocks until final sentences arrive.


class FunASRRealtimeTranscriber(SDKStreamTranscriber):
    name = "fun-asr-stream"

    def make_session(self, callback):
        return _RecognitionSession(self.settings, callback)
