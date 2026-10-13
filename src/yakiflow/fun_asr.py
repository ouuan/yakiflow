"""Fun-ASR file HTTP API and DashScope realtime SDK."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from .batch_transcription import BatchResult, BatchTranscriber
from .cloud import api_key, http_client, save_raw
from .models import Word
from .sdk_stream import SDKStreamTranscriber
from .transcription import normalize_source_language


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


class FunASRTranscriber(BatchTranscriber):
    name = "fun-asr"
    max_file_bytes = 1024**3

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
        self._policy = None

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

    def legacy_task_exists(self):
        return bool(
            self.db.get_checkpoint("fun_asr_task")
            or self.db.get_checkpoint("fun_asr_upload")
        )

    async def _get_policy(self, client):
        response = await self._request(
            client,
            "GET",
            self.settings.fun_asr.api_url.rstrip("/") + "/uploads",
            headers={
                "Authorization": "Bearer "
                + api_key(self.settings, self.name, "DASHSCOPE_API_KEY")
            },
            params={"action": "getPolicy", "model": self.settings.fun_asr.model},
        )
        return response.json()["data"]

    async def prepare_size_limit(self, audio):
        # An old task is already uploaded; resuming must only poll that task.
        if self.legacy_result_exists() or self.legacy_task_exists():
            return self.max_file_bytes
        client = self.client_factory()
        try:
            self._policy = await self._get_policy(client)
            return min(
                self.max_file_bytes, int(self._policy["max_file_size_mb"]) * 1024**2
            )
        finally:
            await asyncio.to_thread(client.close)

    async def transcribe_one(
        self, audio, *, chunk, cache_dir, namespace, diarize, on_progress
    ):
        raw = cache_dir / self.name / "result.json"
        if raw.exists():
            result = json.loads(raw.read_text())
        else:
            client = self.client_factory()
            try:
                result = await self._run_task(
                    client, audio, diarize, namespace=namespace, cache_dir=cache_dir
                )
                save_raw(cache_dir, self.name, "result.json", result)
            finally:
                await asyncio.to_thread(client.close)
        words = words_from_result(result, diarize=diarize)
        if not words and any(
            t.get("text", "").strip() for t in result.get("transcripts", [])
        ):
            raise RuntimeError("Fun-ASR response has text but no word timestamps")
        return BatchResult(words)

    async def _run_task(self, client, audio, diarize, *, namespace, cache_dir):
        base = self.settings.fun_asr.api_url.rstrip("/")
        headers = {
            "Authorization": "Bearer "
            + api_key(self.settings, self.name, "DASHSCOPE_API_KEY")
        }
        task = self.db.get_checkpoint(namespace + "_task")
        if task and (
            task["model"] != self.settings.fun_asr.model or task["api_url"] != base
        ):
            raise ValueError("Fun-ASR saved task uses a different model or API URL")
        if not task:
            upload = self.db.get_checkpoint(namespace + "_upload")
            if (
                not upload
                or upload["expires_at"] <= time.time()
                or upload["model"] != self.settings.fun_asr.model
            ):
                policy = self._policy or await self._get_policy(client)
                self._policy = None
                if audio.stat().st_size >= min(
                    self.max_file_bytes, int(policy["max_file_size_mb"]) * 1024**2
                ):
                    raise ValueError(
                        "chunk exceeds the Fun-ASR upload policy size limit"
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
                self.db.checkpoint(namespace + "_upload", upload)
                save_raw(cache_dir, self.name, "upload.json", upload)
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
            if not submitted.get("output", {}).get("task_id"):
                save_raw(cache_dir, self.name, "submitted.json", submitted)
                raise RuntimeError("Fun-ASR submission did not return a task ID")
            task = {
                "id": submitted["output"]["task_id"],
                "model": self.settings.fun_asr.model,
                "api_url": base,
            }
            self.db.checkpoint(namespace + "_task", task)
            save_raw(cache_dir, self.name, "submitted.json", submitted)
        while True:
            response = await self._request(
                client, "GET", base + "/tasks/" + task["id"], headers=headers
            )
            status = response.json()
            save_raw(cache_dir, self.name, "task.json", status)
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
