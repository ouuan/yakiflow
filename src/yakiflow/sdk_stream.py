"""Async adapter for SDKs that deliver segment results from callback threads."""

from __future__ import annotations

import asyncio
import wave
from dataclasses import replace
from pathlib import Path

from .cloud import save_raw
from .models import Cue, TranscriptEvent
from .transcription import Transcriber, merge_overlap


class SDKStreamTranscriber(Transcriber):
    continuous_feed = True

    def __init__(self, settings, work_dir, db, runner=None, *, session_factory=None):
        super().__init__()
        self.settings, self.work_dir, self.db = settings, work_dir, db
        self.session_factory = session_factory or self.make_session
        self.session = None
        self.queue = asyncio.Queue()
        self.loop = None
        self.base = 0.0
        self.fed_end = 0.0
        self.buffer_start = 0.0
        self.buffer = bytearray()
        self.session_number = 0
        self.reconnects = 0
        self._final = []
        self._partial = None

    def make_session(self, callback):
        raise NotImplementedError

    def enqueue(self, kind, payload=None):
        # Only asyncio's loop owns the queue, timeline and SQLite connection.
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (kind, payload))

    async def start_chunks(self):
        self.loop = asyncio.get_running_loop()
        self.db.checkpoint(
            "diarization_state", {"backend": self.name, "enabled": False}
        )

    async def _open(self, start):
        self.base = start
        self.session_number += 1
        # Late callbacks from a retired connection cannot land in its replacement.
        generation = self.session_number

        def callback(kind, payload=None):
            if generation == self.session_number:
                self.enqueue(kind, payload)

        self.session = self.session_factory(callback)
        await asyncio.to_thread(self.session.start)

    def _drain(self):
        ended = None
        while not self.queue.empty():
            kind, data = self.queue.get_nowait()
            save_raw(
                self.work_dir,
                self.name,
                f"event-{self.session_number}-{self.queue_event_id}.json",
                {"kind": kind, "base": self.base, "data": data},
            )
            self.queue_event_id += 1
            if kind in {"final", "partial"} and data and data.get("text"):
                cue = Cue(
                    str(len(self._final) + 1),
                    self.base + data["start"],
                    self.base + data["end"],
                    data["text"],
                )
                if kind == "partial":
                    self._partial = replace(
                        cue, id="sdk-partial", metadata={"transient": True}
                    )
                else:
                    self._partial = None
                    self._final = merge_overlap(self._final, [cue])
                    self._final = [
                        replace(item, id=str(i + 1))
                        for i, item in enumerate(self._final)
                    ]
                    self.db.checkpoint(
                        self.name + "_final",
                        [
                            {
                                "id": item.id,
                                "start": item.start,
                                "end": item.end,
                                "source": item.source,
                            }
                            for item in self._final
                        ],
                    )
                    # Retain two seconds of overlap and all uncommitted audio.
                    cut = max(self.buffer_start, min(self.fed_end, cue.end - 2))
                    n = int((cut - self.buffer_start) * 32000)
                    del self.buffer[:n]
                    self.buffer_start += n / 32000
            elif kind in {"error", "closed"}:
                if ended is None or kind == "error":
                    ended = str(data or kind)
            elif kind == "nomatch":
                self.db.log(self.name, "stderr", "NoMatch")
        self._chunk_timeline.cues = list(self._final)
        return ended

    queue_event_id = 0

    async def _recover(self, reason):
        self.reconnects += 1
        if self.reconnects > 3:
            raise RuntimeError(
                f"{self.name}: repeated connection failures: {reason}; final results preserved"
            )
        await self._warn(f"{self.name}: reconnecting after {reason}")
        if self.session is not None:
            try:
                await asyncio.wait_for(asyncio.to_thread(self.session.stop), 30)
            except Exception as exc:
                self.db.log(self.name, "stderr", f"retired session stop failed: {exc}")
            self._drain()
        self.session_number += 1
        self._partial = None
        await self._open(self.buffer_start)
        await self._send(bytes(self.buffer))

    async def _send(self, pcm):
        for offset in range(0, len(pcm), 3200):
            await asyncio.to_thread(self.session.send, pcm[offset : offset + 3200])
            await asyncio.sleep(0.1)

    async def submit_chunk(self, wav: Path, offset: float):
        with wave.open(str(wav), "rb") as audio:
            if (audio.getframerate(), audio.getsampwidth(), audio.getnchannels()) != (
                16000,
                2,
                1,
            ):
                raise ValueError(f"{self.name} requires mono 16 kHz 16-bit PCM")
            pcm = audio.readframes(audio.getnframes())
        previous = {(cue.start, cue.end, cue.source) for cue in self._final}
        if self.session is None:
            self.fed_end = offset
            self.buffer_start = offset
            await self._open(offset)
        failure = self._drain()
        if failure:
            await self._recover(failure)
        skip = max(0, round((self.fed_end - offset) * 32000))
        fresh = pcm[skip:]
        if fresh:
            self.buffer.extend(fresh)
            self.fed_end += len(fresh) / 32000
            try:
                await self._send(fresh)
            except Exception as exc:
                await self._recover(str(exc))
        failure = self._drain()
        if failure:
            await self._recover(failure)
        # Intermediate speech is a UI preview only. It is never persisted as a cue.
        result = [
            cue
            for cue in self._final
            if (cue.start, cue.end, cue.source) not in previous
        ]
        if self._partial:
            result.append(self._partial)
        return result

    async def close_chunks(self):
        if self.session is None:
            return
        try:
            error = None
            try:
                await asyncio.wait_for(asyncio.to_thread(self.session.stop), 60)
            except Exception as exc:
                error = exc
            await asyncio.sleep(0)  # land callback-thread handoffs queued by stop
            failure = self._drain()
            if error is not None:
                raise error
            if failure and failure != "closed":
                raise RuntimeError(f"{self.name} stopped with service error: {failure}")
        finally:
            self.session_number += 1
            self.session = None

    async def transcribe(
        self, audio, on_event=None, on_progress=None, *, resume_from=()
    ):
        self._final = [
            Cue(**item) for item in self.db.get_checkpoint(self.name + "_final", [])
        ]
        await self.start_chunks()
        start = max(0.0, max((cue.end for cue in self._final), default=0) - 2)
        self.fed_end = self.buffer_start = start
        await self._open(start)
        try:
            with wave.open(str(audio), "rb") as wav:
                if (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) != (
                    16000,
                    2,
                    1,
                ):
                    raise ValueError(f"{self.name} requires mono 16 kHz 16-bit PCM")
                wav.setpos(min(wav.getnframes(), int(start * 16000)))
                total = wav.getnframes() / 16000
                while pcm := wav.readframes(16000):
                    self.buffer.extend(pcm)
                    self.fed_end += len(pcm) / 32000
                    try:
                        await self._send(pcm)
                    except Exception as exc:
                        await self._recover(str(exc))
                    failure = self._drain()
                    if failure:
                        await self._recover(failure)
                    if on_event:
                        for cue in self._final:
                            if cue.id not in self._emitted:
                                self._emitted.add(cue.id)
                                await on_event(TranscriptEvent(cue))
                        if self._partial:
                            await on_event(TranscriptEvent(self._partial))
                    if on_progress:
                        await on_progress(min(0.99, self.fed_end / total))
        finally:
            await self.close_chunks()
        if on_event:
            for cue in self._final:
                await on_event(TranscriptEvent(cue, final=True))
        if on_progress:
            await on_progress(1)
        return self._final

    @property
    def _emitted(self):
        if not hasattr(self, "_emitted_ids"):
            self._emitted_ids = set()
        return self._emitted_ids
