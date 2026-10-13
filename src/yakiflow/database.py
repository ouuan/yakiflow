from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from .models import Cue, JobStatus, Word, utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
-- Every process-log line and progress checkpoint is its own committed
-- transaction, and under WAL the default FULL setting makes each one an fsync
-- on the event loop. NORMAL still survives a crash of this process, which is
-- the failure the resume machinery is built for.
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, input TEXT NOT NULL, status TEXT NOT NULL,
    config_json TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS media_artifacts (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cues (
    id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, start REAL NOT NULL,
    end REAL NOT NULL, source TEXT NOT NULL, translated TEXT,
    timing_confidence REAL, metadata_json TEXT NOT NULL DEFAULT '{}',
    speaker TEXT,
    stable INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS transcript_words (
    ordinal INTEGER PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL,
    speaker TEXT, text TEXT NOT NULL, logprob REAL,
    stable INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS translation_batches (
    id INTEGER PRIMARY KEY, cue_ids_json TEXT NOT NULL,
    backend TEXT NOT NULL, model TEXT, status TEXT NOT NULL,
    request_json TEXT, response_json TEXT, error TEXT,
    created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
    name TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS process_logs (
    id INTEGER PRIMARY KEY, process TEXT NOT NULL, stream TEXT NOT NULL,
    line TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


class JobDatabase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def create_job(self, job_id: str, input_value: str, config: dict[str, Any]) -> None:
        now = utc_now()
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO jobs VALUES (?, ?, ?, ?, NULL, ?, ?)",
                (job_id, input_value, JobStatus.CREATED, json.dumps(config, default=str), now, now),
            )

    def job(self) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM jobs ORDER BY created_at LIMIT 1").fetchone()

    def set_status(self, status: JobStatus | str, error: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=?",
                (str(status), error, utc_now()),
            )

    def add_artifact(self, kind: str, path: Path) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO media_artifacts(kind,path,created_at) VALUES(?,?,?)",
                (kind, str(path), utc_now()),
            )

    def artifact(self, kind: str) -> Path | None:
        row = self.connection.execute(
            "SELECT path FROM media_artifacts WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()
        return Path(row[0]) if row else None

    def upsert_cues(self, cues: Sequence[Cue], *, stable: bool = True, reset_order: bool = False) -> None:
        with self.connection:
            self._upsert_cues(cues, stable=stable, reset_order=reset_order)

    def _upsert_cues(
        self, cues: Sequence[Cue], *, stable: bool = True, reset_order: bool = False
    ) -> None:
        """Write cues inside the caller's transaction."""
        sql = """
        INSERT INTO cues(id,ordinal,start,end,source,translated,
          timing_confidence,metadata_json,speaker,stable)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          ordinal=excluded.ordinal,start=excluded.start,end=excluded.end,
          source=excluded.source,
          translated=COALESCE(excluded.translated,cues.translated),
          timing_confidence=excluded.timing_confidence,metadata_json=excluded.metadata_json,
          speaker=excluded.speaker,
          stable=excluded.stable
        """
        if reset_order:
            ordinals = list(range(len(cues)))
        else:
            # Look up only the cues being written. Reading every row here made
            # the incremental one-cue-at-a-time transcription path quadratic.
            ids = [cue.id for cue in cues]
            placeholders = ",".join("?" * len(ids))
            existing = {
                row["id"]: row["ordinal"]
                for row in self.connection.execute(
                    f"SELECT id,ordinal FROM cues WHERE id IN ({placeholders})",
                    tuple(ids),
                )
            } if ids else {}
            highest = self.connection.execute(
                "SELECT MAX(ordinal) FROM cues"
            ).fetchone()[0]
            next_ordinal = (highest if highest is not None else -1) + 1
            ordinals = [
                existing.get(cue.id, next_ordinal + i) for i, cue in enumerate(cues)
            ]
        self.connection.executemany(
            sql,
            [
                (
                    cue.id,
                    ordinal,
                    cue.start,
                    cue.end,
                    cue.source,
                    cue.translated,
                    cue.timing_confidence,
                    json.dumps(cue.metadata),
                    cue.speaker,
                    stable,
                )
                for ordinal, cue in zip(ordinals, cues, strict=True)
            ],
        )

    def replace_transcript(self, cues: Sequence[Cue]) -> None:
        """Replace provisional cues with an authoritative transcript."""
        # A draft Agent can finish while Whisper is still producing its final
        # JSON. Preserve that completed source correction and translation while
        # accepting Whisper's authoritative timing and metadata.
        existing = {
            row["id"]: row
            for row in self.connection.execute(
                "SELECT id,source,translated FROM cues"
            ).fetchall()
        }
        merged: list[Cue] = []
        for cue in cues:
            prior = existing.get(cue.id)
            completed = prior is not None and prior["translated"] is not None
            metadata = dict(cue.metadata)
            merged.append(Cue(
                cue.id,
                cue.start,
                cue.end,
                prior["source"] if completed else cue.source,
                prior["translated"] if completed else cue.translated,
                cue.timing_confidence,
                metadata,
                # Diarization always comes from the authoritative transcript;
                # dropping it here would erase every speaker a draft batch
                # had already finished translating.
                cue.speaker,
            ))
        # Retiring the old rows and installing the merged transcript must be one
        # transaction: a crash in between would leave no durable timeline at
        # all, so a resume would re-transcribe the whole file from zero.
        with self.connection:
            self.connection.execute("UPDATE cues SET stable=0, ordinal=ordinal+1000000")
            self._upsert_cues(merged, stable=True, reset_order=True)

    def reset_transcription(self) -> None:
        """Atomically remove all state belonging to the live transcript."""
        with self.connection:
            for table in ("cues", "transcript_words", "translation_batches"):
                self.connection.execute(f"DELETE FROM {table}")
            self.connection.execute("DELETE FROM checkpoints WHERE name NOT IN ("
                "'memory_destination_snapshot', 'temporary_workdir', 'play_res', "
                "'stream_capture_complete', 'stream_completion_choice')")

    def discard_provisional_transcript(self) -> None:
        """Hide live cues and their translations before the authoritative pass."""
        with self.connection:
            self.connection.execute(
                "UPDATE cues SET translated=NULL, stable=0"
            )

    def replace_aligned_timeline(self, cues: Sequence[Cue]) -> None:
        """Atomically install a renumbered timeline and mark alignment durable."""
        sql = """
        INSERT INTO cues(id,ordinal,start,end,source,translated,
          timing_confidence,metadata_json,speaker,stable)
        VALUES(?,?,?,?,?,?,?,?,?,1)
        """
        now = utc_now()
        with self.connection:
            self.connection.execute("DELETE FROM cues")
            self.connection.executemany(
                sql,
                [
                    (
                        cue.id,
                        ordinal,
                        cue.start,
                        cue.end,
                        cue.source,
                        cue.translated,
                        cue.timing_confidence,
                        json.dumps(cue.metadata),
                        cue.speaker,
                    )
                    for ordinal, cue in enumerate(cues)
                ],
            )
            self.connection.execute(
                "INSERT INTO checkpoints VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "value_json=excluded.value_json,updated_at=excluded.updated_at",
                ("alignment_complete", json.dumps(True), now),
            )

    @staticmethod
    def _cue(row: sqlite3.Row) -> Cue:
        return Cue(
            id=row["id"], start=row["start"], end=row["end"], source=row["source"],
            translated=row["translated"],
            timing_confidence=row["timing_confidence"], metadata=json.loads(row["metadata_json"]),
            speaker=row["speaker"],
        )

    def list_cues(self, *, stable_only: bool = False) -> list[Cue]:
        where = " WHERE stable=1" if stable_only else ""
        rows = self.connection.execute(f"SELECT * FROM cues{where} ORDER BY ordinal").fetchall()
        return [self._cue(row) for row in rows]

    def cues_by_id(
        self, ids: Sequence[str], *, stable_only: bool = False
    ) -> dict[str, Cue]:
        """Return only the requested cues, keyed by ID."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        stable = " AND stable=1" if stable_only else ""
        rows = self.connection.execute(
            f"SELECT * FROM cues WHERE id IN ({placeholders}){stable}", tuple(ids)
        ).fetchall()
        return {row["id"]: self._cue(row) for row in rows}

    def replace_transcript_words(self, words: Sequence[Word]) -> None:
        """Install a complete word-level transcript atomically."""
        with self.connection:
            self.connection.execute("DELETE FROM transcript_words")
            self._insert_words(words)

    def append_transcript_words(self, words: Sequence[Word]) -> None:
        """Add or overwrite words by ordinal, for incremental persistence."""
        with self.connection:
            self._insert_words(words)

    def _insert_words(self, words: Sequence[Word]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO transcript_words"
            "(ordinal,start,end,speaker,text,logprob,stable) VALUES(?,?,?,?,?,?,1)",
            [
                (word.ordinal, word.start, word.end, word.speaker, word.text, word.logprob)
                for word in words
            ],
        )

    def list_transcript_words(self) -> list[Word]:
        rows = self.connection.execute(
            "SELECT * FROM transcript_words ORDER BY ordinal"
        ).fetchall()
        return [
            Word(
                ordinal=row["ordinal"], start=row["start"], end=row["end"],
                text=row["text"], speaker=row["speaker"], logprob=row["logprob"],
            )
            for row in rows
        ]

    def rename_cues(self, renames: Sequence[tuple[str, str]]) -> None:
        """Give settled cues their final IDs; ordinal and content stay put."""
        if not renames:
            return
        with self.connection:
            self.connection.executemany(
                "UPDATE cues SET id=? WHERE id=?",
                [(new_id, old_id) for old_id, new_id in renames],
            )

    def delete_cues(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self.connection:
            self.connection.execute(
                f"DELETE FROM cues WHERE id IN ({placeholders})", tuple(ids)
            )

    def checkpoint(self, name: str, value: Any) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO checkpoints VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",
                (name, json.dumps(value), utc_now()),
            )

    def get_checkpoint(self, name: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value_json FROM checkpoints WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def log(self, process: str, stream: str, line: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO process_logs(process,stream,line,created_at) VALUES(?,?,?,?)",
                (process, stream, line.rstrip(), utc_now()),
            )

    def start_batch(self, cue_ids: Sequence[str], backend: str, model: str | None, request: Any) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO translation_batches(cue_ids_json,backend,model,status,request_json,created_at) VALUES(?,?,?,?,?,?)",
                (json.dumps(cue_ids), backend, model, "running", json.dumps(request), utc_now()),
            )
        return int(cursor.lastrowid)

    def finish_batch(self, batch_id: int, response: Any = None, error: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE translation_batches SET status=?,response_json=?,error=?,finished_at=? WHERE id=?",
                ("failed" if error else "complete", json.dumps(response) if response is not None else None,
                 error, utc_now(), batch_id),
            )
