"""SQLite-backed job store.

A single Railway instance running one process is the target deployment, so a
simple file-backed SQLite table (rather than Redis/Postgres) is enough to
persist job state across requests and process restarts. Access is
serialized through a lock and run off the event loop via asyncio.to_thread,
since sqlite3 connections aren't safe to share across threads/async tasks
without care.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

from app.jobs.models import Job

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL
);
"""


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _save_sync(self, job: Job) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, data, created_at, expires_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data=excluded.data, expires_at=excluded.expires_at",
                (job.id, job.model_dump_json(), job.created_at, job.expires_at),
            )

    def _get_sync(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return Job.model_validate_json(row[0])

    def _list_recent_sync(self, limit: int) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [Job.model_validate_json(r[0]) for r in rows]

    def _list_expired_sync(self, now: float | None = None) -> list[Job]:
        now = now if now is not None else time.time()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM jobs WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
            ).fetchall()
        return [Job.model_validate_json(r[0]) for r in rows]

    def _delete_sync(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    async def save(self, job: Job) -> None:
        job.updated_at = time.time()
        async with self._lock:
            await asyncio.to_thread(self._save_sync, job)

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, job_id)

    async def list_recent(self, limit: int = 10) -> list[Job]:
        """Most recently created jobs, newest first. Used to restore the
        /upload page's in-progress/finished job after a browser refresh -
        this is a plain read against the same durable store the pipeline
        already writes to, so it never triggers any AI/processing work."""
        async with self._lock:
            return await asyncio.to_thread(self._list_recent_sync, limit)

    async def list_expired(self) -> list[Job]:
        async with self._lock:
            return await asyncio.to_thread(self._list_expired_sync)

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_sync, job_id)


_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        from app.config import get_settings

        _store = JobStore(get_settings().db_path)
    return _store
