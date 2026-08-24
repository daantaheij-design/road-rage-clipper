from __future__ import annotations

import time

from app.jobs.models import Job, JobStatus
from app.jobs.store import JobStore


async def test_save_and_get_round_trip(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = Job(number_of_clips=4, source_url="https://example.com/video.mp4")
    await store.save(job)

    loaded = await store.get(job.id)
    assert loaded is not None
    assert loaded.id == job.id
    assert loaded.number_of_clips == 4
    assert loaded.source_url == "https://example.com/video.mp4"
    assert loaded.status == JobStatus.QUEUED


async def test_get_missing_returns_none(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    assert await store.get("does-not-exist") is None


async def test_update_persists(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = Job()
    await store.save(job)

    job.status = JobStatus.COMPLETED
    job.progress = 100
    await store.save(job)

    loaded = await store.get(job.id)
    assert loaded.status == JobStatus.COMPLETED
    assert loaded.progress == 100


async def test_list_expired_and_delete(tmp_path):
    store = JobStore(tmp_path / "jobs.db")

    expired = Job(expires_at=time.time() - 10)
    fresh = Job(expires_at=time.time() + 3600)
    forever = Job(expires_at=None)
    await store.save(expired)
    await store.save(fresh)
    await store.save(forever)

    expired_jobs = await store.list_expired()
    ids = {j.id for j in expired_jobs}
    assert expired.id in ids
    assert fresh.id not in ids
    assert forever.id not in ids

    await store.delete(expired.id)
    assert await store.get(expired.id) is None


async def test_list_recent_returns_newest_first(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    older = Job(created_at=100.0)
    newer = Job(created_at=200.0)
    await store.save(older)
    await store.save(newer)

    recent = await store.list_recent(limit=10)
    assert [j.id for j in recent] == [newer.id, older.id]


async def test_list_recent_respects_limit(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    for i in range(5):
        await store.save(Job(created_at=float(i)))

    recent = await store.list_recent(limit=2)
    assert len(recent) == 2
