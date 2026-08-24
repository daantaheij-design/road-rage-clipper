from __future__ import annotations

import pytest

from app.jobs import service as job_service
from app.jobs.models import Job, JobStatus
from app.jobs.store import get_job_store


@pytest.fixture(autouse=True)
def no_real_jobs(settings, monkeypatch):
    # These tests only exercise the service-layer bookkeeping - never let a
    # real background pipeline run start.
    monkeypatch.setattr(job_service, "enqueue_job", lambda job_id: None)


async def test_create_url_job_rejects_unsafe_url(settings):
    with pytest.raises(job_service.JobCreationError):
        await job_service.create_url_job("http://localhost/video.mp4", 3, None)


async def test_create_url_job_rejects_empty_url(settings):
    with pytest.raises(job_service.JobCreationError):
        await job_service.create_url_job("   ", 3, None)


async def test_create_url_job_clamps_clip_count(settings):
    job = await job_service.create_url_job("https://example.com/video.mp4", 99, None)
    assert job.number_of_clips == job_service.MAX_CLIPS

    job2 = await job_service.create_url_job("https://example.com/video.mp4", 0, None)
    assert job2.number_of_clips == job_service.MIN_CLIPS


async def test_get_job_status_missing_returns_none(settings):
    assert await job_service.get_job_status("does-not-exist") is None


async def test_retry_job_missing_returns_none(settings):
    assert await job_service.retry_job("does-not-exist") is None


async def test_retry_job_requires_failed_status(settings):
    job = Job(status=JobStatus.RENDERING)
    await get_job_store().save(job)
    with pytest.raises(job_service.JobNotRetryableError):
        await job_service.retry_job(job.id)

    completed = Job(status=JobStatus.COMPLETED)
    await get_job_store().save(completed)
    with pytest.raises(job_service.JobNotRetryableError):
        await job_service.retry_job(completed.id)


async def test_retry_job_resets_failed_job_and_reenqueues(settings, monkeypatch):
    enqueued = []
    monkeypatch.setattr(job_service, "enqueue_job", lambda job_id: enqueued.append(job_id))

    job = Job(status=JobStatus.FAILED, error="boom", message="Something went wrong.")
    await get_job_store().save(job)

    retried = await job_service.retry_job(job.id)
    assert retried is not None
    assert retried.status == JobStatus.QUEUED
    assert retried.error is None
    assert enqueued == [job.id]

    persisted = await get_job_store().get(job.id)
    assert persisted.status == JobStatus.QUEUED
    assert persisted.error is None


async def test_list_recent_jobs_newest_first(settings):
    older = Job(created_at=100.0)
    newer = Job(created_at=200.0)
    await get_job_store().save(older)
    await get_job_store().save(newer)

    jobs = await job_service.list_recent_jobs()
    assert [j["job_id"] for j in jobs] == [newer.id, older.id]


async def test_job_public_dict_resumable_flag(settings):
    job = Job(status=JobStatus.FAILED, ready_to_render=True)
    assert job.public_dict()["resumable"] is True

    not_ready = Job(status=JobStatus.FAILED, ready_to_render=False)
    assert not_ready.public_dict()["resumable"] is False

    # Only meaningful while actually failed - a completed/ready job isn't
    # something to "retry".
    completed = Job(status=JobStatus.COMPLETED, ready_to_render=True)
    assert completed.public_dict()["resumable"] is False
