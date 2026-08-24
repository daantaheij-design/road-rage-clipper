"""Shared job-creation/status logic used by both the HTTP API (/api/jobs)
and the MCP server tools, so the two surfaces can't drift apart."""

from __future__ import annotations

from app.jobs.models import Job, JobStatus
from app.jobs.runner import enqueue_job
from app.jobs.store import get_job_store
from app.security import UnsafeURLError, validate_url
from app.storage import get_storage

MIN_CLIPS, MAX_CLIPS = 1, 10


class JobCreationError(ValueError):
    pass


class JobNotRetryableError(ValueError):
    pass


async def create_url_job(video_url: str, number_of_clips: int, style: str | None) -> Job:
    video_url = (video_url or "").strip()
    if not video_url:
        raise JobCreationError("video_url is required")
    try:
        validate_url(video_url)
    except UnsafeURLError as exc:
        raise JobCreationError(f"That URL can't be used: {exc}") from exc

    number_of_clips = max(MIN_CLIPS, min(MAX_CLIPS, int(3 if number_of_clips is None else number_of_clips)))
    job = Job(number_of_clips=number_of_clips, style=style, source_url=video_url)
    await get_job_store().save(job)
    enqueue_job(job.id)
    return job


def _job_dict(job: Job) -> dict:
    storage = get_storage()
    urls: dict[str, str] = {}
    for clip in job.clips:
        if clip.storage_key:
            urls[clip.id] = storage.signed_download_url(
                clip.storage_key, expires_in=3600 * 6, filename=clip.filename or f"{clip.id}.mp4"
            )
    return job.public_dict(urls)


async def get_job_status(job_id: str) -> dict | None:
    job = await get_job_store().get(job_id)
    if job is None:
        return None
    return _job_dict(job)


async def list_recent_jobs(limit: int = 10) -> list[dict]:
    """Recent jobs, newest first - a plain read against the job store so
    the /upload page can restore its last job's state after a browser
    refresh without ever calling Anthropic/ElevenLabs again."""
    jobs = await get_job_store().list_recent(limit)
    return [_job_dict(job) for job in jobs]


async def retry_job(job_id: str) -> Job | None:
    """Re-run a failed job. If it already has saved analysis
    (`ready_to_render`), the pipeline picks that up and skips straight to
    rendering - no Anthropic or ElevenLabs calls happen again. Returns None
    if the job doesn't exist; raises JobNotRetryableError if it exists but
    isn't in a failed state (so a caller can't accidentally kick off a
    second concurrent run of an already-active job).
    """
    store = get_job_store()
    job = await store.get(job_id)
    if job is None:
        return None
    if job.status != JobStatus.FAILED:
        raise JobNotRetryableError(f"Job is '{job.status.value}', not failed - nothing to retry")

    job.status = JobStatus.QUEUED
    job.error = None
    job.message = "Retry queued"
    await store.save(job)
    enqueue_job(job.id)
    return job
