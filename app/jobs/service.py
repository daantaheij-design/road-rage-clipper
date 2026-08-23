"""Shared job-creation/status logic used by both the HTTP API (/api/jobs)
and the MCP server tools, so the two surfaces can't drift apart."""

from __future__ import annotations

from app.jobs.models import Job
from app.jobs.runner import enqueue_job
from app.jobs.store import get_job_store
from app.security import UnsafeURLError, validate_url
from app.storage import get_storage

MIN_CLIPS, MAX_CLIPS = 1, 10


class JobCreationError(ValueError):
    pass


async def create_url_job(video_url: str, number_of_clips: int, style: str | None) -> Job:
    video_url = (video_url or "").strip()
    if not video_url:
        raise JobCreationError("video_url is required")
    try:
        validate_url(video_url)
    except UnsafeURLError as exc:
        raise JobCreationError(f"That URL can't be used: {exc}") from exc

    number_of_clips = max(MIN_CLIPS, min(MAX_CLIPS, int(number_of_clips or 3)))
    job = Job(number_of_clips=number_of_clips, style=style, source_url=video_url)
    await get_job_store().save(job)
    enqueue_job(job.id)
    return job


async def get_job_status(job_id: str) -> dict | None:
    job = await get_job_store().get(job_id)
    if job is None:
        return None
    storage = get_storage()
    urls: dict[str, str] = {}
    for clip in job.clips:
        if clip.storage_key:
            urls[clip.id] = storage.signed_download_url(
                clip.storage_key, expires_in=3600 * 6, filename=clip.filename or f"{clip.id}.mp4"
            )
    return job.public_dict(urls)
