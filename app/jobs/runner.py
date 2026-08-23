"""In-process background job runner.

No Redis/Celery: this is a single-user tool meant to run as one Railway
process, so plain asyncio tasks with a concurrency cap are enough. Jobs
survive within the process; on a restart mid-job the job is left in
whatever status it was last saved at (visible to the user as stalled, which
is acceptable for a personal tool - just hit generate again).
"""

from __future__ import annotations

import asyncio
import logging

from app.pipeline.pipeline import process_job

logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = 2

_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_tasks: set[asyncio.Task] = set()


async def _guarded_run(job_id: str) -> None:
    async with _semaphore:
        try:
            await process_job(job_id)
        except Exception:
            logger.exception("unhandled error running job %s", job_id)


def enqueue_job(job_id: str) -> None:
    task = asyncio.create_task(_guarded_run(job_id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
