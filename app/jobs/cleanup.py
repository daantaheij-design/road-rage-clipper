"""Periodic retention cleanup.

Deletes generated clips (and the job record) once a job's expires_at has
passed, default 48 hours after completion (RETENTION_HOURS). Runs as a
background asyncio loop started at app startup - no cron/terminal needed.
"""

from __future__ import annotations

import asyncio
import logging

from app.jobs.store import get_job_store
from app.storage import get_storage

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 60 * 15


async def cleanup_expired_jobs() -> int:
    store = get_job_store()
    storage = get_storage()
    expired = await store.list_expired()
    for job in expired:
        logger.info("retention: deleting expired job %s (%d clip(s))", job.id, len(job.clips))
        storage.delete_prefix(f"jobs/{job.id}/")
        await store.delete(job.id)
    return len(expired)


async def run_cleanup_loop() -> None:
    while True:
        try:
            n = await cleanup_expired_jobs()
            if n:
                logger.info("retention: cleaned up %d expired job(s)", n)
        except Exception:
            logger.exception("retention cleanup loop failed")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
