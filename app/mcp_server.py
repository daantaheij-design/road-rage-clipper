"""Remote MCP server: create_road_rage_clips / get_clip_job.

This lets you connect this tool to Claude as a remote MCP server and ask it
in plain English (from your phone) to turn a video into clips. Both tools
return almost immediately - the actual FFmpeg/Claude/ElevenLabs work runs as
a background job (the same one the /upload web page uses), so the MCP
request never sits open for the minutes that real processing takes.
"""

from __future__ import annotations

import hmac

from mcp.server import MCPServer
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import get_settings
from app.jobs import service as job_service

mcp = MCPServer(
    name="road-rage-clipper",
    title="Road Rage Clipper",
    instructions=(
        "Turns a road-rage / dashcam video into short vertical TikTok-style highlight clips with "
        "AI-selected moments, narration, and captions. Call create_road_rage_clips with a video URL "
        "to start a job, then poll get_clip_job with the returned job_id until status is 'completed' "
        "or 'failed'. Processing takes several minutes; do not block waiting - check back later. If a "
        "job fails, call retry_road_rage_job with the same job_id rather than starting a new job - if "
        "the expensive AI analysis/narration steps already finished (get_clip_job reports "
        "'resumable': true), retrying skips them and only redoes the failed step."
    ),
)


@mcp.tool()
async def create_road_rage_clips(video_url: str, number_of_clips: int = 3, optional_style: str | None = None) -> dict:
    """Start generating road-rage TikTok clips from a video URL.

    Args:
        video_url: Direct URL to the source video (http/https, public - not a local file).
        number_of_clips: How many clips to produce (1-10). Defaults to 3.
        optional_style: Optional free-text style hint for narration tone (e.g. "sarcastic",
            "serious news report"). May be ignored.

    Returns immediately with a job_id. This does NOT wait for processing to finish - call
    get_clip_job with the returned job_id to check progress and retrieve results.
    """
    try:
        job = await job_service.create_url_job(video_url, number_of_clips, optional_style)
    except job_service.JobCreationError as exc:
        return {"error": str(exc)}
    return {"job_id": job.id, "status": job.status.value}


@mcp.tool()
async def get_clip_job(job_id: str) -> dict:
    """Get progress and, once finished, the clips + download URLs for a job started with
    create_road_rage_clips.

    Args:
        job_id: The job_id returned by create_road_rage_clips.
    """
    status = await job_service.get_job_status(job_id)
    if status is None:
        return {"error": f"No job found with id '{job_id}'"}
    return status


@mcp.tool()
async def retry_road_rage_job(job_id: str) -> dict:
    """Retry a failed job started with create_road_rage_clips.

    If transcription, visual analysis, clip selection, story generation, and narration TTS had
    already completed successfully before the job failed (this is reported as `resumable: true`
    by get_clip_job), the retry skips straight to rendering and does NOT call Anthropic or
    ElevenLabs again - only the failed rendering step is redone. If the job failed before that
    point, the retry starts over from the beginning.

    Args:
        job_id: The job_id of a job whose status is 'failed'.

    Returns immediately with the job's new status - poll get_clip_job as usual to track progress.
    """
    try:
        job = await job_service.retry_job(job_id)
    except job_service.JobNotRetryableError as exc:
        return {"error": str(exc)}
    if job is None:
        return {"error": f"No job found with id '{job_id}'"}
    return {"job_id": job.id, "status": job.status.value}


class _BearerAuthMiddleware:
    """Requires `Authorization: Bearer <MCP_API_KEY>` on every request to the MCP app."""

    def __init__(self, app: ASGIApp, expected_token: str):
        self.app = app
        self.expected_token = expected_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""

        if not hmac.compare_digest(token, self.expected_token):
            response = JSONResponse({"error": "Unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def mcp_asgi_app() -> Starlette:
    settings = get_settings()
    inner = mcp.streamable_http_app(streamable_http_path="/", stateless_http=True)
    inner.add_middleware(_BearerAuthMiddleware, expected_token=settings.mcp_api_key)
    return inner
