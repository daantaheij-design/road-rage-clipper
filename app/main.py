from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import auth
from app.config import get_settings
from app.jobs import service as job_service
from app.jobs.cleanup import run_cleanup_loop
from app.jobs.models import Job
from app.jobs.runner import enqueue_job
from app.jobs.store import get_job_store
from app.mcp_server import mcp_asgi_app
from app.security import UnsafeURLError, validate_url
from app.storage import get_storage, verify_local_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_CLIPS, MAX_CLIPS = 1, 10


def create_app() -> FastAPI:
    """Build the FastAPI app.

    This is a factory (rather than a bare module-level `app`) so tests can
    construct a fresh instance - the mounted MCP sub-app owns a
    StreamableHTTPSessionManager that can only be started once per instance,
    so each independent test run (and each real server process) needs its
    own.
    """
    mcp_app = mcp_asgi_app()
    cleanup_task: asyncio.Task | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal cleanup_task
        async with AsyncExitStack() as stack:
            # The mounted MCP sub-app's own lifespan (which starts its
            # session manager) is never triggered by Starlette just from
            # being mounted - it has to be entered explicitly alongside ours.
            await stack.enter_async_context(mcp_app.router.lifespan_context(mcp_app))
            cleanup_task = asyncio.create_task(run_cleanup_loop())
            yield
            cleanup_task.cancel()

    app = FastAPI(title="Road Rage Clipper", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    def require_page_auth(request: Request) -> None:
        if not auth.is_authed(request):
            raise HTTPException(status_code=303, headers={"Location": "/login"})

    def require_api_auth(request: Request) -> None:
        if not auth.is_authed(request):
            raise HTTPException(status_code=401, detail="Not authenticated")

    @app.exception_handler(HTTPException)
    async def _redirect_on_303(request: Request, exc: HTTPException):
        if exc.status_code == 303 and "Location" in (exc.headers or {}):
            return RedirectResponse(exc.headers["Location"], status_code=303)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/upload")

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    async def login_page(request: Request):
        if auth.is_authed(request):
            return RedirectResponse("/upload")
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login", include_in_schema=False)
    async def login_submit(request: Request, password: str = Form(...)):
        if not auth.check_password(password):
            return templates.TemplateResponse(request, "login.html", {"error": "Wrong password"}, status_code=401)
        resp = RedirectResponse("/upload", status_code=303)
        resp.set_cookie(
            auth.SESSION_COOKIE,
            auth.create_session_token(),
            max_age=auth.SESSION_MAX_AGE,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
        )
        return resp

    @app.post("/logout", include_in_schema=False)
    async def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(auth.SESSION_COOKIE)
        return resp

    @app.get(
        "/upload", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(require_page_auth)]
    )
    async def upload_page(request: Request):
        settings = get_settings()
        return templates.TemplateResponse(
            request,
            "upload.html",
            {"min_clips": MIN_CLIPS, "max_clips": MAX_CLIPS, "max_video_mb": int(settings.max_video_mb)},
        )

    # -----------------------------------------------------------------
    # JSON API backing the /upload page (also usable directly, cookie-authed)
    # -----------------------------------------------------------------

    @app.post("/api/jobs", dependencies=[Depends(require_api_auth)])
    async def create_job(
        video_url: str | None = Form(None),
        number_of_clips: int = Form(3),
        style: str | None = Form(None),
        file: UploadFile | None = File(None),
    ):
        settings = get_settings()
        number_of_clips = max(MIN_CLIPS, min(MAX_CLIPS, number_of_clips))
        video_url = (video_url or "").strip() or None

        if not video_url and not (file and file.filename):
            raise HTTPException(400, "Provide either a video file or a video URL")
        if video_url and file and file.filename:
            raise HTTPException(400, "Provide only one of: video file, video URL")

        if video_url:
            try:
                validate_url(video_url)
            except UnsafeURLError as exc:
                raise HTTPException(400, f"That URL can't be used: {exc}") from exc

        job = Job(number_of_clips=number_of_clips, style=style, source_url=video_url)

        if file and file.filename:
            suffix = Path(file.filename).suffix or ".mp4"
            if len(suffix) > 10 or not suffix.replace(".", "").isalnum():
                suffix = ".mp4"
            workdir = settings.tmp_path / job.id
            workdir.mkdir(parents=True, exist_ok=True)
            dest = workdir / f"source{suffix}"
            max_bytes = int(settings.max_video_mb * 1024 * 1024)
            written = 0
            with open(dest, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        f.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(400, f"File exceeds the {settings.max_video_mb:.0f} MB limit")
                    f.write(chunk)
            if written == 0:
                dest.unlink(missing_ok=True)
                raise HTTPException(400, "Uploaded file was empty")
            job.source_filename = file.filename

        await get_job_store().save(job)
        enqueue_job(job.id)
        return {"job_id": job.id}

    @app.get("/api/jobs", dependencies=[Depends(require_api_auth)])
    async def list_jobs(limit: int = Query(10, ge=1, le=50)):
        # Plain read of already-persisted job state - lets the /upload page
        # restore whatever it was showing after a browser refresh without
        # triggering any new download/transcription/analysis/render work.
        return {"jobs": await job_service.list_recent_jobs(limit)}

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_api_auth)])
    async def get_job(job_id: str):
        status = await job_service.get_job_status(job_id)
        if status is None:
            raise HTTPException(404, "Job not found")
        return status

    @app.post("/api/jobs/{job_id}/retry", dependencies=[Depends(require_api_auth)])
    async def retry_job(job_id: str):
        try:
            job = await job_service.retry_job(job_id)
        except job_service.JobNotRetryableError as exc:
            raise HTTPException(409, str(exc)) from exc
        if job is None:
            raise HTTPException(404, "Job not found")
        return {"job_id": job.id, "status": job.status.value}

    # -----------------------------------------------------------------
    # Signed local-storage file serving (only used when R2 isn't configured)
    # -----------------------------------------------------------------

    @app.get("/files/{key:path}", include_in_schema=False)
    async def serve_local_file(
        key: str, exp: int = Query(...), sig: str = Query(...), filename: str | None = Query(None)
    ):
        settings = get_settings()
        if get_storage().backend != "local":
            raise HTTPException(404)
        if not verify_local_signature(settings, key, exp, sig):
            raise HTTPException(403, "Link expired or invalid")
        path = (settings.local_storage_path / key).resolve()
        if not str(path).startswith(str(settings.local_storage_path.resolve())) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4", filename=filename or path.name)

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"ok": True, "time": time.time()}

    app.mount("/mcp", mcp_app)
    return app


app = create_app()
