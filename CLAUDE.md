# CLAUDE.md

Guidance for a future Claude Code session working in this repository.

## What this is

A **private, single-user** tool that turns a road-rage/dashcam video into
short vertical (1080x1920) TikTok-style highlight clips: it finds
interesting moments using two-pass Claude vision analysis (sparse scan, then
dense re-analysis of candidates), writes a short hook/setup/escalation/
main-event/payoff narration for each, generates an English AI voice-over,
burns in captions, and renders a blurred-background vertical composite. It
is used two ways: a tiny mobile web page (`/upload`) and a remote MCP server
(`/mcp`) so it can be driven from Claude on a phone.

There are no accounts, subscriptions, or multi-tenant concerns - it's one
person's tool, gated by a single shared password + MCP bearer token.

## Architecture

```
app/
  main.py            FastAPI app (factory: create_app()) - auth, /upload page, JSON API
  mcp_server.py       MCP tools (create_road_rage_clips, get_clip_job) mounted at /mcp
  auth.py             Password + signed-cookie session auth
  security.py          SSRF-safe URL validation (blocks localhost/private/metadata IPs)
  storage.py            R2 (boto3/S3-compatible) or local-disk storage, signed download URLs
  config.py              Settings (pydantic-settings, all via env vars)
  jobs/
    models.py             Job/Clip/NarrationCue pydantic models
    store.py               SQLite-backed job persistence
    service.py              Shared job-creation logic (used by both HTTP API and MCP)
    runner.py                In-process asyncio background job runner (no Redis/Celery)
    cleanup.py                Retention loop - deletes expired jobs/files
  pipeline/
    download.py               SSRF-safe streaming video download
    ffmpeg_utils.py             probe/extract_audio/extract_frames wrappers around ffmpeg/ffprobe
    transcribe.py                 ElevenLabs Scribe speech-to-text (word timestamps)
    vision.py                      Two-pass Claude vision analysis (tool-use structured output)
    scoring.py                      Turns analyses into a final non-overlapping clip selection
    tts.py                           ElevenLabs text-to-speech narration
    captions.py                      Builds .ass subtitle files (Hook/Narration/Caption styles)
    render.py                         ffmpeg filter-graph: vertical blurred-bg composite + audio ducking/mix
    pipeline.py                       Orchestrates all of the above end-to-end for one job
  templates/            upload.html (the whole mobile UI), login.html
```

### Processing pipeline (per job)

1. Acquire source video (download URL with SSRF checks, or use the browser-uploaded file).
2. `ffprobe` for duration/resolution/audio presence; reject if over the configured duration limit.
3. Extract audio -> ElevenLabs Scribe transcription with word-level timestamps.
4. Extract sparse frames (~1 every 1.5s) across the whole video -> Claude vision pass 1
   (batched, tool-forced JSON) flags rough candidate time windows.
5. For each candidate, extract dense frames (several/sec) in a padded window -> Claude vision
   pass 2 produces: title, observable-only explanation, refined start/end, 10 sub-scores (0-10
   each, summed to a 0-100 score), a hook line, and up to 5 narration cues (hook/setup/
   escalation/main_event/payoff, each optionally skipped so narration doesn't talk over
   everything).
6. `scoring.select_clips` ranks by total score and greedily picks the best non-overlapping set,
   clamping each clip into the 25-90s range (preferring 25-60s).
7. Per selected clip: synthesize narration audio per cue (ElevenLabs TTS) and **upload it to
   storage immediately** (see Resumability below), then build an .ass caption file (original
   transcript captions everywhere narration *isn't* playing, narration captions where it is, a
   hook title card at the very start), then render with `ffmpeg`: blurred/scaled vertical
   composite + ducked-and-mixed audio + burned captions -> H.264/AAC MP4.
8. Upload each clip to storage; job records get a `expires_at` (default 48h) for retention cleanup.

### Resumability: retrying a failed job never repeats paid AI work

Transcription (ElevenLabs) and visual analysis/story/narration (Anthropic + ElevenLabs TTS) are
the expensive, paid steps - rendering (ffmpeg, see the memory-tuning notes below) is by far the
most likely thing to fail, especially on a small container. So every expensive result is
persisted the moment it succeeds, not just at the very end:

- `Job.transcript_words` - the full ElevenLabs transcript, saved right after transcription.
- `Job.clips[]` - one `Clip` per selected moment (title, scores, hook, explanation, start/end),
  populated right after clip selection - **before** rendering starts.
- `Clip.narration_cues[].audio_storage_key` - each cue's synthesized narration audio is uploaded
  to storage (`jobs/{job_id}/narration/{clip_id}/{i}.mp3`) the moment ElevenLabs TTS returns it.
- `Job.ready_to_render` - set `True` once all of the above has happened for every selected clip.

`app/pipeline/pipeline.py::process_job` checks `ready_to_render` (and, as a smaller intermediate
checkpoint, whether `transcript_words` is already populated) at the top of a run: if analysis is
already done, it skips straight to rendering - no Anthropic/ElevenLabs calls happen at all. The
render loop itself only processes clips that don't yet have a `storage_key` (i.e. didn't already
render+upload successfully in a previous attempt), so a failure partway through a multi-clip job
doesn't re-render clips that already finished. `_fetch_narration_tracks` always re-downloads
narration audio from storage before rendering (rather than trusting local scratch files survived)
so this works correctly even after a container restart, not just an in-process retry.

`app/jobs/service.py::retry_job` is what flips a `FAILED` job back to `QUEUED` and re-enqueues it
through the normal `enqueue_job` path - it's intentionally *not* a separate pipeline entry point,
so "retry" and "first attempt" are exactly the same code path with the same resume checks. Only
`FAILED` jobs can be retried (guards against kicking off a second concurrent run of an active
job). Exposed as `POST /api/jobs/{job_id}/retry` and the `retry_road_rage_job` MCP tool; the
`/upload` page shows a "Retry render" button whenever a job's status is `failed`, labelled
differently when `resumable` (i.e. `ready_to_render`) is true so the user knows it won't re-run
the AI steps.

### Two web surfaces, one job engine

`app/jobs/service.py` is the single source of truth for "create a job from a URL" / "get job
status" - both `POST /api/jobs` (HTTP, cookie auth) and the MCP `create_road_rage_clips` /
`get_clip_job` tools call into it, so behavior can't drift between the two. Background
processing always goes through `app/jobs/runner.py::enqueue_job`, which schedules
`app/pipeline/pipeline.py::process_job` as an asyncio task (capped concurrency, no external
queue) - both entry points return almost immediately with just a `job_id`.

### FastAPI app is a factory, not a module-level singleton

`app/main.py` exposes `create_app()` rather than a bare `app = FastAPI()`. This matters because
the mounted MCP sub-app owns a `StreamableHTTPSessionManager` whose `.run()` can only be
entered **once per instance** - reusing one `app` object across multiple test sessions (or
multiple `TestClient` context managers) breaks the second one. Tests call `create_app()` fresh
each time; `app = create_app()` at the bottom of the module is what a real ASGI server imports.

The MCP sub-app's own Starlette `lifespan` (which starts that session manager) is **not**
triggered automatically just by being `app.mount()`-ed - Starlette only sends ASGI lifespan
events to the root app. `create_app()`'s lifespan explicitly enters
`mcp_app.router.lifespan_context(mcp_app)` via an `AsyncExitStack` to work around this. If you
ever see `RuntimeError: Task group is not initialized. Make sure to use run()` from the mcp
package, this wiring is what broke.

## Conventions

- All Claude Vision/analysis calls force a single tool call (`tool_choice={"type": "tool", ...}`)
  so output is always structured JSON - never scrape prose.
- Narration/explanations must describe only what's *visible* - "the driver appears to brake
  sharply", never "the driver wanted to cause an accident". This rule is baked into the vision.py
  system prompts (`OBSERVABLE_ONLY_RULE`) - keep it if you touch those prompts.
- The Claude model is read from `CLAUDE_MODEL` (default `claude-opus-5`) - never hardcode a
  model string elsewhere. Likewise ElevenLabs model IDs come from settings, not literals.
- `app/security.py` is the only place SSRF validation should live; both the HTTP upload endpoint
  and the pipeline's actual downloader call `validate_url`/`resolve_and_validate_host` - re-run
  validation right before connecting (not just at request time) since redirects are followed
  manually hop-by-hop.
- Storage keys are always `jobs/{job_id}/...` so `storage.delete_prefix(f"jobs/{job_id}/")` in
  the retention cleanup loop removes everything for a job in one call, on both the R2 and local
  backends.
- Don't add authentication/accounts/billing - this is intentionally single-user. If asked to add
  multi-user support, treat that as a significant scope change and confirm first.

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # fill in at least APP_PASSWORD, SECRET_KEY, MCP_API_KEY
uvicorn app.main:app --reload
```

ffmpeg/ffprobe must be on PATH locally (`apt-get install ffmpeg` on Debian/Ubuntu). Without
`ANTHROPIC_API_KEY`/`ELEVENLABS_API_KEY` set to real keys, the web UI and job creation still work
end-to-end, but a job will fail once it reaches the analysis/transcription/TTS stages - that's
expected and fine for testing the plumbing.

## Testing

```bash
pytest            # unit tests + an ffmpeg integration suite using a synthetic ffmpeg-generated
                   # test video (tests/conftest.py::synthetic_video) - never a real video
ruff check app tests
```

External APIs (Anthropic, ElevenLabs) are never called in tests - HTTP-layer tests monkeypatch
`enqueue_job` so job creation doesn't actually kick off the real pipeline. The ffmpeg tests are
real (not mocked) since ffmpeg is deterministic, fast, and the whole point of that layer is
correct filter-graph behavior.

Docker builds could not be verified inside the sandbox this project was built in - the sandbox's
egress policy blocks pulls from `docker.io`/CloudFront (a genuine org policy 403, not a bug), so
`docker build` was never run end-to-end there. The Dockerfile mirrors the exact steps validated
locally (apt-get ffmpeg install, pip install -r requirements.txt) - verify with a real
`docker build .` in an environment with normal registry access before relying on it.

## Things to watch out for when changing the pipeline

- `render.py`'s ffmpeg filter graph assumes narration cues don't overlap each other (they're
  sequential story beats) - if you ever let two cues overlap in time, the chained `volume`
  ducking filters and `adelay`+`amix` approach still works, but revisit `alimiter` headroom.
- `ffmpeg_utils.extract_frames` timestamps are computed from the requested `fps`/`start`, not
  read back from ffmpeg - if you change the `-vf fps=...` filter to something more complex
  (e.g. `select=`), you'll need to compute real timestamps differently.
- `scoring._clamp_duration` can shift a clip's start earlier than the original candidate (to hit
  `MIN_CLIP_SECONDS`). `select_clips` compensates by shifting every `narration_cues[].start_seconds`
  by the same amount (`head_shift`) before overwriting `a.start_seconds` - if you touch either
  function, keep that shift in sync or narration will play at the wrong point in the rendered
  clip. Tail truncation (clip too long) doesn't need compensation since
  `pipeline._synthesize_and_store_narration` already clamps each cue's start into
  `[0, clip.duration_seconds]` before it's ever persisted.
- `render.py` deliberately caps ffmpeg/libx264 threading (`FFMPEG_THREADS`, default 2) and blurs
  the background at a small internal resolution (`BG_BLUR_W`/`BG_BLUR_H`) before scaling back up
  to 1080x1920 - small Railway containers report the *host's* full CPU count to ffmpeg, and an
  unconstrained thread count plus a full-resolution `gblur` was enough to get the render process
  OOM-killed (ffmpeg exits with return code -9). If you touch this file, keep the thread caps and
  low-res blur; `ffmpeg_utils._describe_failure` gives OOM-killed renders (negative return code,
  i.e. killed by signal) a distinct, clearly-labeled error message instead of a generic ffmpeg
  failure - preserve that if you change error handling there.
- `app/jobs/store.py::JobStore` uses an `asyncio.Lock` to serialize sqlite access. In production
  that's always used from a single event loop (one uvicorn process), so it's fine - but in tests,
  never seed/mutate job rows via `asyncio.run(get_job_store().save(...))` while a `TestClient` is
  active: the app's background retention cleanup loop (started in `create_app()`'s lifespan) runs
  on `TestClient`'s own portal thread/loop and touches the same lock, and racing a second,
  independent event loop against it can deadlock the cross-thread lock hand-off (the test just
  hangs forever, no exception). Seed test job rows with `get_job_store()._save_sync(job)` instead
  - a plain synchronous sqlite write, no event loop involved. See `tests/test_api.py` and
  `tests/test_mcp_server.py` for the pattern.
