# Road Rage Clipper 🚗

A **private, single-user** tool that turns a road-rage / dashcam video into short vertical
(1080x1920) TikTok-style highlight clips. It watches the actual video (not just the transcript)
using two-pass Claude vision analysis, writes a short factual hook/setup/escalation/payoff story
for each moment it finds, generates an English AI voice-over, burns in captions, and renders a
blurred-background vertical composite so nothing important gets cropped out.

Two ways to use it:

- **`/upload`** - a tiny mobile web page. Pick a video (or paste a URL), choose a clip count, tap
  Generate, watch progress, download the finished MP4s.
- **Remote MCP server** - connect it to Claude (including from your phone) and just say *"Make 5
  TikTok clips from this road-rage video: [url]"*.

This is not a SaaS product. There are no accounts, subscriptions, or payments - just a password
and a bearer token, both of which only you know.

---

## Deploying from an iPhone (no computer required)

Everything below can be done from Safari on your iPhone, using the Railway app/website and the
GitHub app/website. You do **not** need a terminal.

### 1. Get API keys (one-time, ~5 minutes)

You need two API keys. Both providers work fine from Safari on mobile:

1. **Anthropic** (video/story understanding) - sign up at console.anthropic.com, add billing, and
   create an API key under **Settings -> API Keys**. Copy it somewhere safe.
2. **ElevenLabs** (transcription + voice-over) - sign up at elevenlabs.io, go to your profile
   settings, and copy your API key.

### 2. Push this repo to your own GitHub account

If you're reading this from a repo Claude already created for you, it's already on GitHub -
skip to step 3. Otherwise, use the GitHub app to create a new repository and upload this project's
files to it.

### 3. Create a Railway project from GitHub

1. Open **railway.app** in Safari and sign in (or sign up) with your GitHub account.
2. Tap **New Project -> Deploy from GitHub repo**, and pick this repository.
3. Railway will detect the `Dockerfile` in this repo automatically and build from it - you don't
   need to configure anything about the build.
4. The first deploy will fail (or the app will crash-loop) until you set the required environment
   variables in the next step - that's expected.

### 4. Set environment variables

In your Railway project, open the service, go to the **Variables** tab, and add these (tap
"+ New Variable" for each, or use the "Raw Editor" to paste several at once). See
[`.env.example`](.env.example) for the full list with explanations - at minimum you need:

| Variable | Value |
|---|---|
| `APP_PASSWORD` | Any long password you'll remember - protects `/upload` |
| `SECRET_KEY` | A long random string (see tip below) |
| `MCP_API_KEY` | Another long random string (see tip below) |
| `ANTHROPIC_API_KEY` | From step 1 |
| `ELEVENLABS_API_KEY` | From step 1 |

**Generating random strings on your phone:** open any password manager's "generate password"
tool and generate a 40+ character password with letters and numbers - paste that in as
`SECRET_KEY` and generate a second one for `MCP_API_KEY`. They don't need to be memorable, just
long and unique.

Everything else has a sensible default (see `.env.example`) - you can leave the rest unset to
start. Notably:

- Without `R2_ACCOUNT_ID`/`R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` set, clips are stored on
  Railway's local disk instead of object storage. This works, but Railway's filesystem is
  ephemeral across redeploys - **for anything you care about keeping, set up Cloudflare R2**
  (see below). Either way, clips still auto-delete after `RETENTION_HOURS` (default 48).
- `RETENTION_HOURS` controls how long clips stick around before auto-deletion.
- `MAX_VIDEO_MB` / `MAX_VIDEO_DURATION_SECONDS` cap how large/long a source video can be.

Railway redeploys automatically whenever you change a variable.

### 5. (Recommended) Set up Cloudflare R2 storage

R2 is inexpensive (has a generous free tier) and, unlike Railway's disk, survives redeploys.
From Safari:

1. Sign up at **dash.cloudflare.com**, and open **R2 Object Storage**.
2. Create a bucket (any name, e.g. `road-rage-clipper`).
3. Go to **Manage R2 API Tokens -> Create API Token**, grant it read/write access to that bucket.
4. Copy the **Account ID**, **Access Key ID**, and **Secret Access Key** it gives you.
5. Back in Railway, set:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET` (the bucket name you chose)

### 6. Open it

Railway gives your service a public URL under **Settings -> Networking -> Public Networking**
(tap "Generate Domain" if one isn't there yet). Open that URL in Safari, enter your
`APP_PASSWORD`, and you're in. Add it to your Home Screen (Share -> Add to Home Screen) so it
behaves like an app.

### 7. Connect it to Claude as a remote MCP server (optional)

Once deployed, the MCP server is available at `https://<your-railway-domain>/mcp`, authenticated
with the `MCP_API_KEY` you set as a Bearer token. Add it as a remote MCP server connector in
Claude (Settings -> Connectors -> Add custom connector), pointing at that URL with an
`Authorization: Bearer <MCP_API_KEY>` header. Once connected, you can say things like:

> Make 5 TikTok clips from this road-rage video: https://example.com/my-dashcam-clip.mp4

Claude will call `create_road_rage_clips`, get a `job_id` back immediately, and can check back
later with `get_clip_job` - processing (transcription, AI analysis, rendering) happens in the
background and takes a few minutes, so don't expect an instant reply.

---

## What it actually does

1. **Download/accept the video** - either a browser upload or a URL (validated against SSRF:
   localhost, private IP ranges, and the cloud metadata endpoint are all blocked).
2. **Transcribe the audio** with word-level timestamps (ElevenLabs Scribe).
3. **Scan the video visually in two passes**: a cheap wide scan (sparse low-res frames every
   ~1.5s) flags rough candidate windows, then a focused pass (several frames/sec) on just those
   windows produces a detailed, scored analysis - visual action, escalation, surprise, tension,
   hook potential, understandable context, payoff, retention, reaction, and uniqueness, each
   0-10, summed to a 0-100 score.
4. **Pick the best non-overlapping clips**, padding each one with enough lead-in that the setup
   makes sense (never starting exactly on the incident), targeting 25-60s (up to ~90s when the
   story needs it).
5. **Write a short story** for each clip (hook / setup / escalation / main event / payoff) and
   **generate an English AI voice-over** for the beats that benefit from narration - not the
   whole clip, so original reactions, honking, and arguments stay audible. Original audio ducks
   under narration and returns to full volume when it stops.
6. **Render vertically**: horizontal dashcam footage is placed in full inside the 1080x1920
   frame (never cropped to fit), with a blurred, zoomed copy of the same footage filling the
   space above/below.
7. **Burn in captions**: large, high-contrast, TikTok-safe-area captions from the transcript,
   narration captions while the AI voice-over is speaking, and a hook title card at the very
   start.

All narration and captions describe only what's visibly happening ("the driver appears to brake
sharply") - never claims about what someone was thinking or intending.

**If rendering fails, retrying never re-runs the AI steps.** Transcription, visual analysis, story
generation, and narration are the expensive, paid steps (Anthropic + ElevenLabs); rendering
(ffmpeg) is the step most likely to fail, especially on a small container. Every expensive result
is saved the moment it's produced, so if a job fails during/after rendering, the `/upload` page
shows a **Retry render** button (and the MCP `retry_road_rage_job` tool does the same) that skips
straight back to rendering - no Anthropic or ElevenLabs calls happen again, and any clips that
already rendered successfully aren't re-rendered either.

---

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # fill in APP_PASSWORD, SECRET_KEY, MCP_API_KEY at minimum
uvicorn app.main:app --reload
```

Requires `ffmpeg`/`ffprobe` on your PATH (`apt-get install ffmpeg`, `brew install ffmpeg`, etc).

```bash
pytest              # unit tests + ffmpeg integration tests (uses a synthetic test video)
ruff check app tests
docker build -t road-rage-clipper .   # production image
```

See [`CLAUDE.md`](CLAUDE.md) for an architecture overview if you're picking this project back up
for further development.

---

## Privacy & security

- `/upload` and its API are behind a single shared password (`APP_PASSWORD`), stored via a
  signed, httpOnly session cookie.
- The MCP server is behind a separate bearer token (`MCP_API_KEY`).
- Generated files are never publicly listable - downloads are short-lived signed URLs (presigned
  R2 URLs, or a locally-signed URL scheme when running without R2).
- Source videos and clips are automatically deleted after `RETENTION_HOURS` (default 48).
- User-supplied video URLs are validated against SSRF: only `http`/`https`, no credentials in the
  URL, hostname resolution is checked against private/loopback/link-local/metadata ranges before
  *and* at each redirect hop, and downloads are size-capped while streaming (not just via a
  `Content-Length` header, which can lie).
- API keys are only ever used server-side and are never sent to the browser.
- `.env` is git-ignored; only `.env.example` (with placeholder values) is committed.
