"""End-to-end pipeline: source video in, finished vertical clips out.

This is the orchestrator that a background job (see app/jobs/runner.py)
drives. Every stage updates the Job's status/progress in the store so the
/upload page (and the MCP get_clip_job tool) can poll for live progress.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

from app.config import get_settings
from app.jobs.models import Clip, ClipScores, Job, JobStatus, NarrationCue
from app.jobs.store import get_job_store
from app.pipeline import captions as captions_mod
from app.pipeline import ffmpeg_utils, vision
from app.pipeline.download import DownloadError, download_video
from app.pipeline.render import NarrationTrack, render_vertical_clip
from app.pipeline.scoring import select_clips
from app.pipeline.transcribe import Transcript, transcribe_audio
from app.pipeline.tts import synthesize_narration
from app.storage import get_storage

logger = logging.getLogger(__name__)

PASS1_FPS = 2 / 3  # one frame roughly every 1.5s
PASS2_FPS = 4.0
PASS2_PAD_SECONDS = 6.0
PASS2_CONCURRENCY = 3


class PipelineError(Exception):
    pass


async def _update(job: Job, *, status: JobStatus | None = None, progress: int | None = None, message: str | None = None) -> None:
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = max(job.progress, progress)
    if message is not None:
        job.message = message
    await get_job_store().save(job)


def _workdir(job_id: str) -> Path:
    d = get_settings().tmp_path / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _acquire_source(job: Job, workdir: Path) -> Path:
    existing = sorted(workdir.glob("source.*"))
    existing = [p for p in existing if p.is_file()]
    if existing:
        return existing[0]

    if not job.source_url:
        raise PipelineError("No source video URL or uploaded file was provided")

    settings = get_settings()
    dest = workdir / "source.mp4"
    try:
        await download_video(
            job.source_url,
            dest,
            max_bytes=int(settings.max_video_mb * 1024 * 1024),
            timeout_seconds=settings.download_timeout_seconds,
        )
    except DownloadError as exc:
        raise PipelineError(f"Could not download video: {exc}") from exc
    return dest


async def _analyze(
    source_path: Path, workdir: Path, video_duration: float, transcript: Transcript
) -> list[vision.MomentAnalysis]:
    sparse_dir = workdir / "frames_sparse"
    sparse_frames = await ffmpeg_utils.extract_frames(
        source_path, sparse_dir, fps=PASS1_FPS, scale_width=320, prefix="sparse"
    )
    logger.info("extracted %d sparse frames", len(sparse_frames))

    candidates = await vision.scan_for_candidates(sparse_frames, transcript)
    logger.info("pass1 found %d candidate windows", len(candidates))

    # Keep this bounded even on a long, noisy video.
    candidates = sorted(candidates, key=lambda c: c.suspicion, reverse=True)[:20]

    sem = asyncio.Semaphore(PASS2_CONCURRENCY)

    async def run_one(idx: int, cand: vision.CandidateWindow) -> vision.MomentAnalysis | None:
        async with sem:
            start = max(0.0, cand.start_seconds - PASS2_PAD_SECONDS)
            end = min(video_duration, cand.end_seconds + PASS2_PAD_SECONDS)
            dense_dir = workdir / f"frames_dense_{idx}"
            dense_frames = await ffmpeg_utils.extract_frames(
                source_path, dense_dir, fps=PASS2_FPS, start=start, duration=end - start, scale_width=512, prefix="dense"
            )
            return await vision.analyze_candidate(cand, dense_frames, transcript, video_duration=video_duration)

    results = await asyncio.gather(*(run_one(i, c) for i, c in enumerate(candidates)))
    return [r for r in results if r is not None]


async def _render_selected_clip(
    *,
    source_path: Path,
    media_info: ffmpeg_utils.MediaInfo,
    analysis: vision.MomentAnalysis,
    transcript: Transcript,
    workdir: Path,
    clip_index: int,
) -> Clip:
    clip = Clip(
        title=analysis.title,
        score=analysis.total_score,
        hook=analysis.hook_text,
        explanation=analysis.explanation,
        start_seconds=analysis.start_seconds,
        end_seconds=analysis.end_seconds,
        duration_seconds=analysis.end_seconds - analysis.start_seconds,
        scores=ClipScores(**analysis.scores),
    )

    clip_dir = workdir / f"clip_{clip.id}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    narration_tracks: list[NarrationTrack] = []
    narration_for_captions: list[tuple[float, float, str]] = []
    for i, cue in enumerate(analysis.narration_cues):
        if cue.skip or not cue.text.strip():
            continue
        rel_start = max(0.0, min(clip.duration_seconds, cue.start_seconds))
        audio_path = clip_dir / f"narration_{i}.mp3"
        try:
            await synthesize_narration(cue.text, audio_path)
        except Exception:
            logger.exception("narration synthesis failed for clip %s cue %d", clip.id, i)
            continue
        dur = await ffmpeg_utils.duration_of(audio_path)
        narration_tracks.append(NarrationTrack(audio_path=audio_path, start_seconds=rel_start, duration_seconds=dur))
        narration_for_captions.append((rel_start, rel_start + dur, cue.text))
        clip.narration_cues.append(NarrationCue(beat=cue.beat, text=cue.text, start_seconds=rel_start))

    words = [
        captions_mod.Word(text=w.text, start=w.start, end=w.end)
        for w in transcript.words_in_range(analysis.start_seconds, analysis.end_seconds)
    ]
    ass_path = clip_dir / "captions.ass"
    captions_mod.build_ass_captions(
        output_path=ass_path,
        clip_start=analysis.start_seconds,
        clip_end=analysis.end_seconds,
        transcript_words=words,
        narration_cues=narration_for_captions,
        hook_text=analysis.hook_text,
    )

    output_path = clip_dir / "output.mp4"
    await render_vertical_clip(
        source_path=source_path,
        start_seconds=analysis.start_seconds,
        end_seconds=analysis.end_seconds,
        output_path=output_path,
        media_info=media_info,
        narration_tracks=narration_tracks,
        captions_ass_path=ass_path,
    )

    clip.filename = f"road-rage-clip-{clip_index + 1}.mp4"
    return clip


async def process_job(job_id: str) -> None:
    store = get_job_store()
    settings = get_settings()
    job = await store.get(job_id)
    if job is None:
        logger.error("job %s vanished before processing", job_id)
        return

    workdir = _workdir(job_id)
    storage = get_storage()

    try:
        await _update(job, status=JobStatus.DOWNLOADING, progress=5, message="Fetching video")
        source_path = await _acquire_source(job, workdir)

        media_info = await ffmpeg_utils.probe(source_path)
        if media_info.duration_seconds > settings.max_video_duration_seconds:
            raise PipelineError(
                f"Video is {media_info.duration_seconds / 60:.1f} minutes long; the limit is "
                f"{settings.max_video_duration_seconds / 60:.0f} minutes"
            )

        await _update(job, status=JobStatus.TRANSCRIBING, progress=15, message="Transcribing audio")
        if media_info.has_audio:
            audio_path = workdir / "audio.wav"
            await ffmpeg_utils.extract_audio(source_path, audio_path)
            transcript = await transcribe_audio(audio_path)
        else:
            transcript = Transcript(text="", words=[])

        await _update(job, status=JobStatus.ANALYZING, progress=25, message="Scanning footage for road-rage moments")
        analyses = await _analyze(source_path, workdir, media_info.duration_seconds, transcript)
        if not analyses:
            raise PipelineError("Could not find any clear road-rage moments in this video")

        selected = select_clips(analyses, number_of_clips=job.number_of_clips, video_duration=media_info.duration_seconds)
        if not selected:
            raise PipelineError("Found candidate moments but none produced a usable clip length")

        await _update(
            job,
            status=JobStatus.RENDERING,
            progress=40,
            message=f"Rendering {len(selected)} clip(s)",
        )

        clips: list[Clip] = []
        for i, analysis in enumerate(selected):
            await _update(
                job,
                progress=40 + int(40 * i / max(1, len(selected))),
                message=f"Rendering clip {i + 1} of {len(selected)}: {analysis.title}",
            )
            clip = await _render_selected_clip(
                source_path=source_path,
                media_info=media_info,
                analysis=analysis,
                transcript=transcript,
                workdir=workdir,
                clip_index=i,
            )
            clips.append(clip)

        await _update(job, status=JobStatus.UPLOADING, progress=85, message="Uploading finished clips")
        for i, clip in enumerate(clips):
            local_path = workdir / f"clip_{clip.id}" / "output.mp4"
            key = f"jobs/{job.id}/clips/{clip.id}.mp4"
            storage.upload_file(local_path, key, content_type="video/mp4")
            clip.storage_key = key
            job.clips.append(clip)
            await _update(job, progress=85 + int(10 * (i + 1) / len(clips)))

        job.expires_at = time.time() + settings.retention_hours * 3600
        await _update(job, status=JobStatus.COMPLETED, progress=100, message=f"Done - {len(clips)} clip(s) ready")

    except PipelineError as exc:
        logger.warning("job %s failed: %s", job_id, exc)
        job.error = str(exc)
        await _update(job, status=JobStatus.FAILED, message=str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("job %s failed unexpectedly", job_id)
        job.error = "An unexpected error occurred while processing this video."
        await _update(job, status=JobStatus.FAILED, message=job.error)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
