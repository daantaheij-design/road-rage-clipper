"""End-to-end pipeline: source video in, finished vertical clips out.

This is the orchestrator that a background job (see app/jobs/runner.py)
drives. Every stage updates the Job's status/progress in the store so the
/upload page (and the MCP get_clip_job tool) can poll for live progress.

Resumability: transcription (ElevenLabs) and visual analysis/story
generation/narration (Anthropic + ElevenLabs TTS) are the expensive, paid
steps. As soon as each one succeeds its result is persisted on the Job
itself - the transcript as `Job.transcript_words`, and each selected
moment's analysis plus its synthesized narration audio (uploaded to storage
immediately, see `Clip.narration_cues[].audio_storage_key`) as a `Clip` with
`Job.ready_to_render = True`. If FFmpeg rendering then fails (this is the
step most likely to fail on a small container - see render.py), retrying
the job (`app/jobs/service.py::retry_job`) re-enters this function, sees
`ready_to_render`, and skips straight to rendering - no Anthropic or
ElevenLabs calls are made again. Per-clip: any clip that already has a
`storage_key` (i.e. it rendered and uploaded successfully in a previous
attempt) is also skipped, so a failure partway through rendering multiple
clips doesn't re-render the ones that already finished.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

from app.config import get_settings
from app.jobs.models import Clip, ClipScores, Job, JobStatus, NarrationCue, TranscriptWordRecord, WordTimingRecord
from app.jobs.models import CropKeyframe as CropKeyframeRecord
from app.jobs.store import get_job_store
from app.pipeline import captions as captions_mod
from app.pipeline import ffmpeg_utils, vision
from app.pipeline.crop import CropKeyframe
from app.pipeline.download import DownloadError, download_video
from app.pipeline.render import NarrationTrack, render_vertical_clip
from app.pipeline.scoring import select_clips
from app.pipeline.transcribe import Transcript, TranscriptWord, transcribe_audio
from app.pipeline.tts import synthesize_narration
from app.storage import Storage, get_storage

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


def _words_to_transcript(records: list[TranscriptWordRecord]) -> Transcript:
    return Transcript(text="", words=[TranscriptWord(text=r.text, start=r.start, end=r.end, kind=r.kind) for r in records])


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


def _build_clip_from_analysis(analysis: vision.MomentAnalysis, clip_index: int) -> Clip:
    return Clip(
        title=analysis.title,
        score=analysis.total_score,
        hook=analysis.hook_text,
        explanation=analysis.explanation,
        start_seconds=analysis.start_seconds,
        end_seconds=analysis.end_seconds,
        duration_seconds=analysis.end_seconds - analysis.start_seconds,
        scores=ClipScores(**analysis.scores),
        crop_keyframes=[
            CropKeyframeRecord(
                time_seconds=kf.time_seconds, focus_x=kf.focus_x, focus_y=kf.focus_y, confidence=kf.confidence
            )
            for kf in analysis.crop_keyframes
        ],
        filename=f"road-rage-clip-{clip_index + 1}.mp4",
    )


async def _synthesize_and_store_narration(
    analysis: vision.MomentAnalysis, clip: Clip, clip_dir: Path, storage: Storage, job_id: str
) -> None:
    """Generate narration audio for each cue and upload it immediately -
    this is the ElevenLabs TTS cost we want to never pay twice, so it's
    durably saved (storage, not local disk) the moment it's produced."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i, cue in enumerate(analysis.narration_cues):
        if cue.skip or not cue.text.strip():
            continue
        rel_start = max(0.0, min(clip.duration_seconds, cue.start_seconds))
        audio_path = clip_dir / f"narration_{i}.mp3"
        try:
            narration_audio = await synthesize_narration(cue.text, audio_path)
        except Exception:
            logger.exception("narration synthesis failed for clip %s cue %d", clip.id, i)
            continue
        key = f"jobs/{job_id}/narration/{clip.id}/{i}.mp3"
        storage.upload_file(audio_path, key, content_type="audio/mpeg")
        clip.narration_cues.append(
            NarrationCue(
                beat=cue.beat,
                text=cue.text,
                start_seconds=rel_start,
                audio_storage_key=key,
                word_timings=[WordTimingRecord(text=w.text, start=w.start, end=w.end) for w in narration_audio.words],
            )
        )


async def _fetch_narration_tracks(clip: Clip, clip_dir: Path, storage: Storage) -> list[NarrationTrack]:
    """Re-download this clip's already-synthesized narration audio from
    storage. Used for every render (first attempt and retries alike) so
    there's exactly one code path for "get the narration audio ready to
    render with" regardless of whether the local scratch directory still
    has it."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    tracks: list[NarrationTrack] = []
    for i, cue in enumerate(clip.narration_cues):
        if not cue.audio_storage_key:
            continue
        audio_path = clip_dir / f"narration_{i}.mp3"
        storage.download_file(cue.audio_storage_key, audio_path)
        duration = await ffmpeg_utils.duration_of(audio_path)
        tracks.append(NarrationTrack(audio_path=audio_path, start_seconds=cue.start_seconds, duration_seconds=duration))
    return tracks


async def _render_and_upload_clip(
    *,
    clip: Clip,
    source_path: Path,
    media_info: ffmpeg_utils.MediaInfo,
    transcript: Transcript,
    tracks: list[NarrationTrack],
    clip_dir: Path,
    storage: Storage,
    job_id: str,
) -> None:
    words = [
        captions_mod.Word(text=w.text, start=w.start, end=w.end)
        for w in transcript.words_in_range(clip.start_seconds, clip.end_seconds)
    ]
    # clip.narration_cues and tracks are built/fetched in lockstep (same
    # order, same successfully-synthesized subset), so zipping them pairs
    # each cue's text/word timings with its actual audio placement in the
    # clip.
    narration_for_captions = [
        (track.start_seconds, track.start_seconds + track.duration_seconds, cue.text)
        for cue, track in zip(clip.narration_cues, tracks, strict=False)
    ]
    narration_words = [
        captions_mod.Word(text=w.text, start=track.start_seconds + w.start, end=track.start_seconds + w.end)
        for cue, track in zip(clip.narration_cues, tracks, strict=False)
        for w in cue.word_timings
    ]
    narration_words.sort(key=lambda w: w.start)

    ass_path = clip_dir / "captions.ass"
    captions_mod.build_ass_captions(
        output_path=ass_path,
        clip_start=clip.start_seconds,
        clip_end=clip.end_seconds,
        transcript_words=words,
        narration_cues=narration_for_captions,
        narration_words=narration_words,
        hook_text=clip.hook,
    )

    crop_keyframes = [
        CropKeyframe(time_seconds=kf.time_seconds, focus_x=kf.focus_x, focus_y=kf.focus_y, confidence=kf.confidence)
        for kf in clip.crop_keyframes
    ]

    output_path = clip_dir / "output.mp4"
    await render_vertical_clip(
        source_path=source_path,
        start_seconds=clip.start_seconds,
        end_seconds=clip.end_seconds,
        output_path=output_path,
        media_info=media_info,
        narration_tracks=tracks,
        captions_ass_path=ass_path,
        crop_keyframes=crop_keyframes,
    )

    key = f"jobs/{job_id}/clips/{clip.id}.mp4"
    storage.upload_file(output_path, key, content_type="video/mp4")
    clip.storage_key = key


async def process_job(job_id: str) -> None:
    store = get_job_store()
    settings = get_settings()
    job = await store.get(job_id)
    if job is None:
        logger.error("job %s vanished before processing", job_id)
        return

    workdir = _workdir(job_id)
    storage = get_storage()
    job.error = None  # clear any stale error from a previous failed attempt

    try:
        await _update(job, status=JobStatus.DOWNLOADING, progress=5, message="Fetching video")
        source_path = await _acquire_source(job, workdir)

        media_info = await ffmpeg_utils.probe(source_path)
        if media_info.duration_seconds > settings.max_video_duration_seconds:
            raise PipelineError(
                f"Video is {media_info.duration_seconds / 60:.1f} minutes long; the limit is "
                f"{settings.max_video_duration_seconds / 60:.0f} minutes"
            )

        if job.ready_to_render and job.clips:
            # Everything up through narration TTS already succeeded and is
            # saved - skip straight to rendering. No Anthropic/ElevenLabs
            # calls happen on this path.
            transcript = _words_to_transcript(job.transcript_words)
            await _update(
                job,
                progress=max(job.progress, 40),
                message="Resuming from saved analysis (skipping Claude/ElevenLabs re-analysis)",
            )
        else:
            if job.transcript_words:
                # Transcription succeeded in a previous attempt but
                # something after it failed before analysis finished - reuse
                # the transcript rather than paying for ElevenLabs Scribe again.
                transcript = _words_to_transcript(job.transcript_words)
                await _update(
                    job,
                    status=JobStatus.ANALYZING,
                    progress=25,
                    message="Reusing saved transcript - scanning footage for road-rage moments",
                )
            else:
                await _update(job, status=JobStatus.TRANSCRIBING, progress=15, message="Transcribing audio")
                if media_info.has_audio:
                    audio_path = workdir / "audio.wav"
                    await ffmpeg_utils.extract_audio(source_path, audio_path)
                    transcript = await transcribe_audio(audio_path)
                else:
                    transcript = Transcript(text="", words=[])
                job.transcript_words = [
                    TranscriptWordRecord(text=w.text, start=w.start, end=w.end, kind=w.kind) for w in transcript.words
                ]
                await _update(
                    job,
                    status=JobStatus.ANALYZING,
                    progress=25,
                    message="Scanning footage for road-rage moments",
                )

            analyses = await _analyze(source_path, workdir, media_info.duration_seconds, transcript)
            if not analyses:
                raise PipelineError("Could not find any clear road-rage moments in this video")

            selected = select_clips(
                analyses, number_of_clips=job.number_of_clips, video_duration=media_info.duration_seconds
            )
            if not selected:
                raise PipelineError("Found candidate moments but none produced a usable clip length")

            await _update(job, progress=35, message=f"Writing narration for {len(selected)} clip(s)")
            for i, analysis in enumerate(selected):
                clip = _build_clip_from_analysis(analysis, clip_index=i)
                clip_dir = workdir / f"clip_{clip.id}"
                await _synthesize_and_store_narration(analysis, clip, clip_dir, storage, job.id)
                job.clips.append(clip)

            job.ready_to_render = True
            # Start this job's retention clock from the moment the costly AI
            # work is durably saved, so an abandoned failed job still
            # eventually gets cleaned up (along with its uploaded narration
            # audio) even if it's never retried. A later success refreshes it.
            job.expires_at = time.time() + settings.retention_hours * 3600
            await _update(
                job, status=JobStatus.RENDERING, progress=40, message=f"Rendering {len(job.clips)} clip(s)"
            )

        await _update(job, status=JobStatus.RENDERING)
        pending = [c for c in job.clips if not c.storage_key]
        already_done = len(job.clips) - len(pending)
        for i, clip in enumerate(pending):
            await _update(
                job,
                progress=40 + int(40 * (already_done + i) / max(1, len(job.clips))),
                message=f"Rendering clip {already_done + i + 1} of {len(job.clips)}: {clip.title}",
            )
            clip_dir = workdir / f"clip_{clip.id}"
            tracks = await _fetch_narration_tracks(clip, clip_dir, storage)
            await _render_and_upload_clip(
                clip=clip,
                source_path=source_path,
                media_info=media_info,
                transcript=transcript,
                tracks=tracks,
                clip_dir=clip_dir,
                storage=storage,
                job_id=job.id,
            )
            # Save after every clip (not just at the end) so a later clip's
            # render failure doesn't lose this one's finished storage_key.
            await _update(job, progress=40 + int(40 * (already_done + i + 1) / max(1, len(job.clips))))

        job.expires_at = time.time() + settings.retention_hours * 3600
        await _update(job, status=JobStatus.COMPLETED, progress=100, message=f"Done - {len(job.clips)} clip(s) ready")

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
