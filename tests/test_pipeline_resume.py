"""Tests for job resumability: a render-only failure (or an earlier failure
that still saved a transcript) must not repeat the paid Anthropic/ElevenLabs
work on retry.

Every external, paid, or subprocess-based call (download, ffmpeg probe/
frames/audio, ElevenLabs transcription/TTS, Claude vision analysis, ffmpeg
render) is faked out with a call-counting stand-in - only app.pipeline.
pipeline's own orchestration logic, app.jobs.store, and app.storage (local
backend) run for real."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.jobs.models import Job, JobStatus
from app.jobs.store import get_job_store
from app.pipeline import pipeline, vision
from app.pipeline import tts as pipeline_tts
from app.pipeline.ffmpeg_utils import Frame, MediaInfo
from app.pipeline.transcribe import Transcript, TranscriptWord

FAKE_DURATION = 300.0


def _fake_media_info() -> MediaInfo:
    return MediaInfo(duration_seconds=FAKE_DURATION, width=1920, height=1080, fps=30.0, has_audio=True)


def _place_fake_source(job_id: str) -> None:
    workdir = pipeline._workdir(job_id)
    (workdir / "source.mp4").write_bytes(b"fake source video bytes")


class _Harness:
    """Wires up fakes for every external call in the pipeline and counts
    how many times each was actually invoked."""

    def __init__(self, monkeypatch):
        self.calls = {"download": 0, "transcribe": 0, "scan": 0, "analyze": 0, "tts": 0, "render": 0}
        self.render_should_fail = True
        self.effects_to_return: list = []
        self.teaser_to_return = None
        self.last_render_kwargs: dict = {}

        async def fake_download_video(url, dest, *, max_bytes, timeout_seconds):
            self.calls["download"] += 1
            raise AssertionError("download_video should not be called - source is pre-placed in workdir")

        async def fake_probe(path: Path) -> MediaInfo:
            return _fake_media_info()

        async def fake_extract_audio(source_path, out_path, *, sample_rate=16000):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake wav bytes")
            return out_path

        async def fake_extract_frames(*args, **kwargs):
            return [Frame(timestamp=1.0, path=Path("/nonexistent.jpg"))]

        async def fake_transcribe_audio(audio_path):
            self.calls["transcribe"] += 1
            return Transcript(text="", words=[TranscriptWord(text="hello", start=1.0, end=1.5, kind="word")])

        async def fake_scan_for_candidates(frames, transcript, **kwargs):
            self.calls["scan"] += 1
            if self.calls["scan"] == 1 and self.fail_first_scan:
                return []
            return [vision.CandidateWindow(start_seconds=5.0, end_seconds=15.0, suspicion=90, reasons=["test"])]

        async def fake_analyze_candidate(candidate, dense_frames, transcript, *, video_duration, **kwargs):
            self.calls["analyze"] += 1
            return vision.MomentAnalysis(
                is_moment=True,
                title="Test moment",
                explanation="A car cuts off another car.",
                start_seconds=2.0,
                end_seconds=32.0,
                scores={dim: 8 for dim in vision.SCORE_DIMENSIONS},
                hook_text="You won't believe this.",
                narration_cues=[
                    vision.NarrationCueDraft(beat="hook", text="Watch this.", start_seconds=0.0, skip=False),
                ],
                effects=self.effects_to_return,
                teaser=self.teaser_to_return,
            )

        async def fake_synthesize_narration(text, out_path):
            self.calls["tts"] += 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake mp3 bytes")
            return pipeline_tts.NarrationAudio(
                audio_path=out_path,
                words=[pipeline_tts.WordTiming(text=w, start=i * 0.3, end=i * 0.3 + 0.25) for i, w in enumerate(text.split())],
            )

        async def fake_duration_of(path):
            return 2.0

        async def fake_render_vertical_clip(
            *,
            source_path,
            start_seconds,
            end_seconds,
            output_path,
            media_info,
            narration_tracks=None,
            captions_ass_path=None,
            crop_keyframes=None,
            segments=None,
            overlays=None,
            threads=None,
        ):
            self.calls["render"] += 1
            self.last_render_kwargs = {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "narration_tracks": narration_tracks,
                "crop_keyframes": crop_keyframes,
                "segments": segments,
                "overlays": overlays,
            }
            if self.render_should_fail:
                raise RuntimeError("simulated ffmpeg failure")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake mp4 bytes")
            return output_path

        self.fail_first_scan = False

        monkeypatch.setattr(pipeline, "download_video", fake_download_video)
        monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe_audio)
        monkeypatch.setattr(pipeline, "synthesize_narration", fake_synthesize_narration)
        monkeypatch.setattr(pipeline, "render_vertical_clip", fake_render_vertical_clip)
        monkeypatch.setattr(pipeline.ffmpeg_utils, "probe", fake_probe)
        monkeypatch.setattr(pipeline.ffmpeg_utils, "extract_audio", fake_extract_audio)
        monkeypatch.setattr(pipeline.ffmpeg_utils, "extract_frames", fake_extract_frames)
        monkeypatch.setattr(pipeline.ffmpeg_utils, "duration_of", fake_duration_of)
        monkeypatch.setattr(pipeline.vision, "scan_for_candidates", fake_scan_for_candidates)
        monkeypatch.setattr(pipeline.vision, "analyze_candidate", fake_analyze_candidate)


@pytest.fixture
def harness(settings, monkeypatch):
    return _Harness(monkeypatch)


async def test_render_failure_then_retry_skips_ai_reanalysis(settings, harness):
    job = Job(number_of_clips=1, source_url="https://example.com/video.mp4")
    await get_job_store().save(job)

    # --- first attempt: everything succeeds except the render step ---
    _place_fake_source(job.id)
    harness.render_should_fail = True
    await pipeline.process_job(job.id)

    failed = await get_job_store().get(job.id)
    assert failed.status == JobStatus.FAILED
    assert failed.ready_to_render is True
    assert len(failed.clips) == 1
    assert failed.clips[0].storage_key is None
    assert failed.clips[0].narration_cues[0].audio_storage_key is not None  # TTS output was durably saved
    assert harness.calls == {"download": 0, "transcribe": 1, "scan": 1, "analyze": 1, "tts": 1, "render": 1}

    # --- retry (what jobs/service.py::retry_job does): status -> queued ---
    failed.status = JobStatus.QUEUED
    await get_job_store().save(failed)

    _place_fake_source(job.id)  # workdir was wiped after the failed attempt
    harness.render_should_fail = False
    await pipeline.process_job(job.id)

    done = await get_job_store().get(job.id)
    assert done.status == JobStatus.COMPLETED
    assert done.clips[0].storage_key is not None
    assert done.error is None

    # The expensive AI/TTS steps must NOT have run a second time - only
    # rendering (the thing that actually failed) was retried.
    assert harness.calls == {"download": 0, "transcribe": 1, "scan": 1, "analyze": 1, "tts": 1, "render": 2}


async def test_second_of_two_clips_failing_only_reruns_that_clip(settings, monkeypatch, harness):
    # Two candidates -> two clips selected.
    call_n = {"n": 0}

    async def two_candidates(frames, transcript, **kwargs):
        harness.calls["scan"] += 1
        return [
            vision.CandidateWindow(start_seconds=5.0, end_seconds=15.0, suspicion=90, reasons=["a"]),
            vision.CandidateWindow(start_seconds=200.0, end_seconds=210.0, suspicion=80, reasons=["b"]),
        ]

    async def two_analyses(candidate, dense_frames, transcript, *, video_duration, **kwargs):
        harness.calls["analyze"] += 1
        call_n["n"] += 1
        start = candidate.start_seconds
        return vision.MomentAnalysis(
            is_moment=True,
            title=f"Moment {call_n['n']}",
            explanation="Something happens.",
            start_seconds=start,
            end_seconds=start + 30,
            scores={dim: 8 for dim in vision.SCORE_DIMENSIONS},
            hook_text="Hook",
            narration_cues=[vision.NarrationCueDraft(beat="hook", text="Watch.", start_seconds=0.0, skip=False)],
        )

    monkeypatch.setattr(pipeline.vision, "scan_for_candidates", two_candidates)
    monkeypatch.setattr(pipeline.vision, "analyze_candidate", two_analyses)

    render_call_n = {"n": 0}

    async def render_second_clip_fails(**kwargs):
        render_call_n["n"] += 1
        harness.calls["render"] += 1
        if render_call_n["n"] == 2:
            raise RuntimeError("simulated failure on the second clip")
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"fake mp4 bytes")
        return kwargs["output_path"]

    monkeypatch.setattr(pipeline, "render_vertical_clip", render_second_clip_fails)

    job = Job(number_of_clips=2, source_url="https://example.com/video.mp4")
    await get_job_store().save(job)
    _place_fake_source(job.id)
    await pipeline.process_job(job.id)

    failed = await get_job_store().get(job.id)
    assert failed.status == JobStatus.FAILED
    assert len(failed.clips) == 2
    # First clip rendered fine and must not be touched again on retry.
    assert failed.clips[0].storage_key is not None
    assert failed.clips[1].storage_key is None
    assert harness.calls["render"] == 2
    assert harness.calls["tts"] == 2  # narration for both clips was already synthesized

    failed.status = JobStatus.QUEUED
    await get_job_store().save(failed)
    _place_fake_source(job.id)

    async def render_always_succeeds(**kwargs):
        harness.calls["render"] += 1
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"fake mp4 bytes")
        return kwargs["output_path"]

    monkeypatch.setattr(pipeline, "render_vertical_clip", render_always_succeeds)
    await pipeline.process_job(job.id)

    done = await get_job_store().get(job.id)
    assert done.status == JobStatus.COMPLETED
    assert done.clips[0].storage_key is not None
    assert done.clips[1].storage_key is not None
    # Only the one clip that actually needed rendering was rendered again.
    assert harness.calls["render"] == 3
    # No AI/TTS work repeated at all.
    assert harness.calls == {"download": 0, "transcribe": 1, "scan": 1, "analyze": 2, "tts": 2, "render": 3}


async def test_failure_before_ready_to_render_reuses_transcript_but_redoes_analysis(settings, harness):
    harness.fail_first_scan = True
    harness.render_should_fail = False  # this test is only about the analysis stage, not rendering

    job = Job(number_of_clips=1, source_url="https://example.com/video.mp4")
    await get_job_store().save(job)

    _place_fake_source(job.id)
    await pipeline.process_job(job.id)

    failed = await get_job_store().get(job.id)
    assert failed.status == JobStatus.FAILED
    assert failed.ready_to_render is False
    assert failed.clips == []
    assert len(failed.transcript_words) == 1  # transcription succeeded and was saved
    assert harness.calls["transcribe"] == 1
    assert harness.calls["scan"] == 1
    assert harness.calls["analyze"] == 0

    failed.status = JobStatus.QUEUED
    await get_job_store().save(failed)
    _place_fake_source(job.id)
    await pipeline.process_job(job.id)

    done = await get_job_store().get(job.id)
    assert done.status == JobStatus.COMPLETED

    # Transcription was NOT repeated (transcript was reused)...
    assert harness.calls["transcribe"] == 1
    # ...but analysis WAS repeated, since it never completed successfully before.
    assert harness.calls["scan"] == 2
    assert harness.calls["analyze"] == 1


async def test_clip_with_effects_persists_effects_and_uses_segments_render_path(settings, harness):
    harness.render_should_fail = False
    harness.effects_to_return = [
        vision.EffectDraft(type="freeze", start_seconds=5.0, end_seconds=5.4),
    ]
    harness.teaser_to_return = vision.TeaserDraft(enabled=True, source_start=28.0, source_end=29.0)

    job = Job(number_of_clips=1, source_url="https://example.com/video.mp4")
    await get_job_store().save(job)
    _place_fake_source(job.id)
    await pipeline.process_job(job.id)

    done = await get_job_store().get(job.id)
    assert done.status == JobStatus.COMPLETED
    clip = done.clips[0]
    assert len(clip.effects) == 1
    assert clip.effects[0].type == "freeze"
    assert clip.teaser is not None and clip.teaser.enabled
    # The derived render plan was persisted for debugging/inspectability.
    assert clip.timeline_segments
    assert any(s.kind == "freeze" for s in clip.timeline_segments)
    assert any(s.kind == "teaser" for s in clip.timeline_segments)
    # render_vertical_clip was actually called with the segments-based path.
    assert harness.last_render_kwargs["segments"] is not None
    assert harness.last_render_kwargs["crop_keyframes"] is None


async def test_clip_without_effects_uses_legacy_render_path(settings, harness):
    harness.render_should_fail = False
    harness.effects_to_return = []
    harness.teaser_to_return = None

    job = Job(number_of_clips=1, source_url="https://example.com/video.mp4")
    await get_job_store().save(job)
    _place_fake_source(job.id)
    await pipeline.process_job(job.id)

    done = await get_job_store().get(job.id)
    assert done.status == JobStatus.COMPLETED
    assert done.clips[0].effects == []
    assert done.clips[0].timeline_segments == []
    # No effects/teaser -> the plain, pre-effects render path is used.
    assert harness.last_render_kwargs["segments"] is None
    assert harness.last_render_kwargs["crop_keyframes"] is not None


async def test_retry_after_render_failure_with_effects_reuses_persisted_plan_and_makes_zero_new_ai_calls(
    settings, harness
):
    harness.effects_to_return = [
        vision.EffectDraft(
            type="circle",
            start_seconds=1.0,
            end_seconds=2.0,
            target=vision.TargetDraft(
                description="white sedan", confidence=0.9, bbox=vision.BBoxDraft(x=0.5, y=0.4, width=0.2, height=0.2)
            ),
        ),
    ]

    job = Job(number_of_clips=1, source_url="https://example.com/video.mp4")
    await get_job_store().save(job)

    # First attempt: analysis succeeds, render fails.
    _place_fake_source(job.id)
    harness.render_should_fail = True
    await pipeline.process_job(job.id)

    failed = await get_job_store().get(job.id)
    assert failed.status == JobStatus.FAILED
    assert failed.ready_to_render is True
    persisted_effects = failed.clips[0].effects
    assert len(persisted_effects) == 1
    assert persisted_effects[0].type == "circle"

    # Retry: render succeeds this time, using the SAME persisted effect (no
    # new Claude call re-derives it) and zero new Anthropic/ElevenLabs calls.
    harness.render_should_fail = False
    failed.status = JobStatus.QUEUED
    await get_job_store().save(failed)
    _place_fake_source(job.id)
    await pipeline.process_job(job.id)

    done = await get_job_store().get(job.id)
    assert done.status == JobStatus.COMPLETED
    assert done.clips[0].effects[0].type == "circle"
    assert harness.calls == {"download": 0, "transcribe": 1, "scan": 1, "analyze": 1, "tts": 1, "render": 2}
    assert harness.last_render_kwargs["segments"] is not None
    assert harness.last_render_kwargs["overlays"] is not None
    assert len(harness.last_render_kwargs["overlays"]) == 1  # the circle overlay PNG was (re)generated locally
