from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    RENDERING = "rendering"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class WordTimingRecord(BaseModel):
    """Persisted per-word timing for one narration cue's synthesized audio
    (from ElevenLabs' character-level alignment - see
    app.pipeline.tts._characters_to_words). Relative to the start of that
    cue's own audio clip, not the clip or the source video. Saved alongside
    audio_storage_key so word-by-word captions survive a render-only retry
    without ever calling ElevenLabs again."""

    text: str
    start: float
    end: float


class NarrationCue(BaseModel):
    beat: str  # hook | setup | escalation | main_event | payoff
    text: str
    start_seconds: float
    skip: bool = False
    # Storage key of this cue's synthesized ElevenLabs narration audio,
    # uploaded as soon as it's generated. Lets a render-only retry reuse the
    # already-paid-for TTS audio instead of calling ElevenLabs again.
    audio_storage_key: str | None = None
    word_timings: list[WordTimingRecord] = Field(default_factory=list)


class CropKeyframe(BaseModel):
    """One point in a clip's smart-crop pan plan - see app/pipeline/crop.py.
    time_seconds is relative to the clip's own start (0 = clip start)."""

    time_seconds: float
    focus_x: float = 0.5
    focus_y: float = 0.5
    confidence: float = 0.0


class BBox(BaseModel):
    """Normalized (0-1) bounding box relative to the SOURCE frame Claude was
    shown - not the cropped/output frame. See app/pipeline/geometry.py for
    the source -> crop -> output pixel transform."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.02
    height: float = 0.02

    def clamped(self) -> BBox:
        x = max(0.0, min(1.0, self.x))
        y = max(0.0, min(1.0, self.y))
        w = max(0.01, min(1.0, self.width))
        h = max(0.01, min(1.0, self.height))
        if x + w > 1.0:
            x = max(0.0, 1.0 - w)
        if y + h > 1.0:
            y = max(0.0, 1.0 - h)
        return BBox(x=x, y=y, width=w, height=h)


class Target(BaseModel):
    """The visual subject a circle/arrow/punch-zoom effect points at."""

    description: str = ""
    bbox: BBox | None = None
    confidence: float = 0.0


# Effects that only draw/zoom on top of the existing timeline (no change to
# rendered duration) vs. effects that restructure the timeline itself.
OVERLAY_EFFECT_TYPES = {"circle", "arrow", "punch_zoom"}
TIMELINE_EFFECT_TYPES = {"freeze", "slow_motion", "replay"}
EFFECT_TYPES = OVERLAY_EFFECT_TYPES | TIMELINE_EFFECT_TYPES


class Effect(BaseModel):
    """One visual-attention effect, in clip-relative seconds (0 = clip
    start, matching narration_cues/crop_keyframes)."""

    type: str  # circle | arrow | punch_zoom | freeze | slow_motion | replay
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    target: Target | None = None
    zoom: float = 1.2  # punch_zoom only
    speed: float = 0.6  # slow_motion/replay playback speed (1.0 = normal)


class Teaser(BaseModel):
    """Optional cold-open: a brief glimpse of a later moment in this same
    clip's own footage, played before the normal chronological start. In
    ABSOLUTE source-video seconds (unlike everything else on Clip, which is
    clip-relative) since it's drawn from later in the clip's own window."""

    enabled: bool = False
    source_start: float = 0.0
    source_end: float = 0.0


class TimelineSegment(BaseModel):
    """One piece of the final rendered timeline - persisted for
    inspectability/debugging (see app/pipeline/timeline.py, which is the
    authoritative, re-derived-at-render-time source of truth)."""

    kind: str  # normal | freeze | slow_motion | replay | zoom | teaser
    source_start: float = 0.0
    source_end: float = 0.0
    output_duration: float = 0.0
    speed: float = 1.0


class TranscriptWordRecord(BaseModel):
    """Persisted form of app.pipeline.transcribe.TranscriptWord - saved on
    the Job once transcription succeeds so a retry never has to call
    ElevenLabs Scribe again."""

    text: str
    start: float
    end: float
    kind: str = "word"


class ClipScores(BaseModel):
    visual_action: int = 0
    escalation: int = 0
    surprise: int = 0
    tension: int = 0
    hook_potential: int = 0
    understandable_context: int = 0
    payoff: int = 0
    retention: int = 0
    reaction: int = 0
    uniqueness: int = 0

    @property
    def total(self) -> int:
        return sum(self.model_dump().values())


class Clip(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    score: int = 0
    hook: str = ""
    explanation: str = ""
    start_seconds: float = 0
    end_seconds: float = 0
    duration_seconds: float = 0
    scores: ClipScores = Field(default_factory=ClipScores)
    narration_cues: list[NarrationCue] = Field(default_factory=list)
    crop_keyframes: list[CropKeyframe] = Field(default_factory=list)
    effects: list[Effect] = Field(default_factory=list)
    teaser: Teaser | None = None
    # Informational snapshot of the derived render plan - see TimelineSegment.
    timeline_segments: list[TimelineSegment] = Field(default_factory=list)
    storage_key: str | None = None
    filename: str | None = None

    def to_output(self, download_url: str | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "score": self.score,
            "hook": self.hook,
            "explanation": self.explanation,
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "download_url": download_url,
            # Short labels for the UI's "Effects: Arrow · Punch Zoom" chip -
            # not the full structured plan (that stays available server-side
            # for debugging via the persisted Clip, not exposed over the API).
            "effects": sorted({e.type for e in self.effects}),
        }


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    message: str = "Queued"
    source_url: str | None = None
    source_filename: str | None = None
    number_of_clips: int = 3
    style: str | None = None
    clips: list[Clip] = Field(default_factory=list)
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    expires_at: float | None = None
    source_storage_key: str | None = None

    # Set once transcription, Claude visual analysis, clip selection, story
    # generation, and per-cue TTS have all completed successfully and been
    # persisted (transcript_words below, and each clip's narration_cues with
    # their audio_storage_key set). When true, a retry can skip straight to
    # rendering - no Anthropic or ElevenLabs calls needed - re-rendering only
    # the clips that don't have a storage_key yet (some may have already
    # rendered successfully before a later clip's render failed).
    ready_to_render: bool = False
    transcript_words: list[TranscriptWordRecord] = Field(default_factory=list)

    # Which generation of the clip-selection/story-planning logic
    # (app.pipeline.pipeline.ANALYSIS_VERSION) produced this job's clips -
    # stamped once ready_to_render is set. 0 means "predates versioning"
    # (a job analyzed before this field existed). Purely informational: it
    # lets a human/UI recognize that an old job's cached plan came from
    # different selection logic (e.g. the old "shortest possible clip"
    # behavior) - it never triggers an automatic re-analysis. Re-running
    # the paid Anthropic/ElevenLabs steps on an old job requires an
    # explicit new job, not a silent upgrade of stale cached data.
    analysis_version: int = 0

    def public_dict(self, download_urls: dict[str, str] | None = None) -> dict[str, Any]:
        download_urls = download_urls or {}
        return {
            "job_id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "number_of_clips_requested": self.number_of_clips,
            "clips": [c.to_output(download_urls.get(c.id)) for c in self.clips],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # True once the job has saved analysis it can resume from - the
            # web UI and MCP client use this to offer a "Retry render"
            # action that skips paying for AI analysis/TTS again.
            "resumable": self.ready_to_render and self.status == JobStatus.FAILED,
            "analysis_version": self.analysis_version,
        }
