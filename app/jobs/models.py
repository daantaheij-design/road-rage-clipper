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


class NarrationCue(BaseModel):
    beat: str  # hook | setup | escalation | main_event | payoff
    text: str
    start_seconds: float
    skip: bool = False


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
        }
