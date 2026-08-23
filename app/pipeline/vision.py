"""Two-pass Claude vision analysis of the video.

Pass 1 (cheap, wide): sparse low-res frames (~1 every 1-2s) across the whole
video plus the transcript are shown to Claude in batches, asking only "does
anything road-rage-relevant seem to be happening here" - producing a list of
rough candidate time windows.

Pass 2 (focused, detailed): for each surviving candidate, denser frames
(several per second) are extracted just around that window and shown to
Claude again, this time asking for a full breakdown: what's actually
visible, a 0-100 score across ten retention-relevant dimensions, a natural
clip start/end, and a short story (hook / setup / escalation / main event /
payoff) with narration cues.

Both passes force Claude to respond via a single tool call so the output is
always structured, parseable JSON - no prose-scraping.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field

import anthropic

from app.config import get_settings
from app.pipeline.ffmpeg_utils import Frame
from app.pipeline.transcribe import Transcript

logger = logging.getLogger(__name__)

PASS1_BATCH_SIZE = 24
PASS1_BATCH_OVERLAP = 2
PASS2_CONCURRENCY = 3

SCORE_DIMENSIONS = [
    "visual_action",
    "escalation",
    "surprise",
    "tension",
    "hook_potential",
    "understandable_context",
    "payoff",
    "retention",
    "reaction",
    "uniqueness",
]

OBSERVABLE_ONLY_RULE = (
    "Describe only what is visibly happening or audibly said. Never claim to know a "
    "driver's intent, feelings, or motivation. Say 'the driver appears to brake sharply' "
    "or 'the car swerves into the lane', not 'the driver wanted to cause an accident' or "
    "'he was furious'. Use hedged, observation-based language throughout."
)


def _client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)


def _image_block(frame: Frame) -> dict:
    data = base64.standard_b64encode(frame.path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


@dataclass
class CandidateWindow:
    start_seconds: float
    end_seconds: float
    suspicion: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class NarrationCueDraft:
    beat: str
    text: str
    start_seconds: float
    skip: bool


@dataclass
class MomentAnalysis:
    is_moment: bool
    title: str
    explanation: str
    start_seconds: float
    end_seconds: float
    scores: dict[str, int]
    hook_text: str
    narration_cues: list[NarrationCueDraft]

    @property
    def total_score(self) -> int:
        return sum(self.scores.get(k, 0) for k in SCORE_DIMENSIONS)


_FLAG_TOOL = {
    "name": "flag_candidates",
    "description": (
        "Report time windows in this batch of frames that might show a road-rage-relevant "
        "moment worth a closer look (or an empty list if nothing stands out)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_seconds": {"type": "number"},
                        "end_seconds": {"type": "number"},
                        "suspicion": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "How likely this window contains an interesting road-rage moment.",
                        },
                        "reason": {"type": "string", "description": "One short, observation-only sentence."},
                    },
                    "required": ["start_seconds", "end_seconds", "suspicion", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    },
}

_ANALYZE_TOOL = {
    "name": "submit_moment_analysis",
    "description": "Submit the full analysis of one candidate road-rage moment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_moment": {
                "type": "boolean",
                "description": "False if on closer inspection this is not actually an interesting moment.",
            },
            "title": {"type": "string", "description": "Short internal title, e.g. 'Highway cutoff and brake check'."},
            "explanation": {
                "type": "string",
                "description": "2-4 sentences describing what visibly happens, in order. Observable facts only.",
            },
            "start_seconds": {
                "type": "number",
                "description": (
                    "Absolute video timestamp where the CLIP should start - include a few seconds of lead-in "
                    "before the incident so the viewer understands the setup, not just the exact incident frame."
                ),
            },
            "end_seconds": {
                "type": "number",
                "description": "Absolute video timestamp where the clip should end, after a natural payoff/resolution.",
            },
            "scores": {
                "type": "object",
                "properties": {dim: {"type": "integer", "minimum": 0, "maximum": 10} for dim in SCORE_DIMENSIONS},
                "required": SCORE_DIMENSIONS,
                "additionalProperties": False,
            },
            "hook_text": {
                "type": "string",
                "description": (
                    "One punchy on-screen hook sentence (<=12 words) that creates curiosity without spoiling the "
                    "payoff. Original, based on this specific footage - do not reuse generic stock phrases."
                ),
            },
            "narration_cues": {
                "type": "array",
                "description": (
                    "Up to 5 short narration lines, one per story beat (hook, setup, escalation, main_event, "
                    "payoff). Set skip=true for beats where the original audio/video already tells the story and "
                    "narration would just talk over important reactions, honking, or arguments - do not narrate "
                    "the whole clip."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "beat": {"type": "string", "enum": ["hook", "setup", "escalation", "main_event", "payoff"]},
                        "text": {"type": "string", "description": "<= 18 words, natural spoken English, or empty if skip=true."},
                        "start_seconds": {
                            "type": "number",
                            "description": "When this line should start playing, in seconds relative to the CLIP start (0 = clip start).",
                        },
                        "skip": {"type": "boolean"},
                    },
                    "required": ["beat", "text", "start_seconds", "skip"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "is_moment",
            "title",
            "explanation",
            "start_seconds",
            "end_seconds",
            "scores",
            "hook_text",
            "narration_cues",
        ],
        "additionalProperties": False,
    },
}


async def _call_tool(client: anthropic.AsyncAnthropic, *, system: str, content: list[dict], tool: dict, model: str) -> dict:
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError(f"Claude did not return a tool_use block for {tool['name']}")


def _batches(frames: list[Frame], size: int, overlap: int) -> list[list[Frame]]:
    if not frames:
        return []
    batches = []
    step = max(1, size - overlap)
    for i in range(0, len(frames), step):
        batch = frames[i : i + size]
        if batch:
            batches.append(batch)
        if i + size >= len(frames):
            break
    return batches


async def scan_for_candidates(
    frames: list[Frame],
    transcript: Transcript,
    *,
    model: str | None = None,
) -> list[CandidateWindow]:
    """Pass 1: sparse-frame scan of the whole video for candidate windows."""
    settings = get_settings()
    model = model or settings.claude_model
    client = _client()

    system = (
        "You are helping a private individual review their own road/dashcam footage to find "
        "moments worth turning into short highlight clips: cars cutting each other off, dangerous "
        "overtakes, sudden braking, apparent brake checks, drivers getting out of their cars, "
        "arguments, gestures, honking, near collisions, blocking another vehicle, aggressive "
        "driving, unexpected escalation, funny or dramatic reactions, or a clear setup-and-payoff "
        "moment. You are shown a sparse sequence of low-resolution frames sampled roughly every "
        "1-2 seconds, each labelled with its timestamp, plus the transcript for that stretch if "
        "there is speech. This is only a coarse first pass to flag windows for closer review - "
        "bias toward flagging anything plausibly interesting rather than being sure. "
        + OBSERVABLE_ONLY_RULE
    )

    batches = _batches(frames, PASS1_BATCH_SIZE, PASS1_BATCH_OVERLAP)
    all_candidates: list[CandidateWindow] = []

    sem = asyncio.Semaphore(PASS2_CONCURRENCY)

    async def run_batch(batch: list[Frame]) -> list[CandidateWindow]:
        async with sem:
            t0, t1 = batch[0].timestamp, batch[-1].timestamp
            transcript_text = transcript.text_in_range(t0, t1)
            content: list[dict] = []
            for f in batch:
                content.append({"type": "text", "text": f"[t={f.timestamp:.1f}s]"})
                content.append(_image_block(f))
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Transcript for this stretch ({t0:.1f}s-{t1:.1f}s): "
                        f"{transcript_text or '(no speech detected)'}\n\n"
                        "Flag any candidate windows now."
                    ),
                }
            )
            try:
                result = await _call_tool(client, system=system, content=content, tool=_FLAG_TOOL, model=model)
            except Exception:
                logger.exception("pass1 batch %.1f-%.1f failed", t0, t1)
                return []
            out = []
            for c in result.get("candidates", []):
                try:
                    out.append(
                        CandidateWindow(
                            start_seconds=float(c["start_seconds"]),
                            end_seconds=float(c["end_seconds"]),
                            suspicion=int(c["suspicion"]),
                            reasons=[str(c.get("reason", ""))],
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    continue
            return out

    results = await asyncio.gather(*(run_batch(b) for b in batches))
    for r in results:
        all_candidates.extend(r)

    return merge_windows(all_candidates)


def merge_windows(candidates: list[CandidateWindow], gap_seconds: float = 4.0) -> list[CandidateWindow]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.start_seconds)
    merged: list[CandidateWindow] = [ordered[0]]
    for c in ordered[1:]:
        last = merged[-1]
        if c.start_seconds <= last.end_seconds + gap_seconds:
            last.end_seconds = max(last.end_seconds, c.end_seconds)
            last.suspicion = max(last.suspicion, c.suspicion)
            last.reasons.extend(c.reasons)
        else:
            merged.append(c)
    return merged


async def analyze_candidate(
    candidate: CandidateWindow,
    dense_frames: list[Frame],
    transcript: Transcript,
    *,
    video_duration: float,
    model: str | None = None,
) -> MomentAnalysis | None:
    """Pass 2: dense-frame close analysis of one candidate window."""
    settings = get_settings()
    model = model or settings.claude_model
    client = _client()

    if not dense_frames:
        return None

    t0, t1 = dense_frames[0].timestamp, dense_frames[-1].timestamp
    padded_start = max(0.0, t0 - 8)
    transcript_text = transcript.text_in_range(padded_start, t1 + 3)

    system = (
        "You are analyzing one specific moment from a private individual's own road/dashcam "
        "footage in detail, to decide whether it's worth a vertical short-form highlight clip and "
        "to help write a short factual story about it. You are shown a dense sequence of frames "
        "(several per second) covering this moment and its immediate surroundings, each labelled "
        "with its absolute timestamp in the source video, plus the spoken transcript nearby. "
        + OBSERVABLE_ONLY_RULE
        + " Prefer a clip with an understandable beginning (what led to it), middle (the incident), "
        "and end (how it resolved or the reaction to it) - include a few seconds of lead-in before "
        "the incident itself rather than starting exactly on the action. Target a total clip length "
        "of 25-60 seconds, and only go up to about 90 seconds if the story genuinely needs it."
    )

    content: list[dict] = [
        {
            "type": "text",
            "text": f"Reported candidate window: {candidate.start_seconds:.1f}s-{candidate.end_seconds:.1f}s. "
            f"Initial reasons: {'; '.join(candidate.reasons) or 'n/a'}. Source video duration: {video_duration:.1f}s.",
        }
    ]
    for f in dense_frames:
        content.append({"type": "text", "text": f"[t={f.timestamp:.2f}s]"})
        content.append(_image_block(f))
    content.append(
        {
            "type": "text",
            "text": f"Nearby transcript ({padded_start:.1f}s-{t1 + 3:.1f}s): {transcript_text or '(no speech detected)'}\n\n"
            "Submit your full analysis now.",
        }
    )

    try:
        result = await _call_tool(client, system=system, content=content, tool=_ANALYZE_TOOL, model=model)
    except Exception:
        logger.exception("pass2 analysis failed for %.1f-%.1f", candidate.start_seconds, candidate.end_seconds)
        return None

    if not result.get("is_moment", False):
        return None

    scores = {dim: int(result.get("scores", {}).get(dim, 0)) for dim in SCORE_DIMENSIONS}
    cues = [
        NarrationCueDraft(
            beat=str(c.get("beat", "")),
            text=str(c.get("text", "")),
            start_seconds=float(c.get("start_seconds", 0)),
            skip=bool(c.get("skip", False)),
        )
        for c in result.get("narration_cues", [])
    ]

    try:
        start_seconds = float(result["start_seconds"])
        end_seconds = float(result["end_seconds"])
    except (KeyError, ValueError, TypeError):
        return None

    return MomentAnalysis(
        is_moment=True,
        title=str(result.get("title", "Road rage moment")),
        explanation=str(result.get("explanation", "")),
        start_seconds=max(0.0, start_seconds),
        end_seconds=min(video_duration, end_seconds),
        scores=scores,
        hook_text=str(result.get("hook_text", "")),
        narration_cues=cues,
    )
