"""Generate TikTok-style burned-in captions as an .ass subtitle file.

Caption "tracks", never shown at the same time in the same spot so the
screen doesn't get crowded:

- Hook: a short, bold title card for the opening line, top-safe-area.
- Narration / NarrationEmphasis: the AI narrator's words, displayed
  word-by-word synced to ElevenLabs' actual per-word speech timing (see
  `Word` and the `narration_words` param of `build_ass_captions`) rather
  than one static line per sentence - NarrationEmphasis highlights a word
  judged emphatic (see `_is_emphasis_word`). Sits mid-lower in the frame
  (roughly 65% down, within TikTok's UI-safe zone) rather than pinned to
  the very bottom, so it doesn't compete with the bottom-anchored original
  transcript captions or get covered by the platform's own UI chrome.
- Caption: the original speaker's words (from the transcript), shown
  whenever narration isn't active, bottom-safe-area.

If a narration cue's audio was synthesized before word-level alignment was
persisted (or ElevenLabs returned no alignment), `narration_words` falls
back to one whole-line Dialogue event per cue instead - see
`build_ass_captions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_WORDS_PER_PHRASE = 5
MAX_PHRASE_SECONDS = 2.2
MAX_WORD_GAP_SECONDS = 0.6

# How much longer a narration word is displayed past its own spoken end, so
# it doesn't visually disappear mid-pause before the next word starts (but
# never later than the next word's own start, to avoid two words appearing
# to overlap on screen).
NARRATION_WORD_END_PAD_SECONDS = 0.15
MIN_NARRATION_WORD_DISPLAY_SECONDS = 0.08

_HEADER = """[Script Info]
Title: Road Rage Clipper Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,58,&H00FFFFFF,&H000000FF,&H00101010,&H00000000,-1,0,0,0,100,100,0,0,1,4,2,2,70,70,260,1
Style: Narration,DejaVu Sans,64,&H00FFFFFF,&H000000FF,&H00101010,&H00000000,-1,0,0,0,100,100,0,0,1,5,3,2,70,70,660,1
Style: NarrationEmphasis,DejaVu Sans,72,&H0000FFFF,&H000000FF,&H00101010,&H00000000,-1,0,0,0,100,100,0,0,1,5,3,2,70,70,660,1
Style: Hook,DejaVu Sans,68,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,3,0,0,8,60,60,190,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


@dataclass
class Word:
    text: str
    start: float
    end: float


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _escape(text: str) -> str:
    return text.replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N").strip()


def _group_words_into_phrases(words: list[Word]) -> list[tuple[float, float, str]]:
    phrases: list[tuple[float, float, str]] = []
    current: list[Word] = []

    def flush():
        if current:
            start = current[0].start
            end = current[-1].end
            text = " ".join(w.text for w in current)
            phrases.append((start, end, text))
            current.clear()

    for w in words:
        if current:
            gap = w.start - current[-1].end
            span = w.end - current[0].start
            if gap > MAX_WORD_GAP_SECONDS or span > MAX_PHRASE_SECONDS or len(current) >= MAX_WORDS_PER_PHRASE:
                flush()
        current.append(w)
    flush()
    return phrases


def _is_emphasis_word(text: str) -> bool:
    """Simple, deterministic emphasis heuristic - no extra AI call needed.
    A word reads as emphatic if the narrator's own wording marks it that way
    (trailing !/?), it was written in caps, or it's simply a longer/heavier
    word than typical connective/filler words."""
    stripped = text.strip(".,!?;:\"'")
    if not stripped:
        return False
    if text.endswith("!") or text.endswith("?"):
        return True
    if stripped.isupper() and len(stripped) > 1:
        return True
    return len(stripped) >= 7


def _extend_narration_word_ends(words: list[Word]) -> list[Word]:
    """Stretch each word's on-screen time a little past its spoken end (up
    to the next word's start) so word-by-word captions don't blink off
    during a natural micro-pause between words."""
    extended: list[Word] = []
    for i, w in enumerate(words):
        end = w.end + NARRATION_WORD_END_PAD_SECONDS
        if i + 1 < len(words):
            end = min(end, words[i + 1].start)
        end = max(end, w.start + MIN_NARRATION_WORD_DISPLAY_SECONDS)
        extended.append(Word(text=w.text, start=w.start, end=end))
    return extended


def build_ass_captions(
    *,
    output_path: Path,
    clip_start: float,
    clip_end: float,
    transcript_words: list[Word],
    narration_cues: list[tuple[float, float, str]],  # (start_rel, end_rel, text) relative to clip start
    hook_text: str | None,
    narration_words: list[Word] | None = None,  # clip-relative; word-by-word narration timing
) -> Path:
    """narration_cues, narration_words, and hook_text use clip-relative
    seconds. transcript_words use absolute source-video seconds and are
    converted to clip-relative here."""
    clip_duration = clip_end - clip_start

    blocked: list[tuple[float, float]] = [(s, e) for s, e, _ in narration_cues]

    rel_words: list[Word] = []
    for w in transcript_words:
        rel_start = w.start - clip_start
        rel_end = w.end - clip_start
        if rel_end <= 0 or rel_start >= clip_duration:
            continue
        mid = (rel_start + rel_end) / 2
        if any(s <= mid <= e for s, e in blocked):
            continue
        rel_words.append(Word(text=w.text, start=max(0.0, rel_start), end=min(clip_duration, rel_end)))

    lines: list[str] = []

    for start, end, text in _group_words_into_phrases(rel_words):
        lines.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Caption,,0,0,0,,{_escape(text)}"
        )

    if narration_words:
        for w in _extend_narration_word_ends(narration_words):
            if w.end <= 0 or w.start >= clip_duration or not w.text.strip():
                continue
            style = "NarrationEmphasis" if _is_emphasis_word(w.text) else "Narration"
            start = max(0.0, w.start)
            end = min(clip_duration, w.end)
            lines.append(f"Dialogue: 1,{_fmt_time(start)},{_fmt_time(end)},{style},,0,0,0,,{_escape(w.text)}")
    else:
        # No per-word alignment available for this cue (older job, or
        # ElevenLabs returned none) - fall back to one line per cue.
        for start, end, text in narration_cues:
            if not text.strip():
                continue
            lines.append(
                f"Dialogue: 1,{_fmt_time(max(0.0, start))},{_fmt_time(min(clip_duration, end))},Narration,,0,0,0,,{_escape(text)}"
            )

    if hook_text and hook_text.strip():
        hook_end = min(3.5, clip_duration)
        lines.append(f"Dialogue: 2,{_fmt_time(0)},{_fmt_time(hook_end)},Hook,,0,0,0,,{_escape(hook_text)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return output_path
