"""Compose the final vertical (1080x1920) TikTok-ready clip.

Layout: a true full-screen 9:16 crop (see app/pipeline/crop.py) - never a
shrunk-down frame floating inside a blurred box. A vertical slice is cropped
out of the horizontal source and, when Claude's visual analysis provided
crop keyframes for this clip, panned smoothly over time to keep the
important action (the car, the confrontation, whatever the analysis says
matters) inside frame. No keyframes / low confidence -> a plain centered
crop, still full-screen.

Audio: the original clip audio plays throughout. While a narration cue is
speaking, the original audio is ducked (lowered, not muted) so honking,
yelling, and arguments stay audible under the narration; it returns to full
volume as soon as the narration cue ends.

Memory footprint on small containers: this is tuned to run on a
resource-constrained single Railway instance, not a beefy build box. Three
things kept early renders from OOM-killing the process (ffmpeg exits with
-9/SIGKILL, not a normal ffmpeg error, when that happens):

1. libx264 (and ffmpeg's filtergraph) auto-detect the *host's* CPU count,
   not the container's memory/cpu allocation, and will happily spin up
   dozens of threads - each with its own frame buffers. Threads are capped
   everywhere (`-threads`, `-filter_threads`, `-filter_complex_threads`,
   x264 `threads=`) to `settings.ffmpeg_threads` (default 2).
2. x264's default *frame*-threading model holds ~1.5x `threads` full
   uncompressed frame buffers in flight at once. `sliced_threads=1` switches
   to slice-based threading (threads split one frame into row-slices
   instead of encoding whole frames in parallel), which uses a small
   fraction of the memory for a similar speed trade-off at low thread
   counts. `rc-lookahead` (frames buffered ahead for rate-control analysis,
   40 by default) is also capped.
3. The smart-crop pan (crop.py) is a single `crop`+`scale` filter pair
   operating directly at the target resolution - no extra full-resolution
   intermediate frames, no per-pixel blur convolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.pipeline.crop import CropKeyframe, build_crop_filter
from app.pipeline.ffmpeg_utils import MediaInfo, escape_for_filter, probe, run_ffmpeg

TARGET_W = 1080
TARGET_H = 1920
DUCK_VOLUME = 0.35

# How far the actual rendered duration is allowed to drift from the
# requested clip length before we refuse to call the render successful. This
# is the safety net that catches a repeat of the "exported the whole rest of
# the source video" bug (see module docstring) even if some future change
# reintroduces an ffmpeg option-ordering mistake.
MAX_DURATION_DRIFT_SECONDS = 0.5


class RenderValidationError(RuntimeError):
    """Raised when a rendered clip's actual duration doesn't match what was
    requested - never let a mis-trimmed render silently reach storage."""


# x264 rate-control lookahead, in frames. Default is 40, which at 1080x1920
# means dozens of full decoded frames buffered ahead of the encoder purely
# for bitrate-planning purposes. 10 is still enough for veryfast/crf quality
# decisions and cuts that buffer to a quarter of the default.
X264_RC_LOOKAHEAD = 10


@dataclass
class NarrationTrack:
    audio_path: Path
    start_seconds: float  # relative to the rendered clip start
    duration_seconds: float


async def render_vertical_clip(
    *,
    source_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
    media_info: MediaInfo,
    narration_tracks: list[NarrationTrack] | None = None,
    captions_ass_path: Path | None = None,
    crop_keyframes: list[CropKeyframe] | None = None,
    threads: int | None = None,
) -> Path:
    narration_tracks = narration_tracks or []
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError("end_seconds must be greater than start_seconds")

    # Small Railway containers report the host's full CPU count, so ffmpeg's
    # default auto-detected thread count can be wildly higher than what the
    # container's memory allocation can actually support - see module
    # docstring. Cap it everywhere it can spawn threads.
    threads = threads if threads is not None else get_settings().ffmpeg_threads
    threads = max(1, threads)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = ["ffmpeg", "-y", "-threads", str(threads)]

    # Main video/audio input, trimmed to the clip window. Both -ss and -t
    # MUST come before this -i to be unambiguously scoped to this input -
    # ffmpeg binds any option preceding an -i to that specific input, but an
    # option placed *after* -i (and before the next -i) instead binds to
    # whichever input comes next. With narration tracks (additional -i's
    # follow), a `-t` placed after this -i silently became a duration limit
    # on the *next* input (a narration mp3, already short enough that it had
    # no visible effect) rather than on the source video - so the video
    # decoded straight through to EOF with nothing capping it. Input-side
    # duration here is also what keeps a 10-minute source from being fully
    # decoded just to extract a 24-second window.
    args += ["-ss", f"{start_seconds:.3f}", "-t", f"{duration:.3f}", "-i", str(source_path)]

    if not media_info.has_audio:
        # Silent source footage: synthesize a silent base track so narration
        # and the mixing graph below still work unchanged.
        args += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]

    narration_input_start = 2 if not media_info.has_audio else 1
    for track in narration_tracks:
        args += ["-i", str(track.audio_path)]

    filter_parts: list[str] = []

    # ---- explicit trim, belt-and-suspenders against the clip-length bug ----
    # The input-side -ss/-t above should already bound reading to exactly
    # this window, but an explicit `trim`/`atrim` + PTS reset inside the
    # filtergraph is immune to ffmpeg CLI option-ordering pitfalls entirely
    # (this is what caused the original bug) and guarantees the frame/sample
    # count actually fed into the rest of the graph never exceeds `duration`,
    # regardless of how many more -i's or filters get added later.
    filter_parts.append(f"[0:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS[vtrim]")
    audio_src_label = "0:a" if media_info.has_audio else "1:a"
    filter_parts.append(f"[{audio_src_label}]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[atrim]")

    # ---- video: true full-screen 9:16 smart crop ----
    # A single crop+scale pass that always fills the entire target canvas -
    # panned over time to follow crop_keyframes (from Claude's visual
    # analysis of this clip) when available, otherwise a static center crop.
    # See app/pipeline/crop.py for the pan-smoothing/confidence logic.
    crop_filter = build_crop_filter(
        crop_keyframes or [],
        clip_duration=duration,
        source_width=media_info.width,
        source_height=media_info.height,
        target_width=TARGET_W,
        target_height=TARGET_H,
    )
    filter_parts.append(f"[vtrim]{crop_filter}[vcomp]")
    video_out_label = "vcomp"
    if captions_ass_path is not None:
        escaped = escape_for_filter(captions_ass_path)
        filter_parts.append(f"[vcomp]subtitles='{escaped}'[vout]")
        video_out_label = "vout"

    # ---- audio: duck original under narration, then mix narration in ----
    if narration_tracks:
        duck_chain = ",".join(
            f"volume=volume={DUCK_VOLUME}:enable='between(t,{t.start_seconds:.3f},{t.start_seconds + t.duration_seconds:.3f})'"
            for t in narration_tracks
        )
        filter_parts.append(f"[atrim]{duck_chain}[aduck]")

        narration_labels = []
        for i, t in enumerate(narration_tracks):
            in_idx = narration_input_start + i
            delay_ms = max(0, round(t.start_seconds * 1000))
            label = f"an{i}"
            filter_parts.append(f"[{in_idx}:a]adelay=delays={delay_ms}:all=1[{label}]")
            narration_labels.append(f"[{label}]")

        mix_inputs = "".join(["[aduck]"] + narration_labels)
        n_inputs = 1 + len(narration_labels)
        filter_parts.append(
            f"{mix_inputs}amix=inputs={n_inputs}:duration=first:dropout_transition=0:normalize=0[amixed]"
        )
        filter_parts.append("[amixed]alimiter=limit=0.9[aout]")
    else:
        filter_parts.append("[atrim]anull[aout]")

    filter_complex = ";".join(filter_parts)

    # sliced_threads=1 makes x264 split each frame into row-slices across
    # threads instead of encoding multiple whole frames in parallel - the
    # latter (x264's default) holds ~1.5x `threads` full uncompressed frames
    # in memory at once, which is the main thing that was OOM-killing
    # renders on a small container. rc-lookahead is capped for the same
    # reason (see module docstring).
    x264_params = f"threads={threads}:sliced_threads=1:rc-lookahead={X264_RC_LOOKAHEAD}"

    args += [
        "-filter_complex",
        filter_complex,
        "-filter_threads",
        str(threads),
        "-filter_complex_threads",
        str(threads),
        "-map",
        f"[{video_out_label}]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-threads",
        str(threads),
        "-x264-params",
        x264_params,
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-movflags",
        "+faststart",
        # Third, redundant duration cap (input-side -ss/-t plus the
        # trim/atrim filters above should already guarantee this) - an
        # explicit output-side -t here as well is cheap insurance and is an
        # unambiguous output option regardless of input count/ordering.
        "-t",
        f"{duration:.3f}",
        str(output_path),
    ]

    await run_ffmpeg(args, timeout=1800)

    # Never let a mis-trimmed render (e.g. a future ffmpeg-option-ordering
    # mistake like the one this function's docstring describes) reach
    # storage looking successful. Verify the actual output against what was
    # requested before returning.
    actual_info = await probe(output_path)
    drift = abs(actual_info.duration_seconds - duration)
    if drift > MAX_DURATION_DRIFT_SECONDS:
        raise RenderValidationError(
            f"Rendered duration validation failed: expected {duration:.1f}s, got "
            f"{actual_info.duration_seconds:.1f}s (source={source_path.name}, "
            f"start={start_seconds:.1f}s, end={end_seconds:.1f}s)"
        )

    return output_path
