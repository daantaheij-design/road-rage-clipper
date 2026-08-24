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

Visual-attention effects (circle/arrow/punch-zoom/freeze/slow-motion/
replay/teaser - see app.pipeline.timeline): when a clip has any, `segments`
carries the pre-computed app.pipeline.timeline.SegmentPlan list and this
function builds each segment as its own short trim/setpts(/tpad) sub-chain,
concatenated back together via ffmpeg's `concat` filter, instead of the
single crop+scale pass used for the (much more common) no-effects case.
When `segments` is None, the filter graph is *exactly* what it was before
effects existed - callers only pass segments for clips that actually have
freeze/slow_motion/replay/punch_zoom/teaser effects, so the plain case never
pays for or risks the extra complexity.

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
   intermediate frames, no per-pixel blur convolution. Effect overlays are
   single small static PNGs (not per-frame image sequences) composited with
   a plain `overlay` filter, and freeze frames use `tpad` (clone the last
   decoded frame) rather than generating/encoding a run of duplicate PNGs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.pipeline.crop import CropKeyframe, build_crop_filter
from app.pipeline.ffmpeg_utils import MediaInfo, escape_for_filter, probe, run_ffmpeg
from app.pipeline.timeline import SegmentPlan

TARGET_W = 1080
TARGET_H = 1920
DUCK_VOLUME = 0.35

# How far the actual rendered duration is allowed to drift from the
# expected clip length before we refuse to call the render successful. This
# is the safety net that catches a repeat of the "exported the whole rest of
# the source video" bug (see module docstring) even if some future change
# reintroduces an ffmpeg option-ordering mistake. With effects present, the
# "expected" length is the sum of every segment's own output_duration (a
# freeze/slow_motion/replay/teaser can legitimately make the render longer
# than end_seconds-start_seconds) - never just the raw clip window.
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


@dataclass
class OverlaySpec:
    """A circle/arrow PNG (see app.pipeline.overlays) to composite over the
    render for a start/end window - both already in FINAL rendered-timeline
    seconds (i.e. remapped through the clip's RenderPlan, same as
    NarrationTrack.start_seconds). The PNG is a full 1080x1920 transparent
    canvas with the drawing already positioned correctly, so compositing is
    a plain 0,0 overlay - no extra position math needed here."""

    image_path: Path
    start_seconds: float
    end_seconds: float


def _segment_video_chain(seg: SegmentPlan, i: int, media_info: MediaInfo, local_start: float, local_end: float) -> tuple[str, str]:
    """Returns (filter_fragment, output_label) for one segment's video."""
    label = f"vseg{i}"
    if seg.kind == "freeze":
        # Grab a tiny sliver at the freeze point and hold (clone) it for the
        # effect's full on-screen duration via tpad, then trim to exactly
        # that duration (tpad's own duration accounting isn't frame-exact).
        eps = max(1.0 / max(media_info.fps, 1.0), 1.0 / 25.0)
        chain = (
            f"[vtrim{i}]trim=start={local_start:.3f}:end={local_start + eps:.3f},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={seg.output_duration:.3f},"
            f"trim=duration={seg.output_duration:.3f},setpts=PTS-STARTPTS[{label}raw]"
        )
    else:
        speed_expr = f",setpts=PTS/{seg.speed:.4f}" if abs(seg.speed - 1.0) > 1e-6 else ""
        chain = f"[vtrim{i}]trim=start={local_start:.3f}:end={local_end:.3f},setpts=PTS-STARTPTS{speed_expr}[{label}raw]"

    crop_filter = build_crop_filter(
        seg.crop_keyframes,
        clip_duration=seg.output_duration,
        source_width=media_info.width,
        source_height=media_info.height,
        target_width=TARGET_W,
        target_height=TARGET_H,
        zoom=seg.zoom,
    )
    # setsar=1 forces square pixels on every segment - without it, ffmpeg's
    # scale filter can compute slightly different residual SARs for crop
    # windows of different sizes (e.g. a punch-zoom segment's smaller crop
    # vs. a normal segment's), and concat refuses to join video streams
    # whose SAR doesn't match exactly (confirmed empirically: "Input link
    # parameters ... do not match the corresponding output link parameters").
    chain += f";[{label}raw]{crop_filter},setsar=1[{label}]"
    return chain, label


def _segment_audio_chain(seg: SegmentPlan, i: int, local_start: float, local_end: float) -> tuple[str, str]:
    """Returns (filter_fragment, output_label) for one segment's audio.
    Silent (aevalsrc, no extra ffmpeg -i needed - it's a source filter) for
    freeze/muted segments; atempo (a genuine speed change, not just a PTS
    relabel, so it stays pitch-reasonable) for a sped/slowed segment that
    still carries real audio (replay); a plain trim+volume otherwise."""
    label = f"aseg{i}"
    if seg.volume <= 0.0:
        return f"aevalsrc=0:d={seg.output_duration:.3f}:s=44100:c=stereo[{label}]", label
    tempo_expr = f",atempo={seg.speed:.4f}" if abs(seg.speed - 1.0) > 1e-6 else ""
    chain = (
        f"[atrim{i}]atrim=start={local_start:.3f}:end={local_end:.3f},asetpts=PTS-STARTPTS{tempo_expr},"
        f"volume={seg.volume:.3f}[{label}]"
    )
    return chain, label


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
    segments: list[SegmentPlan] | None = None,
    overlays: list[OverlaySpec] | None = None,
    threads: int | None = None,
) -> Path:
    narration_tracks = narration_tracks or []
    overlays = overlays or []
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
    # decoded just to extract a 24-second window. Every effect segment
    # (freeze/slow_motion/replay/zoom/teaser) is built from a sub-range of
    # THIS SAME window - a teaser is validated (app.pipeline.timeline) to be
    # footage from within the clip's own [start_seconds, end_seconds], so no
    # additional source input is ever needed for effects.
    args += ["-ss", f"{start_seconds:.3f}", "-t", f"{duration:.3f}", "-i", str(source_path)]

    if not media_info.has_audio:
        # Silent source footage: synthesize a silent base track so narration
        # and the mixing graph below still work unchanged.
        args += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]

    narration_input_start = 2 if not media_info.has_audio else 1
    for track in narration_tracks:
        args += ["-i", str(track.audio_path)]

    overlay_input_start = narration_input_start + len(narration_tracks)
    for ov in overlays:
        args += ["-i", str(ov.image_path)]

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

    if segments is None:
        # ---- video: true full-screen 9:16 smart crop (no effects) ----
        # A single crop+scale pass that always fills the entire target
        # canvas - panned over time to follow crop_keyframes when available,
        # otherwise a static center crop. See app/pipeline/crop.py.
        crop_filter = build_crop_filter(
            crop_keyframes or [],
            clip_duration=duration,
            source_width=media_info.width,
            source_height=media_info.height,
            target_width=TARGET_W,
            target_height=TARGET_H,
        )
        filter_parts.append(f"[vtrim]{crop_filter}[vcomp]")
        video_label = "vcomp"
        audio_label = "atrim"
        expected_duration = duration
    else:
        # ---- video/audio: per-effect-segment trims, concatenated ----
        # [vtrim]/[atrim] each feed N separate per-segment trim chains, so
        # they need an explicit split/asplit fan-out first - a filter-graph
        # label can't be consumed by more than one downstream filter
        # otherwise ("Invalid stream specifier" / "matches no streams",
        # confirmed empirically).
        # Every segment needs its own video split, but a silent segment
        # (freeze, or any segment with volume<=0) gets its audio from
        # aevalsrc instead and never touches [atrim] at all - asplit
        # requires every one of its outputs to be connected to something,
        # so only split audio for the segments that actually use it
        # (confirmed empirically: an unused asplit output is a hard error).
        n = len(segments)
        filter_parts.append(f"[vtrim]split={n}" + "".join(f"[vtrim{i}]" for i in range(n)))
        audio_needed = [i for i, seg in enumerate(segments) if seg.volume > 0.0]
        if audio_needed:
            filter_parts.append(f"[atrim]asplit={len(audio_needed)}" + "".join(f"[atrim{i}]" for i in audio_needed))

        video_labels: list[str] = []
        audio_labels: list[str] = []
        for i, seg in enumerate(segments):
            local_start = seg.source_start - start_seconds
            local_end = seg.source_end - start_seconds
            vchain, vlabel = _segment_video_chain(seg, i, media_info, local_start, local_end)
            filter_parts.append(vchain)
            video_labels.append(vlabel)
            achain, alabel = _segment_audio_chain(seg, i, local_start, local_end)
            filter_parts.append(achain)
            audio_labels.append(alabel)

        concat_inputs = "".join(f"[{v}][{a}]" for v, a in zip(video_labels, audio_labels, strict=True))
        filter_parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[vcomp][aconcat]")
        video_label = "vcomp"
        audio_label = "aconcat"
        expected_duration = sum(seg.output_duration for seg in segments)

    # ---- effect overlays (circle/arrow): plain full-canvas composites ----
    for i, ov in enumerate(overlays):
        idx = overlay_input_start + i
        new_label = f"vov{i}"
        filter_parts.append(
            f"[{video_label}][{idx}:v]overlay=0:0:enable='between(t,{ov.start_seconds:.3f},{ov.end_seconds:.3f})'[{new_label}]"
        )
        video_label = new_label

    if captions_ass_path is not None:
        escaped = escape_for_filter(captions_ass_path)
        filter_parts.append(f"[{video_label}]subtitles='{escaped}'[vout]")
        video_label = "vout"

    # ---- audio: duck original under narration, then mix narration in ----
    if narration_tracks:
        duck_chain = ",".join(
            f"volume=volume={DUCK_VOLUME}:enable='between(t,{t.start_seconds:.3f},{t.start_seconds + t.duration_seconds:.3f})'"
            for t in narration_tracks
        )
        filter_parts.append(f"[{audio_label}]{duck_chain}[aduck]")

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
        filter_parts.append(f"[{audio_label}]anull[aout]")

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
        f"[{video_label}]",
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
        f"{expected_duration:.3f}",
        str(output_path),
    ]

    await run_ffmpeg(args, timeout=1800)

    # Never let a mis-trimmed render (e.g. a future ffmpeg-option-ordering
    # mistake like the one this function's docstring describes) reach
    # storage looking successful. Verify the actual output against what was
    # requested before returning.
    actual_info = await probe(output_path)
    drift = abs(actual_info.duration_seconds - expected_duration)
    if drift > MAX_DURATION_DRIFT_SECONDS:
        raise RenderValidationError(
            f"Rendered duration validation failed: expected {expected_duration:.1f}s, got "
            f"{actual_info.duration_seconds:.1f}s (source={source_path.name}, "
            f"start={start_seconds:.1f}s, end={end_seconds:.1f}s)"
        )

    return output_path
