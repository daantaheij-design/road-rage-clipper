"""Compose the final vertical (1080x1920) TikTok-ready clip.

Layout: horizontal dashcam footage is never simply center-cropped (that
would cut off cars at the frame edges). Instead the frame is placed in full
inside the vertical canvas, scaled to fit, with a blurred/zoomed copy of the
same frame filling the space above and below it - a common, readable
"letterbox with blurred background" style.

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
3. The background blur used to run gblur(sigma=20) at the full 1080x1920
   output resolution - a large, slow, memory-heavy Gaussian blur. It now
   crops to a small internal working resolution *first*, blurs that (with a
   proportionally smaller sigma - blur radius scales with pixel size), and
   only then scales back up to 1080x1920 for compositing. The output
   resolution and visual result are unchanged; the blur just never
   materializes a full-resolution intermediate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.pipeline.ffmpeg_utils import MediaInfo, escape_for_filter, run_ffmpeg

TARGET_W = 1080
TARGET_H = 1920
DUCK_VOLUME = 0.35

# The background copy is blurred at 1/4 resolution (270x480) rather than the
# full 1080x1920 - a ~16x smaller frame for the expensive gblur step. Sigma
# is scaled down to match (blur radius is relative to pixel size), then the
# result is scaled back up to the full canvas. Visually indistinguishable
# from blurring at full resolution (it's a soft background, not the subject)
# but dramatically cheaper in both CPU and memory.
BG_BLUR_SCALE_DIVISOR = 4
BG_BLUR_W = TARGET_W // BG_BLUR_SCALE_DIVISOR
BG_BLUR_H = TARGET_H // BG_BLUR_SCALE_DIVISOR
BG_BLUR_SIGMA = 20 // BG_BLUR_SCALE_DIVISOR

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

    # Main video/audio input, trimmed to the clip window. Input-side -ss is
    # frame-accurate on modern ffmpeg and much faster than filter-based trim.
    args += ["-ss", f"{start_seconds:.3f}", "-i", str(source_path), "-t", f"{duration:.3f}"]

    if not media_info.has_audio:
        # Silent source footage: synthesize a silent base track so narration
        # and the mixing graph below still work unchanged.
        args += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]

    narration_input_start = 2 if not media_info.has_audio else 1
    for track in narration_tracks:
        args += ["-i", str(track.audio_path)]

    filter_parts: list[str] = []

    # ---- video: blurred-background vertical composition ----
    # The background copy is scaled/cropped straight down to a small working
    # resolution (BG_BLUR_W x BG_BLUR_H) - never materializing a full
    # 1080x1920 frame before blurring - blurred there with a proportionally
    # smaller sigma, then scaled back up to the full canvas. The foreground
    # (the actual dashcam footage people need to read) is untouched, still
    # scaled at full quality straight to fit the target canvas.
    filter_parts.append(
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={BG_BLUR_W}:{BG_BLUR_H}:force_original_aspect_ratio=increase,"
        f"crop={BG_BLUR_W}:{BG_BLUR_H},gblur=sigma={BG_BLUR_SIGMA},"
        f"scale={TARGET_W}:{TARGET_H}:flags=bilinear,eq=brightness=-0.05[bgblur];"
        f"[fg]scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgblur][fgs]overlay=(W-w)/2:(H-h)/2:format=auto[vcomp]"
    )
    video_out_label = "vcomp"
    if captions_ass_path is not None:
        escaped = escape_for_filter(captions_ass_path)
        filter_parts.append(f"[vcomp]subtitles='{escaped}'[vout]")
        video_out_label = "vout"

    # ---- audio: duck original under narration, then mix narration in ----
    audio_src_label = "0:a" if media_info.has_audio else "1:a"
    if narration_tracks:
        duck_chain = ",".join(
            f"volume=volume={DUCK_VOLUME}:enable='between(t,{t.start_seconds:.3f},{t.start_seconds + t.duration_seconds:.3f})'"
            for t in narration_tracks
        )
        filter_parts.append(f"[{audio_src_label}]{duck_chain}[aduck]")

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
        filter_parts.append(f"[{audio_src_label}]anull[aout]")

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
        str(output_path),
    ]

    await run_ffmpeg(args, timeout=1800)
    return output_path
