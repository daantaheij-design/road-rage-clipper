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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.pipeline.ffmpeg_utils import MediaInfo, escape_for_filter, run_ffmpeg

TARGET_W = 1080
TARGET_H = 1920
DUCK_VOLUME = 0.35


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
) -> Path:
    narration_tracks = narration_tracks or []
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError("end_seconds must be greater than start_seconds")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = ["ffmpeg", "-y"]

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
    filter_parts.append(
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},gblur=sigma=20,eq=brightness=-0.05[bgblur];"
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

    args += [
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{video_out_label}]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
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
