"""Thin async wrappers around ffmpeg/ffprobe.

All video/audio manipulation in this project goes through here so there is
one place that knows how to shell out to ffmpeg correctly and safely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Signals under which "the OS killed this process out from under us" is the
# most likely explanation, rather than ffmpeg itself deciding to exit.
# SIGKILL is what the Linux OOM killer sends; SIGBUS commonly shows up when
# the kernel can't back a memory-mapped page (also often memory pressure).
_LIKELY_OOM_SIGNALS = {signal.SIGKILL, signal.SIGBUS}


class FFmpegError(RuntimeError):
    pass


async def run_ffmpeg(args: list[str], *, timeout: float = 900) -> str:
    return await _run(args, timeout=timeout)


async def _run(args: list[str], *, timeout: float = 900) -> str:
    logger.debug("running: %s", " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise FFmpegError(f"Command timed out after {timeout}s: {' '.join(args)}") from exc

    if proc.returncode != 0:
        message = _describe_failure(proc.returncode, args, stderr)
        # Log at error level here too (not just via the raised exception) so
        # it's unmissable in a log aggregator that only surfaces top-level
        # error-level lines rather than full exception tracebacks.
        logger.error("ffmpeg command failed: %s", message)
        raise FFmpegError(message)
    return stdout.decode(errors="replace")


def _describe_failure(returncode: int, args: list[str], stderr: bytes) -> str:
    tail = stderr.decode(errors="replace")[-4000:]

    # On POSIX, a negative returncode means the process was killed by signal
    # `-returncode` rather than exiting normally - ffmpeg never does this to
    # itself, so it's the OS (almost always the kernel OOM killer for -9)
    # terminating the process out from under us. A plain nonzero exit code
    # with no signal is a normal ffmpeg-reported error (bad input, filter
    # error, etc.) and gets no special framing.
    if returncode < 0:
        try:
            sig = signal.Signals(-returncode)
        except ValueError:
            sig = None

        if sig in _LIKELY_OOM_SIGNALS:
            return (
                f"ffmpeg was killed by signal {-returncode} ({sig.name if sig else 'unknown'}) - this almost "
                "always means the process ran out of memory and the kernel's OOM killer terminated it (not a "
                "normal ffmpeg error - there is little/no ffmpeg output below because the process was killed "
                "outright). If this is running in a small/limited container, try lowering FFMPEG_THREADS "
                f"further and/or reducing concurrent renders.\nCommand: {' '.join(args)}\n{tail}"
            )
        sig_desc = sig.name if sig else f"signal {-returncode}"
        return (
            f"ffmpeg was killed by {sig_desc} rather than exiting normally - possibly resource exhaustion "
            f"(OOM, disk full, or the container/host killing it) rather than a normal ffmpeg error.\n"
            f"Command: {' '.join(args)}\n{tail}"
        )

    return f"Command failed ({returncode}): {' '.join(args)}\n{tail}"


@dataclass
class MediaInfo:
    duration_seconds: float
    width: int
    height: int
    fps: float
    has_audio: bool


async def probe(path: Path) -> MediaInfo:
    out = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(out)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0) or 0)

    width = height = 0
    fps = 0.0
    has_audio = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and width == 0:
            width = int(stream.get("width", 0) or 0)
            height = int(stream.get("height", 0) or 0)
            rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
            try:
                num, den = rate.split("/")
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            if not duration:
                duration = float(stream.get("duration", 0) or 0)
        elif stream.get("codec_type") == "audio":
            has_audio = True

    if duration <= 0:
        raise FFmpegError(f"Could not determine duration for {path}")

    return MediaInfo(duration_seconds=duration, width=width, height=height, fps=fps, has_audio=has_audio)


async def extract_audio(video_path: Path, out_path: Path, *, sample_rate: int = 16000) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    await _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )
    return out_path


@dataclass
class Frame:
    timestamp: float
    path: Path


async def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    fps: float,
    start: float = 0.0,
    duration: float | None = None,
    scale_width: int = 320,
    prefix: str = "f",
) -> list[Frame]:
    """Extract frames at a fixed rate, returning them with real timestamps.

    fps=0.67 means "roughly one frame every 1.5 seconds"; fps=4 means dense
    sampling for close analysis of a short window.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%06d.jpg"

    args = ["ffmpeg", "-y"]
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(video_path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        "-vf",
        f"fps={fps},scale={scale_width}:-2",
        "-qscale:v",
        "4",
        str(pattern),
    ]
    await _run(args)

    frames: list[Frame] = []
    for f in sorted(out_dir.glob(f"{prefix}_*.jpg")):
        idx = int(f.stem.split("_")[-1]) - 1
        ts = start + idx / fps
        frames.append(Frame(timestamp=ts, path=f))
    return frames


def escape_for_filter(path: Path) -> str:
    """Escape a filesystem path for safe use inside an ffmpeg filtergraph string
    (e.g. subtitles=<path>). Colons and backslashes need escaping."""
    s = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return s


async def duration_of(path: Path) -> float:
    info = await probe(path)
    return info.duration_seconds
