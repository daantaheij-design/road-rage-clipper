"""Fast unit tests for how render_vertical_clip builds its ffmpeg command -
thread limiting and the low-resolution background blur - without actually
invoking ffmpeg (see test_ffmpeg_pipeline.py for the real-ffmpeg integration
tests)."""

from __future__ import annotations

import pytest

from app.pipeline import render as render_mod
from app.pipeline.ffmpeg_utils import MediaInfo
from app.pipeline.render import NarrationTrack, render_vertical_clip

MEDIA_INFO = MediaInfo(duration_seconds=30.0, width=1920, height=1080, fps=30.0, has_audio=True)


@pytest.fixture
def captured_args(monkeypatch):
    captured: dict[str, list[str]] = {}

    async def fake_run_ffmpeg(args, *, timeout=900):
        captured["args"] = args
        return ""

    monkeypatch.setattr(render_mod, "run_ffmpeg", fake_run_ffmpeg)
    return captured


def _flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


async def test_default_thread_count_comes_from_settings(settings, captured_args, tmp_path):
    settings.ffmpeg_threads = 2
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    args = captured_args["args"]
    assert _flag_value(args, "-threads") == "2"
    assert _flag_value(args, "-filter_threads") == "2"
    assert _flag_value(args, "-filter_complex_threads") == "2"


async def test_ffmpeg_threads_env_var_is_respected(settings, captured_args, tmp_path):
    settings.ffmpeg_threads = 1
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    args = captured_args["args"]
    assert _flag_value(args, "-threads") == "1"
    x264_params = _flag_value(args, "-x264-params")
    assert "threads=1" in x264_params


async def test_explicit_threads_arg_overrides_settings(settings, captured_args, tmp_path):
    settings.ffmpeg_threads = 2
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        threads=4,
    )
    args = captured_args["args"]
    assert _flag_value(args, "-threads") == "4"


async def test_threads_never_goes_below_one(settings, captured_args, tmp_path):
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        threads=0,
    )
    args = captured_args["args"]
    assert _flag_value(args, "-threads") == "1"


async def test_x264_params_use_sliced_threads_and_capped_lookahead(settings, captured_args, tmp_path):
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    x264_params = _flag_value(captured_args["args"], "-x264-params")
    assert "sliced_threads=1" in x264_params
    assert f"rc-lookahead={render_mod.X264_RC_LOOKAHEAD}" in x264_params


async def test_background_blur_runs_at_low_internal_resolution(settings, captured_args, tmp_path):
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")

    # The blur itself must run on the small working resolution, not the full
    # 1080x1920 output canvas - that's the whole point of the optimization.
    assert f"scale={render_mod.BG_BLUR_W}:{render_mod.BG_BLUR_H}" in filter_complex
    assert f"crop={render_mod.BG_BLUR_W}:{render_mod.BG_BLUR_H}" in filter_complex
    assert f"gblur=sigma={render_mod.BG_BLUR_SIGMA}" in filter_complex
    assert render_mod.BG_BLUR_SIGMA < 20  # meaningfully smaller than the old full-res sigma

    # It must still be scaled back up to the full target canvas afterwards.
    assert f"scale={render_mod.TARGET_W}:{render_mod.TARGET_H}:flags=bilinear" in filter_complex


async def test_output_maps_still_target_1080x1920_canvas(settings, captured_args, tmp_path):
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    # Foreground footage still gets scaled to fit the full target canvas at
    # full quality (not the cheap low-res path used for the background).
    assert f"[fg]scale={render_mod.TARGET_W}:{render_mod.TARGET_H}:force_original_aspect_ratio=decrease" in (
        filter_complex
    )


async def test_narration_and_captions_still_wired_up(settings, captured_args, tmp_path):
    narration_path = tmp_path / "narration.mp3"
    narration_path.write_bytes(b"fake")
    captions_path = tmp_path / "captions.ass"
    captions_path.write_text("fake")

    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        narration_tracks=[NarrationTrack(audio_path=narration_path, start_seconds=1.0, duration_seconds=2.0)],
        captions_ass_path=captions_path,
    )
    args = captured_args["args"]
    filter_complex = _flag_value(args, "-filter_complex")

    assert "adelay=delays=1000:all=1" in filter_complex  # narration positioned correctly
    assert "amix=inputs=2" in filter_complex  # original (ducked) + narration track
    assert "volume=volume=0.35" in filter_complex  # ducking still applied
    assert "subtitles=" in filter_complex  # captions still burned in
    assert str(narration_path) in args  # narration audio still passed as an input
