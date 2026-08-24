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

    async def fake_probe(path):
        # render_vertical_clip validates its own output duration against
        # what it requested (see RenderValidationError) - since this fixture
        # never actually runs ffmpeg, fake a MediaInfo matching the last -t
        # (the output-side duration cap, always the final -t in the args)
        # rather than actually reading `path`.
        args = captured["args"]
        last_t_index = len(args) - 1 - args[::-1].index("-t")
        return MediaInfo(duration_seconds=float(args[last_t_index + 1]), width=1080, height=1920, fps=30.0, has_audio=True)

    monkeypatch.setattr(render_mod, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(render_mod, "probe", fake_probe)
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


async def test_no_blurred_background_layout(settings, captured_args, tmp_path):
    """The blurred-background letterbox layout must be gone entirely - the
    normal path is always a true full-screen crop now."""
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    assert "gblur" not in filter_complex
    assert "overlay" not in filter_complex
    assert "split=2" not in filter_complex


async def test_output_fills_full_1080x1920_canvas_via_crop(settings, captured_args, tmp_path):
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    # A crop straight to the target aspect, scaled directly to the full
    # target canvas - no letterboxing, always fills the whole frame.
    assert "crop=w=" in filter_complex
    assert f"scale={render_mod.TARGET_W}:{render_mod.TARGET_H}" in filter_complex


async def test_crop_keyframes_are_passed_through_to_crop_filter(settings, captured_args, tmp_path):
    from app.pipeline.crop import CropKeyframe

    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        crop_keyframes=[
            CropKeyframe(0.0, 0.2, 0.5, 0.9),
            CropKeyframe(10.0, 0.8, 0.5, 0.9),
        ],
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    x_expr_start = filter_complex.index("crop=w=")
    x_expr = filter_complex[x_expr_start : x_expr_start + 400]
    # A moving focus point should produce a time-varying (not fixed) x expression.
    assert "(t-" in x_expr


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


async def test_video_and_audio_filter_chains_start_with_explicit_trim(settings, captured_args, tmp_path):
    """Regression guard for the exact production bug: the filtergraph must
    trim [0:v]/[0:a] explicitly (immune to ffmpeg CLI option-ordering
    mistakes) rather than relying solely on -ss/-t placement."""
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=12.5,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    assert "[0:v]trim=duration=12.500,setpts=PTS-STARTPTS[vtrim]" in filter_complex
    assert "[0:a]atrim=duration=12.500,asetpts=PTS-STARTPTS[atrim]" in filter_complex


async def test_input_and_output_both_carry_explicit_duration_cap(settings, captured_args, tmp_path):
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=5.0,
        end_seconds=17.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    args = captured_args["args"]
    # -ss and -t must both precede -i (unambiguously scoped to that input) -
    # this is what the original bug got wrong.
    i_index = args.index("-i")
    assert "-ss" in args[:i_index]
    assert "-t" in args[:i_index]
    # And the output must also carry its own -t as a third safety net.
    assert args[-3] == "-t"
    assert args[-2] == "12.000"
    assert args[-1] == str(tmp_path / "out.mp4")


async def test_render_raises_when_actual_duration_does_not_match_requested(settings, monkeypatch, tmp_path):
    async def fake_run_ffmpeg(args, *, timeout=900):
        return ""

    async def fake_probe_wrong_duration(path):
        # Simulate exactly what the production bug produced: a render that
        # runs all the way to the end of a long source instead of stopping
        # at the requested window.
        return MediaInfo(duration_seconds=612.3, width=1080, height=1920, fps=30.0, has_audio=True)

    monkeypatch.setattr(render_mod, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(render_mod, "probe", fake_probe_wrong_duration)

    with pytest.raises(render_mod.RenderValidationError, match=r"expected 24\.0s, got 612\.3s"):
        await render_vertical_clip(
            source_path=tmp_path / "in.mp4",
            start_seconds=0.0,
            end_seconds=24.0,
            output_path=tmp_path / "out.mp4",
            media_info=MEDIA_INFO,
        )


async def test_render_with_no_effects_never_touches_segment_machinery(settings, captured_args, tmp_path):
    """segments=None (the default, and what pipeline.py passes for a clip
    with no effects/teaser) must produce the exact same filter graph shape
    as before effects existed - no split/asplit/concat overhead."""
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    assert "split=" not in filter_complex
    assert "concat=" not in filter_complex


async def test_segments_path_produces_split_and_concat(settings, captured_args, tmp_path):
    from app.pipeline.crop import CropKeyframe
    from app.pipeline.timeline import SegmentPlan

    segments = [
        SegmentPlan("normal", 0.0, 5.0, 5.0, crop_keyframes=[CropKeyframe(0, 0.5, 0.5, 0.9)]),
        SegmentPlan("freeze", 5.0, 5.0, 0.5, volume=0.0, crop_keyframes=[CropKeyframe(0, 0.5, 0.5, 0.5)]),
        SegmentPlan("normal", 5.0, 10.0, 5.0, crop_keyframes=[CropKeyframe(0, 0.5, 0.5, 0.9)]),
    ]
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        segments=segments,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    assert "split=3" in filter_complex
    assert "concat=n=3:v=1:a=1" in filter_complex
    assert "tpad=stop_mode=clone" in filter_complex  # the freeze segment
    assert "aevalsrc=" in filter_complex  # silent audio for the freeze


async def test_segments_path_output_duration_cap_uses_expected_duration_not_raw_window(
    settings, captured_args, tmp_path
):
    """A freeze adds real output time beyond end_seconds-start_seconds - the
    output-side -t cap (and duration validation) must use the sum of
    segment output durations, not the raw clip window, or a legitimately
    longer render would get truncated/rejected."""
    from app.pipeline.crop import CropKeyframe
    from app.pipeline.timeline import SegmentPlan

    segments = [
        SegmentPlan("normal", 0.0, 10.0, 10.0, crop_keyframes=[CropKeyframe(0, 0.5, 0.5, 0.9)]),
        SegmentPlan("freeze", 10.0, 10.0, 0.5, volume=0.0, crop_keyframes=[CropKeyframe(0, 0.5, 0.5, 0.5)]),
    ]
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        segments=segments,
    )
    args = captured_args["args"]
    assert args[-3] == "-t"
    assert args[-2] == "10.500"  # 10.0 + 0.5 freeze, not the raw 10.0 window


async def test_overlay_input_added_and_referenced_with_enable_window(settings, captured_args, tmp_path):
    from app.pipeline.render import OverlaySpec

    overlay_png = tmp_path / "circle.png"
    overlay_png.write_bytes(b"fake png")

    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        overlays=[OverlaySpec(image_path=overlay_png, start_seconds=1.5, end_seconds=3.0)],
    )
    args = captured_args["args"]
    filter_complex = _flag_value(args, "-filter_complex")
    assert str(overlay_png) in args
    assert "overlay=0:0:enable='between(t,1.500,3.000)'" in filter_complex


async def test_multiple_overlays_each_get_their_own_input_and_chain(settings, captured_args, tmp_path):
    from app.pipeline.render import OverlaySpec

    png1, png2 = tmp_path / "a.png", tmp_path / "b.png"
    png1.write_bytes(b"a")
    png2.write_bytes(b"b")

    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=10.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        overlays=[
            OverlaySpec(image_path=png1, start_seconds=1.0, end_seconds=2.0),
            OverlaySpec(image_path=png2, start_seconds=4.0, end_seconds=5.0),
        ],
    )
    args = captured_args["args"]
    filter_complex = _flag_value(args, "-filter_complex")
    assert str(png1) in args
    assert str(png2) in args
    assert filter_complex.count("overlay=0:0:enable=") == 2


async def test_setsar_forces_square_pixels_in_segments_path(settings, captured_args, tmp_path):
    """Regression: without setsar=1, differently-sized crop windows across
    segments (e.g. a punch-zoom segment vs. a normal one) can produce
    mismatched SAR and concat refuses to join them."""
    from app.pipeline.crop import CropKeyframe
    from app.pipeline.timeline import SegmentPlan

    segments = [
        SegmentPlan("normal", 0.0, 5.0, 5.0, crop_keyframes=[CropKeyframe(0, 0.5, 0.5, 0.9)]),
        SegmentPlan("zoom", 5.0, 6.0, 1.0, zoom=1.3, crop_keyframes=[CropKeyframe(0, 0.5, 0.5, 0.9)]),
    ]
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=6.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
        segments=segments,
    )
    filter_complex = _flag_value(captured_args["args"], "-filter_complex")
    assert filter_complex.count("setsar=1") == 2


async def test_render_accepts_small_duration_drift(settings, monkeypatch, tmp_path):
    async def fake_run_ffmpeg(args, *, timeout=900):
        return ""

    async def fake_probe_close_enough(path):
        return MediaInfo(duration_seconds=24.3, width=1080, height=1920, fps=30.0, has_audio=True)

    monkeypatch.setattr(render_mod, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(render_mod, "probe", fake_probe_close_enough)

    # Should not raise - 0.3s of drift is within MAX_DURATION_DRIFT_SECONDS.
    await render_vertical_clip(
        source_path=tmp_path / "in.mp4",
        start_seconds=0.0,
        end_seconds=24.0,
        output_path=tmp_path / "out.mp4",
        media_info=MEDIA_INFO,
    )
