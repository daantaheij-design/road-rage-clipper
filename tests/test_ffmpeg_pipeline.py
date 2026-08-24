from __future__ import annotations

import subprocess

import pytest

from app.pipeline import ffmpeg_utils
from app.pipeline.render import NarrationTrack, render_vertical_clip


async def test_probe_reports_duration_and_audio(synthetic_video):
    info = await ffmpeg_utils.probe(synthetic_video)
    assert 7.0 <= info.duration_seconds <= 9.0
    assert info.width == 640
    assert info.height == 360
    assert info.has_audio is True


async def test_extract_audio_produces_wav(tmp_path, synthetic_video):
    out = await ffmpeg_utils.extract_audio(synthetic_video, tmp_path / "audio.wav")
    assert out.exists()
    assert out.stat().st_size > 0


async def test_extract_frames_sparse(tmp_path, synthetic_video):
    frames = await ffmpeg_utils.extract_frames(
        synthetic_video, tmp_path / "frames", fps=1.0, scale_width=160, prefix="f"
    )
    assert 6 <= len(frames) <= 9
    for f in frames:
        assert f.path.exists()
    # timestamps should be monotonically increasing
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)


async def test_extract_frames_windowed(tmp_path, synthetic_video):
    frames = await ffmpeg_utils.extract_frames(
        synthetic_video, tmp_path / "frames_dense", fps=4.0, start=2.0, duration=2.0, scale_width=256, prefix="d"
    )
    assert len(frames) >= 6
    assert frames[0].timestamp >= 2.0
    assert frames[-1].timestamp < 4.5


async def test_render_vertical_clip_basic(tmp_path, synthetic_video):
    info = await ffmpeg_utils.probe(synthetic_video)
    out_path = tmp_path / "out.mp4"
    await render_vertical_clip(
        source_path=synthetic_video,
        start_seconds=1.0,
        end_seconds=5.0,
        output_path=out_path,
        media_info=info,
        narration_tracks=[],
        captions_ass_path=None,
    )
    assert out_path.exists()
    out_info = await ffmpeg_utils.probe(out_path)
    assert out_info.width == 1080
    assert out_info.height == 1920
    assert 3.5 <= out_info.duration_seconds <= 4.5
    assert out_info.has_audio is True


async def test_render_vertical_clip_with_narration_and_captions(tmp_path, synthetic_video, synthetic_audio_clip):
    from app.pipeline.captions import Word, build_ass_captions

    info = await ffmpeg_utils.probe(synthetic_video)
    ass_path = tmp_path / "captions.ass"
    build_ass_captions(
        output_path=ass_path,
        clip_start=1.0,
        clip_end=6.0,
        transcript_words=[Word("test", 1.5, 1.8)],
        narration_cues=[(0.0, 2.0, "This is a narration line.")],
        hook_text="A short hook.",
    )

    out_path = tmp_path / "out_narrated.mp4"
    await render_vertical_clip(
        source_path=synthetic_video,
        start_seconds=1.0,
        end_seconds=6.0,
        output_path=out_path,
        media_info=info,
        narration_tracks=[NarrationTrack(audio_path=synthetic_audio_clip, start_seconds=0.5, duration_seconds=2.0)],
        captions_ass_path=ass_path,
    )
    assert out_path.exists()
    out_info = await ffmpeg_utils.probe(out_path)
    assert out_info.width == 1080
    assert out_info.height == 1920
    assert out_info.has_audio is True
    # Regression test for a real production bug: with narration tracks
    # present, additional -i's after the source video caused a misplaced -t
    # to silently bound a *narration* input instead of the source, so the
    # source read straight through to EOF instead of stopping at 5s.
    assert 4.5 <= out_info.duration_seconds <= 5.5


async def test_render_long_source_with_narration_produces_short_clip_not_full_source(tmp_path, has_ffmpeg):
    """Exact repro of the production bug: a long source video + a short
    selected window + a narration track used to export the *entire
    remainder* of the source video instead of just the requested window."""
    import subprocess

    if not has_ffmpeg:
        pytest.skip("ffmpeg not available")

    from app.pipeline.captions import build_ass_captions

    long_source = tmp_path / "long_source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=10:duration=90",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=90",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(long_source),
        ],
        check=True,
        capture_output=True,
    )
    narration_path = tmp_path / "narration.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=2", str(narration_path)],
        check=True,
        capture_output=True,
    )

    info = await ffmpeg_utils.probe(long_source)
    assert info.duration_seconds >= 89  # sanity: source really is long

    ass_path = tmp_path / "captions.ass"
    build_ass_captions(
        output_path=ass_path,
        clip_start=40.0,
        clip_end=64.0,
        transcript_words=[],
        narration_cues=[(0.0, 2.0, "Watch this.")],
        hook_text="A hook.",
    )

    out_path = tmp_path / "short_clip.mp4"
    await render_vertical_clip(
        source_path=long_source,
        start_seconds=40.0,
        end_seconds=64.0,  # a 24-second window, matching the production report
        output_path=out_path,
        media_info=info,
        narration_tracks=[NarrationTrack(audio_path=narration_path, start_seconds=0.5, duration_seconds=2.0)],
        captions_ass_path=ass_path,
    )

    out_info = await ffmpeg_utils.probe(out_path)
    assert abs(out_info.duration_seconds - 24.0) <= 0.5, (
        f"expected ~24s output, got {out_info.duration_seconds}s - "
        "the render exported far more than the selected window"
    )


async def test_render_rejects_zero_length_clip(tmp_path, synthetic_video):
    info = await ffmpeg_utils.probe(synthetic_video)
    with pytest.raises(ValueError):
        await render_vertical_clip(
            source_path=synthetic_video,
            start_seconds=3.0,
            end_seconds=3.0,
            output_path=tmp_path / "out.mp4",
            media_info=info,
        )


async def test_effects_visibly_render_circle_punch_zoom_and_freeze(tmp_path, has_ffmpeg):
    """The real quality bar (spec section 22): not just "the fields exist in
    a schema" - an actual red circle appears in decoded frames at the right
    time/place, a punch-zoom visibly enlarges the target, a freeze produces
    identical consecutive frames, output stays 1080x1920, and duration
    still passes the same validation as every other render."""
    if not has_ffmpeg:
        pytest.skip("ffmpeg not available")

    from PIL import Image

    from app.jobs.models import BBox, Clip, Effect, Target
    from app.pipeline import pipeline as pl
    from app.pipeline.timeline import build_render_plan

    source = tmp_path / "target_source.mp4"
    # A moving test pattern (so a frozen frame is visibly distinguishable
    # from a normal, still-playing one) with a solid blue box drawn at a
    # fixed, known position throughout - the "target" the circle/punch-zoom
    # effects point at, and something we can visually verify got zoomed in on.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-vf",
            "drawbox=x=1400:y=300:w=300:h=300:color=blue:t=fill",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    media_info = await ffmpeg_utils.probe(source)
    target_bbox = BBox(x=1400 / 1920, y=300 / 1080, width=300 / 1920, height=300 / 1080)
    target = Target(description="blue box", bbox=target_bbox, confidence=0.95)

    clip = Clip(start_seconds=0.0, end_seconds=6.0, duration_seconds=6.0)
    clip.effects = [
        Effect(type="circle", start_seconds=0.5, end_seconds=1.5, target=target),
        Effect(type="punch_zoom", start_seconds=2.0, end_seconds=3.0, zoom=1.4, target=target),
        Effect(type="freeze", start_seconds=4.0, end_seconds=4.5),
    ]

    # Mirrors app.pipeline.pipeline._render_and_upload_clip exactly: bias
    # the pan-crop toward the target before deriving the render plan (spec
    # section 3 - prefer adjusting the crop to include the target).
    augmented_keyframes = pl._augment_crop_keyframes_with_targets(clip, 0.75)
    plan = build_render_plan(clip, video_duration=media_info.duration_seconds, raw_crop_keyframes=augmented_keyframes)
    clip_dir = tmp_path / "clipdir"
    clip_dir.mkdir()
    overlays = pl._build_overlay_specs(clip, plan, media_info, clip_dir, 0.75)
    assert len(overlays) == 1  # only the circle needs a PNG - punch_zoom is a crop-level effect

    out_path = tmp_path / "effects_out.mp4"
    await render_vertical_clip(
        source_path=source,
        start_seconds=0.0,
        end_seconds=6.0,
        output_path=out_path,
        media_info=media_info,
        segments=plan.segments,
        overlays=overlays,
    )

    out_info = await ffmpeg_utils.probe(out_path)
    assert out_info.width == 1080
    assert out_info.height == 1920
    expected_duration = sum(s.output_duration for s in plan.segments)
    assert abs(out_info.duration_seconds - expected_duration) <= 0.5

    def _extract_frame(t: float, name: str):
        frame_path = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(out_path), "-frames:v", "1", str(frame_path)],
            check=True,
            capture_output=True,
        )
        return Image.open(frame_path).convert("RGB")

    def _count_blue(img) -> int:
        return sum(1 for r, g, b in img.getdata() if b > 150 and r < 120 and g < 120)

    def _is_overlay_red(px) -> bool:
        r, g, b = px
        return r > 180 and g < 100 and b < 100

    # --- circle overlay actually visible during its window, gone after ---
    # testsrc2 is a busy color-bar pattern (it has its own reds/greens/
    # blues), so check the SPECIFIC pixel positions the overlay PNG itself
    # painted, not "any reddish pixel anywhere in frame" - a naive global
    # count picks up false positives from the test pattern.
    overlay_img = Image.open(overlays[0].image_path)
    painted_positions = [
        (x, y) for x, y in [(x, y) for y in range(0, 1920, 7) for x in range(0, 1080, 7)] if overlay_img.getpixel((x, y))[3] > 0
    ]
    assert painted_positions, "overlay PNG should have painted something"

    during_circle = _extract_frame(1.0, "f_circle.png")
    after_circle = _extract_frame(5.8, "f_after_circle.png")
    during_matches = sum(1 for x, y in painted_positions if _is_overlay_red(during_circle.getpixel((x, y))))
    after_matches = sum(1 for x, y in painted_positions if _is_overlay_red(after_circle.getpixel((x, y))))
    assert during_matches > 0, "expected red overlay pixels at the painted positions during the circle's window"
    assert after_matches == 0, "the circle overlay must disappear once its window ends"

    # --- punch zoom visibly enlarges the target vs. a non-zoomed frame ---
    normal_frame = _extract_frame(0.2, "f_normal.png")
    zoomed_frame = _extract_frame(2.5, "f_zoom.png")
    assert _count_blue(zoomed_frame) > _count_blue(normal_frame) * 1.15

    def _mean_abs_diff(img_a, img_b) -> float:
        data_a, data_b = list(img_a.getdata()), list(img_b.getdata())
        total = sum(abs(a[c] - b[c]) for a, b in zip(data_a, data_b, strict=True) for c in range(3))
        return total / (len(data_a) * 3)

    # --- freeze produces (near-)identical consecutive frames (a real held
    # frame, not just visually similar motion) - allow a tiny tolerance for
    # H.264 lossy-encoding noise between frames, since it's still lossy
    # compression even when every source frame fed to the encoder is
    # byte-identical.
    freeze_a = _extract_frame(4.05, "f_freeze_a.png")
    freeze_b = _extract_frame(4.4, "f_freeze_b.png")
    assert _mean_abs_diff(freeze_a, freeze_b) < 1.0

    # --- and a frame just after the freeze differs again (playback resumed) ---
    after_freeze = _extract_frame(5.0, "f_after_freeze.png")
    assert _mean_abs_diff(after_freeze, freeze_a) > 5.0
