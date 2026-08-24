from __future__ import annotations

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
