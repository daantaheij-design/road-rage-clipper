from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "testpass")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-not-for-prod")
    monkeypatch.setenv("MCP_API_KEY", "test-mcp-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-fake")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("R2_ACCOUNT_ID", "")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "")
    get_settings.cache_clear()

    # These modules cache a process-wide singleton the first time they're
    # used; reset them so each test gets one bound to its own tmp_path.
    import app.jobs.store as store_mod
    import app.storage as storage_mod

    store_mod._store = None
    storage_mod._storage = None

    s = get_settings()
    yield s
    get_settings.cache_clear()
    store_mod._store = None
    storage_mod._storage = None


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory, has_ffmpeg) -> Path:
    """A short synthetic test video (color bars + a sine wave audio track),
    generated on the fly with ffmpeg's lavfi test sources. Never committed to
    the repo - this exists purely so ffmpeg-pipeline tests have something to
    chew on without a real dashcam clip."""
    if not has_ffmpeg:
        pytest.skip("ffmpeg not available")
    out_dir = tmp_path_factory.mktemp("synthetic")
    out_path = out_dir / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=15:duration=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path


@pytest.fixture(scope="session")
def synthetic_audio_clip(tmp_path_factory, has_ffmpeg) -> Path:
    """A short standalone mp3, standing in for a TTS narration clip in tests
    that don't need real ElevenLabs output - just something with a real
    playable duration."""
    if not has_ffmpeg:
        pytest.skip("ffmpeg not available")
    out_dir = tmp_path_factory.mktemp("synthetic_audio")
    out_path = out_dir / "narration.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=2", str(out_path)],
        check=True,
        capture_output=True,
    )
    return out_path
