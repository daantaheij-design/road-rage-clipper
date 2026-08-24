from __future__ import annotations

from app.storage import Storage, verify_local_signature


def test_local_backend_upload_and_download_round_trip(settings, tmp_path):
    storage = Storage(settings)
    assert storage.backend == "local"

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake mp4 bytes")

    storage.upload_file(src, "jobs/abc/clips/1.mp4", content_type="video/mp4")

    stored = settings.local_storage_path / "jobs/abc/clips/1.mp4"
    assert stored.read_bytes() == b"fake mp4 bytes"

    url = storage.signed_download_url("jobs/abc/clips/1.mp4", expires_in=60, filename="clip.mp4")
    assert "/files/jobs/abc/clips/1.mp4" in url
    assert "sig=" in url and "exp=" in url

    # Extract exp/sig back out and confirm they verify.
    from urllib.parse import parse_qs, urlparse

    q = parse_qs(urlparse(url).query)
    assert verify_local_signature(settings, "jobs/abc/clips/1.mp4", int(q["exp"][0]), q["sig"][0])


def test_local_signature_rejects_tampering(settings):
    assert not verify_local_signature(settings, "jobs/abc/clips/1.mp4", 9999999999, "not-a-real-sig")


def test_local_signature_rejects_expired(settings):
    import time

    from app.storage import _sign

    expired_at = int(time.time()) - 10
    sig = _sign(settings, "jobs/abc/clips/1.mp4", expired_at)
    assert not verify_local_signature(settings, "jobs/abc/clips/1.mp4", expired_at, sig)


def test_delete_and_delete_prefix(settings, tmp_path):
    storage = Storage(settings)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    storage.upload_file(src, "jobs/j1/clips/a.mp4")
    storage.upload_file(src, "jobs/j1/clips/b.mp4")
    storage.upload_file(src, "jobs/j2/clips/c.mp4")

    storage.delete_prefix("jobs/j1/")

    assert not (settings.local_storage_path / "jobs/j1/clips/a.mp4").exists()
    assert not (settings.local_storage_path / "jobs/j1/clips/b.mp4").exists()
    assert (settings.local_storage_path / "jobs/j2/clips/c.mp4").exists()


def test_download_file_round_trip(settings, tmp_path):
    # This is what a job retry uses to pull previously-synthesized narration
    # audio back down into a fresh local scratch directory.
    storage = Storage(settings)
    src = tmp_path / "narration.mp3"
    src.write_bytes(b"synthesized narration audio bytes")
    storage.upload_file(src, "jobs/j1/narration/clip1/0.mp3", content_type="audio/mpeg")

    dest = tmp_path / "downloaded" / "0.mp3"
    result = storage.download_file("jobs/j1/narration/clip1/0.mp3", dest)

    assert result == dest
    assert dest.read_bytes() == b"synthesized narration audio bytes"


def test_download_file_creates_parent_dirs(settings, tmp_path):
    storage = Storage(settings)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video bytes")
    storage.upload_file(src, "jobs/j1/clips/a.mp4")

    dest = tmp_path / "does" / "not" / "exist" / "yet" / "a.mp4"
    storage.download_file("jobs/j1/clips/a.mp4", dest)
    assert dest.read_bytes() == b"video bytes"
