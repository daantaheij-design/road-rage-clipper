from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(settings, monkeypatch):
    import app.main as main_mod

    # Don't actually spin up the real FFmpeg/Claude/ElevenLabs pipeline from
    # HTTP tests - just verify the web layer wires jobs correctly.
    monkeypatch.setattr(main_mod, "enqueue_job", lambda job_id: None)

    # Fresh app per test: the mounted MCP sub-app's session manager can only
    # be started once per instance, and each test needs its own settings
    # (tmp_path) baked into a fresh app anyway.
    test_app = main_mod.create_app()
    with TestClient(test_app) as c:
        yield c


def test_root_redirects_to_upload(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/upload"


def test_upload_requires_auth(client):
    resp = client.get("/upload", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_login_wrong_password(client):
    resp = client.post("/login", data={"password": "nope"})
    assert resp.status_code == 401


def test_login_then_access_upload(client):
    resp = client.post("/login", data={"password": "testpass"}, follow_redirects=False)
    assert resp.status_code == 303
    assert "rrc_session" in resp.cookies

    resp2 = client.get("/upload")
    assert resp2.status_code == 200
    assert "Road Rage Clipper" in resp2.text


def test_api_jobs_requires_auth(client):
    resp = client.get("/api/jobs/nonexistent")
    assert resp.status_code == 401


def test_create_job_requires_source(client):
    client.post("/login", data={"password": "testpass"})
    resp = client.post("/api/jobs", data={"number_of_clips": 3})
    assert resp.status_code == 400


def test_create_job_rejects_unsafe_url(client):
    client.post("/login", data={"password": "testpass"})
    resp = client.post("/api/jobs", data={"video_url": "http://localhost/x.mp4", "number_of_clips": 3})
    assert resp.status_code == 400


def test_create_job_with_url_and_poll_status(client):
    client.post("/login", data={"password": "testpass"})
    resp = client.post(
        "/api/jobs", data={"video_url": "https://example.com/video.mp4", "number_of_clips": 2}
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    status_resp = client.get(f"/api/jobs/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["number_of_clips_requested"] == 2


def test_get_job_not_found(client):
    client.post("/login", data={"password": "testpass"})
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_mcp_requires_bearer_token(client):
    resp = client.post("/mcp/", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert resp.status_code == 401


def test_files_route_rejects_bad_signature(client):
    resp = client.get("/files/jobs/x/clips/1.mp4", params={"exp": 9999999999, "sig": "bad"})
    assert resp.status_code == 403
