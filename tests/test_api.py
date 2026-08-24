from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.jobs.models import Job, JobStatus
from app.jobs.store import get_job_store


def _seed_job(job: Job) -> None:
    # Plain sync sqlite write (JobStore._save_sync), not
    # asyncio.run(store.save(...)): the app's background retention cleanup
    # loop is alive on TestClient's own portal thread/loop for the duration
    # of the `client` fixture and touches the same JobStore's asyncio.Lock -
    # racing a second, independent event loop against it here can deadlock
    # the lock hand-off across threads. Sync-only seeding sidesteps that.
    get_job_store()._save_sync(job)


@pytest.fixture
def client(settings, monkeypatch):
    import app.main as main_mod
    from app.jobs import service as job_service

    # Don't actually spin up the real FFmpeg/Claude/ElevenLabs pipeline from
    # HTTP tests - just verify the web layer wires jobs correctly. Both
    # /api/jobs (main_mod) and /api/jobs/{id}/retry (via job_service) can
    # trigger a background run, so both need patching.
    monkeypatch.setattr(main_mod, "enqueue_job", lambda job_id: None)
    monkeypatch.setattr(job_service, "enqueue_job", lambda job_id: None)

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


def test_retry_route_requires_auth(client):
    resp = client.post("/api/jobs/some-id/retry")
    assert resp.status_code == 401


def test_retry_route_not_found(client):
    client.post("/login", data={"password": "testpass"})
    resp = client.post("/api/jobs/does-not-exist/retry")
    assert resp.status_code == 404


def test_retry_route_conflict_when_not_failed(client):
    client.post("/login", data={"password": "testpass"})

    job = Job(status=JobStatus.RENDERING)
    _seed_job(job)

    resp = client.post(f"/api/jobs/{job.id}/retry")
    assert resp.status_code == 409


def test_retry_route_resets_failed_job(client):
    client.post("/login", data={"password": "testpass"})

    job = Job(status=JobStatus.FAILED, error="boom", ready_to_render=True)
    _seed_job(job)

    resp = client.post(f"/api/jobs/{job.id}/retry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job.id
    assert body["status"] == "queued"

    status_resp = client.get(f"/api/jobs/{job.id}")
    assert status_resp.json()["status"] == "queued"
    assert status_resp.json()["error"] is None


def test_job_status_reports_resumable_flag_when_failed_with_saved_analysis(client):
    client.post("/login", data={"password": "testpass"})

    job = Job(status=JobStatus.FAILED, ready_to_render=True)
    _seed_job(job)

    resp = client.get(f"/api/jobs/{job.id}")
    assert resp.json()["resumable"] is True


def test_list_jobs_requires_auth(client):
    resp = client.get("/api/jobs")
    assert resp.status_code == 401


def test_list_jobs_returns_newest_first_without_touching_pipeline(client):
    client.post("/login", data={"password": "testpass"})

    older = Job(status=JobStatus.COMPLETED, created_at=1000.0)
    newer = Job(status=JobStatus.RENDERING, created_at=2000.0, progress=42, message="Rendering clip 1 of 2")
    _seed_job(older)
    _seed_job(newer)

    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    job_ids = [j["job_id"] for j in jobs]
    assert job_ids.index(newer.id) < job_ids.index(older.id)

    # This is the "browser refresh must not re-trigger AI work" guarantee:
    # a plain read must not change the still-processing job's state at all
    # (no new status, no reset progress) - it's exactly what the store
    # already had.
    restored = next(j for j in jobs if j["job_id"] == newer.id)
    assert restored["status"] == "rendering"
    assert restored["progress"] == 42
    assert restored["message"] == "Rendering clip 1 of 2"


def test_list_jobs_respects_limit(client):
    client.post("/login", data={"password": "testpass"})
    for i in range(3):
        _seed_job(Job(status=JobStatus.COMPLETED, created_at=float(i)))

    resp = client.get("/api/jobs", params={"limit": 1})
    assert len(resp.json()["jobs"]) == 1
