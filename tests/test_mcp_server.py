"""End-to-end tests against the real MCP streamable-http transport (not just
calling the tool Python functions directly) - confirms the tools are
actually registered and reachable over JSON-RPC, including the new
retry_road_rage_job tool.

The mounted MCP app enables DNS-rebinding protection that only allows
Host: 127.0.0.1[:port] - TestClient's default `Host: testserver` gets a 421,
so these tests pin base_url/Host to 127.0.0.1.

Job fixtures are seeded via JobStore._save_sync (plain sqlite3, no asyncio)
rather than `asyncio.run(store.save(...))` - the app's background retention
cleanup loop is alive on TestClient's own portal thread/loop for the
duration of the `with TestClient(...)` block, and touches the same
JobStore's asyncio.Lock; racing a second, independent event loop
(asyncio.run) against it from the test can deadlock the lock hand-off across
threads. Sync-only seeding sidesteps that entirely."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.jobs.models import Job, JobStatus
from app.jobs.store import get_job_store

MCP_BASE_URL = "http://127.0.0.1:8000"
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "Host": "127.0.0.1:8000",
}


def _seed_job(job: Job) -> None:
    get_job_store()._save_sync(job)


@pytest.fixture
def mcp_client(settings, monkeypatch):
    import app.main as main_mod
    from app.jobs import service as job_service

    monkeypatch.setattr(main_mod, "enqueue_job", lambda job_id: None)
    monkeypatch.setattr(job_service, "enqueue_job", lambda job_id: None)

    test_app = main_mod.create_app()
    with TestClient(test_app, base_url=MCP_BASE_URL) as c:
        yield c, settings


def _rpc(client: TestClient, method: str, params: dict, *, mcp_api_key: str, req_id: int = 1) -> dict:
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {mcp_api_key}"}
    resp = client.post("/mcp/", json={"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}, headers=headers)
    assert resp.status_code == 200, resp.text
    # Response is a single SSE event: "event: message\r\ndata: {...}\r\n\r\n"
    data_line = next(line for line in resp.text.splitlines() if line.startswith("data:"))
    return json.loads(data_line[len("data:") :].strip())


def _call_tool(client: TestClient, name: str, arguments: dict, *, mcp_api_key: str) -> dict:
    envelope = _rpc(client, "tools/call", {"name": name, "arguments": arguments}, mcp_api_key=mcp_api_key)
    assert "error" not in envelope, envelope
    content = envelope["result"]["content"]
    text = next(block["text"] for block in content if block["type"] == "text")
    return json.loads(text)


def test_tools_list_includes_retry_tool(mcp_client):
    client, settings = mcp_client
    envelope = _rpc(client, "tools/list", {}, mcp_api_key=settings.mcp_api_key)
    names = {t["name"] for t in envelope["result"]["tools"]}
    assert names == {"create_road_rage_clips", "get_clip_job", "retry_road_rage_job"}


def test_retry_tool_reports_error_for_unknown_job(mcp_client):
    client, settings = mcp_client
    result = _call_tool(client, "retry_road_rage_job", {"job_id": "does-not-exist"}, mcp_api_key=settings.mcp_api_key)
    assert "error" in result


def test_retry_tool_reports_error_when_job_not_failed(mcp_client):
    client, settings = mcp_client

    job = Job(status=JobStatus.RENDERING)
    _seed_job(job)

    result = _call_tool(client, "retry_road_rage_job", {"job_id": job.id}, mcp_api_key=settings.mcp_api_key)
    assert "error" in result


def test_retry_tool_resets_failed_job_over_real_transport(mcp_client):
    client, settings = mcp_client

    job = Job(status=JobStatus.FAILED, error="boom", ready_to_render=True)
    _seed_job(job)

    result = _call_tool(client, "retry_road_rage_job", {"job_id": job.id}, mcp_api_key=settings.mcp_api_key)
    assert result["job_id"] == job.id
    assert result["status"] == "queued"

    status = _call_tool(client, "get_clip_job", {"job_id": job.id}, mcp_api_key=settings.mcp_api_key)
    assert status["status"] == "queued"
    assert status["error"] is None


def test_wrong_bearer_token_is_rejected_before_reaching_the_tool(mcp_client):
    client, _settings = mcp_client
    headers = {**MCP_HEADERS, "Authorization": "Bearer wrong-token"}
    resp = client.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        headers=headers,
    )
    assert resp.status_code == 401
