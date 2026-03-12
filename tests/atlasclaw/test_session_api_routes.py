# -*- coding: utf-8 -*-

from __future__ import annotations

from urllib.parse import quote

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.atlasclaw.api.routes import APIContext, create_router, set_api_context
from app.atlasclaw.session.manager import SessionManager
from app.atlasclaw.session.queue import SessionQueue
from app.atlasclaw.skills.registry import SkillRegistry


def _build_client(tmp_path) -> TestClient:
    ctx = APIContext(
        session_manager=SessionManager(agents_dir=str(tmp_path / "agents")),
        session_queue=SessionQueue(),
        skill_registry=SkillRegistry(),
    )
    set_api_context(ctx)

    app = FastAPI()
    app.include_router(create_router())
    return TestClient(app)


def test_session_routes_use_current_session_manager_interface(tmp_path):
    client = _build_client(tmp_path)

    create_response = client.post("/api/sessions", json={})
    assert create_response.status_code == 200
    session_key = create_response.json()["session_key"]
    encoded_session_key = quote(session_key, safe="")

    get_response = client.get(f"/api/sessions/{encoded_session_key}")
    assert get_response.status_code == 200
    assert get_response.json()["session_key"] == session_key

    reset_response = client.post(
        f"/api/sessions/{encoded_session_key}/reset",
        json={"archive": True},
    )
    assert reset_response.status_code == 200
    assert reset_response.json() == {"status": "reset", "session_key": session_key}

    status_response = client.get(f"/api/sessions/{encoded_session_key}/status")
    assert status_response.status_code == 200
    assert status_response.json()["session_key"] == session_key

    queue_response = client.post(
        f"/api/sessions/{encoded_session_key}/queue",
        json={"mode": "steer"},
    )
    assert queue_response.status_code == 200
    assert queue_response.json() == {"session_key": session_key, "queue_mode": "steer"}

    compact_response = client.post(
        f"/api/sessions/{encoded_session_key}/compact",
        json={},
    )
    assert compact_response.status_code == 200
    assert compact_response.json()["status"] == "compaction_triggered"

    delete_response = client.delete(f"/api/sessions/{encoded_session_key}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"status": "deleted", "session_key": session_key}

    missing_response = client.get(f"/api/sessions/{encoded_session_key}")
    assert missing_response.status_code == 404
