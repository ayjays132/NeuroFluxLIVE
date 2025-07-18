import json
from peer_collab.collaboration_server import CollaborationServer


def test_notes_and_feedback_endpoints():
    server = CollaborationServer(device="cpu")
    client = server.app.test_client()

    # POST notes
    resp = client.post(
        "/notes/test_project",
        json={"notes": "hello"},
        headers={"X-User-Id": "user1"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["notes"] == "hello"

    # GET notes
    resp = client.get("/notes/test_project", headers={"X-User-Id": "user1"})
    assert resp.status_code == 200
    assert resp.get_json()["notes"] == "hello"

    # POST feedback
    resp = client.post(
        "/feedback/test_project",
        json={"comment": "good"},
        headers={"X-User-Id": "user2"},
    )
    assert resp.status_code == 201
    data = resp.get_json()["feedback"]
    assert data["comment"] == "good"

    # GET feedback
    resp = client.get("/feedback/test_project", headers={"X-User-Id": "user1"})
    assert resp.status_code == 200
    assert len(resp.get_json()["feedback"]) == 1
