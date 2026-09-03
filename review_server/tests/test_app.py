import json

import pytest

import review_server.app as review_app


@pytest.fixture()
def app(tmp_path, monkeypatch):
    sent = []

    def fake_discord_request(token, method, path, payload):
        sent.append({"token": token, "method": method, "path": path, "payload": payload})
        return {"id": f"message-{len(sent)}"}

    monkeypatch.setattr(review_app, "discord_api_request", fake_discord_request)
    monkeypatch.setattr(review_app, "verify_discord_signature", lambda *_args: True)
    instance = review_app.create_app({
        "TESTING": True,
        "DATABASE_PATH": str(tmp_path / "reviews.db"),
        "DISCORD_PUBLIC_KEY": "public-key",
        "DISCORD_BOT_TOKEN": "bot-token",
        "DISCORD_CHANNEL_ID": "channel-1",
        "DISCORD_REVIEWER_USER_IDS": {"reviewer-1"},
        "LEARNER_API_KEY": "learner-secret",
        "ALLOWED_ORIGINS": {"https://course.example"},
    })
    instance.sent_messages = sent
    return instance


@pytest.fixture()
def client(app):
    return app.test_client()


def learner_headers():
    return {"X-Learner-Key": "learner-secret", "Origin": "https://course.example"}


def submission_payload(stage=1, content="Reviewed existing work"):
    return {
        "learner_id": "learner_01",
        "stage": stage,
        "work_date": "2026-09-02",
        "start_time": "09:00",
        "end_time": "10:00",
        "minutes": 60,
        "content": content,
        "thought": "Understand before changing",
        "stress": 6,
    }


def interaction(custom_id, *, user_id="reviewer-1", interaction_type=3, components=None):
    data = {"custom_id": custom_id}
    if components is not None:
        data["components"] = components
    return {
        "type": interaction_type,
        "member": {"user": {"id": user_id}},
        "data": data,
    }


def post_interaction(client, payload):
    return client.post(
        "/discord/interactions",
        data=json.dumps(payload),
        content_type="application/json",
        headers={"X-Signature-Ed25519": "signature", "X-Signature-Timestamp": "timestamp"},
    )


def test_health_and_cors(client):
    response = client.get("/health", headers={"Origin": "https://course.example"})
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert response.headers["Access-Control-Allow-Origin"] == "https://course.example"


def test_learner_api_requires_key(client):
    response = client.get("/api/course/state?learner_id=learner_01")
    assert response.status_code == 401


def test_submission_posts_interactive_review_card(app, client):
    response = client.post("/api/submissions", json=submission_payload(), headers=learner_headers())
    assert response.status_code == 201
    body = response.get_json()
    assert body["submission"]["status"] == "pending"
    assert body["current_stage"] == 1

    sent = app.sent_messages[0]
    assert sent["path"] == "/channels/channel-1/messages"
    labels = [button["label"] for button in sent["payload"]["components"][0]["components"]]
    assert labels == ["快速通过", "评语后通过", "返工并写评语"]
    assert sent["payload"]["allowed_mentions"] == {"parse": []}

    state = client.get("/api/course/state?learner_id=learner_01", headers=learner_headers()).get_json()
    assert state["completed"] == 0
    assert state["current_stage"] == 1


def test_only_authorized_reviewer_can_approve(client):
    created = client.post("/api/submissions", json=submission_payload(), headers=learner_headers()).get_json()
    submission_id = created["submission"]["id"]

    denied = post_interaction(client, interaction(f"approve:{submission_id}", user_id="someone-else"))
    assert denied.status_code == 200
    assert denied.get_json()["data"]["flags"] == 64

    state = client.get("/api/course/state?learner_id=learner_01", headers=learner_headers()).get_json()
    assert state["completed"] == 0


def test_approval_unlocks_next_stage(client):
    created = client.post("/api/submissions", json=submission_payload(), headers=learner_headers()).get_json()
    submission_id = created["submission"]["id"]

    reviewed = post_interaction(client, interaction(f"approve:{submission_id}"))
    assert reviewed.status_code == 200
    assert reviewed.get_json()["type"] == 7
    assert reviewed.get_json()["data"]["components"] == []

    state = client.get("/api/course/state?learner_id=learner_01", headers=learner_headers()).get_json()
    assert state["completed"] == 1
    assert state["current_stage"] == 2
    assert state["submissions"][0]["status"] == "approved"


def test_locked_stage_cannot_be_submitted(client):
    response = client.post("/api/submissions", json=submission_payload(stage=2), headers=learner_headers())
    assert response.status_code == 409
    assert response.get_json()["current_stage"] == 1


def test_revision_requires_feedback_and_can_be_resubmitted(client):
    created = client.post("/api/submissions", json=submission_payload(), headers=learner_headers()).get_json()
    submission_id = created["submission"]["id"]

    modal = post_interaction(client, interaction(f"revise:{submission_id}"))
    assert modal.get_json()["type"] == 9
    assert modal.get_json()["data"]["custom_id"] == f"review_modal:revise:{submission_id}"

    components = [{"type": 1, "components": [{"type": 4, "custom_id": "feedback", "value": "Explain the user impact."}]}]
    reviewed = post_interaction(
        client,
        interaction(f"review_modal:revise:{submission_id}", interaction_type=5, components=components),
    )
    assert reviewed.get_json()["type"] == 7

    state = client.get("/api/course/state?learner_id=learner_01", headers=learner_headers()).get_json()
    assert state["completed"] == 0
    assert state["current_stage"] == 1
    assert state["submissions"][0]["status"] == "revision_requested"
    assert state["submissions"][0]["feedback"] == "Explain the user impact."

    resubmission = client.post(
        "/api/submissions",
        json=submission_payload(content="Revised with explicit user impact"),
        headers=learner_headers(),
    )
    assert resubmission.status_code == 201
    final_state = client.get("/api/course/state?learner_id=learner_01", headers=learner_headers()).get_json()
    assert final_state["submissions"][0]["status"] == "pending"
    assert final_state["submissions"][1]["status"] == "superseded"


def test_approve_with_written_feedback(client):
    created = client.post("/api/submissions", json=submission_payload(), headers=learner_headers()).get_json()
    submission_id = created["submission"]["id"]

    modal = post_interaction(client, interaction(f"approve_note:{submission_id}"))
    assert modal.get_json()["type"] == 9
    components = [{"type": 1, "components": [{"type": 4, "custom_id": "feedback", "value": "Clear reasoning and good handoff."}]}]
    reviewed = post_interaction(
        client,
        interaction(f"review_modal:approve:{submission_id}", interaction_type=5, components=components),
    )
    assert reviewed.get_json()["type"] == 7
    state = client.get("/api/course/state?learner_id=learner_01", headers=learner_headers()).get_json()
    assert state["completed"] == 1
    assert state["submissions"][0]["feedback"] == "Clear reasoning and good handoff."


def test_duplicate_pending_submission_is_rejected(client):
    first = client.post("/api/submissions", json=submission_payload(), headers=learner_headers())
    second = client.post("/api/submissions", json=submission_payload(), headers=learner_headers())
    assert first.status_code == 201
    assert second.status_code == 409
    assert second.get_json()["error"] == "stage is already awaiting review"
