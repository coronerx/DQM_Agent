from __future__ import annotations

import hmac
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parent
MAX_STAGE = 30


def load_env_file(path: Path) -> None:
    """Load a small .env file without adding another runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_discord_signature(public_key: str, signature: str, timestamp: str, body: bytes) -> bool:
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey

        VerifyKey(bytes.fromhex(public_key)).verify(timestamp.encode("utf-8") + body, bytes.fromhex(signature))
        return True
    except (ValueError, BadSignatureError, ImportError):
        return False


def discord_api_request(token: str, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://discord.com/api/v10{path}",
        data=encoded,
        method=method,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "DQM-Collaboration-Course/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Discord API returned {exc.code}: {detail}") from exc


def submission_embed(item: dict[str, Any], *, reviewed: bool = False) -> dict[str, Any]:
    status = item.get("status", "pending")
    colors = {
        "pending": 0xD99A48,
        "approved": 0x3D8C6A,
        "revision_requested": 0xD95252,
        "superseded": 0x87918C,
    }
    labels = {
        "pending": "等待批改",
        "approved": "已通过，下一关已解锁",
        "revision_requested": "需要返工",
        "superseded": "已被新提交替代",
    }
    fields = [
        {"name": "工作时间", "value": f"{item['start_time']}–{item['end_time']} · {item['minutes']} 分钟", "inline": True},
        {"name": "压力指数", "value": f"{item['stress']} / 10", "inline": True},
        {"name": "工作内容", "value": item["content"][:1024]},
        {"name": "工作思路", "value": item["thought"][:1024]},
    ]
    if item.get("feedback"):
        fields.append({"name": "批改评语", "value": item["feedback"][:1024]})
    embed = {
        "title": f"Stage {item['stage']}/30 · {labels.get(status, status)}",
        "description": f"学习者：`{item['learner_id']}` · 日期：{item['work_date']}",
        "color": colors.get(status, 0x87918C),
        "fields": fields,
        "footer": {"text": f"Submission {item['id'][:8]}"},
        "timestamp": item.get("reviewed_at") if reviewed and item.get("reviewed_at") else item["created_at"],
    }
    return embed


def pending_components(submission_id: str) -> list[dict[str, Any]]:
    return [{
        "type": 1,
        "components": [
            {"type": 2, "style": 3, "label": "快速通过", "custom_id": f"approve:{submission_id}"},
            {"type": 2, "style": 1, "label": "评语后通过", "custom_id": f"approve_note:{submission_id}"},
            {"type": 2, "style": 4, "label": "返工并写评语", "custom_id": f"revise:{submission_id}"},
        ],
    }]


def message_payload(item: dict[str, Any], *, interactive: bool) -> dict[str, Any]:
    return {
        "embeds": [submission_embed(item, reviewed=not interactive)],
        "components": pending_components(item["id"]) if interactive else [],
        "allowed_mentions": {"parse": []},
    }


def feedback_modal(submission_id: str, decision: str) -> dict[str, Any]:
    approving = decision == "approve"
    return {
        "type": 9,
        "data": {
            "custom_id": f"review_modal:{decision}:{submission_id}",
            "title": "通过并写评语" if approving else "返工评语",
            "components": [{
                "type": 1,
                "components": [{
                    "type": 4,
                    "custom_id": "feedback",
                    "label": "给学习者的评语",
                    "style": 2,
                    "required": True,
                    "min_length": 1,
                    "max_length": 1000,
                    "placeholder": "具体指出做得好的地方，或需要修改的地方。",
                }],
            }],
        },
    }


def extract_modal_value(components: list[dict[str, Any]], custom_id: str) -> str:
    for row in components:
        for component in row.get("components", []):
            if component.get("custom_id") == custom_id:
                return str(component.get("value", "")).strip()
    return ""


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    load_env_file(ROOT / ".env")
    app = Flask(__name__)
    app.config.update(
        DATABASE_PATH=os.environ.get("DATABASE_PATH", str(ROOT / "data" / "reviews.db")),
        DISCORD_PUBLIC_KEY=os.environ.get("DISCORD_PUBLIC_KEY", ""),
        DISCORD_BOT_TOKEN=os.environ.get("DISCORD_BOT_TOKEN", ""),
        DISCORD_CHANNEL_ID=os.environ.get("DISCORD_CHANNEL_ID", ""),
        DISCORD_REVIEWER_USER_IDS={value.strip() for value in os.environ.get("DISCORD_REVIEWER_USER_IDS", "").split(",") if value.strip()},
        LEARNER_API_KEY=os.environ.get("LEARNER_API_KEY", ""),
        ALLOWED_ORIGINS={value.strip() for value in os.environ.get("ALLOWED_ORIGINS", "").split(",") if value.strip()},
    )
    if test_config:
        app.config.update(test_config)

    database_path = Path(app.config["DATABASE_PATH"])
    database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect() -> sqlite3.Connection:
        db = sqlite3.connect(database_path)
        db.row_factory = sqlite3.Row
        return db

    with connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                learner_id TEXT NOT NULL,
                stage INTEGER NOT NULL,
                work_date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                minutes INTEGER NOT NULL,
                content TEXT NOT NULL,
                thought TEXT NOT NULL,
                stress INTEGER NOT NULL,
                status TEXT NOT NULL,
                feedback TEXT NOT NULL DEFAULT '',
                reviewer_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                discord_message_id TEXT NOT NULL DEFAULT ''
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_submissions_learner_stage ON submissions(learner_id, stage, created_at)")

    def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def current_progress(db: sqlite3.Connection, learner_id: str) -> tuple[int, int]:
        approved = {row["stage"] for row in db.execute(
            "SELECT DISTINCT stage FROM submissions WHERE learner_id=? AND status='approved'",
            (learner_id,),
        )}
        completed = 0
        while completed + 1 in approved and completed < MAX_STAGE:
            completed += 1
        return completed, min(completed + 1, MAX_STAGE)

    def learner_authorized() -> bool:
        expected = str(app.config.get("LEARNER_API_KEY", ""))
        supplied = request.headers.get("X-Learner-Key", "")
        return bool(expected) and hmac.compare_digest(expected, supplied)

    def reviewer_id(payload: dict[str, Any]) -> str:
        return str(payload.get("member", {}).get("user", {}).get("id") or payload.get("user", {}).get("id") or "")

    def reviewer_authorized(payload: dict[str, Any]) -> bool:
        reviewers = app.config.get("DISCORD_REVIEWER_USER_IDS", set())
        return reviewer_id(payload) in reviewers

    def discord_send(item: dict[str, Any]) -> str:
        override = app.config.get("DISCORD_SEND_FUNCTION")
        if override:
            return str(override(item))
        token = str(app.config.get("DISCORD_BOT_TOKEN", ""))
        channel_id = str(app.config.get("DISCORD_CHANNEL_ID", ""))
        if not token or not channel_id:
            raise RuntimeError("Discord bot token or channel ID is not configured")
        response = discord_api_request(token, "POST", f"/channels/{channel_id}/messages", message_payload(item, interactive=True))
        message_id = str(response.get("id", ""))
        if not message_id:
            raise RuntimeError("Discord did not return a message ID")
        return message_id

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin", "")
        allowed = app.config.get("ALLOWED_ORIGINS", set())
        if origin and origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Learner-Key"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/api/<path:_path>", methods=["OPTIONS"])
    def api_options(_path: str):
        return ("", 204)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "collaboration-course-review"})

    @app.get("/api/course/state")
    def course_state():
        if not learner_authorized():
            return jsonify({"error": "unauthorized"}), 401
        learner_id = request.args.get("learner_id", "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", learner_id):
            return jsonify({"error": "invalid learner_id"}), 400
        with connect() as db:
            completed, current = current_progress(db, learner_id)
            rows = db.execute(
                "SELECT * FROM submissions WHERE learner_id=? ORDER BY created_at DESC",
                (learner_id,),
            ).fetchall()
        return jsonify({
            "learner_id": learner_id,
            "completed": completed,
            "current_stage": current,
            "course_complete": completed >= MAX_STAGE,
            "submissions": [dict(row) for row in rows],
        })

    @app.post("/api/submissions")
    def create_submission():
        if not learner_authorized():
            return jsonify({"error": "unauthorized"}), 401
        payload = request.get_json(silent=True) or {}
        learner_id = str(payload.get("learner_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", learner_id):
            return jsonify({"error": "invalid learner_id"}), 400
        try:
            stage = int(payload.get("stage"))
            minutes = int(payload.get("minutes"))
            stress = int(payload.get("stress"))
        except (TypeError, ValueError):
            return jsonify({"error": "stage, minutes and stress must be integers"}), 400
        text_fields = {
            key: str(payload.get(key, "")).strip()
            for key in ("work_date", "start_time", "end_time", "content", "thought")
        }
        if not all(text_fields.values()) or not 1 <= stage <= MAX_STAGE or not 0 <= minutes <= 1440 or not 0 <= stress <= 10:
            return jsonify({"error": "invalid submission fields"}), 400
        if len(text_fields["content"]) > 4000 or len(text_fields["thought"]) > 4000:
            return jsonify({"error": "content or thought is too long"}), 400

        with connect() as db:
            completed, current = current_progress(db, learner_id)
            if completed >= MAX_STAGE:
                return jsonify({"error": "course already complete"}), 409
            if stage != current:
                return jsonify({"error": "stage is locked", "current_stage": current}), 409
            pending = db.execute(
                "SELECT id FROM submissions WHERE learner_id=? AND stage=? AND status='pending' LIMIT 1",
                (learner_id, stage),
            ).fetchone()
            if pending:
                return jsonify({"error": "stage is already awaiting review", "submission_id": pending["id"]}), 409

        item = {
            "id": uuid.uuid4().hex,
            "learner_id": learner_id,
            "stage": stage,
            **text_fields,
            "minutes": minutes,
            "stress": stress,
            "status": "pending",
            "feedback": "",
            "reviewer_id": "",
            "created_at": utc_now(),
            "reviewed_at": None,
            "discord_message_id": "",
        }
        try:
            item["discord_message_id"] = discord_send(item)
        except RuntimeError as exc:
            return jsonify({"error": "discord_send_failed", "detail": str(exc)}), 502

        with connect() as db:
            db.execute(
                "UPDATE submissions SET status='superseded' WHERE learner_id=? AND stage=? AND status='revision_requested'",
                (learner_id, stage),
            )
            db.execute("""
                INSERT INTO submissions (
                    id, learner_id, stage, work_date, start_time, end_time, minutes,
                    content, thought, stress, status, feedback, reviewer_id,
                    created_at, reviewed_at, discord_message_id
                ) VALUES (
                    :id, :learner_id, :stage, :work_date, :start_time, :end_time, :minutes,
                    :content, :thought, :stress, :status, :feedback, :reviewer_id,
                    :created_at, :reviewed_at, :discord_message_id
                )
            """, item)
        return jsonify({"submission": item, "completed": completed, "current_stage": current}), 201

    @app.post("/discord/interactions")
    def discord_interactions():
        raw_body = request.get_data(cache=True)
        signature = request.headers.get("X-Signature-Ed25519", "")
        timestamp = request.headers.get("X-Signature-Timestamp", "")
        public_key = str(app.config.get("DISCORD_PUBLIC_KEY", ""))
        signature_verifier = app.config.get("VERIFY_SIGNATURE_FUNCTION", verify_discord_signature)
        if not public_key or not signature_verifier(public_key, signature, timestamp, raw_body):
            return jsonify({"error": "invalid request signature"}), 401
        payload = request.get_json(silent=True) or {}
        interaction_type = payload.get("type")
        if interaction_type == 1:
            return jsonify({"type": 1})
        if not reviewer_authorized(payload):
            return jsonify({"type": 4, "data": {"content": "你没有这个课程的批改权限。", "flags": 64}})

        data = payload.get("data", {})
        custom_id = str(data.get("custom_id", ""))
        parts = custom_id.split(":", 2)
        if len(parts) < 2:
            return jsonify({"type": 4, "data": {"content": "无法识别这个批改操作。", "flags": 64}})

        action = parts[0]
        if action in {"approve_note", "revise"}:
            submission_id = parts[1]
            decision = "approve" if action == "approve_note" else "revise"
            return jsonify(feedback_modal(submission_id, decision))

        if action == "review_modal" and len(parts) == 3:
            decision, submission_id = parts[1], parts[2]
            feedback = extract_modal_value(data.get("components", []), "feedback")
            if not feedback:
                return jsonify({"type": 4, "data": {"content": "评语不能为空。", "flags": 64}})
            status = "approved" if decision == "approve" else "revision_requested"
        elif action == "approve":
            submission_id = parts[1]
            feedback = ""
            status = "approved"
        else:
            return jsonify({"type": 4, "data": {"content": "无法识别这个批改操作。", "flags": 64}})

        now = utc_now()
        reviewer = reviewer_id(payload)
        with connect() as db:
            row = db.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
            item = row_dict(row)
            if not item:
                return jsonify({"type": 4, "data": {"content": "这份提交不存在。", "flags": 64}})
            if item["status"] != "pending":
                return jsonify({"type": 4, "data": {"content": "这份提交已经批改过了。", "flags": 64}})
            db.execute(
                "UPDATE submissions SET status=?, feedback=?, reviewer_id=?, reviewed_at=? WHERE id=?",
                (status, feedback, reviewer, now, submission_id),
            )
            item.update(status=status, feedback=feedback, reviewer_id=reviewer, reviewed_at=now)
        return jsonify({"type": 7, "data": message_payload(item, interactive=False)})

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8787"))
    app.run(host="127.0.0.1", port=port, debug=False)
