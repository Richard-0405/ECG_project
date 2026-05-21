from datetime import datetime
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "ecg_memory.db"
USER_DATA_ROOT = ROOT / "user_data"


def hash_pin(pin):
    if not pin:
        return None
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM chat_messages")
        conn.execute("DELETE FROM ecg_records")
        conn.execute("DELETE FROM users")

        now = datetime.now().isoformat(timespec="seconds")
        for folder in sorted(USER_DATA_ROOT.glob("user_*_*")):
            if not folder.is_dir():
                continue
            profile = read_json(folder / "profile.json", None)
            if not isinstance(profile, dict) or not profile.get("id") or not profile.get("name"):
                continue

            user_id = int(profile["id"])
            conn.execute(
                """
                INSERT INTO users (
                    id, name, id_last4, pin_hash, age, gender, height_cm, weight_kg,
                    medical_history, location, notes, knowledge_level, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    profile.get("name"),
                    profile.get("id_last4"),
                    hash_pin(profile.get("id_last4")),
                    profile.get("age"),
                    profile.get("gender"),
                    profile.get("height_cm"),
                    profile.get("weight_kg"),
                    profile.get("medical_history"),
                    profile.get("location"),
                    profile.get("notes"),
                    profile.get("knowledge_level") or "beginner",
                    profile.get("created_at") or now,
                    profile.get("updated_at") or profile.get("created_at") or now,
                ),
            )

            records = read_json(folder / "ecg_records.json", [])
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict) or not record.get("csv_path"):
                        continue
                    conn.execute(
                        """
                        INSERT INTO ecg_records (
                            id, user_id, csv_path, source, final_result, probability,
                            total_windows, counts_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.get("id"),
                            record.get("user_id") or user_id,
                            record.get("csv_path"),
                            record.get("source"),
                            record.get("final_result"),
                            record.get("probability"),
                            record.get("total_windows"),
                            json.dumps(record.get("counts") or {}, ensure_ascii=False),
                            record.get("created_at") or now,
                        ),
                    )

            chat_path = folder / "chat_messages.jsonl"
            if chat_path.exists():
                for line in chat_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    message = json.loads(line)
                    if message.get("role") not in {"user", "assistant"}:
                        continue
                    conn.execute(
                        """
                        INSERT INTO chat_messages (id, user_id, role, content, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            message.get("id"),
                            message.get("user_id") or user_id,
                            message.get("role"),
                            message.get("content") or "",
                            message.get("created_at") or now,
                        ),
                    )

        conn.commit()
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        record_count = conn.execute("SELECT COUNT(*) FROM ecg_records").fetchone()[0]
        chat_count = conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
        print(f"restored users={user_count} records={record_count} chat={chat_count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
