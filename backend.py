from datetime import datetime, timedelta
import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from export_memory import export_memory


DB_PATH = Path(__file__).with_name("ecg_memory.db")
USER_DATA_ROOT = Path(__file__).with_name("user_data")
CHAT_RETENTION_DAYS = 7

app = FastAPI(title="ECG Memory Backend")


class UserIn(BaseModel):
    name: str
    id_last4: str | None = None
    age: int | None = None
    gender: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    medical_history: str | None = None
    location: str | None = None
    notes: str | None = None
    knowledge_level: str | None = None


class RecordIn(BaseModel):
    csv_path: str
    source: str | None = None
    final_result: str | None = None
    probability: float | None = None
    total_windows: int | None = None
    counts: dict[str, Any] | dict[int, Any] | None = None


class ChatMessageIn(BaseModel):
    role: str
    content: str


class LoginIn(BaseModel):
    name: str
    id_last4: str


def hash_pin(pin: str | None):
    if not pin:
        return None
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                id_last4 TEXT,
                pin_hash TEXT,
                age INTEGER,
                gender TEXT,
                height_cm REAL,
                weight_kg REAL,
                medical_history TEXT,
                location TEXT,
                notes TEXT,
                knowledge_level TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ecg_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                csv_path TEXT NOT NULL,
                source TEXT,
                final_result TEXT,
                probability REAL,
                total_windows INTEGER,
                counts_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "pin_hash" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN pin_hash TEXT")
        if "id_last4" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN id_last4 TEXT")
        if "knowledge_level" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN knowledge_level TEXT")


def row_to_user(row):
    user = dict(row)
    user.pop("pin_hash", None)
    return user


def row_to_record(row):
    record = dict(row)
    record["counts"] = json.loads(record.pop("counts_json") or "{}")
    return record


def row_to_chat_message(row):
    return dict(row)


def chat_retention_cutoff():
    return (datetime.now() - timedelta(days=CHAT_RETENTION_DAYS)).isoformat(timespec="seconds")


def prune_old_chat_messages(user_id: int | None = None):
    cutoff = chat_retention_cutoff()
    with connect() as conn:
        if user_id:
            conn.execute(
                "DELETE FROM chat_messages WHERE user_id = ? AND created_at < ?",
                (user_id, cutoff),
            )
        else:
            conn.execute("DELETE FROM chat_messages WHERE created_at < ?", (cutoff,))


def safe_folder_name(value: str):
    value = re.sub(r'[<>:"/\\|?*\s]+', "_", value).strip("._")
    return value or "user"


def get_user_row(conn, user_id: int):
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def user_folder_name(user):
    return f"user_{user['id']}_{safe_folder_name(user['name'])}"


def read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def import_user_data_folders():
    if not USER_DATA_ROOT.exists():
        return

    with connect() as conn:
        for folder in USER_DATA_ROOT.glob("user_*_*"):
            if not folder.is_dir():
                continue

            profile = read_json_file(folder / "profile.json", None)
            if not isinstance(profile, dict):
                continue

            user_id = profile.get("id")
            name = profile.get("name")
            if not user_id or not name:
                continue

            id_last4 = profile.get("id_last4")
            conn.execute(
                """
                INSERT INTO users (
                    id, name, id_last4, pin_hash, age, gender, height_cm,
                    weight_kg, medical_history, location, notes, knowledge_level,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    id_last4 = excluded.id_last4,
                    pin_hash = excluded.pin_hash,
                    age = excluded.age,
                    gender = excluded.gender,
                    height_cm = excluded.height_cm,
                    weight_kg = excluded.weight_kg,
                    medical_history = excluded.medical_history,
                    location = excluded.location,
                    notes = excluded.notes,
                    knowledge_level = excluded.knowledge_level,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    name,
                    id_last4,
                    hash_pin(id_last4),
                    profile.get("age"),
                    profile.get("gender"),
                    profile.get("height_cm"),
                    profile.get("weight_kg"),
                    profile.get("medical_history"),
                    profile.get("location"),
                    profile.get("notes"),
                    profile.get("knowledge_level") or "beginner",
                    profile.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                    profile.get("updated_at") or profile.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                ),
            )

            records = read_json_file(folder / "ecg_records.json", [])
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict) or not record.get("csv_path"):
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO ecg_records (
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
                            record.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                        ),
                    )

            chat_path = folder / "chat_messages.jsonl"
            if chat_path.exists():
                for line in chat_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line)
                    except Exception:
                        continue
                    if message.get("role") not in {"user", "assistant"}:
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO chat_messages (
                            id, user_id, role, content, created_at
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            message.get("id"),
                            message.get("user_id") or user_id,
                            message.get("role"),
                            message.get("content") or "",
                            message.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                        ),
                    )


def sync_user_folder(user_id: int):
    prune_old_chat_messages(user_id)
    with connect() as conn:
        user = get_user_row(conn, user_id)
        if not user:
            return None

        folder = USER_DATA_ROOT / user_folder_name(user)
        if USER_DATA_ROOT.exists():
            for old_folder in USER_DATA_ROOT.glob(f"user_{user_id}_*"):
                if old_folder != folder and old_folder.is_dir():
                    shutil.rmtree(old_folder)

        csv_folder = folder / "csv"
        csv_folder.mkdir(parents=True, exist_ok=True)

        profile = row_to_user(user)
        (folder / "profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        records = conn.execute(
            """
            SELECT *
            FROM ecg_records
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        ).fetchall()
        record_rows = [row_to_record(row) for row in records]
        (folder / "ecg_records.json").write_text(
            json.dumps(record_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for record in record_rows:
            csv_path = Path(record["csv_path"])
            if not csv_path.is_absolute():
                csv_path = Path(__file__).parent / csv_path
            if csv_path.exists():
                destination = csv_folder / csv_path.name
                if csv_path.resolve() != destination.resolve():
                    shutil.copy2(csv_path, destination)

        chat_rows = conn.execute(
            """
            SELECT *
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        ).fetchall()
        chat_messages = [row_to_chat_message(row) for row in chat_rows]
        with (folder / "chat_messages.jsonl").open("w", encoding="utf-8") as f:
            for message in chat_messages:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")

        lines = [
            "個人資料匯出",
            "=" * 60,
            f"使用者 ID：{profile.get('id')}",
            f"姓名：{profile.get('name')}",
            f"身分證後四碼：{profile.get('id_last4') or ''}",
            f"年齡：{profile.get('age') or ''}",
            f"性別：{profile.get('gender') or ''}",
            f"身高：{profile.get('height_cm') or ''}",
            f"體重：{profile.get('weight_kg') or ''}",
            f"病史：{profile.get('medical_history') or ''}",
            f"位置：{profile.get('location') or ''}",
            f"備註：{profile.get('notes') or ''}",
            "",
            f"ECG 紀錄數量：{len(record_rows)}",
        ]
        for record in record_rows:
            lines.extend([
                "",
                f"  紀錄 ID：{record.get('id')}",
                f"  時間：{record.get('created_at')}",
                f"  來源：{record.get('source') or ''}",
                f"  CSV：{record.get('csv_path')}",
                f"  最大機率病症：{record.get('final_result') or ''}",
                f"  機率：{record.get('probability') if record.get('probability') is not None else ''}",
            ])
        lines.extend(["", f"對話紀錄數量：{len(chat_messages)}"])
        for message in chat_messages:
            role_label = "使用者" if message["role"] == "user" else "AI"
            lines.extend([
                "",
                f"  時間：{message.get('created_at')}",
                f"  角色：{role_label}",
                f"  內容：{message.get('content')}",
            ])

        (folder / "user_export.txt").write_text("\n".join(lines), encoding="utf-8")
        return folder


def delete_user_folder(user_id: int):
    if not USER_DATA_ROOT.exists():
        return
    for folder in USER_DATA_ROOT.glob(f"user_{user_id}_*"):
        if folder.is_dir():
            shutil.rmtree(folder)


def sync_all_user_folders():
    with connect() as conn:
        rows = conn.execute("SELECT id FROM users ORDER BY id ASC").fetchall()
    for row in rows:
        sync_user_folder(row["id"])


def refresh_export():
    try:
        prune_old_chat_messages()
        export_memory()
    except Exception:
        pass


@app.on_event("startup")
def startup():
    init_db()
    import_user_data_folders()
    prune_old_chat_messages()
    refresh_export()
    sync_all_user_folders()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/users")
def list_users():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY updated_at DESC, id DESC").fetchall()
    return [row_to_user(row) for row in rows]


@app.post("/login")
def login(login_data: LoginIn):
    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM users
            WHERE name = ? AND pin_hash = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (login_data.name, hash_pin(login_data.id_last4)),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid name or code")
    return row_to_user(row)


@app.post("/users")
def create_user(user: UserIn):
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (
                name, id_last4, pin_hash, age, gender, height_cm, weight_kg,
                medical_history, location, notes, knowledge_level, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.name,
                user.id_last4,
                hash_pin(user.id_last4),
                user.age,
                user.gender,
                user.height_cm,
                user.weight_kg,
                user.medical_history,
                user.location,
                user.notes,
                user.knowledge_level or "beginner",
                now,
                now,
            ),
        )
        user_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    refresh_export()
    sync_user_folder(user_id)
    return row_to_user(row)


@app.put("/users/{user_id}")
def update_user(user_id: int, user: UserIn):
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        exists = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="User not found")
        conn.execute(
            """
            UPDATE users
            SET name = ?,
                id_last4 = COALESCE(?, id_last4),
                pin_hash = COALESCE(?, pin_hash),
                age = ?, gender = ?, height_cm = ?, weight_kg = ?,
                medical_history = ?, location = ?, notes = ?, knowledge_level = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                user.name,
                user.id_last4,
                hash_pin(user.id_last4),
                user.age,
                user.gender,
                user.height_cm,
                user.weight_kg,
                user.medical_history,
                user.location,
                user.notes,
                user.knowledge_level or "beginner",
                now,
                user_id,
            ),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    refresh_export()
    sync_user_folder(user_id)
    return row_to_user(row)


@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    delete_user_folder(user_id)
    with connect() as conn:
        conn.execute("DELETE FROM ecg_records WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
    refresh_export()
    return {"ok": True}


@app.get("/users/{user_id}/records")
def list_records(user_id: int):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ecg_records
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    return [row_to_record(row) for row in rows]


@app.post("/users/{user_id}/records")
def create_record(user_id: int, record: RecordIn):
    now = datetime.now().isoformat(timespec="seconds")
    counts_json = json.dumps(record.counts or {}, ensure_ascii=False)
    with connect() as conn:
        exists = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="User not found")
        cursor = conn.execute(
            """
            INSERT INTO ecg_records (
                user_id, csv_path, source, final_result, probability,
                total_windows, counts_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                record.csv_path,
                record.source,
                record.final_result,
                record.probability,
                record.total_windows,
                counts_json,
                now,
            ),
        )
        record_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM ecg_records WHERE id = ?", (record_id,)).fetchone()
    refresh_export()
    sync_user_folder(user_id)
    return row_to_record(row)


@app.get("/users/{user_id}/chat")
def list_chat_messages(user_id: int):
    prune_old_chat_messages(user_id)
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        ).fetchall()
    return [row_to_chat_message(row) for row in rows]


@app.post("/users/{user_id}/chat")
def create_chat_message(user_id: int, message: ChatMessageIn):
    if message.role not in {"user", "assistant"}:
        raise HTTPException(status_code=400, detail="Only user and assistant messages are saved")

    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        exists = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="User not found")
        cursor = conn.execute(
            """
            INSERT INTO chat_messages (user_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, message.role, message.content, now),
        )
        message_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM chat_messages WHERE id = ?", (message_id,)).fetchone()
    prune_old_chat_messages(user_id)
    refresh_export()
    sync_user_folder(user_id)
    return row_to_chat_message(row)


@app.delete("/users/{user_id}/chat")
def delete_chat_messages(user_id: int):
    with connect() as conn:
        conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
    refresh_export()
    sync_user_folder(user_id)
    return {"ok": True}
