from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # Local SQLite mode does not need psycopg.
    psycopg = None
    dict_row = None

try:
    from export_memory import export_memory
except Exception:
    export_memory = None


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("ECG_SQLITE_PATH", ROOT / "ecg_memory.db"))
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("ECG_DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")))
USER_DATA_ROOT = Path(os.getenv("ECG_USER_DATA_ROOT", ROOT / "user_data"))
CHAT_RETENTION_DAYS = int(os.getenv("ECG_CHAT_RETENTION_DAYS", "7"))


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


IMPORT_USER_DATA_ON_STARTUP = env_flag("ECG_IMPORT_USER_DATA_ON_STARTUP", not USE_POSTGRES)
LOCAL_EXPORT_ENABLED = env_flag("ECG_LOCAL_EXPORT_ENABLED", not USE_POSTGRES)

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
    csv_filename: str | None = None
    csv_content: str | None = None
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


class DbCursor:
    def __init__(self, cursor, lastrowid=None):
        self.cursor = cursor
        self._lastrowid = lastrowid

    @property
    def lastrowid(self):
        return self._lastrowid if self._lastrowid is not None else getattr(self.cursor, "lastrowid", None)

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()


class DbConnection:
    def __init__(self, raw_conn, use_postgres: bool):
        self.raw_conn = raw_conn
        self.use_postgres = use_postgres

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.raw_conn.rollback()
        else:
            self.raw_conn.commit()
        self.raw_conn.close()

    def execute(self, sql: str, params: tuple | list | None = None):
        params = tuple(params or ())
        cursor = self.raw_conn.cursor()
        if self.use_postgres:
            sql = sql.replace("?", "%s")
        cursor.execute(sql, params)
        return DbCursor(cursor)


def connect() -> DbConnection:
    if USE_POSTGRES:
        if psycopg is None:
            raise RuntimeError("psycopg is required when DATABASE_URL is set")
        raw_conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        return DbConnection(raw_conn, True)

    raw_conn = sqlite3.connect(DB_PATH)
    raw_conn.row_factory = sqlite3.Row
    raw_conn.execute("PRAGMA foreign_keys = ON")
    return DbConnection(raw_conn, False)


def execute_insert(conn: DbConnection, sql: str, params: tuple):
    if conn.use_postgres:
        cursor = conn.execute(f"{sql.rstrip()} RETURNING id", params)
        row = cursor.fetchone()
        return row["id"]
    cursor = conn.execute(sql, params)
    return cursor.lastrowid


def get_columns(conn: DbConnection, table_name: str) -> set[str]:
    if conn.use_postgres:
        rows = conn.execute(
            """
            SELECT column_name AS name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = ?
            """,
            (table_name,),
        ).fetchall()
        return {row["name"] for row in rows}

    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def ensure_column(conn: DbConnection, table_name: str, column_name: str, column_type: str):
    if column_name not in get_columns(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def init_db():
    user_id_type = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    record_id_type = user_id_type
    chat_id_type = user_id_type

    with connect() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS users (
                id {user_id_type},
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
            f"""
            CREATE TABLE IF NOT EXISTS ecg_records (
                id {record_id_type},
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                csv_path TEXT NOT NULL,
                csv_filename TEXT,
                csv_content TEXT,
                source TEXT,
                final_result TEXT,
                probability REAL,
                total_windows INTEGER,
                counts_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id {chat_id_type},
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        ensure_column(conn, "users", "pin_hash", "TEXT")
        ensure_column(conn, "users", "id_last4", "TEXT")
        ensure_column(conn, "users", "knowledge_level", "TEXT")
        ensure_column(conn, "ecg_records", "csv_filename", "TEXT")
        ensure_column(conn, "ecg_records", "csv_content", "TEXT")


def hash_pin(pin: str | None):
    if not pin:
        return None
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


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


def get_user_row(conn: DbConnection, user_id: int):
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


def read_csv_content_from_path(csv_path: str | None):
    if not csv_path:
        return None
    path = Path(csv_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return None


def upsert_user(conn: DbConnection, profile: dict[str, Any]):
    user_id = profile.get("id")
    name = profile.get("name")
    id_last4 = profile.get("id_last4")
    now = datetime.now().isoformat(timespec="seconds")
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
            profile.get("created_at") or now,
            profile.get("updated_at") or profile.get("created_at") or now,
        ),
    )


def upsert_record(conn: DbConnection, record: dict[str, Any], user_id: int):
    record_id = record.get("id")
    now = datetime.now().isoformat(timespec="seconds")
    csv_path = record.get("csv_path")
    csv_filename = record.get("csv_filename") or Path(csv_path).name if csv_path else None
    csv_content = record.get("csv_content") or read_csv_content_from_path(csv_path)
    params = (
        record_id,
        record.get("user_id") or user_id,
        csv_path,
        csv_filename,
        csv_content,
        record.get("source"),
        record.get("final_result"),
        record.get("probability"),
        record.get("total_windows"),
        json.dumps(record.get("counts") or {}, ensure_ascii=False),
        record.get("created_at") or now,
    )
    if record_id:
        conn.execute(
            """
            INSERT INTO ecg_records (
                id, user_id, csv_path, csv_filename, csv_content, source,
                final_result, probability, total_windows, counts_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id = excluded.user_id,
                csv_path = excluded.csv_path,
                csv_filename = excluded.csv_filename,
                csv_content = excluded.csv_content,
                source = excluded.source,
                final_result = excluded.final_result,
                probability = excluded.probability,
                total_windows = excluded.total_windows,
                counts_json = excluded.counts_json,
                created_at = excluded.created_at
            """,
            params,
        )
    else:
        execute_insert(
            conn,
            """
            INSERT INTO ecg_records (
                user_id, csv_path, csv_filename, csv_content, source,
                final_result, probability, total_windows, counts_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params[1:],
        )


def upsert_chat_message(conn: DbConnection, message: dict[str, Any], user_id: int):
    message_id = message.get("id")
    now = datetime.now().isoformat(timespec="seconds")
    if message_id:
        conn.execute(
            """
            INSERT INTO chat_messages (id, user_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id = excluded.user_id,
                role = excluded.role,
                content = excluded.content,
                created_at = excluded.created_at
            """,
            (
                message_id,
                message.get("user_id") or user_id,
                message.get("role"),
                message.get("content") or "",
                message.get("created_at") or now,
            ),
        )
    else:
        execute_insert(
            conn,
            """
            INSERT INTO chat_messages (user_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                message.get("user_id") or user_id,
                message.get("role"),
                message.get("content") or "",
                message.get("created_at") or now,
            ),
        )


def reset_postgres_sequences(conn: DbConnection):
    if not conn.use_postgres:
        return
    for table_name in ("users", "ecg_records", "chat_messages"):
        conn.execute(
            f"""
            SELECT setval(
                pg_get_serial_sequence('{table_name}', 'id'),
                COALESCE((SELECT MAX(id) FROM {table_name}), 1),
                (SELECT MAX(id) IS NOT NULL FROM {table_name})
            )
            """
        )


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

            upsert_user(conn, profile)

            records = read_json_file(folder / "ecg_records.json", [])
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict) or not record.get("csv_path"):
                        continue
                    if not record.get("csv_content"):
                        csv_candidate = folder / "csv" / Path(record["csv_path"]).name
                        if csv_candidate.exists():
                            record["csv_content"] = csv_candidate.read_text(encoding="utf-8", errors="replace")
                    upsert_record(conn, record, user_id)

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
                    upsert_chat_message(conn, message, user_id)

        reset_postgres_sequences(conn)


def sync_user_folder(user_id: int):
    if not LOCAL_EXPORT_ENABLED:
        return None

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
            filename = record.get("csv_filename") or Path(record["csv_path"]).name
            destination = csv_folder / filename
            if record.get("csv_content"):
                destination.write_text(record["csv_content"], encoding="utf-8")
                continue

            csv_path = Path(record["csv_path"])
            if not csv_path.is_absolute():
                csv_path = ROOT / csv_path
            if csv_path.exists() and csv_path.resolve() != destination.resolve():
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
            "User export",
            "=" * 60,
            f"User ID: {profile.get('id')}",
            f"Name: {profile.get('name')}",
            f"ID last 4: {profile.get('id_last4') or ''}",
            f"Age: {profile.get('age') or ''}",
            f"Gender: {profile.get('gender') or ''}",
            f"Height cm: {profile.get('height_cm') or ''}",
            f"Weight kg: {profile.get('weight_kg') or ''}",
            f"Medical history: {profile.get('medical_history') or ''}",
            f"Location: {profile.get('location') or ''}",
            f"Notes: {profile.get('notes') or ''}",
            f"Knowledge level: {profile.get('knowledge_level') or ''}",
            "",
            f"ECG records: {len(record_rows)}",
        ]
        for record in record_rows:
            lines.extend([
                "",
                f"  Record ID: {record.get('id')}",
                f"  Created at: {record.get('created_at')}",
                f"  Source: {record.get('source') or ''}",
                f"  CSV: {record.get('csv_path')}",
                f"  Result: {record.get('final_result') or ''}",
                f"  Probability: {record.get('probability') if record.get('probability') is not None else ''}",
            ])
        lines.extend(["", f"Chat messages: {len(chat_messages)}"])
        for message in chat_messages:
            role_label = "User" if message["role"] == "user" else "AI"
            lines.extend([
                "",
                f"  Created at: {message.get('created_at')}",
                f"  Role: {role_label}",
                f"  Content: {message.get('content')}",
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
    if not LOCAL_EXPORT_ENABLED:
        return
    with connect() as conn:
        rows = conn.execute("SELECT id FROM users ORDER BY id ASC").fetchall()
    for row in rows:
        sync_user_folder(row["id"])


def refresh_export():
    if not LOCAL_EXPORT_ENABLED or export_memory is None:
        return
    try:
        prune_old_chat_messages()
        export_memory()
    except Exception:
        pass


@app.on_event("startup")
def startup():
    init_db()
    if IMPORT_USER_DATA_ON_STARTUP:
        import_user_data_folders()
    prune_old_chat_messages()
    refresh_export()
    sync_all_user_folders()


@app.get("/health")
def health():
    with connect() as conn:
        conn.execute("SELECT 1")
    return {
        "ok": True,
        "database": "postgresql" if USE_POSTGRES else "sqlite",
        "local_export_enabled": LOCAL_EXPORT_ENABLED,
    }


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
        user_id = execute_insert(
            conn,
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
    csv_filename = record.csv_filename or Path(record.csv_path).name
    with connect() as conn:
        exists = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="User not found")
        record_id = execute_insert(
            conn,
            """
            INSERT INTO ecg_records (
                user_id, csv_path, csv_filename, csv_content, source, final_result,
                probability, total_windows, counts_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                record.csv_path,
                csv_filename,
                record.csv_content,
                record.source,
                record.final_result,
                record.probability,
                record.total_windows,
                counts_json,
                now,
            ),
        )
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
        message_id = execute_insert(
            conn,
            """
            INSERT INTO chat_messages (user_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, message.role, message.content, now),
        )
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
