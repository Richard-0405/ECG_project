import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).parent
DB_PATH = ROOT / "ecg_memory.db"
OUTPUT_PATH = ROOT / "ecg_memory_export.txt"
CHAT_RETENTION_DAYS = 7


def export_memory():
    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now() - timedelta(days=CHAT_RETENTION_DAYS)).isoformat(timespec="seconds")

    users = conn.execute(
        """
        SELECT id, name, age, gender, height_cm, weight_kg,
               id_last4, medical_history, location, notes, created_at, updated_at
        FROM users
        ORDER BY updated_at DESC, id DESC
        """
    ).fetchall()

    lines = []
    lines.append("ECG 後台儲存資訊")
    lines.append("=" * 60)
    lines.append(f"使用者數量：{len(users)}")
    lines.append("")

    for user in users:
        lines.append(f"使用者 ID：{user['id']}")
        lines.append(f"姓名：{user['name']}")
        lines.append(f"身分證後四碼：{user['id_last4'] or ''}")
        lines.append(f"年齡：{user['age'] or ''}")
        lines.append(f"性別：{user['gender'] or ''}")
        lines.append(f"身高：{user['height_cm'] or ''}")
        lines.append(f"體重：{user['weight_kg'] or ''}")
        lines.append(f"過去病史：{user['medical_history'] or ''}")
        lines.append(f"目前位置：{user['location'] or ''}")
        lines.append(f"備註：{user['notes'] or ''}")
        lines.append(f"建立時間：{user['created_at']}")
        lines.append(f"更新時間：{user['updated_at']}")

        records = conn.execute(
            """
            SELECT *
            FROM ecg_records
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user["id"],),
        ).fetchall()

        lines.append(f"ECG 紀錄數量：{len(records)}")
        for record in records:
            counts = json.loads(record["counts_json"] or "{}")
            lines.append("")
            lines.append(f"  紀錄 ID：{record['id']}")
            lines.append(f"  時間：{record['created_at']}")
            lines.append(f"  來源：{record['source'] or ''}")
            lines.append(f"  CSV：{record['csv_path']}")
            lines.append(f"  最大機率結果：{record['final_result'] or ''}")
            probability = record["probability"]
            lines.append(f"  機率：{probability:.1%}" if probability is not None else "  機率：")
            lines.append(f"  視窗數：{record['total_windows'] or ''}")
            lines.append(f"  類別統計：{counts}")

        chat_messages = conn.execute(
            """
            SELECT *
            FROM chat_messages
            WHERE user_id = ? AND created_at >= ?
            ORDER BY created_at ASC, id ASC
            """,
            (user["id"], cutoff),
        ).fetchall()

        lines.append("")
        lines.append(f"對話紀錄數量：{len(chat_messages)}")
        for message in chat_messages:
            role_label = "使用者" if message["role"] == "user" else "助理"
            lines.append("")
            lines.append(f"  訊息 ID：{message['id']}")
            lines.append(f"  時間：{message['created_at']}")
            lines.append(f"  角色：{role_label}")
            lines.append(f"  內容：{message['content']}")

        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return OUTPUT_PATH


def main():
    output_path = export_memory()
    if output_path is None:
        raise SystemExit(f"找不到資料庫：{DB_PATH}")
    print(f"已匯出：{output_path}")


if __name__ == "__main__":
    main()
