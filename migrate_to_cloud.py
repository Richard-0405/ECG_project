import os
import sys


def main():
    database_url = os.getenv("DATABASE_URL") or os.getenv("ECG_DATABASE_URL")
    if not database_url:
        print("Please set DATABASE_URL to your cloud PostgreSQL external connection string first.")
        print("PowerShell example:")
        print('$env:DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DATABASE"')
        sys.exit(1)

    # Import after environment variables are ready, because backend.py reads them at import time.
    import backend

    if not backend.USE_POSTGRES:
        print("DATABASE_URL does not look like a PostgreSQL URL.")
        sys.exit(1)

    print("Initializing cloud database tables...")
    backend.init_db()

    print(f"Importing local user folders from: {backend.USER_DATA_ROOT}")
    backend.import_user_data_folders()
    backend.prune_old_chat_messages()

    with backend.connect() as conn:
        users = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
        records = conn.execute("SELECT COUNT(*) AS count FROM ecg_records").fetchone()["count"]
        chats = conn.execute("SELECT COUNT(*) AS count FROM chat_messages").fetchone()["count"]

    print("Migration completed.")
    print(f"Users: {users}")
    print(f"ECG records: {records}")
    print(f"Chat messages: {chats}")


if __name__ == "__main__":
    main()
