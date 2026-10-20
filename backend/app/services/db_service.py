import sqlite3
from pathlib import Path

# Thư mục lưu trữ database (sẽ được mount ra ngoài Docker)
APP_DATA_DIR = Path("app_data")
APP_DATA_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DATA_DIR / "database.db"

class DBService:
    def __init__(self):
        self._init_db()

    def _get_connection(self):
        # Thiết lập connect với check_same_thread=False để dùng trong FastAPI (bất đồng bộ)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Trả về dict thay vì tuple
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    size TEXT NOT NULL,
                    status TEXT DEFAULT 'ready',
                    chunk_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Tự động di trú cột (Schema Migration) cho các DB SQLite đã tồn tại
            cursor.execute("PRAGMA table_info(documents)")
            columns = [col["name"] if isinstance(col, sqlite3.Row) else col[1] for col in cursor.fetchall()]
            if "status" not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN status TEXT DEFAULT 'ready'")
            if "chunk_count" not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN chunk_count INTEGER DEFAULT 0")
            if "error_message" not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN error_message TEXT")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    thoughts_json TEXT,
                    checkpoint_id TEXT,
                    order_index INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def get_all_documents(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, size, status, chunk_count, error_message, created_at FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_document(self, doc_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, size, status, chunk_count, error_message, created_at FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def insert_document(self, doc_id: str, name: str, size: str, status: str = "ready", chunk_count: int = 0, error_message: str = None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (id, name, size, status, chunk_count, error_message) VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, name, size, status, chunk_count, error_message)
            )
            conn.commit()

    def update_document_status(self, doc_id: str, status: str, chunk_count: int = None, error_message: str = None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if chunk_count is not None and error_message is not None:
                cursor.execute(
                    "UPDATE documents SET status = ?, chunk_count = ?, error_message = ? WHERE id = ?",
                    (status, chunk_count, error_message, doc_id)
                )
            elif chunk_count is not None:
                cursor.execute(
                    "UPDATE documents SET status = ?, chunk_count = ? WHERE id = ?",
                    (status, chunk_count, doc_id)
                )
            elif error_message is not None:
                cursor.execute(
                    "UPDATE documents SET status = ?, error_message = ? WHERE id = ?",
                    (status, error_message, doc_id)
                )
            else:
                cursor.execute(
                    "UPDATE documents SET status = ? WHERE id = ?",
                    (status, doc_id)
                )
            conn.commit()

    def delete_document(self, doc_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

    def get_all_document_ids(self) -> list[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM documents")
            return [row["id"] for row in cursor.fetchall()]

    # =========================
    # CHAT SESSIONS & MESSAGES
    # =========================

    def get_all_sessions(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_session(self, session_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_session(self, session_id: str = None, title: str = "Cuộc trò chuyện mới"):
        import uuid
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:10]}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO chat_sessions (id, title, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (session_id, title)
            )
            conn.commit()
            return self.get_session(session_id)

    def update_session_title(self, session_id: str, title: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id)
            )
            conn.commit()

    def touch_session(self, session_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()
            return True

    def save_message(self, session_id: str, role: str, content: str, message_id: str = None, thoughts: list = None, checkpoint_id: str = None, order_index: int = None) -> str:
        import json
        import uuid
        if not message_id:
            message_id = f"msg_{uuid.uuid4().hex[:12]}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure session exists
            cursor.execute(
                "INSERT OR IGNORE INTO chat_sessions (id, title, updated_at) VALUES (?, 'Cuộc trò chuyện mới', CURRENT_TIMESTAMP)",
                (session_id,)
            )

            if order_index is None:
                cursor.execute("SELECT COALESCE(MAX(order_index), 0) + 1 AS next_idx FROM chat_messages WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                order_index = row["next_idx"] if row else 1

            thoughts_json = json.dumps(thoughts, ensure_ascii=False) if thoughts else None
            cursor.execute("""
                INSERT INTO chat_messages (id, session_id, role, content, thoughts_json, checkpoint_id, order_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    thoughts_json = excluded.thoughts_json,
                    checkpoint_id = COALESCE(excluded.checkpoint_id, chat_messages.checkpoint_id)
            """, (str(message_id), session_id, role, content, thoughts_json, checkpoint_id, order_index))

            # Update session touch time
            cursor.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
            conn.commit()
            return str(message_id)

    def update_message_checkpoint(self, message_id: str, checkpoint_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE chat_messages SET checkpoint_id = ? WHERE id = ?", (checkpoint_id, message_id))
            conn.commit()

    def get_session_messages(self, session_id: str):
        import json
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, session_id, role, content, thoughts_json, checkpoint_id, order_index, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY order_index ASC, created_at ASC
            """, (session_id,))
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                item = dict(row)
                if item.get("thoughts_json"):
                    try:
                        item["thoughts"] = json.loads(item["thoughts_json"])
                    except:
                        item["thoughts"] = []
                else:
                    item["thoughts"] = []
                messages.append(item)
            return messages

    def delete_messages_from_and_after(self, session_id: str, target_message_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT order_index FROM chat_messages WHERE id = ? AND session_id = ?", (target_message_id, session_id))
            row = cursor.fetchone()
            if row:
                target_order = row["order_index"]
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ? AND order_index >= ?", (session_id, target_order))
                cursor.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
                conn.commit()

    def delete_messages_after(self, session_id: str, target_message_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT order_index FROM chat_messages WHERE id = ? AND session_id = ?", (target_message_id, session_id))
            row = cursor.fetchone()
            if row:
                target_order = row["order_index"]
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ? AND order_index > ?", (session_id, target_order))
                cursor.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
                conn.commit()

# Khởi tạo instance dùng chung
db_service = DBService()
