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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_all_documents(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, size FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def insert_document(self, doc_id: str, name: str, size: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (id, name, size) VALUES (?, ?, ?)",
                (doc_id, name, size)
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

# Khởi tạo instance dùng chung
db_service = DBService()
