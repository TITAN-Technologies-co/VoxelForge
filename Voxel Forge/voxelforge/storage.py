import hashlib
import json
import secrets
import sqlite3
from pathlib import Path

from .config import logger


class VoxelForgeStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL DEFAULT '',
                auth_provider TEXT NOT NULL DEFAULT 'email',
                password_salt TEXT,
                password_hash TEXT,
                google_sub TEXT UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            );
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                path TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS blobs (
                key TEXT PRIMARY KEY,
                user_id INTEGER,
                kind TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                data BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            """
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def save_state(self, key, value):
        self.conn.execute(
            """
            INSERT INTO app_state(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(key), json.dumps(value)),
        )
        self.conn.commit()

    def load_state(self, key, default=None):
        row = self.conn.execute("SELECT value FROM app_state WHERE key = ?", (str(key),)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            logger.exception("Failed to decode app_state value for %s", key)
            return default

    def save_blob(self, key, data, kind, user_id=None, content_type="application/octet-stream"):
        self.conn.execute(
            """
            INSERT INTO blobs(key, user_id, kind, content_type, data, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                user_id = excluded.user_id,
                kind = excluded.kind,
                content_type = excluded.content_type,
                data = excluded.data,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(key), user_id, str(kind), str(content_type), sqlite3.Binary(data)),
        )
        self.conn.commit()

    def load_blob(self, key):
        row = self.conn.execute("SELECT data FROM blobs WHERE key = ?", (str(key),)).fetchone()
        return bytes(row["data"]) if row is not None else None

    def save_script(self, name, content, user_id=None, path=""):
        self.conn.execute(
            """
            INSERT INTO scripts(user_id, name, path, content, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (user_id, str(name), str(path), str(content)),
        )
        self.conn.commit()

    def _hash_password(self, password, salt=None):
        salt = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt.encode("utf-8"), 200000)
        return salt, digest.hex()

    def create_email_user(self, email, password, display_name=""):
        email = str(email).strip().lower()
        if not email or "@" not in email:
            raise ValueError("Enter a valid email address.")
        if len(str(password)) < 8:
            raise ValueError("Password must be at least 8 characters.")
        salt, password_hash = self._hash_password(password)
        cur = self.conn.execute(
            """
            INSERT INTO users(email, display_name, auth_provider, password_salt, password_hash, last_login_at)
            VALUES (?, ?, 'email', ?, ?, CURRENT_TIMESTAMP)
            """,
            (email, str(display_name).strip(), salt, password_hash),
        )
        self.conn.commit()
        return self.get_user(cur.lastrowid)

    def authenticate_email_user(self, email, password):
        email = str(email).strip().lower()
        row = self.conn.execute(
            "SELECT * FROM users WHERE email = ? AND auth_provider = 'email'",
            (email,),
        ).fetchone()
        if row is None:
            raise ValueError("No email account was found.")
        _, candidate_hash = self._hash_password(password, row["password_salt"])
        if not secrets.compare_digest(candidate_hash, row["password_hash"] or ""):
            raise ValueError("Incorrect password.")
        self.conn.execute("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (row["id"],))
        self.conn.commit()
        return self.get_user(row["id"])

    def create_or_update_google_user(self, email, google_sub="", display_name=""):
        email = str(email).strip().lower()
        google_sub = str(google_sub).strip() or email
        if not email or "@" not in email:
            raise ValueError("Enter the Google account email.")
        row = self.conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            cur = self.conn.execute(
                """
                INSERT INTO users(email, display_name, auth_provider, google_sub, last_login_at)
                VALUES (?, ?, 'google', ?, CURRENT_TIMESTAMP)
                """,
                (email, str(display_name).strip(), google_sub),
            )
            user_id = cur.lastrowid
        else:
            user_id = row["id"]
            self.conn.execute(
                """
                UPDATE users
                SET auth_provider = 'google',
                    google_sub = ?,
                    display_name = COALESCE(NULLIF(?, ''), display_name),
                    last_login_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (google_sub, str(display_name).strip(), user_id),
            )
        self.conn.commit()
        return self.get_user(user_id)

    def get_user(self, user_id):
        if not user_id:
            return None
        row = self.conn.execute(
            "SELECT id, email, display_name, auth_provider, google_sub, created_at, last_login_at FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        return dict(row) if row is not None else None

