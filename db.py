
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        credits INTEGER NOT NULL DEFAULT 3,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_at TEXT,
        input_file TEXT,
        is_public INTEGER DEFAULT 1,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS credit_purchases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stripe_session_id TEXT UNIQUE NOT NULL,
        user_id INTEGER NOT NULL,
        package_code TEXT NOT NULL,
        credits INTEGER NOT NULL,
        amount_eur INTEGER NOT NULL,
        payment_status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        completed_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()

    ensure_column_exists(conn, "users", "credits", "INTEGER NOT NULL DEFAULT 3")

    conn.commit()
    conn.close()


def ensure_column_exists(conn, table_name, column_name, column_def):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [row["name"] for row in cur.fetchall()]
    if column_name not in columns:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def insert_user(username, password_salt, password_hash, credits=3):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO users (username, password_salt, password_hash, credits)
    VALUES (?, ?, ?, ?)
    """, (username, password_salt, password_hash, credits))

    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def fetch_user_by_username(username):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM users
    WHERE username = ?
    """, (username,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def fetch_user_by_id(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM users
    WHERE id = ?
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def update_user_credits(user_id, credits):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    UPDATE users
    SET credits = ?
    WHERE id = ?
    """, (credits, user_id))

    conn.commit()
    conn.close()


def add_credits(user_id, amount):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    UPDATE users
    SET credits = credits + ?
    WHERE id = ?
    """, (amount, user_id))

    conn.commit()
    conn.close()


def consume_credit(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT credits
    FROM users
    WHERE id = ?
    """, (user_id,))

    row = cur.fetchone()

    if not row:
        conn.close()
        return {"ok": False, "reason": "user_not_found"}

    current_credits = row["credits"]

    if current_credits <= 0:
        conn.close()
        return {"ok": False, "reason": "no_credits"}

    cur.execute("""
    UPDATE users
    SET credits = credits - 1
    WHERE id = ? AND credits > 0
    """, (user_id,))

    conn.commit()

    cur.execute("""
    SELECT credits
    FROM users
    WHERE id = ?
    """, (user_id,))

    updated_row = cur.fetchone()
    conn.close()

    return {
        "ok": True,
        "remaining_credits": updated_row["credits"]
    }


def insert_job(job_id, user_id, created_at, input_file, is_public=1):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO jobs (job_id, user_id, created_at, input_file, is_public)
    VALUES (?, ?, ?, ?, ?)
    """, (job_id, user_id, created_at, input_file, is_public))

    conn.commit()
    conn.close()


def fetch_job(job_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM jobs
    WHERE job_id = ?
    """, (job_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def fetch_jobs_for_user(user_id, limit=50):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM jobs
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT ?
    """, (user_id, limit))

    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def insert_credit_purchase(stripe_session_id, user_id, package_code, credits, amount_eur, payment_status="pending"):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT OR IGNORE INTO credit_purchases
    (stripe_session_id, user_id, package_code, credits, amount_eur, payment_status)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (stripe_session_id, user_id, package_code, credits, amount_eur, payment_status))

    conn.commit()
    conn.close()


def fetch_credit_purchase_by_session_id(stripe_session_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM credit_purchases
    WHERE stripe_session_id = ?
    """, (stripe_session_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def mark_credit_purchase_completed(stripe_session_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    UPDATE credit_purchases
    SET payment_status = 'paid',
        completed_at = CURRENT_TIMESTAMP
    WHERE stripe_session_id = ?
    """, (stripe_session_id,))

    conn.commit()
    conn.close()