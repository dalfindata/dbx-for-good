"""
Persistence for user actions (shortlists, notes, overrides, review decisions).

Satisfies the hackathon MUST: "Persist user actions".

Backend selection:
  - If a Postgres/Lakebase connection is available (env DATABASE_URL, or PGHOST/PGUSER/...,
    or Databricks Apps' injected PG* vars) and `psycopg` is installed -> Postgres (Lakebase).
  - Otherwise -> a local SQLite file (planner_actions.db) so the app runs anywhere immediately.

One table, upsert by (district_name, state_ut):
    shortlist(district_name, state_ut, status, note, priority_override, reviewed, updated_by, updated_at)
"""
import os
import datetime as dt

_PG_DSN = None


def _pg_dsn():
    """Return a Postgres DSN if the environment provides one, else None."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    host = os.environ.get("PGHOST")
    if host:
        user = os.environ.get("PGUSER", "")
        pw = os.environ.get("PGPASSWORD", "")
        db = os.environ.get("PGDATABASE", "databricks_postgres")
        port = os.environ.get("PGPORT", "5432")
        auth = f"{user}:{pw}@" if pw else f"{user}@"
        return f"postgresql://{auth}{host}:{port}/{db}?sslmode=require"
    return None


def _backend():
    global _PG_DSN
    _PG_DSN = _pg_dsn()
    if _PG_DSN:
        try:
            import psycopg  # noqa: F401
            return "postgres"
        except ImportError:
            pass
    return "sqlite"


BACKEND = _backend()
_SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planner_actions.db")

_DDL = """
CREATE TABLE IF NOT EXISTS shortlist (
    district_name     TEXT NOT NULL,
    state_ut          TEXT NOT NULL,
    status            TEXT,
    note              TEXT,
    priority_override INTEGER,
    reviewed          INTEGER DEFAULT 0,
    updated_by        TEXT,
    updated_at        TEXT,
    PRIMARY KEY (district_name, state_ut)
);
"""


def _connect():
    if BACKEND == "postgres":
        import psycopg
        return psycopg.connect(_PG_DSN, autocommit=True)
    import sqlite3
    con = sqlite3.connect(_SQLITE_PATH)
    return con


def _ph(n):
    """Placeholder token for the active backend."""
    return "%s" if BACKEND == "postgres" else "?"


def init_db():
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_DDL)
        if BACKEND == "sqlite":
            con.commit()
    finally:
        con.close()


def upsert(district_name, state_ut, *, status=None, note=None,
           priority_override=None, reviewed=None, user="planner"):
    """Insert or update one district's saved action."""
    now = dt.datetime.utcnow().isoformat(timespec="seconds")
    con = _connect()
    try:
        cur = con.cursor()
        p = _ph(0)
        if BACKEND == "postgres":
            cur.execute(f"""
                INSERT INTO shortlist (district_name, state_ut, status, note,
                    priority_override, reviewed, updated_by, updated_at)
                VALUES ({p},{p},{p},{p},{p},{p},{p},{p})
                ON CONFLICT (district_name, state_ut) DO UPDATE SET
                    status = COALESCE(EXCLUDED.status, shortlist.status),
                    note = COALESCE(EXCLUDED.note, shortlist.note),
                    priority_override = COALESCE(EXCLUDED.priority_override, shortlist.priority_override),
                    reviewed = COALESCE(EXCLUDED.reviewed, shortlist.reviewed),
                    updated_by = EXCLUDED.updated_by,
                    updated_at = EXCLUDED.updated_at
            """, (district_name, state_ut, status, note,
                  priority_override, reviewed, user, now))
        else:
            cur.execute("SELECT status, note, priority_override, reviewed "
                        "FROM shortlist WHERE district_name=? AND state_ut=?",
                        (district_name, state_ut))
            row = cur.fetchone()
            if row:
                status = status if status is not None else row[0]
                note = note if note is not None else row[1]
                priority_override = priority_override if priority_override is not None else row[2]
                reviewed = reviewed if reviewed is not None else row[3]
            cur.execute("""
                INSERT OR REPLACE INTO shortlist (district_name, state_ut, status, note,
                    priority_override, reviewed, updated_by, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (district_name, state_ut, status, note,
                  priority_override, reviewed, user, now))
            con.commit()
    finally:
        con.close()


def remove(district_name, state_ut):
    con = _connect()
    try:
        cur = con.cursor()
        p = _ph(0)
        cur.execute(f"DELETE FROM shortlist WHERE district_name={p} AND state_ut={p}",
                    (district_name, state_ut))
        if BACKEND == "sqlite":
            con.commit()
    finally:
        con.close()


def get_all():
    """Return list of dict rows."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("SELECT district_name, state_ut, status, note, priority_override, "
                    "reviewed, updated_by, updated_at FROM shortlist ORDER BY updated_at DESC")
        cols = ["district_name", "state_ut", "status", "note", "priority_override",
                "reviewed", "updated_by", "updated_at"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def get_one(district_name, state_ut):
    for r in get_all():
        if r["district_name"] == district_name and r["state_ut"] == state_ut:
            return r
    return None
