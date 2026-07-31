import sqlite3
from datetime import datetime, timezone
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    model_hash    TEXT NOT NULL,
    gpu           TEXT,
    calib_hash    TEXT,
    label         TEXT,
    ngl           INTEGER,
    threads       INTEGER,
    ctk           TEXT,
    ot_pattern    TEXT,
    gen_tps       REAL,
    quality       REAL,
    kl_mean       REAL,
    vram_mb       REAL,
    spilled       INTEGER
);

CREATE TABLE IF NOT EXISTS sensitivity (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    model_hash    TEXT NOT NULL,
    gpu           TEXT NOT NULL,
    group_name    TEXT NOT NULL,
    size_mb       REAL,
    gain_tps      REAL,
    value_per_mb  REAL
);

CREATE TABLE IF NOT EXISTS models (
    model_hash    TEXT PRIMARY KEY,
    arch          TEXT,
    n_layers      INTEGER,
    file_size_mb  REAL,
    path          TEXT
);
"""

def _migrate(con: sqlite3.Connection) -> None:
    """Add columns that didn't exist in earlier versions of this schema,
    so an existing results.db from before this change keeps working."""
    cols = {r["name"] for r in con.execute("PRAGMA table_info(models)").fetchall()}
    if "path" not in cols:
        con.execute("ALTER TABLE models ADD COLUMN path TEXT")
        con.commit()

def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _migrate(con)
    return con

def save_model(con, model_hash: str, info) -> None:
    con.execute(
        "INSERT OR REPLACE INTO models (model_hash, arch, n_layers, file_size_mb, path)"
        " VALUES (?,?,?,?,?)",
        (model_hash, info.arch, info.n_layers, info.file_size_mb, str(info.path)),
    )
    con.commit()

def save_evaluation(con, model_hash: str, hw, cand, ev, calib_hash: str = "") -> int:
    cur = con.execute(
        "INSERT INTO runs (ts, model_hash, gpu, calib_hash, label, ngl, threads,"
        " ctk, ot_pattern, gen_tps, quality, kl_mean, vram_mb, spilled)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"),
         model_hash, hw.name if hw else "", calib_hash, cand.label,
         cand.ngl, cand.threads, cand.ctk, cand.ot_pattern,
         ev.gen_tps, ev.quality, ev.kl_mean, ev.vram_mb, int(ev.spilled)),
    )
    con.commit()
    return int(cur.lastrowid)

def save_sensitivity(con, model_hash: str, gpu_name: str, groups: dict) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.executemany(
        "INSERT INTO sensitivity (ts, model_hash, gpu, group_name, size_mb,"
        " gain_tps, value_per_mb) VALUES (?,?,?,?,?,?,?)",
        [(ts, model_hash, gpu_name, v.name, v.size_mb, v.gain_tps, v.value_per_mb)
         for v in groups.values()],
    )
    con.commit()

def load_all(con, table: str = "runs") -> list[dict]:
    order_by = "model_hash" if table == "models" else "id"
    rows = con.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
    return [dict(r) for r in rows]

def load_sensitivity(con, model_hash: str, gpu_name: str) -> dict | None:
    rows = con.execute(
        "SELECT * FROM sensitivity WHERE model_hash=? AND gpu=?"
        " AND ts = (SELECT MAX(ts) FROM sensitivity WHERE model_hash=? AND gpu=?)",
        (model_hash, gpu_name, model_hash, gpu_name),
    ).fetchall()
    if not rows:
        return None
    return {r["group_name"]: dict(r) for r in rows}
