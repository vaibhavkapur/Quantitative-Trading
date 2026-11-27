import json
import sqlite3
from contextlib import closing
from pathlib import Path


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def publish(path, meta, report, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=20)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, meta TEXT NOT NULL, report TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS history (id TEXT REFERENCES runs(id), available_at TEXT,
                   body TEXT NOT NULL, PRIMARY KEY(id, available_at))""")
        if db.execute("SELECT 1 FROM runs WHERE id=?", (meta["run_id"],)).fetchone():
            return
        db.execute("INSERT INTO runs VALUES(?,?,?)", (meta["run_id"], packed(meta), packed(report)))
        db.executemany("INSERT INTO history VALUES(?,?,?)",
                       ((meta["run_id"], r["available_at"], packed(r)) for r in records))


def reader(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def page(db, run_id, after, as_of, limit, features_only):
    rows = db.execute("""SELECT body FROM history WHERE id=? AND available_at>? AND available_at<=?
                      ORDER BY available_at LIMIT ?""", (run_id, after, as_of, limit + 1)).fetchall()
    items = [json.loads(row[0]) for row in rows[:limit]]
    if features_only:
        keys = ("available_at", "event_at", "features", "feature_status", "series")
        items = [{k: item[k] for k in keys} for item in items]
    return {"items": items, "next_after": items[-1]["available_at"] if len(rows) > limit else None}
