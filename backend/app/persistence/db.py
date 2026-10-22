"""Local SQLite store: hash-chained append-only event log + immutable artifacts."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, name TEXT, created REAL, state TEXT, config TEXT);
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, ts REAL,
  type TEXT, payload TEXT, prev_hash TEXT, hash TEXT);
CREATE INDEX IF NOT EXISTS ev_exp ON events(experiment_id, seq);
CREATE TABLE IF NOT EXISTS artifacts(experiment_id TEXT, kind TEXT, key TEXT, body TEXT,
  PRIMARY KEY(experiment_id, kind, key));
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER IF NOT EXISTS art_no_update BEFORE UPDATE ON artifacts BEGIN SELECT RAISE(ABORT,'immutable'); END;
CREATE TRIGGER IF NOT EXISTS art_no_delete BEFORE DELETE ON artifacts BEGIN SELECT RAISE(ABORT,'immutable'); END;
"""


def _hash(prev: str, etype: str, payload: str) -> str:
    return hashlib.sha256(f"{prev}|{etype}|{payload}".encode()).hexdigest()


class Store:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = str(path)
        self._lock = threading.Lock()
        if readonly:
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.executescript(SCHEMA)
        self.conn.row_factory = sqlite3.Row

    # ---- writes (engine process only) ----
    def create_experiment(self, eid: str, name: str, config: dict):
        with self._lock, self.conn:
            self.conn.execute("INSERT INTO experiments VALUES(?,?,?,?,?)",
                              (eid, name, time.time(), "CREATED", json.dumps(config)))

    def set_state(self, eid: str, state: str):
        with self._lock, self.conn:
            self.conn.execute("UPDATE experiments SET state=? WHERE id=?", (state, eid))

    def append(self, eid: str, etype: str, payload: dict) -> int:
        body = json.dumps(payload, sort_keys=True, default=str)
        with self._lock, self.conn:
            row = self.conn.execute("SELECT hash FROM events WHERE experiment_id=? ORDER BY seq DESC LIMIT 1",
                                    (eid,)).fetchone()
            prev = row["hash"] if row else "genesis"
            cur = self.conn.execute("INSERT INTO events(experiment_id,ts,type,payload,prev_hash,hash) VALUES(?,?,?,?,?,?)",
                                    (eid, time.time(), etype, body, prev, _hash(prev, etype, body)))
            return cur.lastrowid

    def put(self, eid: str, kind: str, key: str, body: dict):
        """First write wins; artifacts are immutable."""
        with self._lock, self.conn:
            self.conn.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                              (eid, kind, key, json.dumps(body, default=str)))

    # ---- reads (serialised: one sqlite connection is shared across server threads) ----
    def _all(self, sql: str, args: tuple) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, args).fetchall()

    def experiments(self) -> list[dict]:
        return [dict(r) | {"config": json.loads(r["config"])} for r in
                self._all("SELECT * FROM experiments ORDER BY created DESC", ())]

    def events(self, eid: str, after: int = 0, limit: int = 500) -> list[dict]:
        rows = self._all("SELECT seq,ts,type,payload,hash FROM events WHERE experiment_id=? AND seq>? "
                         "ORDER BY seq LIMIT ?", (eid, after, limit))
        return [{"seq": r["seq"], "ts": r["ts"], "type": r["type"], "payload": json.loads(r["payload"]),
                 "hash": r["hash"]} for r in rows]

    def artifacts(self, eid: str, kind: str) -> list[dict]:
        return [json.loads(r["body"]) for r in
                self._all("SELECT body FROM artifacts WHERE experiment_id=? AND kind=? ORDER BY key", (eid, kind))]

    def artifact(self, eid: str, kind: str, key: str) -> dict | None:
        r = self._all("SELECT body FROM artifacts WHERE experiment_id=? AND kind=? AND key=?", (eid, kind, key))
        return json.loads(r[0]["body"]) if r else None

    def verify_chain(self, eid: str) -> bool:
        prev = "genesis"
        for r in self._all("SELECT type,payload,prev_hash,hash FROM events WHERE experiment_id=? ORDER BY seq", (eid,)):
            if r["prev_hash"] != prev or _hash(prev, r["type"], r["payload"]) != r["hash"]:
                return False
            prev = r["hash"]
        return True
