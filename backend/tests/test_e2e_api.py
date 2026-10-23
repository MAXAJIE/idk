import sqlite3

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient

from app.api.read_only import create_app
from app.orchestration.runner import run
from app.persistence.db import Store


@pytest.fixture(scope="module")
def experiment(tmp_path_factory, request):
    import json
    from pathlib import Path
    from app.config import ExperimentConfig
    root = Path(__file__).parents[1]
    d = json.loads((root / "configs/default.json").read_text())
    d["data"]["n_bars"] = 1400
    d["wfa"] |= {"n_folds": 2, "is_bars": 500, "oos_bars": 200, "holdout_bars": 150}
    d["ga"] |= {"population": 6, "generations": 2, "elite_size": 2}
    d["gates"]["min_trades"] = 3
    cfg = ExperimentConfig.model_validate(d)
    db = tmp_path_factory.mktemp("db") / "t.sqlite"
    store = Store(db)
    eid = run(cfg, store)
    return cfg, db, store, eid


def test_run_completes_with_audit_trail(experiment):
    _, _, store, eid = experiment
    types = [e["type"] for e in store.events(eid, 0, 5000)]
    assert store.experiments()[0]["state"] == "COMPLETED"
    assert types.count("generation") == 4 and types.count("mutation") == 2
    assert "candidates_frozen" in types and "judge_verdict" in types and "final_ranking" in types
    assert store.verify_chain(eid)


def test_oos_never_inside_is(experiment):
    _, _, store, eid = experiment
    for c in store.artifacts(eid, "candidate"):
        assert c["oos"]["start"] >= c["is"]["end"]


def test_replay_is_deterministic(experiment, tmp_path):
    cfg, _, store, eid = experiment
    s2 = Store(tmp_path / "r.sqlite")
    eid2 = run(cfg, s2)
    a, b = store.artifact(eid, "final", "ranking"), s2.artifact(eid2, "final", "ranking")
    assert [r["id"] for r in a["ranking"]] == [r["id"] for r in b["ranking"]]
    assert store.artifact(eid, "dataset", "snapshot")["checksum"] == s2.artifact(eid2, "dataset", "snapshot")["checksum"]


def test_event_log_append_only(experiment):
    _, db, _, _ = experiment
    con = sqlite3.connect(db)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE events SET type='x'")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("DELETE FROM events")


def test_api_exposes_only_reads(experiment):
    _, db, _, eid = experiment
    app = create_app(str(db))
    for r in app.routes:
        if isinstance(r, APIRoute):
            assert r.methods <= {"GET", "HEAD"}, r.path
        else:
            assert isinstance(r, APIWebSocketRoute) or r.path.startswith("/"), r
    c = TestClient(app)
    assert c.get("/api/experiments").status_code == 200
    assert c.get(f"/api/experiments/{eid}").json()["chain_ok"] is True
    assert c.get(f"/api/experiments/{eid}/final").status_code == 200
    for m in ("post", "put", "patch", "delete"):
        assert getattr(c, m)(f"/api/experiments/{eid}").status_code == 405
    assert c.post("/api/experiments/run").status_code in (404, 405)


def test_api_db_handle_is_read_only(experiment):
    _, db, _, eid = experiment
    ro = Store(str(db), readonly=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.append(eid, "x", {})


def test_websocket_streams_events(experiment):
    _, db, _, eid = experiment
    c = TestClient(create_app(str(db)))
    with c.websocket_connect(f"/ws/{eid}") as ws:
        first = ws.receive_json()
        assert first["type"] == "experiment_created"
