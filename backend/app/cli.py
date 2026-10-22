"""Operator CLI (local terminal only — not reachable from the browser).

  python -m app.cli run                       # fetches BTCUSDT from Binance automatically
  python -m app.cli run   --config configs/default.json   # offline synthetic demo
  python -m app.cli fetch --symbol BTCUSDT --tf 60 --n 3000 --out data/btc.csv [--backend ccxt]
  python -m app.cli trade --config configs/binance.json --experiment <exp_id> [--dry-run] [--once]
  python -m app.cli serve
  python -m app.cli llm-test                  # checks LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.env import load_env

load_env()  # backend/.env or backend/app/.env, before anything reads os.environ

from app.config import ExperimentConfig, load_config  # noqa: E402
from app.persistence.db import Store  # noqa: E402

BACKEND = Path(os.path.abspath(__file__)).parents[1]
DB = os.environ.get("CGTRS_DB", str(BACKEND / "data/cgtrs.sqlite"))
os.environ.setdefault("CGTRS_SNAPSHOT_DIR", str(BACKEND / "data/snapshots"))


def backend_path(value):
    path = Path(value)
    return str(path if path.is_absolute() else BACKEND / path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--config", default="configs/binance.json"); r.add_argument("--db", default=DB)
    f = sub.add_parser("fetch")
    f.add_argument("--symbol", default="BTCUSDT"); f.add_argument("--tf", type=int, default=60)
    f.add_argument("--n", type=int, default=3000); f.add_argument("--out", required=True)
    f.add_argument("--backend", choices=["rest", "ccxt"], default="rest")
    t = sub.add_parser("trade")
    t.add_argument("--config"); t.add_argument("--db", default=DB)
    t.add_argument("--experiment", required=True); t.add_argument("--dry-run", action="store_true")
    t.add_argument("--once", action="store_true"); t.add_argument("--allow-unqualified", action="store_true")
    t.add_argument("--state"); t.add_argument("--kill-file", default="KILL")
    s = sub.add_parser("serve"); s.add_argument("--db", default=DB); s.add_argument("--port", type=int, default=8000)
    sub.add_parser("llm-test")
    x = sub.add_parser("experiments"); x.add_argument("--db", default=DB)
    x = sub.add_parser("explain"); x.add_argument("--db", default=DB)
    x.add_argument("--experiment", default="latest")
    a = ap.parse_args()
    if hasattr(a, "db"):
        a.db = backend_path(a.db)
        print(f"database: {a.db}", flush=True)
    if a.cmd == "run":
        from app.orchestration.runner import run
        cfg = load_config(backend_path(a.config))
        if cfg.data.source == "binance":
            print(f"fetching {cfg.data.n_bars} x {cfg.data.timeframe_minutes}m {cfg.data.symbol} from Binance ...", flush=True)
        eid = run(cfg, Store(a.db))
        st = next((e["state"] for e in Store(a.db, readonly=True).experiments() if e["id"] == eid), "?")
        print(eid, st)
        print(f'Inspect: python -m app.cli experiments --db "{a.db}"')
        if st != "COMPLETED":
            raise SystemExit(1)
    elif a.cmd == "fetch":
        from app.data import binance
        bars = binance.fetch(a.symbol, a.tf, a.n, a.backend)
        out = backend_path(a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        binance.write_csv(bars, out)
        print(f"{len(bars)} closed bars -> {out}")
    elif a.cmd == "trade":
        from app.live.runner import LiveTrader, pick_chromosome, resolve_experiment
        from app.live.testnet import TestnetClient
        if a.allow_unqualified and not a.dry_run:
            ap.error("--allow-unqualified requires --dry-run; unqualified strategies may not place orders")
        store = Store(a.db)
        eid = resolve_experiment(store, a.experiment)
        exp = next(e for e in store.experiments() if e["id"] == eid)
        cfg = load_config(backend_path(a.config)) if a.config else ExperimentConfig.model_validate(exp["config"])
        if cfg.data.source != "binance":
            ap.error("trade requires a Binance research experiment, not synthetic/CSV data")
        state = backend_path(a.state or ("data/dry_state.json" if a.dry_run else "data/live_state.json"))
        print(f"source experiment: {eid}; mode: {'DRY_RUN (no orders)' if a.dry_run else 'BINANCE_TESTNET'}", flush=True)
        lt = LiveTrader(cfg, pick_chromosome(store, eid, a.allow_unqualified), store,
                        TestnetClient(cfg.live.base_url), state, a.dry_run, backend_path(a.kill_file), eid)
        print("live experiment", lt.eid)
        print(lt.loop(1 if a.once else None))
    elif a.cmd == "llm-test":
        from app.agents.provider import LLMError, OpenAICompatibleProvider, llm_env
        env = llm_env()
        print(f"LLM_BASE_URL={env['base_url'] or '(missing)'}  LLM_MODEL={env['model'] or '(missing)'}  "
              f"LLM_API_KEY={'set (' + str(len(env['api_key'])) + ' chars)' if env['api_key'] else '(missing)'}")
        try:
            p = OpenAICompatibleProvider(30.0)
            print("reply:", p.chat("Reply with exactly: OK", "ping", max_tokens=128))
        except LLMError as e:
            raise SystemExit(f"LLM FAILED: {e}")
    elif a.cmd == "experiments":
        store = Store(a.db)
        xs = store.experiments()
        if not xs:
            print("No experiments in this database. Run python -m app.cli run first.")
        for e in xs:
            fin = store.artifact(e["id"], "final", "ranking")
            rows = fin["ranking"] if fin else []
            print(f"{e['id']}  {e['state']}  ranking={'yes' if fin is not None else 'no'} "
                  f"qualified={sum(r['status'] == 'QUALIFIED' for r in rows)}  name={e['name']}")
            states = [v["payload"] for v in store.events(e["id"], limit=100000) if v["type"] == "state"]
            if states and states[-1].get("reason"):
                print("  reason:", states[-1]["reason"])
    elif a.cmd == "explain":
        from app.live.runner import resolve_experiment
        from app.agents.narration import explain_experiment
        store = Store(a.db)
        eid = resolve_experiment(store, a.experiment)
        exp = next(e for e in store.experiments() if e["id"] == eid)
        cfg = ExperimentConfig.model_validate(exp["config"])
        cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"provider": "openai_compatible", "narrate": "elite_only"})})
        rows = explain_experiment(store, eid, cfg)
        for n in rows:
            print(n["decision_id"], n["text"] or n["error"])
        if not rows or any(n["error"] for n in rows):
            raise SystemExit("LLM explanation incomplete; run python -m app.cli llm-test for diagnostics.")
    else:
        import uvicorn
        from app.api.read_only import create_app
        Store(a.db)
        print(f"Console: http://127.0.0.1:{a.port}/  (tabs: Training / Live testnet / Debate & indicators)", flush=True)
        print("Do not use the old frontend dev server on :5173 - it is the legacy spectator UI.", flush=True)
        uvicorn.run(create_app(a.db), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
