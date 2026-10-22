"""Spectator API. GET + WebSocket only; there is no route that can start,
stop, tune or trade. The DB is opened read-only."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

from app.api import analysis, live_market
from app.env import load_env
from app.persistence.db import Store

load_env()

DB = os.environ.get("CGTRS_DB", "data/cgtrs.sqlite")
# built console (npm run build) is served at / when present
CONSOLE = Path(__file__).with_name("console.html")  # built-in console (default UI)
USE_DIST = os.environ.get("CGTRS_UI", "builtin") == "dist"
UI_DIST = Path(os.environ.get("CGTRS_UI_DIST", Path(__file__).resolve().parents[3] / "frontend" / "dist"))

LANDING = """<!doctype html><meta charset=utf-8><title>CGTRS API</title>
<body style="font:14px system-ui;max-width:640px;margin:40px auto;line-height:1.5">
<h2>CGTRS read-only API is running</h2>
<p>This port serves data only. To see the console either:</p>
<ol><li><code>cd frontend &amp;&amp; npm ci &amp;&amp; npm run dev</code> then open
<a href="http://127.0.0.1:5173">http://127.0.0.1:5173</a></li>
<li>or <code>npm run build</code> once, restart <code>serve</code>, and reload this page.</li></ol>
<p>Experiments in DB: <b>{n}</b>{hint}</p>
<p>Endpoints: <a href="/api/health">/api/health</a> · <a href="/api/experiments">/api/experiments</a></p>
</body>"""


def create_app(db_path: str = DB) -> FastAPI:
    app = FastAPI(title="CGTRS Spectator (read-only)", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET"], allow_headers=[])
    store = Store(db_path, readonly=True)

    def need(x):
        if x is None:
            raise HTTPException(404)
        return x

    @app.get("/api/health")
    def health():
        return {"ok": True, "db": str(Path(db_path).resolve()), "experiments": len(store.experiments()),
                "ui_built": (UI_DIST / "index.html").exists(), "ui": "dist" if USE_DIST else "builtin"}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/api/experiments")
    def experiments():
        return [{k: e[k] for k in ("id", "name", "created", "state")} for e in store.experiments()]

    @app.get("/api/experiments/{eid}")
    def experiment(eid: str):
        e = next((x for x in store.experiments() if x["id"] == eid), None)
        need(e)
        return e | {"dataset": store.artifact(eid, "dataset", "snapshot"), "folds": store.artifact(eid, "folds", "all"),
                    "chain_ok": store.verify_chain(eid)}

    @app.get("/api/experiments/{eid}/events")
    def events(eid: str, after: int = 0, limit: int = 500):
        return store.events(eid, after, min(limit, 2000))

    @app.get("/api/experiments/{eid}/bars")
    def bars(eid: str):
        return need(store.artifact(eid, "market", "bars"))

    @app.get("/api/experiments/{eid}/chromosomes")
    def chromosomes(eid: str):
        return store.artifacts(eid, "chromosome")

    @app.get("/api/experiments/{eid}/candidates")
    def candidates(eid: str):
        return store.artifacts(eid, "candidate")

    @app.get("/api/experiments/{eid}/final")
    def final(eid: str):
        return {"final": store.artifact(eid, "final", "ranking"), "holdout": store.artifacts(eid, "holdout"),
                "narration": store.artifacts(eid, "narration")}

    @app.get("/api/experiments/{eid}/training")
    def training(eid: str):
        """Per-generation curve + current state; readable while the run is still going."""
        e = need(next((x for x in store.experiments() if x["id"] == eid), None))
        gens, states, verdicts, after = [], [], [], 0
        while True:
            rows = store.events(eid, after, 2000)
            if not rows:
                break
            after = rows[-1]["seq"]
            for r in rows:
                if r["type"] == "generation":
                    gens.append(r["payload"] | {"ts": r["ts"]})
                elif r["type"] == "state":
                    states.append(r["payload"] | {"ts": r["ts"]})
                elif r["type"] == "judge_verdict":
                    verdicts.append(r["payload"])
        cfg = e["config"].get("ga", {})
        return {"id": eid, "state": e["state"], "generations": gens, "states": states, "verdicts": verdicts,
                "plan": {"generations": cfg.get("generations"), "population": cfg.get("population")}}

    LIVE_TYPES = {"live_started", "decision", "order", "position", "position_closed", "flatten", "halted",
                  "defer", "exchange_error", "protective_order_failed"}

    @app.get("/api/experiments/{eid}/live")
    def live_events(eid: str, limit: int = 300):
        out, after = [], 0
        while True:
            rows = store.events(eid, after, 2000)
            if not rows:
                break
            after = rows[-1]["seq"]
            out += [r for r in rows if r["type"] in LIVE_TYPES]
        return out[-min(limit, 2000):]

    @app.get("/api/analysis/{symbol}")
    def live_analysis(symbol: str, tf: int = 60, limit: int = 300):
        try:
            return analysis.analyse(symbol, tf, limit)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except Exception as e:  # noqa: BLE001 upstream/network
            raise HTTPException(502, f"Binance unreachable: {type(e).__name__}: {str(e)[:160]}") from None

    @app.get("/api/llm")
    def llm_status():
        from app.agents.provider import llm_status
        return llm_status()  # configuration is not connectivity; never returns the key

    @app.get("/api/live/{symbol}")
    def live(symbol: str, tf: int = 60, limit: int = 120):
        try:
            return live_market.snapshot(symbol, tf, limit)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except Exception as e:  # noqa: BLE001 upstream/network
            raise HTTPException(502, f"Binance unreachable: {type(e).__name__}: {str(e)[:160]}") from None

    @app.websocket("/ws/live/{symbol}")
    async def live_stream(ws: WebSocket, symbol: str, tf: int = 60):
        await ws.accept()
        try:
            while True:
                try:
                    d = await asyncio.to_thread(live_market.snapshot, symbol, tf, 120)
                    await ws.send_json(d)
                except (ValueError, OSError) as e:
                    await ws.send_json({"error": f"{type(e).__name__}: {str(e)[:160]}"})
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            return

    @app.websocket("/ws/{eid}")
    async def stream(ws: WebSocket, eid: str, after: int = 0):
        await ws.accept()
        try:
            while True:
                rows = store.events(eid, after, 200)
                for r in rows:
                    await ws.send_json(r)
                    after = r["seq"]
                await asyncio.sleep(1.0)  # inbound messages are never read
        except WebSocketDisconnect:
            return

    @app.get("/{path:path}", include_in_schema=False)
    def ui(path: str = ""):
        if path.startswith(("api/", "ws/")):
            raise HTTPException(404)
        if path in ("", "console", "index.html") and CONSOLE.exists():
            return FileResponse(CONSOLE, media_type="text/html")  # new console always wins at /
        if not USE_DIST or not (UI_DIST / "index.html").exists():
            if CONSOLE.exists():
                return FileResponse(CONSOLE, media_type="text/html")
        if (UI_DIST / "index.html").exists():
            f = (UI_DIST / path).resolve()
            if path and f.is_file() and UI_DIST.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(UI_DIST / "index.html")
        n = len(store.experiments())
        hint = "" if n else " — run <code>python -m app.cli run --config configs/default.json</code> first"
        return HTMLResponse(LANDING.format(n=n, hint=hint))

    return app


app = create_app() if os.path.exists(DB) else None
