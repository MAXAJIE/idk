"""Forward test on Binance USDT-M Futures TESTNET.

Per closed bar:  data -> evidence -> discussion -> Trader proposal ->
deterministic Risk gate (+ live spread / exchange filters, which can only
shrink or block) -> entry MARKET + exchange-side STOP & TAKE-PROFIT
(closePosition). Fail closed: if protective orders cannot be placed, the
position is flattened immediately.

The LLM explains persisted decisions only and never changes orders. Every step is appended to the hash-chained event
log so the read-only console can display it.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from app.agents import discussion as disc_mod
from app.agents import trader
from app.agents.narration import explain_records
from app.config import ExperimentConfig
from app.data import binance
from app.data.ingest import Bar
from app.data.point_in_time import PITView
from app.features import pipeline
from app.features.trend import ema
from app.genetic.schema import Chromosome
from app.live.testnet import ExchangeError, TestnetClient
from app.persistence.db import Store
from app.risk import gate

CONDITIONAL = ("STOP_MARKET", "TAKE_PROFIT_MARKET")


def resolve_experiment(store: Store, requested: str) -> str:
    experiments = store.experiments()
    if requested == "latest":
        eligible = [e for e in experiments if e["state"] == "COMPLETED"
                    and store.artifact(e["id"], "final", "ranking") is not None]
        if eligible:
            return eligible[0]["id"]
        raise SystemExit("No completed research experiment. Run: python -m app.cli run; then experiments.")
    if any(e["id"] == requested for e in experiments):
        return requested
    raise SystemExit(f"Unknown experiment '{requested}'. IDs are exact, not sequence numbers like 0001. "
                     "Run: python -m app.cli experiments (use the same --db).")


def pick_chromosome(store: Store, eid: str, allow_unqualified: bool = False) -> Chromosome:
    fin = store.artifact(eid, "final", "ranking")
    if not fin:
        exp = next((e for e in store.experiments() if e["id"] == eid), None)
        state = exp["state"] if exp else "NOT_FOUND"
        states = [e["payload"] for e in store.events(eid, limit=100000) if e["type"] == "state"]
        reason = states[-1].get("reason", "experiment has not completed") if states else "experiment not found"
        raise SystemExit(f"experiment {eid} has no final ranking: state={state}; {reason}. "
                         "Run python -m app.cli experiments, then run a new experiment if aborted.")
    rows = [r for r in fin["ranking"] if r["status"] == "QUALIFIED"] or (fin["ranking"] if allow_unqualified else [])
    if not rows and allow_unqualified:
        candidates = [c for c in store.artifacts(eid, "candidate") if c["status"] != "FAILED_RISK"]
        candidates.sort(key=lambda c: -c["oos"]["scorecard"]["net_pnl"])
        rows = candidates
    if not rows:
        raise SystemExit("No QUALIFIED chromosome; no orders allowed. Use explain --experiment <id> for LLM analysis; "
                         "research preview only: trade --dry-run --once --allow-unqualified --experiment <id>.")
    d = rows[0]["chromosome"]
    return Chromosome(genes=d["genes"], generation=d["generation"], parents=d["parents"], operator=d["operator"])


class LiveTrader:
    def __init__(self, cfg: ExperimentConfig, chromo: Chromosome, store: Store, client: TestnetClient,
                 state_path: str, dry_run: bool = False, kill_file: str = "KILL", source_eid: str | None = None):
        self.cfg, self.c, self.store, self.x = cfg, chromo, store, client
        self.sym, self.tf = cfg.data.symbol, cfg.data.timeframe_minutes
        self.dry, self.kill = dry_run, Path(kill_file)
        self.state_path = Path(state_path)
        self.st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "peak": None, "opened_ts": None, "halted": False, "last_bar": None}
        self.eid = f"live_{chromo.id}_{time.time_ns()}"
        store.create_experiment(self.eid, f"testnet:{self.sym}", cfg.model_dump() | {"chromosome": chromo.to_dict()})
        store.set_state(self.eid, "LIVE_TESTNET" if not dry_run else "LIVE_DRY_RUN")
        self.rules = client.rules(self.sym)
        self.emit("live_started", {"mode": "DRY_RUN" if dry_run else "BINANCE_TESTNET", "base": client.base,
                                   "symbol": self.sym, "chromosome": chromo.id, "source_experiment": source_eid,
                                   "rules": self.rules.__dict__})

    def emit(self, t: str, p: dict):
        self.store.append(self.eid, t, p)
        if t in ("decision", "order") and "proposal" in p:
            try:
                explain_records(self.store, self.eid, self.cfg, [p], "forward_test")
            except Exception as e:
                # Optional explanation cannot interrupt order/state management.
                self.store.append(self.eid, "narration", {"scope": "forward_test", "n": 0,
                                                        "ok": 0, "errors": [type(e).__name__]})

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.st))

    # ---------- data
    def bars(self, n: int = 300) -> list[Bar]:
        now_ms = int(time.time() * 1000)
        if self.cfg.live.signal_source == "testnet":
            rows = self.x.klines(self.sym, binance.interval(self.tf), n + 1)
            kl = binance.parse_klines(rows, self.tf, now_ms)
            fr = self.x.request("GET", "/fapi/v1/fundingRate", {"symbol": self.sym, "limit": 100})
            funding = sorted((int(f["fundingTime"]) // 1000, float(f["fundingRate"])) for f in fr)
            return binance.attach_funding(kl, funding)
        return binance.fetch(self.sym, self.tf, n, self.cfg.data.backend)

    # ---------- exchange helpers
    def _conditional(self, side: str, typ: str, trigger: float):
        px = self.rules.round_px(trigger)
        try:
            return self.x.order(symbol=self.sym, side=side, type=typ, stopPrice=px, closePosition="true",
                                workingType="MARK_PRICE", priceProtect="true")
        except ExchangeError as e:
            if "-4120" not in e.body:  # Binance moved conditional orders to the Algo API
                raise
            return self.x.request("POST", "/fapi/v1/algoOrder", {
                "algoType": "CONDITIONAL", "symbol": self.sym, "side": side, "type": typ, "triggerPrice": px,
                "closePosition": "true", "workingType": "MARK_PRICE", "priceProtect": "true"}, signed=True)

    def _cancel_all(self):
        self.x.cancel_all(self.sym)
        try:
            self.x.request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": self.sym}, signed=True)
        except ExchangeError:
            pass

    def flatten(self, reason: str):
        if self.dry:
            self.emit("flatten", {"reason": reason, "dry_run": True})
            return
        self._cancel_all()
        pos = self.x.position(self.sym)
        if pos["qty"]:
            r = self.x.order(symbol=self.sym, side="SELL" if pos["qty"] > 0 else "BUY", type="MARKET",
                             quantity=abs(pos["qty"]), reduceOnly="true")
            self.emit("flatten", {"reason": reason, "qty": pos["qty"], "order": r.get("orderId")})
        self.st["opened_ts"] = None
        self.save()

    # ---------- one closed bar
    def step(self) -> str:
        if self.kill.exists():
            self.flatten("kill switch file present")
            self.emit("halted", {"reason": "kill switch"})
            return "KILLED"
        if self.st["halted"]:
            return "HALTED"
        bars = self.bars()
        if len(bars) < pipeline.MIN_BARS:
            self.emit("defer", {"reason": f"only {len(bars)} closed bars"})
            return "DEFER"
        last = bars[-1]
        if self.st["last_bar"] == last.ts:
            return "SAME_BAR"
        step_s = self.tf * 60
        staleness = int((time.time() - (last.ts + step_s)) // step_s)

        equity = self.cfg.risk.initial_equity if self.dry else self.x.equity()
        pos = {"qty": 0.0} if self.dry else self.x.position(self.sym)
        self.st["peak"] = max(self.st["peak"] or equity, equity)
        dd = 1 - equity / self.st["peak"]
        if dd >= self.cfg.risk.hard_drawdown_frac:
            self.flatten(f"hard drawdown {dd:.4f}")
            self.st["halted"] = True
            self.save()
            self.emit("halted", {"reason": "hard_drawdown", "drawdown": round(dd, 6), "equity": equity})
            return "HALTED"

        self.st["last_bar"] = last.ts
        if pos["qty"]:
            held = (last.ts - (self.st["opened_ts"] or last.ts)) // step_s
            protect = [o for o in self.x.open_orders(self.sym) if o.get("type") in CONDITIONAL]
            self.emit("position", {"bar_ts": last.ts, "qty": pos["qty"], "entry": pos.get("entry"),
                                   "upnl": pos.get("upnl"), "held_bars": held, "equity": equity})
            if held >= self.c.g("stop_target", "max_hold_bars"):
                self.flatten("time exit")
            self.save()
            return "MANAGED"
        if self.st["opened_ts"]:  # position closed on exchange by stop / target
            self._cancel_all() if not self.dry else None
            self.emit("position_closed", {"bar_ts": last.ts, "equity": equity})
            self.st["opened_ts"] = None

        t = len(bars) - 1
        eb = pipeline.build(PITView(bars, t))
        closes = [b.close for b in bars]
        ef = ema(closes, self.c.g("trend", "ema_fast"))[-1]
        es = ema(closes, self.c.g("trend", "ema_slow"))[-1]
        disc = disc_mod.discuss(eb) if eb.valid else None
        prop = trader.propose(self.c, eb, disc, ef, es, equity)
        risk_cfg = self.cfg.risk.model_copy(update={"max_cost_to_reward": min(
            self.cfg.risk.max_cost_to_reward, self.c.g("cost", "max_cost_to_reward"))})
        rd = gate.evaluate(prop, gate.AccountState(equity, 0, self.st["halted"]),
                           gate.MarketState(last, eb.valid, staleness), risk_cfg, self.cfg.costs)
        rec = {"bar_ts": last.ts, "equity": equity, "proposal": prop.model_dump(), "risk": rd.model_dump(),
               "discussion": disc.model_dump() if disc else None}
        if rd.outcome not in ("APPROVE", "RESIZE"):
            self.emit("decision", rec)
            self.save()
            return rd.outcome

        # live-only checks (may only block/shrink)
        bid, ask = self.x.book(self.sym)
        spread_bps = (ask - bid) / ((ask + bid) / 2) * 1e4
        qty = self.rules.floor_qty(rd.approved_qty)
        ref = ask if prop.side == "long" else bid
        live = {"spread_bps": round(spread_bps, 3), "qty_after_step": qty, "notional": round(qty * ref, 4)}
        block = (spread_bps > self.cfg.risk.max_spread_bps and "live spread too wide") or \
                (qty <= 0 and "below exchange min qty") or \
                (qty * ref < max(self.rules.min_notional, gate.MIN_NOTIONAL) and "below exchange min notional")
        if block:
            self.emit("decision", rec | {"live": live, "blocked": block})
            self.save()
            return "DEFER"
        if self.dry:
            self.emit("decision", rec | {"live": live, "dry_run_order": True})
            self.save()
            return "DRY_APPROVED"

        side, exit_side = ("BUY", "SELL") if prop.side == "long" else ("SELL", "BUY")
        cid = "cg" + hashlib.sha256(prop.decision_id.encode()).hexdigest()[:20]
        lev = max(1, math.ceil(self.cfg.risk.max_leverage))
        self.x.set_leverage(self.sym, lev)
        entry = self.x.order(symbol=self.sym, side=side, type="MARKET", quantity=qty, newClientOrderId=cid,
                             newOrderRespType="RESULT")
        fill = float(entry.get("avgPrice") or ref) or ref
        shift = fill - prop.ref_price  # keep stop/target distances relative to the fill, as in the simulator
        stop, target = prop.stop + shift, prop.target + shift
        self.st["opened_ts"] = last.ts
        self.save()
        try:
            so = self._conditional(exit_side, "STOP_MARKET", stop)
            to = self._conditional(exit_side, "TAKE_PROFIT_MARKET", target)
        except ExchangeError as e:
            self.emit("protective_order_failed", {"error": str(e)})
            self.flatten("protective order failed")
            return "FLATTENED"
        self.emit("order", rec | {"live": live, "entry": {"id": entry.get("orderId"), "avg": fill, "qty": qty},
                                  "stop": {"px": stop, "id": so.get("orderId") or so.get("algoId")},
                                  "target": {"px": target, "id": to.get("orderId") or to.get("algoId")},
                                  "leverage": lev})
        return "ORDERED"

    def loop(self, max_steps: int | None = None):
        n = 0
        while max_steps is None or n < max_steps:
            try:
                out = self.step()
            except ExchangeError as e:
                self.emit("exchange_error", {"error": str(e)})
                out = "ERROR"
            print(time.strftime("%H:%M:%S"), out, flush=True)
            if out in ("KILLED", "HALTED"):
                return out
            n += 1
            if max_steps is not None and n >= max_steps:
                break
            step_s = self.tf * 60
            time.sleep(step_s - time.time() % step_s + self.cfg.live.close_grace_s)
        return "DONE"
