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
from app.learning.reflection import ReflectionMemory
from app.learning.online_evolution import evaluate_rolling_challenger

CONDITIONAL = ("STOP_MARKET", "TAKE_PROFIT_MARKET")


def resolve_experiment(store: Store, requested: str) -> str:
    experiments = store.experiments()
    if requested == "latest":
        eligible = [e for e in experiments if e["state"] == "COMPLETED"
                    and store.artifact(e["id"], "final", "ranking") is not None]
        if eligible:
            return eligible[0]["id"]
        raise SystemExit("No completed research experiment. Run: python -m app.cli run; then experiments.")
    if requested in ("best", "auto"):
        try:
            _, best_eid, _ = find_best_chromosome(store)
            return best_eid
        except SystemExit:
            raise SystemExit("No completed qualified training chromosome found. Run: python -m app.cli train --config configs/binance.json --auto.")
    if any(e["id"] == requested for e in experiments):
        return requested
    raise SystemExit(f"Unknown experiment '{requested}'. IDs are exact, not sequence numbers like 0001. "
                     "Run: python -m app.cli experiments (use the same --db).")


def pick_chromosome(store: Store, eid: str, allow_unqualified: bool = False) -> Chromosome:
    fin = store.artifact(eid, "final", "ranking")
    arena_res = store.artifact(eid, "arena", "result")

    if not fin and not arena_res:
        exp = next((e for e in store.experiments() if e["id"] == eid), None)
        state = exp["state"] if exp else "NOT_FOUND"
        states = [e["payload"] for e in store.events(eid, limit=100000) if e["type"] == "state"]
        reason = states[-1].get("reason", "experiment has not completed") if states else "experiment not found"
        raise SystemExit(f"experiment {eid} has no final ranking: state={state}; {reason}. "
                         "Run python -m app.cli experiments, then run a new experiment if aborted.")

    # If arena experiment result exists
    if arena_res and isinstance(arena_res, dict):
        champ = arena_res.get("champion")
        if champ and isinstance(champ, dict) and "genes" in champ:
            return Chromosome(genes=champ["genes"], generation=champ.get("generation", 0),
                              parents=champ.get("parents", []), operator=champ.get("operator", "elite"))
        for comb in (arena_res.get("combatants") or []):
            if isinstance(comb, dict) and comb.get("chromosome"):
                if comb.get("qualified") or allow_unqualified:
                    cd = comb["chromosome"]
                    if isinstance(cd, dict) and "genes" in cd:
                        return Chromosome(genes=cd["genes"], generation=cd.get("generation", 0),
                                          parents=cd.get("parents", []), operator=cd.get("operator", "elite"))

    rows = [r for r in fin["ranking"] if r["status"] == "QUALIFIED"] or (fin["ranking"] if allow_unqualified else []) if fin else []
    if not rows and allow_unqualified:
        candidates = [c for c in store.artifacts(eid, "candidate") if c["status"] != "FAILED_RISK"]
        candidates.sort(key=lambda c: -c["oos"]["scorecard"]["net_pnl"])
        rows = candidates
    if not rows:
        raise SystemExit("No QUALIFIED chromosome; no orders allowed. Use explain --experiment <id> for LLM analysis; "
                         "research preview only: trade --dry-run --once --allow-unqualified --experiment <id>.")
    d = rows[0]["chromosome"]
    return Chromosome(genes=d["genes"], generation=d["generation"], parents=d["parents"], operator=d["operator"])


def find_best_chromosome(store: Store, requested: str | None = None, allow_unqualified: bool = False) -> tuple[Chromosome, str, dict]:
    """Automatically find and return (Chromosome, source_eid, metadata) of the best qualified strategy.

    Scans completed Arena experiments and WFA research experiments, ranking them by
    anti-overfitting validation score (OOS PnL weighted by consistency).
    """
    experiments = store.experiments()
    if not experiments:
        raise SystemExit("No experiments in database. Run python -m app.cli train --config configs/binance.json --auto first.")

    if requested and requested not in ("best", "latest", "auto"):
        c = pick_chromosome(store, requested, allow_unqualified)
        return c, requested, {"type": "explicit_experiment", "source": requested}

    candidates = []
    completed = [e for e in experiments if e.get("state") in ("COMPLETED", "LIVE_TESTNET", "LIVE_DRY_RUN")]
    search_pool = completed if completed else experiments

    for exp in search_pool:
        eid = exp["id"]
        created = exp.get("created") or 0

        # 1. Check arena result
        arena_res = store.artifact(eid, "arena", "result")
        if arena_res and isinstance(arena_res, dict):
            champ = arena_res.get("champion")
            if champ and isinstance(champ, dict) and "genes" in champ:
                oos_pnl = float(arena_res.get("champion_oos_pnl") or 0.0)
                cons = float(arena_res.get("champion_consistency") or 0.0)
                score = oos_pnl * (0.5 + 0.5 * cons)
                ch = Chromosome(genes=champ["genes"], generation=champ.get("generation", 0),
                                parents=champ.get("parents", []), operator=champ.get("operator", "elite"))
                candidates.append({
                    "chromosome": ch, "source_eid": eid, "qualified": True,
                    "score": score, "created": created,
                    "meta": {"type": "arena_champion", "oos_pnl": oos_pnl, "consistency": cons, "source": eid}
                })
            for comb in (arena_res.get("combatants") or []):
                if isinstance(comb, dict) and comb.get("chromosome"):
                    cd = comb["chromosome"]
                    if isinstance(cd, dict) and "genes" in cd:
                        is_qual = bool(comb.get("qualified"))
                        oos_pnl = float(comb.get("oos_pnl") or 0.0)
                        cons = float(comb.get("consistency") or 0.0)
                        score = oos_pnl * (0.5 + 0.5 * cons)
                        if is_qual or allow_unqualified:
                            ch = Chromosome(genes=cd["genes"], generation=cd.get("generation", 0),
                                            parents=cd.get("parents", []), operator=cd.get("operator", "elite"))
                            candidates.append({
                                "chromosome": ch, "source_eid": eid, "qualified": is_qual,
                                "score": score, "created": created,
                                "meta": {"type": "arena_combatant", "oos_pnl": oos_pnl, "consistency": cons, "source": eid}
                            })

        # 2. Check WFA final ranking
        fin = store.artifact(eid, "final", "ranking")
        if fin and isinstance(fin, dict) and "ranking" in fin:
            for r in fin["ranking"]:
                if isinstance(r, dict) and r.get("chromosome"):
                    cd = r["chromosome"]
                    if isinstance(cd, dict) and "genes" in cd:
                        is_qual = r.get("status") == "QUALIFIED"
                        scorecard = r.get("oos", {}).get("scorecard", {})
                        oos_pnl = float(scorecard.get("net_pnl") or 0.0)
                        sharpe = float(scorecard.get("sharpe") or 0.0)
                        score = oos_pnl if oos_pnl != 0 else sharpe
                        if is_qual or allow_unqualified:
                            ch = Chromosome(genes=cd["genes"], generation=cd.get("generation", 0),
                                            parents=cd.get("parents", []), operator=cd.get("operator", "elite"))
                            candidates.append({
                                "chromosome": ch, "source_eid": eid, "qualified": is_qual,
                                "score": score, "created": created,
                                "meta": {"type": "wfa_ranking", "oos_pnl": oos_pnl, "sharpe": sharpe, "source": eid}
                            })

    if requested == "latest" and candidates:
        candidates.sort(key=lambda c: (c["created"], 1 if c["qualified"] else 0, c["score"]), reverse=True)
        top = candidates[0]
        return top["chromosome"], top["source_eid"], top["meta"]

    qualified = [c for c in candidates if c["qualified"]]
    pool = qualified if qualified else (candidates if allow_unqualified else [])

    if not pool:
        if allow_unqualified:
            for exp in search_pool:
                for cand in store.artifacts(exp["id"], "candidate"):
                    if cand.get("status") != "FAILED_RISK" and cand.get("chromosome"):
                        cd = cand["chromosome"]
                        ch = Chromosome(genes=cd["genes"], generation=cd.get("generation", 0),
                                        parents=cd.get("parents", []), operator=cd.get("operator", "elite"))
                        return ch, exp["id"], {"type": "candidate_fallback", "source": exp["id"]}
        raise SystemExit("No QUALIFIED chromosome found in any experiment. "
                         "Run python -m app.cli train --config configs/binance.json --auto "
                         "to evolve an anti-overfitting champion, or specify --allow-unqualified for research preview.")

    pool.sort(key=lambda c: (1 if c["qualified"] else 0, c["score"], c["created"]), reverse=True)
    best = pool[0]
    return best["chromosome"], best["source_eid"], best["meta"]


class LiveTrader:
    def __init__(self, cfg: ExperimentConfig, chromo: Chromosome, store: Store, client: TestnetClient,
                 state_path: str, dry_run: bool = False, kill_file: str = "KILL", source_eid: str | None = None):
        self.cfg, self.c, self.store, self.x = cfg, chromo, store, client
        self.sym, self.tf = cfg.data.symbol, cfg.data.timeframe_minutes
        self.dry, self.kill = dry_run, Path(kill_file)
        self.state_path = Path(state_path)
        self.st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "peak": None, "opened_ts": None, "halted": False, "last_bar": None,
            "bars_since_evolution": 0, "closed_trade_count": 0, "consecutive_losses": 0,
            "last_closed_trade_id": None}
        # Restore reflection memory independently from the core trader state.
        self.memory_path = self.state_path.with_suffix(".memory.json")
        self.memory = ReflectionMemory.load(self.memory_path) if cfg.live.enable_reflection_memory else ReflectionMemory()
        self._pending_signature = None
        self._pending_entry_t = None
        # Backward compatibility with state files created before learning support.
        self.st.setdefault("bars_since_evolution", 0)
        self.st.setdefault("closed_trade_count", 0)
        self.st.setdefault("consecutive_losses", 0)
        self.st.setdefault("last_closed_trade_id", None)
        self.eid = f"live_{chromo.id}_{time.time_ns()}"
        store.create_experiment(self.eid, f"testnet:{self.sym}", cfg.model_dump() | {"chromosome": chromo.to_dict()})
        store.set_state(self.eid, "LIVE_TESTNET" if not dry_run else "LIVE_DRY_RUN")
        self.rules = client.rules(self.sym)
        self.emit("live_started", {"mode": "DRY_RUN" if dry_run else "BINANCE_TESTNET", "base": client.base,
                                   "symbol": self.sym, "chromosome": chromo.id, "source_experiment": source_eid,
                                   "rules": self.rules.__dict__})
        if not self.dry:
            self._initial_reconcile()

    def _initial_reconcile(self):
        """Immediately reconcile exchange position and account balance on startup."""
        if self.dry:
            return
        try:
            equity = self.x.equity()
            pos = self.x.position(self.sym)
            protect = [o for o in self.x.open_orders(self.sym) if o.get("type") in CONDITIONAL]
        except Exception as e:
            self.emit("exchange_error", {"error": f"Initial sync failed: {e}"})
            return

        self.st["peak"] = max(self.st.get("peak") or equity, equity)
        now_ts = int(time.time())

        qty = float(pos.get("qty") or 0.0)
        if qty != 0:
            entry = float(pos.get("entry") or 0.0)
            upnl = float(pos.get("upnl") or 0.0)
            side = "LONG" if qty > 0 else "SHORT"
            if self.st.get("opened_ts") is None:
                self.st["opened_ts"] = now_ts
            self.st["position_anchor"] = {"entry": entry, "realized_pnl": None}
            self.st["position_qty"] = qty
            self.st["position_entry"] = entry
            self.save()

            rec = {
                "bar_ts": now_ts,
                "symbol": self.sym,
                "qty": qty,
                "side": side,
                "entry": entry,
                "upnl": upnl,
                "equity": equity,
                "protective_orders": len(protect),
                "reconciled": True,
            }
            self.emit("position_reconnected", rec)
            self.emit("position", rec | {"held_bars": 0})
            print(f"[{time.strftime('%H:%M:%S')}] [RECONNECT] Active {side} position restored: {qty:+.4f} {self.sym} @ {entry:.2f} (uPnL: {upnl:+.2f} USDT, equity: , protect: {len(protect)} orders)", flush=True)
            if len(protect) == 0:
                print(f"[{time.strftime('%H:%M:%S')}] [WARNING] No exchange-side protective stop/take-profit orders found for open position!", flush=True)
        else:
            self.emit("account_sync", {
                "bar_ts": now_ts,
                "equity": equity,
                "position": "FLAT",
                "qty": 0.0,
            })
            print(f"[{time.strftime('%H:%M:%S')}] [SYNC] Account equity: , position: FLAT", flush=True)

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
        if self.cfg.live.enable_reflection_memory:
            self.memory.save(self.memory_path)

    def _record_learning_outcome(self, pos: dict, last_ts: int):
        """Record one exchange-confirmed closed position once; never infer PnL from proposal."""
        qty = float(pos.get("qty") or 0.0)
        entry = float(pos.get("entry") or 0.0)
        if qty == 0 or entry <= 0:
            return
        key = f"{self.sym}:{entry:.12g}:{abs(qty):.12g}:{self.st.get('opened_ts')}"
        if self.st.get("last_closed_trade_id") == key:
            return
        # Live exchange position response reports unrealized PnL while open.
        # At closure, the preceding position snapshot was saved as a lightweight
        # anchor; absent an anchor, skip rather than fabricate a realized result.
        anchor = self.st.get("position_anchor") or {}
        realized = anchor.get("realized_pnl")
        if realized is None:
            return
        pnl = float(realized)
        sig = self.st.get("position_signature", "unknown:unknown:unknown:unknown")
        self.memory.record_outcome(sig, pnl, "exchange_closed", last_ts)
        self.st["closed_trade_count"] = self.st.get("closed_trade_count", 0) + 1
        self.st["consecutive_losses"] = self.st.get("consecutive_losses", 0) + 1 if pnl < 0 else 0
        self.st["last_closed_trade_id"] = key
        self.st["position_anchor"] = None
        self.st["position_signature"] = None
        self.emit("learning_outcome", {"signature": sig, "net_pnl": pnl,
                                        "consecutive_losses": self.st["consecutive_losses"],
                                        "closed_trade_count": self.st["closed_trade_count"]})

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

    def _check_and_run_re_evolution(self, bars: list[Bar], last_ts: int):
        """Periodically or conditionally run rolling continuous evolution and candidate hot-swap."""
        if not self.cfg.live.auto_re_evolve:
            return

        self.st["bars_since_evolution"] = self.st.get("bars_since_evolution", 0) + 1
        bars_since = self.st["bars_since_evolution"]
        consec_losses = self.st.get("consecutive_losses", 0)
        
        # Trigger conditions: periodic interval reached OR adverse loss streak encountered
        interval_trigger = bars_since >= self.cfg.live.re_evolve_interval_bars
        loss_trigger = consec_losses >= self.cfg.live.loss_trigger_count

        if not (interval_trigger or loss_trigger):
            return

        # Do not hot-swap while holding an active position on exchange
        if self.st.get("opened_ts") is not None:
            return

        trigger_reason = f"loss streak ({consec_losses})" if loss_trigger else f"interval ({bars_since} bars)"
        self.emit("online_evolution_triggered", {
            "reason": trigger_reason,
            "incumbent_id": self.c.id,
            "bars_since": bars_since,
            "consecutive_losses": consec_losses,
            "bar_ts": last_ts
        })

        # Evaluate rolling challenger using window of latest bars
        eval_bars = bars[-self.cfg.live.re_evolve_lookback_bars:] if len(bars) > self.cfg.live.re_evolve_lookback_bars else bars
        eval_res = evaluate_rolling_challenger(
            self.c,
            eval_bars,
            self.cfg,
            min_improvement=self.cfg.live.hot_swap_min_improvement,
            pop_size=12,
            seed=int(last_ts % 100000)
        )

        self.emit("online_evolution_evaluation", eval_res.to_dict() | {"bar_ts": last_ts})

        if eval_res.won and eval_res.challenger is not None:
            old_id = self.c.id
            self.c = eval_res.challenger
            self.st["bars_since_evolution"] = 0
            self.st["consecutive_losses"] = 0
            self.emit("candidate_hot_swapped", {
                "previous_id": old_id,
                "new_id": self.c.id,
                "improvement_frac": eval_res.improvement_frac,
                "incumbent_pnl": eval_res.incumbent_net_pnl,
                "challenger_pnl": eval_res.challenger_net_pnl,
                "bar_ts": last_ts
            })
        else:
            self.st["bars_since_evolution"] = 0

    # ---------- one closed bar
    def step(self) -> str:
        if self.kill.exists():
            self.flatten("kill switch file present")
            self.emit("halted", {"reason": "kill switch"})
            return "KILLED"
        if self.st["halted"]:
            return "HALTED"

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

        step_s = self.tf * 60
        now_ts = int(time.time())

        # If exchange position closed (e.g. hit stop or target while running)
        if not pos["qty"] and self.st.get("opened_ts"):
            self._cancel_all() if not self.dry else None
            self.emit("position_closed", {"bar_ts": now_ts, "equity": equity})
            self.st["opened_ts"] = None
            self.st["position_qty"] = 0.0
            self._record_learning_outcome(self.st.get("position_anchor") or {}, now_ts)
            self.save()

        # If holding an active position on exchange
        if pos["qty"]:
            qty = float(pos["qty"])
            side = "LONG" if qty > 0 else "SHORT"
            if self.st.get("opened_ts") is None:
                self.st["opened_ts"] = now_ts
                self.emit("position_reconnected", {
                    "bar_ts": now_ts, "qty": qty, "side": side,
                    "entry": pos.get("entry"), "upnl": pos.get("upnl"), "equity": equity,
                    "reconciled": True
                })
            held = (now_ts - (self.st["opened_ts"] or now_ts)) // step_s
            protect = [o for o in self.x.open_orders(self.sym) if o.get("type") in CONDITIONAL] if not self.dry else []
            self.emit("position", {
                "bar_ts": now_ts, "qty": qty, "side": side,
                "entry": pos.get("entry"), "upnl": pos.get("upnl"), "held_bars": held, "equity": equity,
                "protective_orders": len(protect)
            })
            self.st["position_anchor"] = {"entry": pos.get("entry"), "realized_pnl": pos.get("realized_pnl")}
            self.st["position_qty"] = qty
            self.st["position_entry"] = float(pos.get("entry") or 0.0)
            if held >= self.c.g("stop_target", "max_hold_bars"):
                self.flatten("time exit")
            self.save()

        bars = self.bars()
        if len(bars) < pipeline.MIN_BARS:
            self.emit("defer", {"reason": f"only {len(bars)} closed bars"})
            return "DEFER"
        last = bars[-1]
        staleness = int((time.time() - (last.ts + step_s)) // step_s)

        if self.st["last_bar"] == last.ts:
            return "MANAGED" if pos["qty"] else "SAME_BAR"

        self.st["last_bar"] = last.ts
        self._check_and_run_re_evolution(bars, last.ts)

        if pos["qty"]:
            return "MANAGED"

        t = len(bars) - 1
        eb = pipeline.build(PITView(bars, t))
        closes = [b.close for b in bars]
        ef = ema(closes, self.c.g("trend", "ema_fast"))[-1]
        es = ema(closes, self.c.g("trend", "ema_slow"))[-1]
        disc = disc_mod.discuss(eb) if eb.valid else None
        prop = trader.propose(self.c, eb, disc, ef, es, equity)

        # Reflection Memory check (negative constraint inhibitor)
        inhibition = None
        if self.cfg.live.enable_reflection_memory and prop.side in ("long", "short"):
            entry_mode = getattr(prop, 'entry_mode', None) or self.c.genes.get('topology', {}).get('entry_mode', 'unknown')
            sig = ReflectionMemory.signature_from_context(entry_mode, prop.side, eb)
            inhibition = self.memory.check_inhibition(sig)
            if inhibition.action == "VETO":
                self.emit("reflection_inhibition", {"signature": sig, "action": "VETO",
                                                    "reason": inhibition.reason, "bar_ts": last.ts})
                self.emit("decision", {"bar_ts": last.ts, "equity": equity, "proposal": prop.model_dump(),
                                       "risk": {"outcome": "REJECT", "reasons": [inhibition.reason]},
                                       "reflection": inhibition.to_dict()})
                self.save()
                return "REJECT"

        risk_cfg = self.cfg.risk.model_copy(update={"max_cost_to_reward": min(
            self.cfg.risk.max_cost_to_reward, self.c.g("cost", "max_cost_to_reward"))})
        rd = gate.evaluate(prop, gate.AccountState(equity, 0, self.st["halted"]),
                           gate.MarketState(last, eb.valid, staleness), risk_cfg, self.cfg.costs)
        
        # Apply inhibition downsizing penalty if active
        if inhibition and inhibition.action == "RESIZE_PENALTY" and rd.outcome in ("APPROVE", "RESIZE"):
            penalized_qty = rd.approved_qty * inhibition.size_mult
            rd = rd.model_copy(update={"approved_qty": penalized_qty, "outcome": "RESIZE"})
            self.emit("reflection_inhibition", {"signature": inhibition.signature, "action": "RESIZE_PENALTY",
                                                "reason": inhibition.reason, "size_mult": inhibition.size_mult, "bar_ts": last.ts})

        rec = {"bar_ts": last.ts, "equity": equity, "proposal": prop.model_dump(), "risk": rd.model_dump(),
               "discussion": disc.model_dump() if disc else None,
               "reflection": inhibition.to_dict() if inhibition else None}
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
        entry_mode = getattr(prop, "entry_mode", None) or self.c.genes.get("topology", {}).get("entry_mode", "unknown")
        self.st["position_signature"] = ReflectionMemory.signature_from_context(entry_mode, prop.side, eb)
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
        consecutive_errors = 0
        while max_steps is None or n < max_steps:
            try:
                out = self.step()
                consecutive_errors = 0
            except (ExchangeError, Exception) as e:
                consecutive_errors += 1
                self.emit("exchange_error", {"error": str(e), "consecutive": consecutive_errors})
                out = "ERROR"
                # If network/exchange error, wait backoff time before next attempt
                if consecutive_errors <= 5:
                    time.sleep(min(30.0, 2.0 * consecutive_errors))
                    continue
            print(time.strftime("%H:%M:%S"), out, flush=True)
            if out in ("KILLED", "HALTED"):
                return out
            n += 1
            if max_steps is not None and n >= max_steps:
                break
            step_s = self.tf * 60
            time_to_bar = max(1.0, step_s - (time.time() % step_s) + self.cfg.live.close_grace_s)
            sleep_s = min(time_to_bar, 15.0) if (self.st.get("opened_ts") and not self.dry) else time_to_bar
            time.sleep(sleep_s)
        return "DONE"
