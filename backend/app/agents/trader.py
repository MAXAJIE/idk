"""Trader: chromosome rules + discussion -> typed proposal.
Requests a size only; Risk Control decides the permitted size."""
from __future__ import annotations

from app.agents.schemas import DiscussionReport, TradeProposal
from app.features.pipeline import EvidenceBundle
from app.genetic.schema import Chromosome


def signal(c: Chromosome, eb: EvidenceBundle, ema_f: float, ema_s: float) -> tuple[int, list[str]]:
    r, s, a, rsi = eb.get("regime"), eb.get("smc"), eb.get("amt"), eb.get("rsi")["rsi14"]
    mode = c.g("topology", "entry_mode")
    g = c.g("trend", "rsi_guard")
    if mode == "trend_follow":
        side = 1 if ema_f > ema_s and r["trend"] != "down" else -1 if ema_f < ema_s and r["trend"] != "up" else 0
    elif mode == "breakout":
        side = 1 if s["event"] in ("bos_up", "choch_up") else -1 if s["event"] in ("bos_down", "choch_down") else 0
    else:
        side = 1 if a["location"] == "below_va" and rsi < 100 - g else -1 if a["location"] == "above_va" and rsi > g else 0
    if not side:
        return 0, []
    filters: list[bool] = []
    if c.g("amt", "use_value_area") and mode != "mean_revert":
        accept = c.g("amt", "va_policy") == "accept_outside"
        bad = ("below_va" if side > 0 else "above_va") if accept else ("above_va" if side > 0 else "below_va")
        filters.append(a["location"] != bad)
    if c.g("smc", "require_structure"):
        filters.append(s["bias"] != ("bear" if side > 0 else "bull"))
    if c.g("smc", "use_fvg"):
        filters.append(eb.get("fvg")["bear_fvg" if side > 0 else "bull_fvg"] is None)
    if c.g("smc", "respect_sweep"):
        filters.append(s["sweep"] != ("buy_side_swept" if side > 0 else "sell_side_swept"))
    if mode != "mean_revert":
        filters.append(rsi <= g if side > 0 else rsi >= 100 - g)
    if c.g("entry", "skip_high_vol"):
        filters.append(r["vol_pct"] <= 0.9)
    filters.append(abs(eb.get("price")["close"] - ema_f) <= c.g("entry", "max_chase_atr") * eb.get("atr")["atr14"])
    ok = all(filters) if c.g("topology", "compose") == "all" else sum(filters) * 2 > len(filters)
    return (side if ok else 0), [eb.id("regime"), eb.id("smc"), eb.id("amt"), eb.id("rsi"), eb.id("atr")]


def _claim_keys(turn) -> list[str]:
    return [f"{cl.evidence_id}|{cl.text}" for cl in turn.claims]


def propose(c: Chromosome, eb: EvidenceBundle, disc: DiscussionReport | None, ema_f: float, ema_s: float,
            equity: float) -> TradeProposal:
    base = dict(decision_id=f"dec:{c.id}:{eb.t}", chromosome_id=c.id, t=eb.t,
                ref_price=eb.get("price")["close"] if eb.valid else 0.0)
    if not eb.valid:
        return TradeProposal(**base, side="no_trade", evidence_ids=[], discussion_adoption="NOT_EVALUATED",
                             rationale="insufficient evidence")
    side, ev = signal(c, eb, ema_f, ema_s)
    if not side or disc is None:
        return TradeProposal(**base, side="no_trade", evidence_ids=ev, discussion_adoption="NOT_EVALUATED",
                             rationale="chromosome rules produced no entry")
    aligned = disc.score * side
    thr, strict = c.g("discussion", "agreement_threshold"), c.g("discussion", "mapping") == "strict"
    mine, theirs = disc.turns[0 if side > 0 else 1], disc.turns[1 if side > 0 else 0]
    size_mult, adopted = 1.0, _claim_keys(mine)
    if aligned >= thr:
        adoption = "ADOPTED"
    elif aligned <= -thr and thr > 0:
        if strict:
            return TradeProposal(**base, side="no_trade", evidence_ids=ev, discussion_adoption="ADOPTED",
                                 adopted_claims=_claim_keys(theirs),
                                 rationale=f"discussion opposed signal (score {disc.score}); stood aside")
        adoption, size_mult = "PARTIALLY_ADOPTED", 0.5
    elif strict:
        adoption, adopted = "REJECTED", []
    else:
        adoption, size_mult = "PARTIALLY_ADOPTED", 0.5

    close, atr, s = base["ref_price"], eb.get("atr")["atr14"], eb.get("smc")
    stop_dist = c.g("stop_target", "stop_atr") * atr
    if c.g("stop_target", "stop_method") == "structure":
        lvl = s["last_swing_low"] if side > 0 else s["last_swing_high"]
        if lvl is not None and side * (close - lvl) > 0.2 * atr:
            stop_dist = side * (close - lvl) + 0.1 * atr
    target = close + side * c.g("stop_target", "rr") * stop_dist
    if c.g("stop_target", "target_method") == "snr":
        lvl = eb.get("snr")["resistance" if side > 0 else "support"]
        if lvl and side * (lvl["price"] - close) >= stop_dist:
            target = lvl["price"]
    qty = equity * c.g("sizing", "risk_frac") * size_mult / stop_dist if stop_dist > 0 else 0.0
    return TradeProposal(**base, side="long" if side > 0 else "short", stop=round(close - side * stop_dist, 6),
                         target=round(target, 6), qty=qty, evidence_ids=ev + [eb.id("snr")],
                         discussion_adoption=adoption, adopted_claims=adopted,
                         rationale=f"{c.g('topology', 'entry_mode')} signal; discussion score {disc.score}")
