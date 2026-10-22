"""Deterministic Data Aggregator, Technical Analyst and sequential bull/bear
discussion. Every claim cites an evidence ID from the bundle; no new facts."""
from __future__ import annotations

from app.agents.schemas import AnalystReport, Claim, DiscussionReport, DiscussionTurn
from app.features.pipeline import EvidenceBundle

REQUIRED = ["price", "atr", "regime", "amt", "smc", "snr"]


def aggregate(eb: EvidenceBundle) -> AnalystReport:
    missing = [k for k in REQUIRED if eb.get(k) is None]
    return AnalystReport(role="data_aggregator", ts=eb.ts, summary=f"{len(eb.items)} evidence items",
                         evidence_ids=list(eb.items), missing=missing,
                         insufficient_evidence=bool(missing) or not eb.valid)


def technical(eb: EvidenceBundle) -> AnalystReport:
    if not eb.valid:
        return AnalystReport(role="technical_analyst", ts=eb.ts, summary="insufficient evidence",
                             evidence_ids=[], insufficient_evidence=True)
    r, s, a = eb.get("regime"), eb.get("smc"), eb.get("amt")
    return AnalystReport(role="technical_analyst", ts=eb.ts,
                         summary=f"trend={r['trend']} structure={s['bias']}/{s['event']} auction={a['location']}",
                         evidence_ids=[eb.id("regime"), eb.id("smc"), eb.id("amt")])


def _claims(eb: EvidenceBundle, bull: bool) -> list[Claim]:
    r, s, a, f, n, rsi = (eb.get(k) for k in ("regime", "smc", "amt", "fvg", "snr", "rsi"))
    want = "up" if bull else "down"
    out = []
    if r["trend"] == want:
        out.append(Claim(text=f"regime trend {want} (slope {r['slope_atr']} ATR)", evidence_id=eb.id("regime"), weight=0.8))
    if s["bias"] == ("bull" if bull else "bear"):
        out.append(Claim(text=f"structure bias {s['bias']} ({s['event']})", evidence_id=eb.id("smc"), weight=0.7))
    if s["sweep"] == ("sell_side_swept" if bull else "buy_side_swept"):
        out.append(Claim(text=f"liquidity {s['sweep']}", evidence_id=eb.id("smc"), weight=0.5))
    if a["location"] == ("above_va" if bull else "below_va"):
        out.append(Claim(text=f"price accepted {a['location']}", evidence_id=eb.id("amt"), weight=0.5))
    if f["bull_fvg" if bull else "bear_fvg"]:
        out.append(Claim(text=f"unmitigated {'bull' if bull else 'bear'} FVG", evidence_id=eb.id("fvg"), weight=0.4))
    lvl = n["support" if bull else "resistance"]
    if lvl and lvl["touches"] >= 2:
        out.append(Claim(text=f"{'support' if bull else 'resistance'} {lvl['price']} x{lvl['touches']}",
                         evidence_id=eb.id("snr"), weight=0.3))
    if (rsi["rsi14"] < 30) if bull else (rsi["rsi14"] > 70):
        out.append(Claim(text=f"RSI stretched {rsi['rsi14']}", evidence_id=eb.id("rsi"), weight=0.3))
    return out


def discuss(eb: EvidenceBundle) -> DiscussionReport:
    """Bull speaks first; bear second and must answer bull's strongest claim."""
    bull, bear = _claims(eb, True), _claims(eb, False)
    s = eb.get("smc")

    def top(cs):
        return max(cs, key=lambda c: c.weight).text if cs else "no claim"

    bw, sw = sum(c.weight for c in bull), sum(c.weight for c in bear)
    score = (bw - sw) / (bw + sw) if bw + sw else 0.0
    turns = [
        DiscussionTurn(role="bull", claims=bull, counterarguments=[f"bear may cite: {top(bear)}"],
                       invalidation=f"close below swing low {s['last_swing_low']}", uncertainty=round(1 - min(bw, 1), 3)),
        DiscussionTurn(role="bear", claims=bear, counterarguments=[f"rebuts bull: {top(bull)}"],
                       invalidation=f"close above swing high {s['last_swing_high']}", uncertainty=round(1 - min(sw, 1), 3)),
    ]
    rec = "long" if score > 0.2 else "short" if score < -0.2 else "no_trade"
    return DiscussionReport(turns=turns, score=round(score, 4), recommendation=rec)
