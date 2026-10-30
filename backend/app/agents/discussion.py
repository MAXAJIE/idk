"""Deterministic Data Aggregator, Technical Analyst and sequential bull/bear
discussion. Every claim cites an evidence ID from the bundle; no new facts."""
from __future__ import annotations

import json
import re

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


def _cross_examine(own: list[Claim], opposing: list[Claim]) -> list[str]:
    """Make each side answer the strongest contrary, evidence-backed claim."""
    if not opposing:
        return ["No evidence-backed opposing claim was found; no counterexample is asserted."]
    rebuttals = []
    for claim in sorted(opposing, key=lambda item: item.weight, reverse=True)[:2]:
        rebuttals.append(
            f"Counterevidence to {own[0].text if own else 'this side'}: "
            f"{claim.text} [{claim.evidence_id}]."
        )
    return rebuttals


def discuss(eb: EvidenceBundle) -> DiscussionReport:
    """Two rule-based analysts exchange evidence-backed challenges."""
    bull, bear = _claims(eb, True), _claims(eb, False)
    s = eb.get("smc")
    bw, sw = sum(c.weight for c in bull), sum(c.weight for c in bear)
    score = (bw - sw) / (bw + sw) if bw + sw else 0.0
    turns = [
        DiscussionTurn(role="bull", claims=bull, counterarguments=_cross_examine(bull, bear),
                       invalidation=f"close below swing low {s['last_swing_low']}", uncertainty=round(1 - min(bw, 1), 3)),
        DiscussionTurn(role="bear", claims=bear, counterarguments=_cross_examine(bear, bull),
                       invalidation=f"close above swing high {s['last_swing_high']}", uncertainty=round(1 - min(sw, 1), 3)),
    ]
    rec = "long" if score > 0.2 else "short" if score < -0.2 else "no_trade"
    return DiscussionReport(turns=turns, score=round(score, 4), recommendation=rec)


def _parse_llm_payload(raw: str, role: str) -> dict:
    """Robustly parse and repair LLM JSON turns, handling markdown fences, unescaped quotes, and truncation."""
    cleaned = raw.strip()
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
        if m:
            cleaned = m.group(1).strip()
        else:
            m2 = re.search(r"```(?:json)?\s*(\{.*)", cleaned, re.S)
            if m2:
                cleaned = m2.group(1).strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()

    first_brace = cleaned.find("{")
    if first_brace != -1:
        last_brace = cleaned.rfind("}")
        if last_brace != -1 and last_brace > first_brace:
            candidate = cleaned[first_brace:last_brace + 1]
        else:
            candidate = cleaned[first_brace:]
    else:
        candidate = cleaned

    for attempt in [
        candidate,
        re.sub(r",\s*([\}\]])", r"", candidate),
        re.sub(r'("(?:[^"\\]|\\.)*")\s*\n\s*(")', r'\1,\n\2', re.sub(r",\s*([\}\]])", r"", candidate)),
    ]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Truncated or malformed repair: close quotes and open braces/brackets
    stack = []
    in_string = False
    escape = False
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    repaired = candidate
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*([\}\]])", r"", repaired)
    repaired = re.sub(r",\s*$", "", repaired.strip())
    while stack:
        top = stack.pop()
        repaired += "}" if top == "{" else "]"

    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Regex extraction fallback
    out: dict = {"claims": [], "counterarguments": [], "invalidation": "uncertainty elevated", "uncertainty": 0.5}
    for m in re.finditer(r'"text"\s*:\s*"(.*?)"\s*,\s*"evidence_id"\s*:\s*"(ev:[^"]+)"\s*,\s*"weight"\s*:\s*([0-9.]+)', candidate, re.S):
        try:
            out["claims"].append({"text": m.group(1), "evidence_id": m.group(2), "weight": min(1.0, max(0.0, float(m.group(3))))})
        except ValueError:
            pass

    ca_match = re.search(r'"counterarguments"\s*:\s*\[(.*?)(?:\n\s*\]|\s*\]\s*,)', candidate, re.S)
    if ca_match:
        for line in ca_match.group(1).splitlines():
            line = line.strip().rstrip(",").strip()
            if line.startswith('"') and line.endswith('"') and len(line) >= 2:
                out["counterarguments"].append(line[1:-1])
            elif line and not line.startswith("[") and not line.startswith("]"):
                clean = re.sub(r'^"+|"+$', '', line)
                if clean:
                    out["counterarguments"].append(clean)
    else:
        out["counterarguments"] = re.findall(r'"([^"\n]*?ev:[a-z_]+:\d+[^"\n]*?)"', candidate)

    inv_m = re.search(r'"invalidation"\s*:\s*"(.*?)"', candidate)
    if inv_m:
        out["invalidation"] = inv_m.group(1)
    unc_m = re.search(r'"uncertainty"\s*:\s*([0-9.]+)', candidate)
    if unc_m:
        try:
            out["uncertainty"] = min(1.0, max(0.0, float(unc_m.group(1))))
        except ValueError:
            pass

    if out["claims"] or out["counterarguments"]:
        return out
    raise ValueError(f"{role} agent did not return JSON")


def _llm_turn(provider, role: str, eb: EvidenceBundle, previous: dict | None, stage: str = "opening") -> DiscussionTurn:
    import os
    evidence = {key: item["value"] for key, item in eb.items.items()}
    instructions = (
        "You are one side of a simulated market research debate, not a trader. "
        "Use only the supplied point-in-time evidence. Make no price predictions or trade instructions. "
        "Return RAW JSON only with keys: claims, counterarguments, invalidation, uncertainty. "
        "Do NOT wrap with markdown code blocks (no ```json). Do NOT use unescaped double quotes inside text values (use single quotes). "
        "Each claim is {text, evidence_id, weight}; every evidence_id must exactly match a supplied key. "
        "Each counterargument is a short string and must include its supporting evidence ID verbatim (e.g. [ev:regime:123]). "
        "If there is no support, return an empty claims list and say so in counterarguments. "
        "Do not invent measurements, evidence, citations, or future outcomes."
    )
    if stage == "opening":
        task = ("Present the strongest cautious bullish opening thesis with evidence." if role == "bull" else
                "Present the strongest cautious bearish opening thesis and directly challenge the bullish turn.")
    elif stage in ("rebuttal", "cross_examination"):
        task = (f"Rebut and cross-examine the opposing arguments directly. "
                "Defend your thesis using surviving evidence or concede where opposing evidence is stronger.")
    else:
        task = "Deliver closing summary on whether the thesis remains supported or invalidated by evidence."

    user = {"role": role, "stage": stage, "evidence": evidence, "opposing_turn": previous, "task": task,
            "schema_example": {"claims": [{"text": "...", "evidence_id": "ev:regime:123", "weight": 0.5}],
                               "counterarguments": ["... [ev:regime:123]"],
                               "invalidation": "...", "uncertainty": 0.5}}

    try:
        budget = max(1200, int(os.environ.get("LLM_MAX_TOKENS", "1500")))
    except ValueError:
        budget = 1500

    raw = provider.chat(instructions, json.dumps(user, sort_keys=True), max_tokens=budget)
    payload = _parse_llm_payload(raw, role)
    payload["stage"] = stage
    turn = DiscussionTurn(role=role, **payload)
    valid_ids = set(eb.items)
    cited = {claim.evidence_id for claim in turn.claims}
    if not cited.issubset(valid_ids):
        raise ValueError(f"{role} agent cited evidence outside this candle")
    referenced = set(re.findall(r"ev:[a-z_]+:\d+", " ".join(turn.counterarguments)))
    if not referenced.issubset(valid_ids):
        raise ValueError(f"{role} agent counterargument cited evidence outside this candle")
    return turn


def discuss_with_llm(eb: EvidenceBundle, provider, rounds: int = 1) -> DiscussionReport:
    """Run sequential LLM analyst turns; output is display-only and never gates trades.

    rounds=1: 2 turns (Bull opening, Bear challenge), strictly backward-compatible.
    rounds>=2: 4 turns full cross-examination (Bull opening, Bear cross-exam, Bull rebuttal, Bear closing) + Judge synthesis.
    """
    if rounds <= 1:
        bull = _llm_turn(provider, "bull", eb, None, stage="opening")
        bear = _llm_turn(provider, "bear", eb, bull.model_dump(), stage="cross_examination")
        bw, sw = sum(c.weight for c in bull.claims), sum(c.weight for c in bear.claims)
        score = (bw - sw) / (bw + sw) if bw + sw else 0.0
        rec = "long" if score > 0.2 else "short" if score < -0.2 else "no_trade"
        judge = f"Evidence score {score:+.3f}: bull weight {bw:.2f} vs bear weight {sw:.2f} -> {rec.upper()}."
        return DiscussionReport(turns=[bull, bear], score=round(score, 4), recommendation=rec, judge_summary=judge)

    # 2-round cross-examination debate
    bull_1 = _llm_turn(provider, "bull", eb, None, stage="opening")
    bear_1 = _llm_turn(provider, "bear", eb, bull_1.model_dump(), stage="cross_examination")
    bull_2 = _llm_turn(provider, "bull", eb, {"opening": bull_1.model_dump(), "opposing": bear_1.model_dump()}, stage="rebuttal")
    bear_2 = _llm_turn(provider, "bear", eb, {"opposing": bull_2.model_dump(), "counter": bear_1.model_dump()}, stage="closing")

    bw = sum(c.weight for c in bull_1.claims + bull_2.claims)
    sw = sum(c.weight for c in bear_1.claims + bear_2.claims)
    score = (bw - sw) / (bw + sw) if bw + sw else 0.0
    rec = "long" if score > 0.2 else "short" if score < -0.2 else "no_trade"
    judge = (f"2-round cross-examination: Bull presented {len(bull_1.claims + bull_2.claims)} claims (weight {bw:.2f}), "
             f"Bear countered with {len(bear_1.claims + bear_2.claims)} claims (weight {sw:.2f}). "
             f"Evidence balance {score:+.3f} -> {rec.upper()}. "
             f"Key invalidations: bull [{bull_2.invalidation or bull_1.invalidation}], bear [{bear_2.invalidation or bear_1.invalidation}].")
    return DiscussionReport(turns=[bull_1, bear_1, bull_2, bear_2], score=round(score, 4), recommendation=rec, judge_summary=judge)
