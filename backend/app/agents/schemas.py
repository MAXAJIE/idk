"""Typed agent contracts. Every structured output is validated before use."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Adoption = Literal["ADOPTED", "PARTIALLY_ADOPTED", "REJECTED", "NOT_EVALUATED"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(Strict):
    text: str
    evidence_id: str
    weight: float = Field(ge=0, le=1)


class AnalystReport(Strict):
    role: Literal["data_aggregator", "technical_analyst"]
    ts: int
    summary: str
    evidence_ids: list[str]
    missing: list[str] = []
    insufficient_evidence: bool = False


class DiscussionTurn(Strict):
    role: Literal["bull", "bear"]
    claims: list[Claim]
    counterarguments: list[str]
    invalidation: str
    uncertainty: float = Field(ge=0, le=1)


class DiscussionReport(Strict):
    turns: list[DiscussionTurn]
    score: float = Field(ge=-1, le=1)
    recommendation: Literal["long", "short", "no_trade"]


class TradeProposal(Strict):
    decision_id: str
    chromosome_id: str
    t: int
    side: Literal["long", "short", "no_trade"]
    ref_price: float
    stop: float | None = None
    target: float | None = None
    qty: float = 0.0
    evidence_ids: list[str]
    discussion_adoption: Adoption
    adopted_claims: list[str] = []
    rationale: str = ""

    @field_validator("ref_price", "qty")
    @classmethod
    def finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class RiskCheck(Strict):
    name: str
    passed: bool
    value: float | str | None = None
    limit: float | str | None = None
    detail: str = ""


class RiskDecision(Strict):
    decision_id: str
    outcome: Literal["APPROVE", "RESIZE", "DEFER", "REJECT", "NO_TRADE"]
    requested_qty: float
    approved_qty: float
    checks: list[RiskCheck]
    reasons: list[str]
    risk_version: str


class JudgeVerdict(Strict):
    chromosome_id: str
    verdict: Literal["RETAIN", "REJECT", "DEFER"]
    nullified_trade_ids: list[str]
    flags: list[str]
    risk_concerns: list[str]
    evidence_refs: list[str]
    adjusted_net_pnl: float
    adjusted_max_drawdown: float
    uncertainty: float = Field(ge=0, le=1)
    rationale: str
