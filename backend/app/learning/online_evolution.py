"""Online Continuous Evolution & Candidate Hot-Swap Engine.

Evaluates an incumbent chromosome against fresh market data by running a
rolling walk-forward challenge: generates local mutations and diverse exploration
chromosomes on a rolling IS window, validates survivors on rolling OOS, and
promotes a qualified challenger only if it strictly outperforms the incumbent
under risk gates.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from app.config import ExperimentConfig
from app.data.ingest import Bar
from app.data.point_in_time import PITView
from app.features import pipeline
from app.genetic.operators import offspring, random_chromosome
from app.genetic.schema import Chromosome
from app.simulator.engine import Context, simulate


@dataclass
class ChallengerEvaluation:
    won: bool
    challenger: Chromosome | None
    incumbent_id: str
    incumbent_net_pnl: float
    incumbent_max_drawdown: float
    challenger_net_pnl: float | None = None
    challenger_max_drawdown: float | None = None
    improvement_frac: float = 0.0
    reason: str = "incumbent maintained"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.challenger:
            d["challenger"] = self.challenger.to_dict()
        return d


def evaluate_rolling_challenger(
    incumbent: Chromosome,
    bars: list[Bar],
    cfg: ExperimentConfig,
    min_improvement: float = 0.05,
    pop_size: int = 12,
    seed: int = 42,
) -> ChallengerEvaluation:
    lookback = pipeline.LOOKBACK
    if len(bars) < lookback + 40:
        return ChallengerEvaluation(
            won=False,
            challenger=None,
            incumbent_id=incumbent.id,
            incumbent_net_pnl=0.0,
            incumbent_max_drawdown=0.0,
            reason=f"insufficient bars ({len(bars)} < {lookback + 40})",
        )

    # 1. Build causal evidence pipeline for the bars
    evidence = [pipeline.build(PITView(bars, t)) for t in range(len(bars))]
    ctx = Context(cfg, bars, evidence)

    # 2. Split into rolling in-sample (70%) and out-of-sample (30%)
    split_idx = int(lookback + (len(bars) - lookback) * 0.7)
    is_start, is_end = lookback, split_idx
    oos_start, oos_end = split_idx, len(bars)

    # 3. Simulate incumbent on the rolling OOS window
    inc_res = simulate(incumbent, ctx, oos_start, oos_end)
    inc_sc = inc_res["scorecard"]
    inc_pnl = inc_sc["net_pnl"]
    inc_dd = inc_sc["max_drawdown"]

    # 4. Breed candidates from incumbent + mutations + exploration
    rng = random.Random(seed)
    candidates: list[Chromosome] = [incumbent]
    attempts = 0
    while len(candidates) < pop_size and attempts < 100:
        attempts += 1
        partner = random_chromosome(rng) if rng.random() < 0.3 else incumbent
        child = offspring(
            rng,
            incumbent,
            partner,
            generation=incumbent.generation + 1,
            crossover_rate=0.7,
            mutation_rate=0.25,
            operator_version=cfg.ga.operator_version,
        )
        if child is not None:
            candidates.append(child)

    # 5. Evaluate all candidates on rolling IS
    is_scored = []
    for cand in candidates:
        res = simulate(cand, ctx, is_start, is_end)
        if res["halted"]:
            continue
        sc = res["scorecard"]
        is_scored.append((cand, sc["net_pnl"], sc["max_drawdown"]))

    is_scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [c for c, _, _ in is_scored[:4] if c.id != incumbent.id]

    if not top_candidates:
        return ChallengerEvaluation(
            won=False,
            challenger=None,
            incumbent_id=incumbent.id,
            incumbent_net_pnl=round(inc_pnl, 4),
            incumbent_max_drawdown=round(inc_dd, 4),
            reason="no viable candidate survived rolling in-sample",
        )

    # 6. Evaluate top candidates on rolling OOS
    best_challenger = None
    best_cand_pnl = inc_pnl
    best_cand_dd = inc_dd

    for cand in top_candidates:
        oos_res = simulate(cand, ctx, oos_start, oos_end)
        if oos_res["halted"]:
            continue
        cand_sc = oos_res["scorecard"]
        pnl = cand_sc["net_pnl"]
        dd = cand_sc["max_drawdown"]

        # Risk gate checks
        if dd > cfg.gates.max_drawdown_frac:
            continue

        # Strictly outperforms incumbent
        margin = max(1.0, abs(inc_pnl) * min_improvement)
        if pnl >= inc_pnl + margin and dd <= max(inc_dd * 1.15, 0.05):
            if pnl > best_cand_pnl:
                best_challenger = cand
                best_cand_pnl = pnl
                best_cand_dd = dd

    if best_challenger is not None:
        imp = (best_cand_pnl - inc_pnl) / (abs(inc_pnl) + 1e-6)
        return ChallengerEvaluation(
            won=True,
            challenger=best_challenger,
            incumbent_id=incumbent.id,
            incumbent_net_pnl=round(inc_pnl, 4),
            incumbent_max_drawdown=round(inc_dd, 4),
            challenger_net_pnl=round(best_cand_pnl, 4),
            challenger_max_drawdown=round(best_cand_dd, 4),
            improvement_frac=round(imp, 4),
            reason=f"challenger outperformed incumbent (net_pnl: {inc_pnl:.2f} -> {best_cand_pnl:.2f})",
        )

    return ChallengerEvaluation(
        won=False,
        challenger=None,
        incumbent_id=incumbent.id,
        incumbent_net_pnl=round(inc_pnl, 4),
        incumbent_max_drawdown=round(inc_dd, 4),
        reason="incumbent remained superior or competitive on OOS",
    )
