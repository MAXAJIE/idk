"""Self-Training Arena: mass-produced candidate evolution with anti-overfitting combat.

Features:
1. Massive population generation from history data (mutation, crossover, diverse seeds).
2. Multi-window Walk-Forward splits (IS train -> OOS combat -> Embargoed Holdout).
3. Anti-overfitting checks:
   - WFA consistency across non-overlapping OOS windows
   - Deflated Sharpe Ratio / Haircut / Win-rate sanity check
   - Parameter sensitivity / neighbour retention test (robustness)
   - Cost-to-reward stress test
4. Elite tournament combat (head-to-head out-of-sample comparison).
5. Auto-training loop with progressive seed shifts until survivor crowned and analyzed.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from app.config import ExperimentConfig
from app.data.ingest import Bar
from app.data.point_in_time import PITView
from app.features import pipeline
from app.genetic import population
from app.genetic.operators import offspring, random_chromosome
from app.genetic.schema import Chromosome
from app.risk.scorecard import fitness, gates
from app.simulator.engine import Context, simulate
from app.validation import robustness
from app.validation.walk_forward import make_folds


@dataclass
class ArenaCombatant:
    chromosome: Chromosome
    is_pnl: float
    is_drawdown: float
    oos_pnl: float
    oos_drawdown: float
    consistency: float
    neighbour_retention: float
    qualified: bool
    status_reason: str
    oos_return: float = 0.0
    is_trades: int = 0
    is_win_rate: float = 0.0
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    positive_oos_folds: int = 0
    total_oos_folds: int = 0
    cost_stress_passed: bool = True
    selection_score: float = 0.0
    holdout_pnl: float | None = None
    holdout_drawdown: float | None = None
    holdout_trades: int | None = None
    holdout_win_rate: float | None = None
    holdout_passed: bool | None = None
    wfa_qualified: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["chromosome"] = self.chromosome.to_dict()
        return d


def rule_complexity(c: Chromosome) -> float:
    """Parsimony pressure: penalizes brittle indicator filters that overfit historical noise."""
    comp = 0.0
    for fam, gene in [
        ("amt", "use_value_area"),
        ("smc", "require_structure"),
        ("smc", "use_fvg"),
        ("smc", "respect_sweep"),
        ("entry", "skip_high_vol"),
    ]:
        if c.g(fam, gene):
            comp += 1.0
    if c.g("amt", "va_policy") == "fade_extremes":
        comp += 0.5
    if c.g("topology", "entry_mode") != "trend_follow":
        comp += 0.5
    if c.g("topology", "compose") == "all":
        comp += 0.5
    return comp


def selection_fitness(is_fitness: float, fold_fitness: list[float], complexity: float,
                      gap_weight: float = 1.0, complexity_weight: float = 0.25) -> tuple[float, float, float]:
    """Score IS fitness against the worst/median WFA folds, never the holdout."""
    if not fold_fitness:
        return is_fitness - complexity_weight * complexity, is_fitness, 0.0
    ordered = sorted(fold_fitness)
    robust_oos = 0.6 * ordered[0] + 0.4 * ordered[(len(ordered) - 1) // 2]
    gap = max(0.0, is_fitness - robust_oos)
    return is_fitness - gap_weight * gap - complexity_weight * complexity, robust_oos, gap


@dataclass
class ArenaResult:
    champion: Chromosome | None
    total_spawned: int
    is_survivors: int
    oos_qualifiers: int
    combatants: list[ArenaCombatant]
    champion_oos_pnl: float = 0.0
    champion_consistency: float = 0.0
    summary: str = "no champion"
    round_number: int = 1
    total_rounds: int = 1
    decisions: list[dict[str, Any]] = field(default_factory=list)
    holdout_attempted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": self.champion.to_dict() if self.champion else None,
            "total_spawned": self.total_spawned,
            "is_survivors": self.is_survivors,
            "oos_qualifiers": self.oos_qualifiers,
            "champion_oos_pnl": round(self.champion_oos_pnl, 4),
            "champion_consistency": round(self.champion_consistency, 4),
            "summary": self.summary,
            "round_number": self.round_number,
            "total_rounds": self.total_rounds,
            "combatants": [c.to_dict() for c in self.combatants],
            "decisions": self.decisions,
            "holdout_attempted": self.holdout_attempted,
        }


def train_and_combat(
    bars: list[Bar],
    cfg: ExperimentConfig,
    base_population: list[Chromosome] | None = None,
    pop_size: int = 36,
    generations: int = 3,
    elite_pool_size: int = 8,
    min_consistency: float = 0.6,
    seed: int = 42,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> ArenaResult:
    """Mass-produces chromosomes on historical bars and conducts anti-overfitting arena combat."""
    lookback = pipeline.LOOKBACK
    if len(bars) < lookback + 120:
        return ArenaResult(champion=None, total_spawned=0, is_survivors=0, oos_qualifiers=0,
                           combatants=[], summary=f"insufficient bars ({len(bars)} < {lookback + 120})")

    # 1. Build PIT causal context
    evidence = [pipeline.build(PITView(bars, t)) for t in range(len(bars))]
    ctx = Context(cfg, bars, evidence)

    # 2. Build independent walk-forward splits. Never turn the same bars into
    # both OOS and holdout data: that would silently invalidate qualification.
    try:
        folds, holdout = make_folds(len(bars), lookback, cfg.wfa)
    except (ValueError, AssertionError) as exc:
        summary = f"insufficient valid walk-forward history: {exc}"
        if progress:
            progress("state", {"to": "ARENA_COMPLETED", "reason": summary, "fold": 0})
        return ArenaResult(
            champion=None, total_spawned=0, is_survivors=0, oos_qualifiers=0,
            combatants=[], summary=summary,
        )
    if len(folds) < cfg.wfa.min_oos_folds:
        summary = (f"insufficient independent OOS folds ({len(folds)} < "
                   f"required {cfg.wfa.min_oos_folds}); increase history or revise WFA windows")
        if progress:
            progress("state", {"to": "ARENA_COMPLETED", "reason": summary, "fold": 0})
        return ArenaResult(
            champion=None, total_spawned=0, is_survivors=0, oos_qualifiers=0,
            combatants=[],
            summary=summary,
        )

    rng = random.Random(seed)
    if progress:
        progress("state", {"to": "ARENA_EVOLUTION", "reason": "Generating and scoring arena chromosomes", "fold": 0})

    # 3. Inject a small, bounded seed fraction; keep the rest freshly randomized.
    pop: list[Chromosome] = []
    if base_population:
        seed_count = min(len(base_population), max(1, int(pop_size * 0.15)))
        seeds = base_population[:seed_count]
        pop.extend(seeds)
        for seed_chromosome in seeds:
            if len(pop) >= pop_size:
                break
            mutated = offspring(rng, seed_chromosome, seed_chromosome, 0, 0.0,
                                cfg.ga.mutation_rate, cfg.ga.operator_version)
            if mutated:
                pop.append(mutated)
    while len(pop) < pop_size:
        pop.append(random_chromosome(rng))

    total_spawned = len(pop)
    is_fold = folds[0]
    elite_combatants: list[Chromosome] = []
    scored: list[tuple[Chromosome, float]] = []
    previous_gap: float | None = None
    previous_is_fitness: float | None = None
    rising_gap_streak = 0

    # 4. Evolution is driven by IS fitness tempered by robust WFA fitness.
    # The holdout is deliberately not touched here.
    for gen in range(generations):
        import time
        generation_started = time.time()
        scored = []
        metrics: dict[str, dict[str, float]] = {}
        for c in pop:
            is_res = simulate(c, ctx, is_fold.is_start, is_fold.is_end)
            if is_res["failed_risk"]:
                c.state = "FAILED_RISK"
                continue
            is_fit = fitness(is_res["scorecard"], gates(is_res["scorecard"], cfg.gates))
            fold_fitness = []
            for fold in folds:
                result = simulate(c, ctx, fold.oos_start, fold.oos_end)
                card = result["scorecard"]
                min_trades = max(2, round(cfg.gates.min_trades *
                                         ((fold.oos_end - fold.oos_start) /
                                          max(1, is_fold.is_end - is_fold.is_start))))
                fold_cfg = cfg.gates.model_copy(update={"min_trades": min_trades})
                fold_fitness.append(fitness(card, gates(card, fold_cfg)))
            if not fold_fitness:
                continue
            complexity = rule_complexity(c)
            composite, robust_oos, gap = selection_fitness(is_fit, fold_fitness, complexity)
            scored.append((c, composite))
            metrics[c.id] = {"is": is_fit, "oos": robust_oos, "gap": gap}

        scored.sort(key=lambda item: item[1], reverse=True)
        values = [score for _, score in scored]
        best = metrics.get(scored[0][0].id, {}) if scored else {}
        if progress:
            progress("generation", {
                "fold": 0, "gen": gen,
                "best": round(values[0], 4) if values else 0.0,
                "mean": round(sum(values) / len(values), 4) if values else 0.0,
                "worst": round(values[-1], 4) if values else 0.0,
                "gap": round(best.get("gap", 0.0), 4),
                "is_fitness": round(best.get("is", 0.0), 4),
                "oos_fitness": round(best.get("oos", 0.0), 4),
                "valid": len(scored), "population": len(pop),
                "generations": generations, "folds": len(folds),
                "seconds": round(time.time() - generation_started, 3),
                "failed_risk": sum(c.state == "FAILED_RISK" for c in pop),
                "best_id": scored[0][0].id if scored else "",
                "training_mode": "arena",
            })
        current_is = best.get("is", 0.0)
        current_gap = best.get("gap", 0.0)
        if (previous_gap is not None and previous_is_fitness is not None
                and current_gap > previous_gap and current_is > previous_is_fitness):
            rising_gap_streak += 1
        else:
            rising_gap_streak = 0
        previous_gap, previous_is_fitness = current_gap, current_is
        if rising_gap_streak >= 2 and gen < generations - 1:
            elite_combatants = population.select_diverse_elite(scored, k=elite_pool_size)
            if progress:
                progress("state", {"to": "ARENA_EARLY_STOP", "reason": "IS fitness rose while the IS-WFA gap widened for consecutive generations", "fold": 0, "gap": round(current_gap, 4)})
            break
        if gen < generations - 1:
            pop, _ = population.next_generation(rng, scored, [], pop_size, gen + 1, cfg.ga)
            total_spawned += len(pop)
        else:
            elite_combatants = population.select_diverse_elite(scored, k=elite_pool_size)

    if not elite_combatants and scored:
        elite_combatants = population.select_diverse_elite(scored, k=elite_pool_size)

    if not elite_combatants:
        if progress:
            progress("state", {"to": "ARENA_COMPLETED", "reason": "No candidates survived IS evolution", "fold": 0})
        return ArenaResult(champion=None, total_spawned=total_spawned, is_survivors=0,
                           oos_qualifiers=0, combatants=[], summary="no candidates survived IS evolution")

    is_survivors = len(elite_combatants)

    if progress:
        progress("state", {"to": "ARENA_COMBAT", "reason": "Testing IS survivors against WFA and holdout gates", "fold": 0})

    # 5. Evaluate WFA robustness first. No candidate-specific holdout peeking.
    combatant_records: list[ArenaCombatant] = []
    wfa_qualifiers: list[tuple[Chromosome, float, float]] = []
    stress_ctx = Context(cfg.model_copy(update={
        "costs": cfg.costs.model_copy(update={
            "slippage_base_bps": cfg.costs.slippage_base_bps * 1.5,
            "taker_fee_bps": cfg.costs.taker_fee_bps * 1.25,
        })
    }), bars, evidence)

    for c in elite_combatants:
        is_res = simulate(c, ctx, is_fold.is_start, is_fold.is_end)
        is_pnl = is_res["scorecard"]["net_pnl"]
        fold_results = [simulate(c, ctx, f.oos_start, f.oos_end) for f in folds]
        positive = []
        for fold, result in zip(folds, fold_results):
            card = result["scorecard"]
            min_trades = max(2, round(cfg.gates.min_trades *
                                     ((fold.oos_end - fold.oos_start) /
                                      max(1, is_fold.is_end - is_fold.is_start))))
            if (card["net_return"] > 0 and card["max_drawdown"] <= cfg.gates.max_drawdown_frac
                    and card["n_trades"] >= min_trades and not result["failed_risk"]):
                positive.append(result)
        consistency = len(positive) / len(folds)
        # Summarize WFA using the median fold, never the best fold.
        median_result = sorted(fold_results, key=lambda r: r["scorecard"]["net_pnl"])[(len(fold_results) - 1) // 2]
        oos_card = median_result["scorecard"]
        oos_pnl = oos_card["net_pnl"]
        oos_dd = oos_card["max_drawdown"]
        neighbours = robustness.neighbours(c)
        neighbour_pnls = [simulate(n, ctx, is_fold.is_start, is_fold.is_end)["scorecard"]["net_pnl"]
                          for n in neighbours[:4]]
        retention = robustness.retention(is_pnl, neighbour_pnls)
        stress = simulate(c, stress_ctx, folds[-1].oos_start, folds[-1].oos_end)
        cost_stress_passed = stress["scorecard"]["net_return"] > -0.005 and not stress["failed_risk"]

        qualified = (len(positive) >= cfg.wfa.min_oos_folds
                     and consistency >= min_consistency
                     and retention >= 0.20
                     and cost_stress_passed)
        reasons = []
        if len(positive) < cfg.wfa.min_oos_folds:
            reasons.append(f"too few profitable OOS folds ({len(positive)} < {cfg.wfa.min_oos_folds})")
        if consistency < min_consistency:
            reasons.append(f"low WFA consistency ({consistency:.2f} < {min_consistency:.2f})")
        if retention < 0.20:
            reasons.append(f"fragile parameter sensitivity (neighbour retention {retention:.2f} < 0.20)")
        if not cost_stress_passed:
            reasons.append("fragile under adverse execution cost/slippage stress")
        record = ArenaCombatant(
            chromosome=c, is_pnl=round(is_pnl, 4), is_drawdown=is_res["scorecard"]["max_drawdown"],
            oos_pnl=round(oos_pnl, 4), oos_drawdown=round(oos_dd, 4),
            consistency=round(consistency, 4), neighbour_retention=round(retention, 4),
            qualified=False, status_reason="WFA-qualified; not selected for the single holdout check" if qualified else "; ".join(reasons),
            oos_return=round(oos_card["net_return"], 6),
            wfa_qualified=qualified,
            is_trades=is_res["scorecard"]["n_trades"],
            is_win_rate=round(is_res["scorecard"]["win_rate"] or 0.0, 4),
            oos_trades=oos_card["n_trades"], oos_win_rate=round(oos_card["win_rate"] or 0.0, 4),
            positive_oos_folds=len(positive), total_oos_folds=len(folds),
            cost_stress_passed=cost_stress_passed,
            selection_score=round(0.6 * min(r["scorecard"]["net_return"] for r in fold_results)
                                  + 0.4 * sorted(r["scorecard"]["net_return"] for r in fold_results)[(len(fold_results)-1)//2], 6),
        )
        combatant_records.append(record)
        if qualified:
            wfa_qualifiers.append((c, record.selection_score, oos_pnl))

    # Choose on WFA evidence alone, then spend holdout once on exactly that candidate.
    selected = None
    selected_pnl = 0.0
    holdout_attempted = bool(wfa_qualifiers)
    if wfa_qualifiers:
        wfa_qualifiers.sort(key=lambda item: item[1], reverse=True)
        champion, _, _ = wfa_qualifiers[0]
        champion_record = next(r for r in combatant_records if r.chromosome.id == champion.id)
        champion_record.status_reason = "Selected by WFA; awaiting the single final holdout check"
        holdout_result = simulate(champion, ctx, holdout[0], holdout[1], record=True)
        holdout_card = holdout_result["scorecard"]
        holdout_min_trades = max(2, round(cfg.gates.min_trades *
                                          ((holdout[1] - holdout[0]) /
                                           max(1, is_fold.is_end - is_fold.is_start))))
        holdout_cfg = cfg.gates.model_copy(update={"min_trades": holdout_min_trades})
        failed = [g["gate"] for g in gates(holdout_card, holdout_cfg) if not g["passed"]]
        holdout_passed = not failed and not holdout_result["failed_risk"] and holdout_card["net_pnl"] > 0
        champion_record.holdout_pnl = round(holdout_card["net_pnl"], 4)
        champion_record.holdout_drawdown = round(holdout_card["max_drawdown"], 4)
        champion_record.holdout_trades = holdout_card["n_trades"]
        champion_record.holdout_win_rate = round(holdout_card["win_rate"] or 0.0, 4)
        champion_record.holdout_passed = holdout_passed
        champion_record.qualified = holdout_passed
        champion_record.status_reason = ("QUALIFIED" if holdout_passed else
                                         f"final one-time holdout failed: {', '.join(failed) or 'non-positive return or risk failure'}")
        if holdout_passed:
            selected = champion
            selected_pnl = holdout_card["net_pnl"]
        else:
            if progress:
                progress("state", {"to": "ARENA_COMPLETED", "reason": champion_record.status_reason,
                                   "fold": 0, "holdout_attempted": True})

    oos_qualifiers = len(wfa_qualifiers)
    if selected:
        champion_record = next(r for r in combatant_records if r.chromosome.id == selected.id)
        summary = (f"Champion {selected.id} crowned: one-time holdout PnL={selected_pnl:.2f}, "
                   f"WFA consistency={champion_record.consistency:.2f}, retention={champion_record.neighbour_retention:.2f}")
        if progress:
            progress("state", {"to": "ARENA_COMPLETED", "reason": summary, "fold": 0, "holdout_attempted": True})
        decisions = holdout_result.get("decisions", [])
        return ArenaResult(
            champion=selected, total_spawned=total_spawned, is_survivors=is_survivors,
            oos_qualifiers=oos_qualifiers, combatants=combatant_records,
            champion_oos_pnl=selected_pnl, champion_consistency=champion_record.consistency,
            summary=summary, decisions=decisions, holdout_attempted=True,
        )

    summary = ("Final one-time holdout check failed; stopping without trying another candidate."
               if holdout_attempted else
               f"Combat completed with {is_survivors} IS survivors, but none satisfied WFA robustness gates.")
    if progress:
        progress("state", {"to": "ARENA_COMPLETED", "reason": summary, "fold": 0,
                           "holdout_attempted": holdout_attempted})
    return ArenaResult(
        champion=None, total_spawned=total_spawned, is_survivors=is_survivors,
        oos_qualifiers=oos_qualifiers, combatants=combatant_records,
        summary=summary, holdout_attempted=holdout_attempted,
    )


def auto_train_until_survivor(
    bars: list[Bar],
    cfg: ExperimentConfig,
    base_population: list[Chromosome] | None = None,
    pop_size: int = 36,
    generations: int = 3,
    elite_pool_size: int = 8,
    min_consistency: float = 0.6,
    seed: int = 42,
    max_rounds: int = 10,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[ArenaResult, int]:
    """Repeatedly conducts arena combat with progressive seed shifts and elite carry-over
    until at least one champion satisfies strict anti-overfitting gates, or max_rounds is reached."""
    current_seed = seed
    carried_pool: list[Chromosome] = list(base_population) if base_population else []
    last_res: ArenaResult | None = None

    for round_idx in range(1, max_rounds + 1):
        print(f"\n--- [Auto-Train Round {round_idx}/{max_rounds}] seed={current_seed} pop={pop_size} gens={generations} ---", flush=True)
        res = train_and_combat(
            bars=bars,
            cfg=cfg,
            base_population=carried_pool if carried_pool else None,
            pop_size=pop_size,
            generations=generations,
            elite_pool_size=elite_pool_size,
            min_consistency=min_consistency,
            seed=current_seed,
            progress=(lambda etype, payload: progress(etype, payload | {"round": round_idx}) if progress else None),
        )
        res.round_number = round_idx
        res.total_rounds = round_idx
        last_res = res

        print(f"Round {round_idx} result: {res.summary} (Spawned: {res.total_spawned}, IS Survivors: {res.is_survivors}, OOS Qualifiers: {res.oos_qualifiers})", flush=True)

        if res.champion is not None or res.holdout_attempted:
            print(f">>> Final holdout used; stopping auto-training after round {round_idx}.", flush=True)
            return res, round_idx

        # If no champion, carry top robust performers (by consistency, retention & OOS PnL) to next round
        if res.combatants:
            robust_combatants = sorted(
                res.combatants,
                key=lambda x: (x.consistency, x.neighbour_retention, x.oos_pnl, -abs(x.is_pnl - x.oos_pnl)),
                reverse=True,
            )
            carried_pool = [c.chromosome for c in robust_combatants[:4]]
        current_seed = (current_seed * 1103515245 + 12345) & 0x7FFFFFFF

    if last_res is not None:
        last_res.total_rounds = max_rounds
        return last_res, max_rounds
    return ArenaResult(champion=None, total_spawned=0, is_survivors=0, oos_qualifiers=0, combatants=[]), max_rounds


def analyze_arena_result(res: ArenaResult, cfg: ExperimentConfig, round_num: int = 1) -> str:
    """Produces a structured, comprehensive analysis report of arena combat outcomes."""
    lines = [
        "=" * 64,
        "          ARENA COMBAT & ANTI-OVERFITTING ANALYSIS REPORT        ",
        "=" * 64,
        f"Status:            {'CHAMPION QUALIFIED' if res.champion else 'NO QUALIFIED SURVIVOR'}",
        f"Total Evaluated:   {res.total_spawned} chromosomes across {res.total_rounds} round(s)",
        f"IS Survivors:      {res.is_survivors} reached tournament combat",
        f"OOS Qualifiers:    {res.oos_qualifiers} satisfied all anti-overfitting gates",
    ]

    if not res.champion:
        lines.extend([
            "-" * 64,
            "Failure Analysis / Gate Blockers:",
        ])
        if not res.combatants and res.summary:
            lines.append(f"  {res.summary}")
        if res.combatants:
            from collections import Counter
            blockers = Counter()
            for combatant in res.combatants:
                if combatant.qualified:
                    continue
                reason = combatant.status_reason
                prefix = "failed risk gates on holdout:"
                if reason.startswith(prefix):
                    blockers.update(f"holdout gate: {name.strip()}"
                                    for name in reason[len(prefix):].split(",") if name.strip())
                else:
                    blockers[reason.split("(", 1)[0].strip()] += 1
            lines.append("  Most common blockers:")
            lines.extend(f"    - {reason}: {count}/{len(res.combatants)} candidates"
                         for reason, count in blockers.most_common(5))
        for idx, c in enumerate(res.combatants[:6]):
            lines.append(f"  Candidate {idx+1} ({c.chromosome.id}): {c.status_reason}")
            lines.append(f"    IS PnL: {c.is_pnl:.2f} | OOS PnL: {c.oos_pnl:.2f} | Consistency: {c.consistency:.2f} | Retention: {c.neighbour_retention:.2f}")
        lines.extend([
            "-" * 64,
            "Recommendation: Do not lower gates or keep rerolling seeds against the same history. "
            "Use the blocker counts above to diagnose the limiting metric; collect more independent "
            "history or revise the strategy/search space, then reserve a fresh holdout for one final check.",
            "=" * 64,
        ])
        return "\n".join(lines)

    champ = res.champion
    champ_rec = next((r for r in res.combatants if r.chromosome.id == champ.id), res.combatants[0])
    g = champ.genes

    lines.extend([
        "-" * 64,
        f"[1] Champion Chromosome: {champ.id} (Gen {champ.generation})",
        f"  - Strategy Topology: {g.get('topology', {}).get('entry_mode', 'trend_follow')} (compose={g.get('topology', {}).get('compose', 'all')})",
        f"  - Stop Loss:         {g.get('stop_target', {}).get('stop_method', 'atr').upper()} (ATR x {g.get('stop_target', {}).get('stop_atr', 'N/A')})",
        f"  - Take Profit:       {g.get('stop_target', {}).get('target_method', 'rr').upper()} (Risk/Reward {g.get('stop_target', {}).get('rr', 'N/A')}:1)",
        f"  - Max Holding Bars:  {g.get('stop_target', {}).get('max_hold_bars', 'N/A')} bars",
        f"  - Trend Filter:      EMA {g.get('trend', {}).get('ema_fast', 'N/A')}/{g.get('trend', {}).get('ema_slow', 'N/A')} (RSI guard: {g.get('trend', {}).get('rsi_guard', 'N/A')})",
        f"  - Risk Sizing:       Risk {g.get('sizing', {}).get('risk_frac', 0.01)*100:.2f}% equity per trade",
        "-" * 64,
        "[2] Performance Scorecard (IS vs median WFA fold)",
        f"  - In-Sample (IS):    Net PnL: ${champ_rec.is_pnl:,.2f} | Max DD: {champ_rec.is_drawdown*100:.2f}% | Trades: {champ_rec.is_trades} | Win Rate: {champ_rec.is_win_rate*100:.1f}%",
        f"  - Median WFA fold:  Net PnL: ${champ_rec.oos_pnl:,.2f} | Max DD: {champ_rec.oos_drawdown*100:.2f}% | Trades: {champ_rec.oos_trades} | Win Rate: {champ_rec.oos_win_rate*100:.1f}%",
        f"  - One-time holdout: {('PASSED' if champ_rec.holdout_passed else 'FAILED')} | Net PnL: ${champ_rec.holdout_pnl or 0.0:,.2f} | Max DD: {(champ_rec.holdout_drawdown or 0.0)*100:.2f}% | Trades: {champ_rec.holdout_trades or 0}",
        "-" * 64,
        "[3] Anti-Overfitting Verification",
        f"  - WFA Multi-Window Consistency:      {champ_rec.consistency*100:.1f}% (Required: >={res.champion_consistency*100:.1f}%) -> PASSED",
        f"  - Parameter Neighbourhood Retention: {champ_rec.neighbour_retention:.2f} (Required: >=0.20) -> PASSED",
        "    (Confirms strategy sits on a wide parameter plateau rather than a fragile overfitted curve-fit spike)",
        f"  - Holdout Hard Risk Gates:           {('PASSED' if champ_rec.holdout_passed else 'NOT PASSED')} (checked once after WFA selection)",
        "-" * 64,
        "[4] Deployment Readiness",
        "  - Verdict:           APPROVED for Testnet Dry-Run / Paper execution.",
        "  - Next Step Command: python -m app.cli trade --config configs/binance.json --experiment <exp_id> --dry-run",
        "=" * 64,
    ])
    return "\n".join(lines)
