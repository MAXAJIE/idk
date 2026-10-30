"""Experiment configuration.

All numeric thresholds live in a versioned JSON config file (configs/*.json),
never in code. The defaults shipped in configs/default.json are EXAMPLE values
for research only; they are not recommendations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(Frozen):
    source: Literal["synthetic", "csv", "binance"] = "synthetic"
    backend: Literal["rest", "ccxt"] = "rest"
    csv_path: str | None = None
    symbol: str = "BTC-PERP"
    exchange: str = "SIM"
    timeframe_minutes: int = 60
    n_bars: int = 3000
    synthetic_seed: int = 7
    max_gap_bars: int = 0


class CostConfig(Frozen):
    taker_fee_bps: float
    maker_fee_bps: float
    half_spread_bps: float
    slippage_base_bps: float
    slippage_impact_coef: float  # bps per 1% of bar volume consumed
    slippage_vol_coef: float  # multiplier on bar range in bps
    funding_interval_bars: int


class RiskConfig(Frozen):
    version: str
    initial_equity: float
    max_risk_per_trade_frac: float
    max_leverage: float
    max_concurrent_positions: int
    max_bar_volume_frac: float
    max_spread_bps: float
    max_cost_to_reward: float
    min_liq_distance_to_stop: float
    maintenance_margin_frac: float
    stress_gap_mult: float
    max_stress_loss_frac: float
    hard_drawdown_frac: float
    max_data_staleness_bars: int


class GateConfig(Frozen):
    min_trades: int
    max_drawdown_frac: float
    min_net_return: float
    min_stop_compliance: float
    max_cost_share_of_gross: float


class JudgeConfig(Frozen):
    windfall_dominance_frac: float
    windfall_range_atr_mult: float
    systemic_windfall_count: int


class WFAConfig(Frozen):
    n_folds: int
    is_bars: int
    oos_bars: int
    embargo_bars: int
    holdout_bars: int
    min_consistency_frac: float
    min_oos_folds: int


class GAConfig(Frozen):
    population: int
    generations: int
    elite_size: int
    tournament_k: int
    crossover_rate: float
    mutation_rate: float
    seed: int
    operator_version: str = "ga-ops-1"


class RewardConfig(Frozen):
    downside_penalty: float
    drawdown_penalty: float
    cost_penalty: float
    violation_penalty: float


class LLMConfig(Frozen):
    provider: Literal["offline", "openai_compatible", "auto"] = "auto"
    narrate: Literal["none", "elite_only"] = "elite_only"
    max_narrated_decisions: int = 6
    timeout_s: float = 30.0


class LiveConfig(Frozen):
    base_url: str = "https://testnet.binancefuture.com"
    signal_source: Literal["testnet", "mainnet"] = "testnet"
    close_grace_s: float = 5.0
    recv_window_ms: int = 10000
    auto_re_evolve: bool = False
    re_evolve_interval_bars: int = 48
    re_evolve_lookback_bars: int = 300
    loss_trigger_count: int = 3
    hot_swap_min_improvement: float = 0.05
    enable_reflection_memory: bool = True


class ExperimentConfig(Frozen):
    name: str
    data: DataConfig
    costs: CostConfig
    risk: RiskConfig
    gates: GateConfig
    judge: JudgeConfig
    wfa: WFAConfig
    ga: GAConfig
    reward: RewardConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    feature_version: str = "features-1"
    schema_version: str = "chromosome-1"

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(json.loads(Path(path).read_text()))
