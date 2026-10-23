import json
from pathlib import Path

import pytest

from app.config import ExperimentConfig

ROOT = Path(__file__).parents[1]


@pytest.fixture
def cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(json.loads((ROOT / "configs/default.json").read_text()))


@pytest.fixture
def small_cfg(cfg) -> ExperimentConfig:
    d = cfg.model_dump()
    d["data"]["n_bars"] = 1400
    d["wfa"] |= {"n_folds": 2, "is_bars": 500, "oos_bars": 200, "holdout_bars": 150}
    d["ga"] |= {"population": 6, "generations": 2, "elite_size": 2}
    d["gates"]["min_trades"] = 3
    return ExperimentConfig.model_validate(d)


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Tests never call a real LLM even if backend/.env holds a key."""
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
