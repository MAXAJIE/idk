"""Typed, symbolic chromosome schema. Genes are enums or bounded numbers grouped
by family; a chromosome is data, never code or prompts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

GENE_SPEC: dict[str, dict[str, tuple]] = {
    "topology": {"entry_mode": ("enum", ["trend_follow", "breakout", "mean_revert"]),
                 "compose": ("enum", ["all", "majority"])},
    "amt": {"use_value_area": ("bool",), "va_policy": ("enum", ["accept_outside", "fade_extremes"])},
    "smc": {"require_structure": ("bool",), "use_fvg": ("bool",), "respect_sweep": ("bool",)},
    "trend": {"ema_fast": ("int", 5, 30), "ema_slow": ("int", 20, 120), "rsi_guard": ("float", 55.0, 85.0)},
    "entry": {"max_chase_atr": ("float", 0.5, 3.0), "skip_high_vol": ("bool",)},
    "stop_target": {"stop_method": ("enum", ["atr", "structure"]), "stop_atr": ("float", 0.8, 4.0),
                    "target_method": ("enum", ["rr", "snr"]), "rr": ("float", 1.0, 5.0),
                    "max_hold_bars": ("int", 6, 120)},
    "sizing": {"risk_frac": ("float", 0.002, 0.02)},
    "cost": {"max_cost_to_reward": ("float", 0.05, 0.5)},
    "discussion": {"agreement_threshold": ("float", 0.0, 0.8), "mapping": ("enum", ["strict", "lenient"])},
}


def gene_distance(a: "Chromosome", b: "Chromosome") -> float:
    diff = n = 0.0
    for fam, genes in GENE_SPEC.items():
        for g, spec in genes.items():
            n += 1
            x, y = a.genes[fam][g], b.genes[fam][g]
            diff += abs(x - y) / (spec[2] - spec[1]) if spec[0] in ("int", "float") else float(x != y)
    return diff / n


@dataclass
class Chromosome:
    genes: dict[str, dict[str, Any]]
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    operator: str = "init"
    changed: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    state: str = "ACTIVE"
    schema_version: str = "chromosome-1"

    @property
    def id(self) -> str:
        blob = json.dumps(self.genes, sort_keys=True) + "|" + ",".join(self.parents) + f"|{self.generation}|{self.operator}"
        return "chr_" + hashlib.sha256(blob.encode()).hexdigest()[:10]

    def g(self, fam: str, gene: str) -> Any:
        return self.genes[fam][gene]

    def to_dict(self) -> dict:
        return {"id": self.id, "genes": self.genes, "generation": self.generation, "parents": self.parents,
                "operator": self.operator, "changed": self.changed, "repairs": self.repairs,
                "state": self.state, "schema_version": self.schema_version}


def validate(genes: dict) -> list[str]:
    errs = []
    for fam, spec in GENE_SPEC.items():
        if fam not in genes:
            errs.append(f"missing family {fam}")
            continue
        for g, s in spec.items():
            v = genes[fam].get(g)
            if s[0] == "enum" and v not in s[1]:
                errs.append(f"{fam}.{g}={v!r} not in enum")
            elif s[0] == "bool" and not isinstance(v, bool):
                errs.append(f"{fam}.{g} not bool")
            elif s[0] in ("int", "float") and (not isinstance(v, (int, float)) or isinstance(v, bool) or not s[1] <= v <= s[2]):
                errs.append(f"{fam}.{g}={v!r} out of [{s[1]},{s[2]}]")
    return errs


def repair(genes: dict) -> tuple[dict, list[str]]:
    """Deterministic, logged repair of numeric genes. Enum/bool errors are
    not repairable; callers discard such offspring."""
    log = []
    out = json.loads(json.dumps(genes))
    for fam, spec in GENE_SPEC.items():
        for g, s in spec.items():
            if s[0] in ("int", "float"):
                v = out[fam][g]
                c = min(max(v, s[1]), s[2])
                c = int(round(c)) if s[0] == "int" else round(float(c), 4)
                if c != v:
                    log.append(f"clamp {fam}.{g} {v}->{c}")
                out[fam][g] = c
    t = out["trend"]
    if t["ema_fast"] >= t["ema_slow"]:
        new_slow = min(GENE_SPEC["trend"]["ema_slow"][2], t["ema_fast"] + 5)
        log.append(f"ema_slow {t['ema_slow']}->{new_slow} (must exceed ema_fast)")
        t["ema_slow"] = new_slow
    return out, log
