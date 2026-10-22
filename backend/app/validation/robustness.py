"""Parameter-neighbourhood sensitivity, evaluated on the IS segment only."""
from __future__ import annotations

import copy

from app.genetic.schema import GENE_SPEC, Chromosome, repair


def neighbours(c: Chromosome, rel: float = 0.1) -> list[Chromosome]:
    out = []
    for sign in (-1, 1):
        for fam in ("stop_target", "trend"):
            g = copy.deepcopy(c.genes)
            for gene, spec in GENE_SPEC[fam].items():
                if spec[0] in ("int", "float"):
                    g[fam][gene] = g[fam][gene] * (1 + sign * rel)
            g, _ = repair(g)
            out.append(Chromosome(g, generation=c.generation, parents=[c.id], operator=f"neighbour:{fam}:{sign}"))
    return out


def retention(base_net: float, neighbour_nets: list[float]) -> float:
    if base_net <= 0:
        return 0.0
    return round(sum(max(0.0, n) for n in neighbour_nets) / (len(neighbour_nets) * base_net), 4)
