"""GA operators. Mutation is mandatory every generation and does not consult the Judge."""
from __future__ import annotations

import copy
import random

from app.genetic.schema import GENE_SPEC, Chromosome, repair, validate


def random_gene(rng: random.Random, spec: tuple):
    if spec[0] == "enum":
        return rng.choice(spec[1])
    if spec[0] == "bool":
        return rng.random() < 0.5
    if spec[0] == "int":
        return rng.randint(spec[1], spec[2])
    return round(rng.uniform(spec[1], spec[2]), 4)


def random_chromosome(rng: random.Random, generation: int = 0) -> Chromosome:
    genes = {f: {g: random_gene(rng, s) for g, s in spec.items()} for f, spec in GENE_SPEC.items()}
    genes, log = repair(genes)
    return Chromosome(genes, generation=generation, operator="random", repairs=log)


def crossover(rng: random.Random, a: Chromosome, b: Chromosome) -> dict:
    """Family-level uniform crossover keeps gene groups coherent."""
    return {f: copy.deepcopy((a if rng.random() < 0.5 else b).genes[f]) for f in GENE_SPEC}


def mutate(rng: random.Random, genes: dict, rate: float) -> tuple[dict, list[str]]:
    genes = copy.deepcopy(genes)
    flat = [(f, g) for f, spec in GENE_SPEC.items() for g in spec]
    chosen = [fg for fg in flat if rng.random() < rate] or [rng.choice(flat)]  # at least one gene
    changed = []
    for f, g in chosen:
        s = GENE_SPEC[f][g]
        old = genes[f][g]
        if s[0] in ("int", "float"):
            step = (s[2] - s[1]) * 0.15
            new = old + rng.gauss(0, step)
            new = int(round(new)) if s[0] == "int" else round(new, 4)
            new = min(max(new, s[1]), s[2])
            if new == old:  # guarantee an effective change, staying in bounds
                d = 1 if s[0] == "int" else round(step / 4, 4)
                new = old + d if old + d <= s[2] else old - d
        elif s[0] == "bool":
            new = not old
        else:
            new = rng.choice([v for v in s[1] if v != old])
        genes[f][g] = new
        changed.append(f"{f}.{g}: {old}->{new}")
    return genes, changed


def offspring(rng: random.Random, a: Chromosome, b: Chromosome, gen: int, cx_rate: float, mut_rate: float,
              op_version: str) -> Chromosome | None:
    use_cx = rng.random() < cx_rate
    base = crossover(rng, a, b) if use_cx else copy.deepcopy(a.genes)
    genes, changed = mutate(rng, base, mut_rate)
    genes, log = repair(genes)
    if validate(genes):
        return None
    parents = [a.id, b.id] if use_cx else [a.id]
    op = f"{op_version}:{'crossover+' if use_cx else ''}mutation"
    return Chromosome(genes, generation=gen, parents=parents, operator=op, changed=changed, repairs=log)
