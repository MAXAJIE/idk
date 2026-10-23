"""Generational population management (no mid-generation spawning)."""
from __future__ import annotations

import random

from app.genetic.operators import offspring, random_chromosome
from app.genetic.schema import Chromosome, gene_distance


def initial(rng: random.Random, n: int) -> list[Chromosome]:
    return [random_chromosome(rng) for _ in range(n)]


def tournament(rng: random.Random, pool: list[tuple[Chromosome, float]], k: int) -> Chromosome:
    picks = [pool[rng.randrange(len(pool))] for _ in range(min(k, len(pool)))]
    return max(picks, key=lambda x: x[1])[0]


def select_diverse_elite(ranked: list[tuple[Chromosome, float]], k: int, min_dist: float = 0.08) -> list[Chromosome]:
    elite: list[Chromosome] = []
    for c, _ in ranked:
        if all(gene_distance(c, e) >= min_dist for e in elite):
            elite.append(c)
        if len(elite) == k:
            break
    return elite


def next_generation(rng: random.Random, scored: list[tuple[Chromosome, float]], archive: list[Chromosome],
                    n: int, gen: int, ga) -> tuple[list[Chromosome], int]:
    """Parents only from non-failed individuals plus the elite archive.
    Returns (population of exactly n, number of discarded invalid offspring)."""
    survivors = [(c, f) for c, f in scored if c.state != "FAILED_RISK"]
    floor = min((f for _, f in survivors), default=0.0)
    pool = survivors + [(c, floor) for c in archive]
    if not pool:
        return [random_chromosome(rng, gen) for _ in range(n)], 0
    out: list[Chromosome] = []
    discarded = 0
    while len(out) < n:
        p1 = tournament(rng, pool, ga.tournament_k)
        p2 = tournament(rng, pool, ga.tournament_k)
        # Anti-incest / anti-clone check: prevent identical parents from collapsing diversity
        if len(pool) > 2 and gene_distance(p1, p2) < 0.04:
            p2 = tournament(rng, pool, ga.tournament_k)
        child = offspring(rng, p1, p2, gen, ga.crossover_rate, ga.mutation_rate, ga.operator_version)
        if child is None:
            discarded += 1
            continue
        out.append(child)
    return out, discarded
