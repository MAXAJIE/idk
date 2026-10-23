"""Continuous learning, reflection memory, online re-evolution, and arena combat."""
from app.learning.reflection import ReflectionMemory, InhibitionVerdict
from app.learning.online_evolution import evaluate_rolling_challenger, ChallengerEvaluation
from app.learning.arena import (
    train_and_combat,
    auto_train_until_survivor,
    analyze_arena_result,
    ArenaResult,
    ArenaCombatant,
)

__all__ = [
    "ReflectionMemory",
    "InhibitionVerdict",
    "evaluate_rolling_challenger",
    "ChallengerEvaluation",
    "train_and_combat",
    "auto_train_until_survivor",
    "analyze_arena_result",
    "ArenaResult",
    "ArenaCombatant",
]
