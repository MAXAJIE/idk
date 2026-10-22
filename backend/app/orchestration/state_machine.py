"""Experiment state machine; illegal transitions raise."""
from __future__ import annotations

TRANSITIONS = {
    "CREATED": {"DATA_VALIDATION"},
    "DATA_VALIDATION": {"FEATURE_BUILD", "ABORTED_INTEGRITY"},
    "FEATURE_BUILD": {"IS_EVOLUTION", "ABORTED_INTEGRITY"},
    "IS_EVOLUTION": {"CANDIDATE_FREEZE", "ABORTED_INTEGRITY"},
    "CANDIDATE_FREEZE": {"OOS_EVALUATION"},
    "OOS_EVALUATION": {"JUDGE_REVIEW", "ABORTED_INTEGRITY"},
    "JUDGE_REVIEW": {"FOLD_DONE"},
    "FOLD_DONE": {"IS_EVOLUTION", "WFA_GATE"},
    "WFA_GATE": {"HOLDOUT"},
    "HOLDOUT": {"COMPLETED"},
}
TERMINAL = {"COMPLETED", "ABORTED_INTEGRITY"}


class StateMachine:
    def __init__(self, on_change):
        self.state = "CREATED"
        self._on = on_change

    def to(self, new: str, **info):
        if new not in TRANSITIONS.get(self.state, set()):
            raise RuntimeError(f"illegal transition {self.state} -> {new}")
        old, self.state = self.state, new
        self._on(old, new, info)
