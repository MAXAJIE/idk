"""Deterministic ledger for one isolated perp account."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Position:
    trade_id: str
    side: int
    qty: float
    entry_mid: float
    entry_px: float
    stop: float
    target: float
    opened_t: int
    planned_risk: float
    entry_spread: float
    entry_slip: float
    entry_fee: float
    decision_id: str
    adoption: str
    funding: float = 0.0


@dataclass
class Trade:
    trade_id: str
    decision_id: str
    side: int
    qty: float
    opened_t: int
    closed_t: int
    entry_mid: float
    exit_mid: float
    entry_px: float
    exit_px: float
    exit_reason: str
    gross_mid_pnl: float
    spread_cost: float
    slippage_cost: float
    fees: float
    funding: float
    net_pnl: float
    planned_risk: float
    stop_compliant: bool
    bars_held: int
    adoption: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ledger:
    initial: float
    cash: float = 0.0
    position: Position | None = None
    trades: list[Trade] = field(default_factory=list)
    fees: float = 0.0
    funding: float = 0.0
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    exposure_curve: list[tuple[int, float]] = field(default_factory=list)
    peak: float = 0.0
    halted: bool = False
    raw_net: list[float] = field(default_factory=list)  # unrounded, for reconciliation

    def __post_init__(self):
        self.cash = self.initial
        self.peak = self.initial

    def equity(self, mark: float) -> float:
        p = self.position
        return self.cash + (p.side * p.qty * (mark - p.entry_px) if p else 0.0)

    def mark(self, t: int, price: float) -> float:
        eq = self.equity(price)
        self.peak = max(self.peak, eq)
        self.equity_curve.append((t, round(eq, 6)))
        p = self.position
        self.exposure_curve.append((t, round((p.qty * price / eq) if p and eq > 0 else 0.0, 6)))
        return eq

    def drawdown(self, price: float) -> float:
        return 1 - self.equity(price) / self.peak if self.peak > 0 else 0.0

    def open(self, pos: Position):
        assert self.position is None, "one position per account"
        self.cash -= pos.entry_fee
        self.fees += pos.entry_fee
        self.position = pos

    def apply_funding(self, rate: float, mark: float):
        p = self.position
        if not p:
            return
        amt = p.side * p.qty * mark * rate  # longs pay positive funding
        self.cash -= amt
        self.funding += amt
        p.funding += amt

    def close(self, t: int, exit_mid: float, exit_px: float, spread: float, slip: float, fee: float, reason: str) -> Trade:
        p = self.position
        assert p is not None
        exec_pnl = p.side * p.qty * (exit_px - p.entry_px)
        self.cash += exec_pnl - fee
        self.fees += fee
        fe = p.entry_fee + fee
        net = exec_pnl - fe - p.funding
        tr = Trade(p.trade_id, p.decision_id, p.side, p.qty, p.opened_t, t, p.entry_mid, exit_mid, p.entry_px,
                   exit_px, reason, round(p.side * p.qty * (exit_mid - p.entry_mid), 6),
                   round(p.entry_spread + spread, 6), round(p.entry_slip + slip, 6), round(fe, 6),
                   round(p.funding, 6), round(net, 6), round(p.planned_risk, 6),
                   net >= -1.5 * p.planned_risk, t - p.opened_t, p.adoption)
        self.trades.append(tr)
        self.raw_net.append(net)
        self.position = None
        return tr

    def reconcile(self) -> float:
        """Absolute discrepancy between cash and the trade-level decomposition."""
        expected = self.initial + sum(self.raw_net)
        if self.position:
            expected -= self.position.entry_fee + self.position.funding
        return abs(self.cash - expected)
