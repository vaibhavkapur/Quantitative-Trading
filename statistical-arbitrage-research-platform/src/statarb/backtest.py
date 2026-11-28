"""Walk-forward backtest engine with realistic costs, borrow modeling, and exact reconciliation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from statarb.cointegration import (
    FoldSelectionResult,
    select_spreads_on_training_window,
)
from statarb.config import PlatformConfig, WalkForwardConfig
from statarb.exceptions import (
    BorrowUnavailableError,
    LeakageViolationError,
    MissingDataError,
    ReconciliationError,
)
from statarb.signals import (
    SpreadPositionState,
    compute_target_leg_weights,
    compute_zscore_causal,
    step_spread_signal,
)
from statarb.universe import UniverseFilterResult, filter_universe


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_days: int
    n_train_days: int
    n_test_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "embargo_days": self.embargo_days,
            "n_train_days": self.n_train_days,
            "n_test_days": self.n_test_days,
        }


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    date: date
    fold_id: int
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: float  # signed shares (+ buy, - sell)
    price_adj: float
    notional: float
    commission: float
    slippage: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "date": str(self.date),
            "fold_id": self.fold_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": round(self.quantity, 8),
            "price_adj": round(self.price_adj, 6),
            "notional": round(self.notional, 4),
            "commission": round(self.commission, 6),
            "slippage": round(self.slippage, 6),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PositionSnapshot:
    date: date
    fold_id: int
    symbol: str
    quantity: float
    price_adj: float
    market_value: float


@dataclass(frozen=True)
class DailyEquityRecord:
    date: date
    fold_id: int
    equity: float
    cash: float
    holdings_value: float
    gross_notional: float
    net_notional: float
    daily_return_net: float
    daily_return_gross: float
    commission: float
    slippage: float
    borrow_cost: float
    turnover: float
    active_spreads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": str(self.date),
            "fold_id": self.fold_id,
            "equity": round(self.equity, 4),
            "cash": round(self.cash, 4),
            "holdings_value": round(self.holdings_value, 4),
            "gross_notional": round(self.gross_notional, 4),
            "net_notional": round(self.net_notional, 4),
            "daily_return_net": round(self.daily_return_net, 8),
            "daily_return_gross": round(self.daily_return_gross, 8),
            "commission": round(self.commission, 6),
            "slippage": round(self.slippage, 6),
            "borrow_cost": round(self.borrow_cost, 6),
            "turnover": round(self.turnover, 6),
            "active_spreads": self.active_spreads,
        }


@dataclass(frozen=True)
class ReconciliationSummary:
    passed: bool
    total_trades: int
    total_position_checks: int
    total_equity_checks: int
    max_position_diff: float
    max_equity_diff: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_trades": self.total_trades,
            "total_position_checks": self.total_position_checks,
            "total_equity_checks": self.total_equity_checks,
            "max_position_diff": self.max_position_diff,
            "max_equity_diff": self.max_equity_diff,
        }


@dataclass(frozen=True)
class FoldExecutionRecord:
    fold: WalkForwardFold
    universe: UniverseFilterResult
    selection: FoldSelectionResult
    oos_net_return: float
    oos_gross_return: float
    n_trades: int


@dataclass(frozen=True)
class BacktestResult:
    folds: tuple[FoldExecutionRecord, ...]
    daily_equity: tuple[DailyEquityRecord, ...]
    trades: tuple[TradeRecord, ...]
    positions: tuple[PositionSnapshot, ...]
    reconciliation: ReconciliationSummary
    diagnostics: dict[str, int]
    config_hash: str


def generate_walk_forward_folds(
    dates: list[date],
    cfg: WalkForwardConfig,
) -> list[WalkForwardFold]:
    """Generate non-overlapping walk-forward folds with explicit train/test boundaries."""
    sorted_dates = sorted(set(dates))
    n = len(sorted_dates)
    min_needed = cfg.train_window_days + cfg.embargo_days + cfg.test_window_days
    if n < min_needed:
        raise MissingDataError(
            f"Insufficient trading dates ({n}) for walk-forward schedule requiring at least "
            f"{min_needed} days (train={cfg.train_window_days}, embargo={cfg.embargo_days}, "
            f"test={cfg.test_window_days})."
        )

    folds: list[WalkForwardFold] = []
    fold_id = 1
    train_end_idx = cfg.train_window_days - 1

    while True:
        train_start_idx = (
            0 if cfg.window_type == "expanding" else (train_end_idx - cfg.train_window_days + 1)
        )
        test_start_idx = train_end_idx + 1 + cfg.embargo_days
        test_end_idx = test_start_idx + cfg.test_window_days - 1
        if test_end_idx >= n:
            break

        train_start = sorted_dates[train_start_idx]
        train_end = sorted_dates[train_end_idx]
        test_start = sorted_dates[test_start_idx]
        test_end = sorted_dates[test_end_idx]

        if not (train_end < test_start):
            raise LeakageViolationError(
                f"Fold {fold_id} has overlapping train_end ({train_end}) "
                f"and test_start ({test_start})"
            )

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=cfg.embargo_days,
                n_train_days=train_end_idx - train_start_idx + 1,
                n_test_days=test_end_idx - test_start_idx + 1,
            )
        )
        fold_id += 1
        train_end_idx += cfg.step_days

    if not folds:
        raise MissingDataError("Walk-forward configuration produced zero valid folds")
    return folds


def verify_trade_position_reconciliation(
    trades: list[TradeRecord],
    positions: list[PositionSnapshot],
    daily_equity: list[DailyEquityRecord],
    initial_capital: float,
    *,
    position_tol: float = 1e-7,
    equity_tol: float = 1e-4,
) -> ReconciliationSummary:
    """Independently replay all trades & cashflows and verify exact reconciliation."""
    trades_by_date: dict[date, list[TradeRecord]] = {}
    for tr in trades:
        trades_by_date.setdefault(tr.date, []).append(tr)

    pos_by_date: dict[date, dict[str, PositionSnapshot]] = {}
    for snap in positions:
        pos_by_date.setdefault(snap.date, {})[snap.symbol] = snap

    replayed_shares: dict[str, float] = {}
    replayed_cash = float(initial_capital)
    max_pos_diff = 0.0
    max_eq_diff = 0.0
    pos_checks = 0
    eq_checks = 0

    for eq in daily_equity:
        d = eq.date
        replayed_cash -= eq.borrow_cost
        for tr in trades_by_date.get(d, []):
            replayed_shares[tr.symbol] = replayed_shares.get(tr.symbol, 0.0) + tr.quantity
            replayed_cash -= tr.quantity * tr.price_adj + tr.commission + tr.slippage

        day_snaps = pos_by_date.get(d, {})
        all_syms = set(replayed_shares.keys()) | set(day_snaps.keys())
        replayed_holdings = 0.0

        for sym in sorted(all_syms):
            rep_q = replayed_shares.get(sym, 0.0)
            snap = day_snaps.get(sym)
            snap_q = snap.quantity if snap is not None else 0.0
            diff = abs(rep_q - snap_q)
            max_pos_diff = max(max_pos_diff, diff)
            pos_checks += 1
            if diff > position_tol:
                raise ReconciliationError(
                    f"Position reconciliation failure on {d} for {sym}: "
                    f"replayed={rep_q:.8f}, snapshot={snap_q:.8f}, diff={diff:.8e}"
                )
            if snap is not None:
                replayed_holdings += rep_q * snap.price_adj

        replayed_equity = replayed_cash + replayed_holdings
        eq_diff = abs(replayed_equity - eq.equity)
        max_eq_diff = max(max_eq_diff, eq_diff)
        eq_checks += 1
        if eq_diff > equity_tol:
            raise ReconciliationError(
                f"Equity reconciliation failure on {d}: "
                f"replayed={replayed_equity:.6f}, ledger={eq.equity:.6f}, diff={eq_diff:.6e}"
            )

    return ReconciliationSummary(
        passed=True,
        total_trades=len(trades),
        total_position_checks=pos_checks,
        total_equity_checks=eq_checks,
        max_position_diff=max_pos_diff,
        max_equity_diff=max_eq_diff,
    )


def _index_market_tables(
    prices: pl.DataFrame,
    borrow: pl.DataFrame,
) -> tuple[
    list[date],
    dict[date, dict[str, float]],
    dict[date, dict[str, tuple[bool, float, float]]],
]:
    all_dates = prices["date"].unique().sort().to_list()
    price_lookup: dict[date, dict[str, float]] = {}
    for row in prices.select(["date", "symbol", "close_adj"]).iter_rows():
        d, sym, px = row
        price_lookup.setdefault(d, {})[sym] = float(px)

    borrow_lookup: dict[date, dict[str, tuple[bool, float, float]]] = {}
    for row in borrow.select(
        ["date", "symbol", "borrow_available", "borrow_fee_bps_annual", "max_short_notional"]
    ).iter_rows():
        d, sym, avail, fee, cap = row
        borrow_lookup.setdefault(d, {})[sym] = (bool(avail), float(fee), float(cap))

    return all_dates, price_lookup, borrow_lookup


def run_walk_forward_backtest(
    prices: pl.DataFrame,
    borrow: pl.DataFrame,
    cfg: PlatformConfig,
    *,
    zero_cost: bool = False,
) -> BacktestResult:
    """Execute full point-in-time walk-forward backtest with strict separation of train/test."""
    all_dates, price_lookup, borrow_lookup = _index_market_tables(prices, borrow)
    folds = generate_walk_forward_folds(all_dates, cfg.walk_forward)

    commission_bps = 0.0 if zero_cost else cfg.execution.commission_bps
    slippage_bps = 0.0 if zero_cost else cfg.execution.slippage_bps
    default_borrow_bps = 0.0 if zero_cost else cfg.execution.default_borrow_fee_bps_annual
    ann_days = float(cfg.evaluation.annualization_days)

    cash = float(cfg.execution.initial_capital)
    shares: dict[str, float] = {}
    last_known_price: dict[str, float] = {}

    trades: list[TradeRecord] = []
    positions: list[PositionSnapshot] = []
    daily_equity: list[DailyEquityRecord] = []
    fold_records: list[FoldExecutionRecord] = []

    diagnostics: dict[str, int] = {
        "folds_executed": 0,
        "folds_with_selected_spreads": 0,
        "spread_entries": 0,
        "spread_exits_mean_revert": 0,
        "spread_exits_stop_loss": 0,
        "spread_exits_time_stop": 0,
        "borrow_blocked_entries": 0,
        "borrow_forced_recalls": 0,
        "borrow_capped_trades": 0,
        "missing_price_skips": 0,
        "delist_liquidations": 0,
        "fold_end_flattens": 0,
    }
    trade_seq = 0

    for fold in folds:
        diagnostics["folds_executed"] += 1

        # STRICT BOUNDARY: pass ONLY rows with date <= fold.train_end to universe & selection
        train_slice = prices.filter(pl.col("date") <= fold.train_end)
        if train_slice["date"].max() > fold.train_end:
            raise LeakageViolationError(
                f"Train slice leaked beyond fold {fold.fold_id} train_end={fold.train_end}"
            )

        uni = filter_universe(
            train_slice,
            formation_start=fold.train_start,
            formation_end=fold.train_end,
            cfg=cfg.universe,
            max_allowed_date=fold.train_end,
        )
        sel = select_spreads_on_training_window(
            train_slice,
            eligible_symbols=uni.eligible_symbols,
            sector_map=uni.sector_map,
            train_start=fold.train_start,
            train_end=fold.train_end,
            cfg=cfg.selection,
            use_log_prices=cfg.data.use_log_prices,
            same_sector_only=cfg.universe.same_sector_only,
            max_allowed_date=fold.train_end,
        )

        models = sel.selected_models
        if models:
            diagnostics["folds_with_selected_spreads"] += 1

        # Seed spread histories from training window for causal rolling z-score mode
        spread_histories: dict[str, list[float]] = {m.spread_id: [] for m in models}
        train_dates = [d for d in all_dates if fold.train_start <= d <= fold.train_end]
        for d in train_dates:
            day_px = price_lookup.get(d, {})
            for m in models:
                if all(s in day_px for s in m.symbols):
                    spread_histories[m.spread_id].append(m.compute_spread_value(day_px))

        # Per-spread state machine and executed active directions for this fold
        signal_states: dict[str, SpreadPositionState] = {
            m.spread_id: SpreadPositionState() for m in models
        }
        active_spread_weights: dict[str, dict[str, float]] = {m.spread_id: {} for m in models}

        # Execution delay queue: items are (target_spread_directions, reason_tag)
        delay = cfg.execution.execution_delay_days
        pending_queue: deque[tuple[dict[str, int], str]] = deque()
        for _ in range(delay - 1):
            pending_queue.append(({m.spread_id: 0 for m in models}, "INIT_FLAT"))

        test_dates = [d for d in all_dates if fold.test_start <= d <= fold.test_end]
        per_spread_gross = (
            min(cfg.execution.max_pair_weight, cfg.execution.gross_leverage_target / len(models))
            if models
            else 0.0
        )

        fold_start_equity = (
            daily_equity[-1].equity if daily_equity else float(cfg.execution.initial_capital)
        )
        fold_trades_start = len(trades)
        fold_gross_cum = 1.0

        for idx_in_fold, d in enumerate(test_dates):
            day_px = price_lookup.get(d, {})
            for sym, px in day_px.items():
                last_known_price[sym] = px
            day_borrow = borrow_lookup.get(d, {})

            prev_equity = (
                daily_equity[-1].equity if daily_equity else float(cfg.execution.initial_capital)
            )

            # 1. Accrue daily short borrow cost on overnight positions
            borrow_cost_today = 0.0
            for sym, qty in list(shares.items()):
                if qty < -1e-12:
                    px = day_px.get(sym, last_known_price.get(sym, 0.0))
                    b_info = day_borrow.get(sym)
                    fee_bps = (
                        0.0
                        if zero_cost
                        else (b_info[1] if b_info is not None else default_borrow_bps)
                    )
                    borrow_cost_today += abs(qty) * px * (fee_bps / 10_000.0) / ann_days
            cash -= borrow_cost_today

            # Mark-to-market equity before today's trades (post-borrow)
            holdings_pre = sum(
                qty * day_px.get(sym, last_known_price.get(sym, 0.0)) for sym, qty in shares.items()
            )
            equity_pre = cash + holdings_pre

            # 2. Execute scheduled target positions for today
            is_last_day = idx_in_fold == len(test_dates) - 1
            if is_last_day:
                scheduled_dirs = {m.spread_id: 0 for m in models}
                exec_reason = "FOLD_END_FLATTEN"
                if any(abs(q) > 1e-10 for q in shares.values()):
                    diagnostics["fold_end_flattens"] += 1
            elif pending_queue:
                scheduled_dirs, exec_reason = pending_queue.popleft()
            else:
                scheduled_dirs = {m.spread_id: 0 for m in models}
                exec_reason = "HOLD"

            # Check borrow recalls on currently held shorts & borrow availability on new shorts
            target_symbol_dollars: dict[str, float] = {}
            new_active_spread_weights: dict[str, dict[str, float]] = {}

            for m in models:
                desired_dir = scheduled_dirs.get(m.spread_id, 0)
                if desired_dir == 0:
                    new_active_spread_weights[m.spread_id] = {}
                    continue

                if not all(s in day_px for s in m.symbols):
                    diagnostics["missing_price_skips"] += 1
                    # Unwind spread cleanly if a constituent symbol is missing/delisted
                    signal_states[m.spread_id] = SpreadPositionState()
                    new_active_spread_weights[m.spread_id] = {}
                    continue

                leg_weights = compute_target_leg_weights(m, desired_dir, per_spread_gross)
                was_open = bool(active_spread_weights.get(m.spread_id))

                # Evaluate borrow constraints across all short legs of this spread
                scale_factor = 1.0
                blocked = False
                recalled = False

                if cfg.execution.enforce_borrow_availability:
                    for sym, w in leg_weights.items():
                        if w < 0.0:
                            b_info = day_borrow.get(sym)
                            b_avail = b_info[0] if b_info is not None else True
                            max_short = b_info[2] if b_info is not None else float("inf")
                            req_short_notional = abs(w) * max(equity_pre, 0.0)

                            if not b_avail or max_short <= 0.0:
                                if cfg.execution.strict_borrow_errors:
                                    raise BorrowUnavailableError(
                                        f"Borrow unavailable for {sym} on {d} "
                                        f"(spread={m.spread_id})"
                                    )
                                if was_open and cfg.execution.recall_on_borrow_loss:
                                    recalled = True
                                else:
                                    blocked = True
                                break
                            if req_short_notional > max_short and req_short_notional > 0.0:
                                scale_factor = min(scale_factor, max_short / req_short_notional)

                if recalled:
                    diagnostics["borrow_forced_recalls"] += 1
                    signal_states[m.spread_id] = SpreadPositionState()
                    new_active_spread_weights[m.spread_id] = {}
                    continue
                if blocked:
                    diagnostics["borrow_blocked_entries"] += 1
                    signal_states[m.spread_id] = SpreadPositionState()
                    new_active_spread_weights[m.spread_id] = {}
                    continue

                if scale_factor < 1.0 - 1e-12:
                    diagnostics["borrow_capped_trades"] += 1
                    leg_weights = {s: w * scale_factor for s, w in leg_weights.items()}

                new_active_spread_weights[m.spread_id] = leg_weights
                for sym, w in leg_weights.items():
                    target_symbol_dollars[sym] = target_symbol_dollars.get(sym, 0.0) + w * max(
                        equity_pre, 0.0
                    )

            active_spread_weights = new_active_spread_weights

            # Convert target dollars to target shares and execute trades deterministically
            commission_today = 0.0
            slippage_today = 0.0
            traded_notional_today = 0.0

            symbols_to_rebalance = sorted(set(shares.keys()) | set(target_symbol_dollars.keys()))
            for sym in symbols_to_rebalance:
                px = day_px.get(sym)
                cur_q = shares.get(sym, 0.0)
                tgt_dollars = target_symbol_dollars.get(sym, 0.0)
                trade_reason = exec_reason
                if px is None:
                    if abs(cur_q) > 1e-10 and abs(tgt_dollars) <= 1e-9:
                        # Liquidate delisted / missing symbol at last known adjusted price
                        px = last_known_price.get(sym)
                        if px is None or px <= 0.0:
                            diagnostics["missing_price_skips"] += 1
                            continue
                        diagnostics["delist_liquidations"] += 1
                        trade_reason = "DELIST_LIQUIDATION"
                    else:
                        if abs(cur_q) > 1e-12:
                            diagnostics["missing_price_skips"] += 1
                        continue
                tgt_q = tgt_dollars / px if abs(tgt_dollars) > 1e-9 else 0.0
                dq = tgt_q - cur_q
                if abs(dq) <= 1e-10:
                    continue

                notional = abs(dq) * px
                comm = notional * (commission_bps / 10_000.0)
                slip = notional * (slippage_bps / 10_000.0)

                cash -= dq * px + comm + slip
                new_q = cur_q + dq
                if abs(new_q) <= 1e-10:
                    shares.pop(sym, None)
                else:
                    shares[sym] = new_q

                commission_today += comm
                slippage_today += slip
                traded_notional_today += notional

                trade_seq += 1
                trades.append(
                    TradeRecord(
                        trade_id=f"T{trade_seq:06d}",
                        date=d,
                        fold_id=fold.fold_id,
                        symbol=sym,
                        side="BUY" if dq > 0 else "SELL",
                        quantity=float(dq),
                        price_adj=float(px),
                        notional=float(notional),
                        commission=float(comm),
                        slippage=float(slip),
                        reason=trade_reason,
                    )
                )

            # 3. End-of-day mark-to-market and position snapshots
            holdings_post = 0.0
            gross_notional = 0.0
            net_notional = 0.0
            for sym in sorted(shares.keys()):
                qty = shares[sym]
                px = day_px.get(sym, last_known_price[sym])
                mv = qty * px
                holdings_post += mv
                gross_notional += abs(mv)
                net_notional += mv
                positions.append(
                    PositionSnapshot(
                        date=d,
                        fold_id=fold.fold_id,
                        symbol=sym,
                        quantity=float(qty),
                        price_adj=float(px),
                        market_value=float(mv),
                    )
                )

            equity_post = cash + holdings_post
            total_costs_today = commission_today + slippage_today + borrow_cost_today
            ret_net = (equity_post - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            ret_gross = (
                (equity_post + total_costs_today - prev_equity) / prev_equity
                if prev_equity > 0
                else 0.0
            )
            turnover = traded_notional_today / prev_equity if prev_equity > 0 else 0.0
            fold_gross_cum *= 1.0 + ret_gross

            active_spread_count = sum(1 for wmap in active_spread_weights.values() if wmap)
            daily_equity.append(
                DailyEquityRecord(
                    date=d,
                    fold_id=fold.fold_id,
                    equity=float(equity_post),
                    cash=float(cash),
                    holdings_value=float(holdings_post),
                    gross_notional=float(gross_notional),
                    net_notional=float(net_notional),
                    daily_return_net=float(ret_net),
                    daily_return_gross=float(ret_gross),
                    commission=float(commission_today),
                    slippage=float(slippage_today),
                    borrow_cost=float(borrow_cost_today),
                    turnover=float(turnover),
                    active_spreads=active_spread_count,
                )
            )

            # 4. Generate end-of-day signals for future execution (t + delay)
            if not is_last_day:
                next_dirs: dict[str, int] = {}
                actions_seen: set[str] = set()
                for m in models:
                    if not all(s in day_px for s in m.symbols):
                        next_dirs[m.spread_id] = signal_states[m.spread_id].position
                        continue
                    s_val = m.compute_spread_value(day_px)
                    spread_histories[m.spread_id].append(s_val)
                    z = compute_zscore_causal(
                        spread_histories[m.spread_id],
                        m.spread_mean,
                        m.spread_std,
                        cfg.signals,
                    )
                    new_st, trans = step_spread_signal(signal_states[m.spread_id], z, cfg.signals)
                    signal_states[m.spread_id] = new_st
                    next_dirs[m.spread_id] = new_st.position
                    actions_seen.add(trans.action)

                    if trans.action in ("ENTER_LONG", "ENTER_SHORT"):
                        diagnostics["spread_entries"] += 1
                    elif trans.action == "EXIT_MEAN_REVERT":
                        diagnostics["spread_exits_mean_revert"] += 1
                    elif trans.action == "EXIT_STOP_LOSS":
                        diagnostics["spread_exits_stop_loss"] += 1
                    elif trans.action == "EXIT_TIME_STOP":
                        diagnostics["spread_exits_time_stop"] += 1

                reason_tag = "SIGNAL_REBALANCE"
                if "EXIT_STOP_LOSS" in actions_seen:
                    reason_tag = "STOP_LOSS_OR_REBALANCE"
                pending_queue.append((next_dirs, reason_tag))

        fold_end_equity = daily_equity[-1].equity if daily_equity else fold_start_equity
        fold_net_ret = (fold_end_equity / fold_start_equity) - 1.0 if fold_start_equity > 0 else 0.0
        fold_records.append(
            FoldExecutionRecord(
                fold=fold,
                universe=uni,
                selection=sel,
                oos_net_return=float(fold_net_ret),
                oos_gross_return=float(fold_gross_cum - 1.0),
                n_trades=len(trades) - fold_trades_start,
            )
        )

    recon = verify_trade_position_reconciliation(
        trades=trades,
        positions=positions,
        daily_equity=daily_equity,
        initial_capital=cfg.execution.initial_capital,
    )

    return BacktestResult(
        folds=tuple(fold_records),
        daily_equity=tuple(daily_equity),
        trades=tuple(trades),
        positions=tuple(positions),
        reconciliation=recon,
        diagnostics=diagnostics,
        config_hash=cfg.config_hash(),
    )
