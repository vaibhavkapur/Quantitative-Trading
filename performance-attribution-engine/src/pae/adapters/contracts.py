"""Clean, documented adapter contracts isolating upstream foundations and optional stacks.

IP & Licensing Boundary Note:
Per plan.md Section 2, external/personal foundation repositories
(Trading-Software/09-transaction-cost-analysis, High-Frequency-Trading-Engine,
Quantum-Trading/02-hybrid-quantum-portfolio-optimizer) are NOT copied directly.
Instead, PAE defines clean, versioned adapter contracts so verified upstream feeds
can be mapped into canonical domain models without coupling business logic to external code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pae.domain.models import TradeExecution
from pae.errors import ValidationError


class UpstreamTCAAdapter:
    """Maps generic TCA execution-log records (e.g. project 09-transaction-cost-analysis) into TradeExecution."""

    contract_version = "tca-adapter-v1"

    def adapt_records(
        self,
        records: List[Dict[str, Any]],
        *,
        portfolio_id: str,
        default_currency: str = "USD",
    ) -> List[TradeExecution]:
        out: List[TradeExecution] = []
        for idx, rec in enumerate(records):
            try:
                qty = float(rec["filled_qty"])
                order_qty = float(rec.get("order_qty", qty))
                out.append(
                    TradeExecution(
                        trade_id=str(rec["fill_id"]),
                        order_id=str(rec.get("parent_order_id", f"ORD-{idx}")),
                        trade_date=str(rec["trade_date"]),
                        execution_ts_utc=str(rec["fill_ts_utc"]),
                        portfolio_id=portfolio_id,
                        strategy_id=str(rec["strategy_id"]),
                        instrument_id=str(rec["symbol"]),
                        sector=str(rec.get("sector", "UNCLASSIFIED")),
                        side=str(rec["side"]).upper(),
                        order_quantity=order_qty,
                        quantity=qty,
                        local_currency=str(rec.get("currency", default_currency)).upper(),
                        decision_price_local=float(rec["decision_px"]),
                        arrival_price_local=float(rec.get("arrival_px", rec["decision_px"])),
                        execution_price_local=float(rec["exec_px"]),
                        close_price_local=float(rec.get("close_px", rec["exec_px"])),
                        benchmark_vwap_local=(
                            float(rec["vwap_px"]) if rec.get("vwap_px") is not None else None
                        ),
                        commission_local=float(rec.get("commission", 0.0)),
                        exchange_fee_local=float(rec.get("exchange_fee", 0.0)),
                        tax_local=float(rec.get("tax", 0.0)),
                        venue=str(rec.get("venue", "UPSTREAM_TCA")),
                        availability_utc=str(rec.get("availability_utc", rec["fill_ts_utc"])),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Malformed upstream TCA record at index {idx}: {exc}",
                    details={"index": idx, "adapter": self.contract_version},
                ) from exc
        return out


class HFTFillAdapter:
    """Maps low-latency fill events (e.g. High-Frequency-Trading-Engine) into canonical TradeExecution."""

    contract_version = "hft-fill-adapter-v1"

    def adapt_fills(
        self,
        fills: List[Dict[str, Any]],
        *,
        portfolio_id: str,
        strategy_id: str,
        sector_lookup: Optional[Dict[str, str]] = None,
    ) -> List[TradeExecution]:
        sectors = sector_lookup or {}
        out: List[TradeExecution] = []
        for idx, fl in enumerate(fills):
            try:
                sym = str(fl["instrument"])
                exec_px = float(fl["price"])
                mid_px = float(fl.get("arrival_mid_price", exec_px))
                qty = float(fl["qty"])
                out.append(
                    TradeExecution(
                        trade_id=str(fl["exec_id"]),
                        order_id=str(fl["order_id"]),
                        trade_date=str(fl["ts_utc"])[:10],
                        execution_ts_utc=str(fl["ts_utc"]),
                        portfolio_id=portfolio_id,
                        strategy_id=str(fl.get("strategy_id", strategy_id)),
                        instrument_id=sym,
                        sector=sectors.get(sym, str(fl.get("sector", "UNCLASSIFIED"))),
                        side="BUY" if str(fl["side"]).upper() in {"B", "BUY", "1"} else "SELL",
                        order_quantity=float(fl.get("leaves_qty", 0.0)) + qty,
                        quantity=qty,
                        local_currency=str(fl.get("currency", "USD")).upper(),
                        decision_price_local=float(fl.get("signal_price", mid_px)),
                        arrival_price_local=mid_px,
                        execution_price_local=exec_px,
                        close_price_local=float(fl.get("eod_close", exec_px)),
                        commission_local=float(fl.get("fee", 0.0)),
                        venue=str(fl.get("venue", "HFT_SIM")),
                        availability_utc=str(fl["ts_utc"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Malformed HFT fill record at index {idx}: {exc}",
                    details={"index": idx, "adapter": self.contract_version},
                ) from exc
        return out


class PortfolioOptimizerTargetAdapter:
    """Normalizes target weights from portfolio optimizers (e.g. Quantum-Trading project 02) for rebalance drift checks."""

    contract_version = "optimizer-target-adapter-v1"

    def compute_rebalance_drift(
        self,
        target_weights: Dict[str, float],
        actual_end_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        all_keys = sorted(set(target_weights.keys()) | set(actual_end_weights.keys()))
        drifts = {
            k: actual_end_weights.get(k, 0.0) - target_weights.get(k, 0.0)
            for k in all_keys
        }
        l1_drift = sum(abs(v) for v in drifts.values())
        max_abs_drift = max((abs(v) for v in drifts.values()), default=0.0)
        return {
            "adapter_version": self.contract_version,
            "target_weight_sum": sum(target_weights.values()),
            "actual_weight_sum": sum(actual_end_weights.values()),
            "l1_weight_drift": l1_drift,
            "max_abs_weight_drift": max_abs_drift,
            "by_entity_drift": drifts,
        }


class OptionalPolarsAdapter:
    """Optional Polars DataFrame bridge (zero-dependency fallback when polars is absent)."""

    @staticmethod
    def is_available() -> bool:
        try:
            import polars  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def to_polars_or_rows(rows: List[Dict[str, Any]]) -> Any:
        try:
            import polars as pl

            return pl.DataFrame(rows)
        except ImportError:
            return rows
