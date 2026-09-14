"""Low-float penny-stock screening and trade-plan scoring.

The scanner is deliberately provider-agnostic. Feed it a DataFrame containing
snapshot/technical/news-risk fields from the existing DSA providers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PennyScannerConfig:
    price_min: float = 1.0
    price_max: float = 5.0
    market_cap_max: float = 300_000_000.0
    float_max: float = 30_000_000.0
    relative_volume_min: float = 2.0
    gap_min_pct: float = 3.0
    min_premarket_volume: int = 100_000
    min_score: float = 60.0
    stop_atr_multiple: float = 1.0
    tp1_rr: float = 1.5
    tp2_rr: float = 2.5
    tp3_rr: float = 4.0


@dataclass
class ScanCandidate:
    ticker: str
    name: str
    price: float
    score: float
    catalyst_score: float
    volume_score: float
    float_score: float
    technical_score: float
    kronos_score: float
    dilution_penalty: float
    market_score: float
    entry: float | None
    stop: float | None
    tp1: float | None
    tp2: float | None
    tp3: float | None
    risk_reward_tp2: float | None
    catalyst: str
    risks: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PennyScanner:
    """Two-stage scanner: hard filters, then weighted score and trade plan."""

    def __init__(self, config: PennyScannerConfig | None = None) -> None:
        self.config = config or PennyScannerConfig()

    @staticmethod
    def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
        value = row.get(key, default)
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def _hard_filter(self, row: pd.Series) -> bool:
        c = self.config
        price = self._num(row, "price")
        market_cap = self._num(row, "market_cap", float("inf"))
        float_shares = self._num(row, "float_shares", float("inf"))
        rel_vol = self._num(row, "relative_volume")
        gap = self._num(row, "gap_pct")
        pm_vol = self._num(row, "premarket_volume")

        return (
            c.price_min <= price <= c.price_max
            and market_cap <= c.market_cap_max
            and float_shares <= c.float_max
            and rel_vol >= c.relative_volume_min
            and gap >= c.gap_min_pct
            and pm_vol >= c.min_premarket_volume
        )

    @staticmethod
    def _clip_score(value: float) -> float:
        return float(max(0.0, min(100.0, value)))

    def _score(self, row: pd.Series) -> tuple[float, dict[str, float]]:
        rel_vol = self._num(row, "relative_volume")
        float_shares = self._num(
            row, "float_shares", self.config.float_max
        )
        gap = self._num(row, "gap_pct")
        pm_vol = self._num(row, "premarket_volume")
        catalyst = self._num(row, "catalyst_score")
        kronos = self._num(row, "kronos_score")
        market = self._num(row, "market_score", 50.0)
        dilution = self._num(row, "dilution_risk")

        volume = self._clip_score(
            50
            + min(rel_vol, 10) * 5
            + min(pm_vol / 1_000_000, 5) * 5
        )

        float_score = self._clip_score(
            100 * (1 - float_shares / max(self.config.float_max, 1))
        )

        technical = self._clip_score(
            35
            + min(max(gap, 0), 20) * 2
            + (15 if self._num(row, "price_vs_vwap_pct") >= 0 else 0)
            + (10 if self._num(row, "ema9_above_ema20") else 0)
            + (10 if self._num(row, "above_prev_day_high") else 0)
        )

        catalyst = self._clip_score(catalyst)
        kronos = self._clip_score(kronos)
        market = self._clip_score(market)
        dilution_penalty = self._clip_score(dilution)

        total = (
            catalyst * 0.20
            + volume * 0.20
            + float_score * 0.15
            + technical * 0.15
            + kronos * 0.15
            + market * 0.05
            + (100 - dilution_penalty) * 0.10
        )

        return self._clip_score(total), {
            "catalyst": catalyst,
            "volume": volume,
            "float": float_score,
            "technical": technical,
            "kronos": kronos,
            "market": market,
            "dilution_penalty": dilution_penalty,
        }

    def _trade_plan(
        self, row: pd.Series
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        price = self._num(row, "price")
        vwap = self._num(row, "vwap")
        prev_high = self._num(row, "previous_day_high")
        atr = self._num(row, "atr", max(price * 0.05, 0.01))

        entry = (
            max(price, vwap, prev_high)
            if max(vwap, prev_high) > 0
            else price
        )

        stop = entry - self.config.stop_atr_multiple * atr

        if stop <= 0 or entry <= stop:
            return None, None, None, None, None, None

        risk = entry - stop

        return (
            entry,
            stop,
            entry + self.config.tp1_rr * risk,
            entry + self.config.tp2_rr * risk,
            entry + self.config.tp3_rr * risk,
            self.config.tp2_rr,
        )

    def scan(
        self, df: pd.DataFrame, limit: int = 10
    ) -> list[ScanCandidate]:
        if df.empty:
            return []

        required = {
            "ticker",
            "price",
            "market_cap",
            "float_shares",
            "relative_volume",
            "gap_pct",
            "premarket_volume",
        }

        missing = sorted(required - set(df.columns))

        if missing:
            raise ValueError(
                "Missing scanner columns: " + ", ".join(missing)
            )

        rows: list[ScanCandidate] = []

        for _, row in df.iterrows():
            if not self._hard_filter(row):
                continue

            score, parts = self._score(row)

            if score < self.config.min_score:
                continue

            entry, stop, tp1, tp2, tp3, rr = self._trade_plan(row)

            risks: list[str] = []

            if parts["dilution_penalty"] >= 50:
                risks.append("high dilution risk")

            if self._num(row, "float_shares") <= 5_000_000:
                risks.append("very low float")

            if self._num(row, "premarket_volume") < 500_000:
                risks.append("premarket liquidity may be thin")

            catalyst = str(row.get("catalyst", ""))

            reason = (
                "passed price/market-cap/float/relative-volume/gap "
                f"filters, technical score {parts['technical']:.0f}, "
                f"Kronos {parts['kronos']:.0f}"
            )

            rows.append(
                ScanCandidate(
                    ticker=str(row["ticker"]),
                    name=str(row.get("name", row["ticker"])),
                    price=self._num(row, "price"),
                    score=score,
                    catalyst_score=parts["catalyst"],
                    volume_score=parts["volume"],
                    float_score=parts["float"],
                    technical_score=parts["technical"],
                    kronos_score=parts["kronos"],
                    dilution_penalty=parts["dilution_penalty"],
                    market_score=parts["market"],
                    entry=entry,
                    stop=stop,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    risk_reward_tp2=rr,
                    catalyst=catalyst,
                    risks=risks,
                    reason=reason,
                )
            )

        rows.sort(key=lambda x: x.score, reverse=True)

        return rows[:limit]
