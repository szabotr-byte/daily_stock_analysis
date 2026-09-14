"""Optional Kronos integration.

Kronos remains an optional dependency. The main DSA environment does not need
PyTorch just to run the ordinary scanner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class KronosForecast:
    score: float
    direction: str
    forecast: pd.DataFrame | None = None
    error: str = ""


class KronosAdapter:
    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        max_context: int = 512,
    ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_context = max_context
        self._predictor: Any = None

    def load(self) -> None:
        from model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        model = Kronos.from_pretrained(self.model_name)
        self._predictor = KronosPredictor(
            model,
            tokenizer,
            max_context=self.max_context,
        )

    def predict(
        self,
        candles: pd.DataFrame,
        pred_len: int = 20,
    ) -> KronosForecast:
        if self._predictor is None:
            self.load()

        required = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamps",
        }

        missing = sorted(required - set(candles.columns))

        if missing:
            raise ValueError(
                "Missing Kronos columns: " + ", ".join(missing)
            )

        x = candles.tail(self.max_context).copy()
        ts = pd.to_datetime(x["timestamps"])

        if len(ts) < 64:
            return KronosForecast(
                50.0,
                "neutral",
                error="insufficient candle history",
            )

        freq = pd.infer_freq(ts)

        y_ts = pd.date_range(
            ts.iloc[-1],
            periods=pred_len + 1,
            freq=freq or "D",
        )[1:]

        try:
            pred = self._predictor.predict(
                df=x[
                    ["open", "high", "low", "close", "volume"]
                ],
                x_timestamp=ts,
                y_timestamp=y_ts,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )

            last_close = float(x["close"].iloc[-1])
            final_close = float(pred["close"].iloc[-1])

            change = (
                (final_close / last_close - 1.0) * 100
                if last_close
                else 0.0
            )

            score = max(
                0.0,
                min(100.0, 50.0 + change * 5.0),
            )

            direction = (
                "bullish"
                if change > 1
                else "bearish"
                if change < -1
                else "neutral"
            )

            return KronosForecast(
                score,
                direction,
                pred,
            )

        except Exception as exc:
            return KronosForecast(
                50.0,
                "neutral",
                error=str(exc),
            )
