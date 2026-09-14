import pandas as pd

from src.penny_scanner.engine import PennyScanner


def test_low_float_breakout_passes():
    df = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "name": "Test Company",
                "price": 3.20,
                "market_cap": 120_000_000,
                "float_shares": 8_000_000,
                "relative_volume": 4.0,
                "gap_pct": 8.0,
                "premarket_volume": 500_000,
                "catalyst_score": 90,
                "kronos_score": 80,
                "dilution_risk": 10,
                "market_score": 70,
                "vwap": 3.10,
                "previous_day_high": 3.15,
                "atr": 0.20,
                "price_vs_vwap_pct": 3.2,
                "ema9_above_ema20": True,
                "above_prev_day_high": True,
                "catalyst": "positive company catalyst",
            }
        ]
    )

    result = PennyScanner().scan(df)

    assert len(result) == 1
    assert result[0].ticker == "TEST"
    assert result[0].score >= 60
    assert result[0].entry is not None
    assert result[0].stop is not None
    assert result[0].tp2 is not None


def test_bad_price_and_float_are_rejected():
    df = pd.DataFrame(
        [
            {
                "ticker": "BAD",
                "price": 7.50,
                "market_cap": 500_000_000,
                "float_shares": 60_000_000,
                "relative_volume": 5.0,
                "gap_pct": 10.0,
                "premarket_volume": 1_000_000,
            }
        ]
    )

    result = PennyScanner().scan(df)

    assert result == []
