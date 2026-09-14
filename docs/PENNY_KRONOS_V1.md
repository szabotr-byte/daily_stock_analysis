# Penny Scanner + Kronos V1

## Overview

This module adds an optional penny-stock scanner to the DSA project.

The scanner is designed for US stocks and focuses on low-float breakout setups.

## Hard Filters

Default V1 filters:

- Price: $1 to $5
- Market cap: below $300M
- Float: below 30M shares
- Relative volume: at least 2x
- Gap: at least 3%
- Premarket volume: at least 100,000 shares

A candidate must pass the hard filters before scoring.

## Scoring

The maximum score is 100 points.

- Catalyst: 20
- Volume: 20
- Float: 15
- Technical setup: 15
- Kronos confirmation: 15
- Dilution risk: 10
- Broader market: 5

Dilution risk is treated as a penalty.

## Technical Inputs

The scanner can use:

- Premarket high
- Previous day high
- VWAP
- EMA 9
- EMA 20
- EMA 50
- ATR
- Price versus VWAP
- Breakout confirmation

## Trade Plan

For candidates that pass the minimum score, the scanner can produce:

- Entry
- Stop loss
- TP1
- TP2
- TP3
- Risk/reward
- Score
- Catalyst
- Main risks

The default targets are based on R multiples:

- TP1: 1.5R
- TP2: 2.5R
- TP3: 4R

## Kronos

Kronos is an optional confirmation layer.

It analyzes historical OHLCV candlestick data and provides directional confirmation.

Kronos does not execute trades and does not place orders.

The Kronos integration is optional so the main DSA installation does not require PyTorch.

Default model:

NeoQuasar/Kronos-small

Default context:

512 candles

## Safety

The scanner is an analytical tool only.

It does not place orders automatically.

Kronos confirmation must never override the hard filters or risk controls.

## V1 Integration

The main implementation is located in:

src/penny_scanner/

Files:

- engine.py
- kronos_adapter.py
- __init__.py

Tests are located in:

tests/test_penny_scanner.py

The GitHub Actions test workflow is:

.github/workflows/penny-scanner-test.yml
