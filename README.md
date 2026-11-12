# Monte Carlo Survival Forecaster (v2)

## Overview
This entry forecasts the top 5 new Solana coins by market cap, 24 hours out, using a Monte Carlo survival model. It is built entirely with Python 3.13 standard library routines, runs deterministically with a fixed seed, and uses no network access.

## Model Description
- **Monte Carlo survival simulation**: For each candidate coin, the model simulates many future paths. Survival probability comes from an exponential hazard (per-10-minute survival ~0.97, adjusted by a coin's recent growth and staleness). Simulated market caps grow with a horizon-dependent drift and lognormal noise.
- **Newcomer field**: At long horizons most of the top 5 are coins not yet in the history. The model reserves probability mass for unnamed newcomers via a retention factor that decays with horizon, so the forecast does not over-commit to coins that will not survive.
- **Order probabilities**: Each place's named probabilities use a floor of 0.002 per coin and a softmax temperature that grows with horizon, so long-horizon forecasts stay diffuse instead of collapsing onto one coin.
- **Size distributions**: Each place's size distribution is a 3-component mixture over log10 market cap in SOL: 55% Monte Carlo Gaussian (mean from simulated caps, std grows with horizon), 30% broad Gaussian centered at 3.25 log10 SOL (std 0.35), 15% uniform over the 82 bins. Components are mixed as true probabilities (density x bin width 0.05), floored at 0.0018 per bin after normalization so no bin is ever scored at the absolute floor.
- **Top 5 selection**: Coins are chosen by maximizing total probability weighted by 1 / sqrt(place).

Inputs are read from `$MAKEFUN_INPUT_DIR/task.json` and `$MAKEFUN_INPUT_DIR/history.ndjson`, and outputs are written to `$MAKEFUN_OUTPUT_DIR/forecast.json` with keys `{"probabilities", "top5", "sizes"}`.

## Execution
```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Backtest Results & Caveats
Evaluated with a faithful backtest that mimics the real practice setup (truncated history + answers at the task reads). Pooled mean log score **-36.385** across 10 reads in 4 splits:

- **Split A (history 1 mark, 4 reads)**: -39.332
- **Split B (history 2 marks, 3 reads)**: -40.064
- **Split C (history 3 marks, 2 reads)**: -29.244
- **Split D (history 4 marks, 1 read)**: -27.839

*Caveat*: the faithful backtest is a proxy, not the real practice score. The real practice uses 5 marks of history and reads at 10-70 minute horizons; the backtest splits use 1-4 marks and shorter horizons. An oracle analysis (size distribution built from the actual answer caps) shows the pooled proxy has a ceiling near -34 even with perfect size knowledge, because at long horizons 4 of the top 5 coins are newcomers that cannot be named. The true practice score can only be measured by submitting to the live platform.
