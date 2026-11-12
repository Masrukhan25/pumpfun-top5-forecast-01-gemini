# Horizon-Aware Heuristic Forecaster

## Overview
This entry provides probability forecasts and size distributions for cryptocurrency market rankings and capitalization profiles over dynamic time horizons. The model is built entirely using Python 3.13 standard library routines, operating deterministically with a fixed seed.

## Model Description
The forecaster uses a horizon-aware heuristic approach to evaluate candidates and project market dynamics:

- **Candidate Scoring**: For each target position, candidate coins (top 20 per place selected via softmax filtering) receive a composite score. The score combines:
  - Log market capitalization (weighted at 2.5)
  - Target rank distance penalty
  - Pool-venue and complete-curve feature bonuses
  - Damped momentum term (weight 0.08)
  - Staleness penalty (-3.0 per missed mark)
- **Probability Scaling & Horizon Softmax**: Softmax temperature expands with the forecast horizon $h$ according to $T(h) = 0.45 + 0.40 \sqrt{h}$. Derived probabilities are scaled down by a retention factor decaying from 0.70 to 0.10 (decay rate 0.60 per hour) to allocate probability mass for unnamed incoming assets.
- **Size Distributions**: Modeled as per-place Gaussian distributions. Means are anchored to the most recent mark's top market caps with a $+0.30/\text{hour}$ drift, while standard deviations scale as $0.10 + 0.30 \sqrt{h}$. A uniform mixture of 0.06 is applied across 82 bins.
- **Top 5 Selection**: Coins are chosen by maximizing total probability weighted by $1 / \sqrt{\text{place}}$.

Inputs are read from `$MAKEFUN_INPUT_DIR/task.json` and `$MAKEFUN_INPUT_DIR/history.ndjson`, and outputs are written to `$MAKEFUN_OUTPUT_DIR/forecast.json` structured as `{"probabilities", "top5", "sizes"}`.

## Execution
To run the forecasting model locally:

```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Backtest Results & Caveats
The entry achieved a **mean backtest score of -35.669** across 4 evaluation splits:

- **Split 1 (hist=2)**: -41.943
- **Split 2 (hist=3)**: -39.291
- **Split 3 (hist=4)**: -31.495
- **Split 4 (hist=5)**: -29.948

*Caveat*: True practice scores can only be accurately measured by submitting to the live platform. Note that the backtest evaluation spans shorter horizons (10–40 minutes) compared to the real practice environment setup (10–70 minutes).
