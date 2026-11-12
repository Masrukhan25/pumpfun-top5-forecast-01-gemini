# Monte Carlo Survival Forecaster (v3)

## Overview
This entry forecasts the top 5 new Solana coins by market cap, 24 hours out, using a Monte Carlo survival model. It is built entirely with Python 3.13 standard library routines, runs deterministically with a fixed seed, and uses no network access.

## Model Description
- **Monte Carlo survival simulation**: For each candidate coin, the model simulates many future paths. Survival probability comes from an exponential hazard (per-10-minute survival ~0.97, adjusted by a coin's recent growth and staleness). Simulated market caps grow with a horizon-dependent drift and lognormal noise.
- **Saturating newcomer cap model**: Newcomer coins rise fast then saturate (mean log10 cap 3.30 + 0.35*(1-exp(-h/40)), std 0.35-0.40, capped at ~3.65). A linear-growth newcomer model overshoots badly at long horizons; real newcomers (e.g. Google, MIHAO VI) reach ~3.3 in 10-30 minutes and then plateau.
- **Size distributions**: Adaptive blend by history richness. Each place's size distribution mixes (a) an empirical histogram of Monte Carlo simulated place caps (smoothed, clamped to [2.0, 5.0]) and (b) a 3-component normal mixture (55% Monte Carlo Gaussian, 30% broad Gaussian centered at 3.25 log10 SOL, 15% uniform over the 82 bins). Blend weight w_emp = clamp((n_nonempty_marks - 1)/3, 0, 1): with thin history the robust normal mixture dominates; with rich history the empirical histogram dominates. Floored at 0.0015 per bin.
- **Order probabilities**: Raw Monte Carlo counts with adaptive sharpening. Power = 0.80 + 0.50 * confidence, where confidence = w_emp * (1 - horizon_factor). Thin history / long horizon -> power ~0.80 (spread out); rich history / short horizon -> power up to 1.30 (sharpen). Each place's named probabilities use a floor of 0.002 per coin and a top-20 cutoff.
- **Adaptive minimum unnamed share**: min_unnamed = 0.50 * (1 - confidence) + 0.15 * confidence per place. When the model is likely to miss the actual coin (thin history, long horizon) more mass is reserved for unnamed newcomers; when it is likely to name the actual coin (rich history, short horizon) less mass is reserved.
- **Top 5 selection**: Coins are chosen by maximizing total probability weighted by 1 / sqrt(place).

Inputs are read from `$MAKEFUN_INPUT_DIR/task.json` and `$MAKEFUN_INPUT_DIR/history.ndjson`, and outputs are written to `$MAKEFUN_OUTPUT_DIR/forecast.json` with keys `{"probabilities", "top5", "sizes"}`.

## Execution
```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Backtest Results & Caveats
Evaluated with a faithful backtest that mimics the real practice setup (truncated history + answers at the task reads). Pooled mean log score **-35.442** across 10 reads in 4 splits:

- **Split A (history 1 mark, 4 reads)**: -39.133 (order -22.022, size -17.111)
- **Split B (history 2 marks, 3 reads)**: -39.028 (order -23.628, size -15.400)
- **Split C (history 3 marks, 2 reads)**: -28.060 (order -15.172, size -12.889)
- **Split D (history 4 marks, 1 read)**: -24.679 (order -13.525, size -11.153)

For reference, the previous v2 entry scored pooled -36.385 on the same backtest, and the real v2 practice score was -39.3 (the backtest proxy is optimistic by roughly 3 points versus the real practice, which uses longer horizons).

*Caveat*: the faithful backtest is a proxy, not the real practice score. The real practice uses 5 marks of history and reads at 10-70 minute horizons; the backtest splits use 1-4 marks and shorter horizons. An oracle analysis shows the pooled proxy has a ceiling near -34 with perfect size knowledge alone, because at long horizons most of the top 5 coins are newcomers that cannot be named. The true practice score can only be measured by submitting to the live platform.
