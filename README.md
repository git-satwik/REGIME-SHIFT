# REGIME-SHIFT
MACRO-AWARE TACTICAL ASSET ALLOCATION ENGINE

## Overview

REGIME-SHIFT is a dynamic tactical asset allocation engine that detects hidden market regimes
using a Gaussian Hidden Markov Model and re-optimizes a multi-asset portfolio via convex
optimization, validated with a strict walk-forward backtest that eliminates look-ahead bias and
explicitly models transaction costs.

## Architecture & Mathematical Rationale

### 1. Regime Detection - Hidden Markov Model

We use a **Gaussian Hidden Markov Model (GaussianHMM)** to discover latent market regimes from
daily features:
- Log returns of the multi-asset universe (SPY, QQQ, TLT, AGG, GLD, SHV).
- VIX closing level.
- FRED macro indicators: BAA-10Y credit spread, 10Y-2Y term spread, CPI YoY change.

The model assumes the market evolves through a finite number of unobserved states
`s_t in {0,1,2}` with state-specific Gaussian emission distributions. The **Viterbi algorithm**
yields the most likely sequence of hidden states. No manual labels are used; instead, states are
automatically mapped to *Bull*, *Bear*, and *Crisis* by ranking each state's mean equity return
and mean VIX level.

### 2. Dynamic Objective Mapping in Portfolio Optimization

The optimizer switches its objective function based on the predicted regime:
- **Bull regime**: Maximise the Sharpe ratio by solving a convex QP. The classic non-convex
  fractional program is transformed into a standard QP:
  `min_v  0.5 * v^T Sigma v   s.t.   v^T (mu - rf) = 1,  v >= 0`.
  The optimal unscaled vector `v*` is then normalised to sum to one, giving the maximum Sharpe
  portfolio under long-only and max-weight constraints.
- **Bear / Crisis regime**: Minimise portfolio variance directly:
  `min_w  w^T Sigma w   s.t.   sum(w) = 1,  0 <= w_i <= 0.4`.

Both formulations are solved with `cvxpy` (ECOS solver) and are numerically stable, with
equal-weight fallbacks on solver failure.

### 3. Walk-Forward Validation - Eliminating Look-Ahead Bias

The backtester uses a **strict temporal expanding window**:
- At each monthly rebalance date `t`, all models (HMM, mean/covariance estimators) are fitted
  *only* on data up to `t-1`.
- The HMM's last in-sample state is taken as the forecast regime for the holding period
  `t -> t+1`.
- Portfolio weights are then computed using those historically available estimates.

No future information leaks into the optimisation. This mirrors the exact operational
constraints of a real trading system.

### 4. Transaction Cost Modelling

A friction penalty of **5 bps per unit of turnover** is applied inside the backtester:

`cost_t = tau * sum_i |w_i,t - w_i,t-1|`, where `tau = 5e-4`.

The cost is deducted multiplicatively from the portfolio value at each rebalance. This penalises
excessive trading and ensures the strategy is economically viable, not just statistically
appealing on a gross-of-cost basis.

### 5. Benchmarks

The engine compares against two static, monthly-rebalanced portfolios:
- **60/40**: 60% SPY, 40% AGG.
- **Equal Weight**: uniform allocation across all six tickers.

Performance metrics (Sharpe, Sortino, Max Drawdown, Calmar, annualised return, average turnover)
are reported in a tear sheet for all three portfolios side by side.

## Repository Structure

```
regime-shift/
|-- regime_shift.ipynb      # Full pipeline notebook (this project)
|-- regime_shift.py         # Script version of the same pipeline
|-- README.md                # This file
|-- requirements.txt         # Pinned dependencies
```

## Reproducibility Guide

1. **Clone** the repository and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd regime-shift
   pip install -r requirements.txt
   ```

   Or install directly:
   ```bash
   pip install numpy pandas yfinance pandas-datareader hmmlearn cvxpy matplotlib scipy jupyter
   ```

2. **Run the notebook end to end**:
   ```bash
   jupyter nbconvert --to notebook --execute regime_shift.ipynb --output regime_shift_executed.ipynb
   ```
   or open it interactively with `jupyter lab regime_shift.ipynb` and Run All.

3. **Determinism**: The HMM is seeded (`random_state=42`) so state assignments are reproducible
   given identical input data. Note that yfinance/FRED data can be revised or extended over time
   (new trading days appended, occasional data vendor corrections), so exact numerical
   reproduction is guaranteed only for a fixed `start_date`/`end_date` window and a fixed data
   pull timestamp - re-running the notebook months later may reflect minor data revisions from
   the upstream sources.

4. **Configuration**: All parameters (asset universe, date range, HMM state count, rebalance
   frequency, transaction cost, lookback window) live in the single `config` dictionary in
   Section 1. Change one parameter there to re-run a full sensitivity analysis.

## Testing Instructions

This project does not ship a formal `pytest` suite, but the notebook itself is structured to be
independently verifiable at each stage:

- **Data integrity check**: after Section 2, `engine.features.isna().sum().sum()` should be `0`
  (all rows are fully aligned and dropna'd).
- **HMM sanity check**: after Section 3, `transition_df` should have a strong diagonal
  (probability of staying in the same regime should exceed the probability of switching), and
  the price-overlay chart's red (Crisis) shading should visually align with known drawdown
  periods (2008, 2020, 2022).
- **Optimizer unit check**: call `DynamicPortfolioOptimizer().optimize(mu, Sigma, 'Bull')` and
  `... .optimize(mu, Sigma, 'Bear')` on a small synthetic `mu`/`Sigma` and confirm weights sum to
  1, are non-negative, and respect the 0.4 max-weight cap.
- **Look-ahead bias check**: confirm that inside `WalkForwardBacktester.run_backtest`, the HMM
  and `mu`/`Sigma` estimation always use `hist_ret`/`hist_feat` sliced strictly before
  `available_end < rebal_date`, and that realized returns applied to compute `port_growth` always
  come from `hold_start > rebal_date`.
- **Cost model check**: verify `turnover` values are in `[0, 2]` (bounded by the L1 distance
  between two simplex-constrained weight vectors) and that `cost = turnover * 5e-4` is being
  applied multiplicatively, not additively, to `port_val`.

## Known Limitations & Future Work

- Only 3 HMM states are modeled; a 4-5 state model could separate "recovery" from "bull" regimes.
- Transaction costs are a flat bps assumption; a more realistic model would scale cost with
  trade size / market liquidity.
- The Max Sharpe QP reformulation assumes long-only weights; extending to long/short would
  require revisiting the reformulation's non-negativity constraint.
- Macro data (FRED) is monthly/lagged in reality (e.g., CPI is reported with a delay); this
  implementation forward-fills as of the print date, not the actual availability date, which is
  a mild simplification versus true real-time data vintages.
