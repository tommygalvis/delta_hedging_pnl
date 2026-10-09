# Discrete-Time Delta Hedging: Replication Error, Transaction Costs and Model Risk

A Monte Carlo study of what it costs to replicate an option with a delta hedge, and where the Black-Scholes hedge breaks down. The running example is a one-year at-the-money call (S₀ = K = 100, r = 2%, σ = 20%), held long and delta-hedged.

## What it covers

1. **Pricing engine:** Black-Scholes price and Greeks, checked against finite differences and put-call parity
2. **Gamma-theta identity:** one-step hedged P&L = ½Γ(δS² − σ²S²δt), verified against full repricing
3. **Discrete hedging:** a self-financing hedge engine, and hedging error shrinking like 1/√N with the number of rebalances
4. **Exact P&L attribution:** gamma (convexity) and theta (time decay) streams that sum to the hedge P&L to machine precision
5. **Transaction costs:** the cost-risk frontier, the optimal rebalancing frequency, band hedging, and a Leland-style volatility adjustment
6. **Model risk:** the jump floor under Merton jump-diffusion, the vega leak under Heston stochastic volatility, and drift invariance
7. **Calibration:** Merton jump parameters estimated from 3 years of SPY data using robust jump detection
8. **Risk report:** mean, standard deviation, VaR and CVaR of the hedged P&L across the Black-Scholes, Merton, Heston and SPY-calibrated worlds

The theory behind each section, with derivations and references, is in the comment blocks above the code.

## Usage

```bash
pip install -r requirements.txt
python delta_hedging_pnl.py
```

The script takes a few seconds to run. It prints results for each section and opens 12 charts one at a time; close each chart to continue. It downloads SPY prices with `yfinance` and falls back to a built-in price snapshot if that fails. All simulations use fixed seeds, so the results are reproducible.
