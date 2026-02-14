# Claude Code Droplet Prompt — A-S Regime-Switching Backtest

Paste this into your fresh Claude Code session on the droplet:

---

```
Build a regime-switching Avellaneda-Stoikov market making backtest engine for WETH/USDm on Canonic DEX (MegaETH chain). This needs to run server-side for extended parameter sweeps.

## REAL ORDERBOOK DATA (from live canonic.trade):
- Chain: MegaETH (sub-second blocks)
- Pair: WETH/USDm at $2,052.36
- Spread: 3.2 bps (0.032%) — extremely tight
- Maker fee: 0% | Taker fee: 0.03%
- Ask side: ~0.6 WETH per level at +3, +5, +10, +20, +30, +50, +100 bps
- Bid side: THIN top-of-book (0.055 WETH at -0.2 bps, dust at -0.3 bps), then 0.597 WETH per level from -3 bps down
- CLP Vault TVL: $30,894 (7.55 WETH / 15,417 USDm)
- 24h Volume: 1.12 WETH / $2,234 USDm
- Contract: 0x23469683e25b780DFDC11410a8e83c923caDF125

## KEY INSIGHT — BID ASYMMETRY:
The bid top-of-book is paper-thin (0.055 WETH) before the real depth wall at -3 bps. Ask side is uniformly thick. An MM parking limit bids at -1 to -2 bps with 0% maker fee gets free queue priority. This is the "hole you can't patent" (Coulter).

## WHAT TO BUILD:

### 1. Regime-Switching Engine
Two modes that auto-switch based on volatility detection:

**CONSOLIDATION MODE** (current regime — ETH in 4-year base):
- Strategy: A-S Optimal
- γ (risk aversion): 0.05
- k (order intensity): 1.5
- q_max: 5 WETH
- Rebalance: @ 85% inventory
- Micro-pulse: OFF
- Expected: ~$0.62/trade, ~558 trades/2hr

**BREAKOUT MODE** (when vol spikes or ETH approaches $5K):
- Strategy: Fibonacci Adaptive
- γ: 1.00 (defensive)
- k: 5.0 (patient)
- q_max: 5 WETH (keep tight)
- Rebalance: @ 85%
- Micro-pulse: ON (shock/bounce detection)
- Fib spread tiers: 0.382× (tight) → 1.618× (max defensive)

**REGIME DETECTION TRIGGERS:**
- Realized vol over 20-tick window exceeds 2× baseline → switch to breakout
- Consecutive tick run of 4+ in same direction (Coulter counting) → pre-switch warning
- Price crosses $5,000 → maximum defensive posture, long-only bias
- Vol drops below 1.5× baseline for 100+ ticks → revert to consolidation

### 2. Fibonacci Spread Tiers
Base spread adapts by inventory pressure:
- Flat (inv < 20% of max): use 0.382× base spread (tight, aggressive)
- Building (20-50%): use 0.618× (golden ratio)
- Heavy (50-80%): use 1.0× (full spread)
- Near limit (80%+): use 1.618× (max defensive)

At breakout, multiply all tiers by 1.5× additional.

### 3. Micro-Pulse Timing (Coulter-style)
Count consecutive up/down price ticks. Detect:
- **Shock**: 4+ consecutive ticks same direction → widen spread 40%
- **Bounce**: Large run followed by reversal → tighten spread 30% on reversal side to capture mean reversion
- **Fibonacci retracement**: If price retraces to 0.382 or 0.618 of recent move → tighten (high-probability fill zone)

### 4. Parameter Sweep Grid
Run a FULL sweep (the server can handle it):
- gammas: [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
- ks: [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
- spread_modes: ["as_optimal", "fib_adaptive", "regime_switching"]
- q_maxs: [3, 5, 8, 10, 15]
- rebalance_thresholds: [0.7, 0.8, 0.85, 0.9, 1.0]
- micro_pulse: [False, True]
- vol_regimes: ["low" (σ=0.02), "normal" (σ=0.055), "high" (σ=0.12), "breakout" (σ=0.25)]

Simulation: 14,400 ticks per run (4 hours MegaETH), 10 Monte Carlo seeds per combo.

### 5. Output
- Save all results to backtest_results_full.json
- Print top 20 configs per vol regime ranked by Sharpe
- Print the SINGLE best regime-switching config that maximizes Sharpe across ALL vol regimes (robust config)
- Print anti-patterns (configs that lose money)
- Generate a summary markdown file with the optimal Hummingbot-ready parameters

### 6. Asymmetric Fill Model
Calibrate fills to the real book:
- Bid fills: 1.3× boost (thin competition at top-of-book)
- Ask fills: 0.9× dampen (thicker ask wall)
- No maker fee on fills (0%)
- Volume-cap: realistic daily fill limit based on $2,300/day observed volume

### 7. Volume-Capped Reality Check
After the unconstrained sweep, re-run the top 10 configs with a volume cap:
- Max fills per day: ~1.12 WETH each side (from observed 24h volume)
- Report "realistic daily PnL" alongside "theoretical daily PnL"
- This is the number that matters for actual deployment

Save everything to ~/canonic_backtest/ on the server. This is a long-running computation — take your time and be thorough.
```

---

## ALSO SAVE THESE FILES TO THE DROPLET:

The fast_sweep.py from this session is already proven to work. You can copy it to the droplet as a starting point:

```bash
scp fast_sweep.py root@YOUR_DROPLET_IP:~/canonic_backtest/
```

Or just paste the prompt above into a fresh Claude Code session on the droplet and let it build from scratch with the full parameter grid.
