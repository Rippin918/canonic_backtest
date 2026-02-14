"""
Quick Regime-Switching Test
============================
Reduced parameter grid for faster validation:
- 5 gammas × 3 ks × 3 spread modes × 2 q_maxs × 2 rebalance × 2 micro × 4 vol regimes
- = 720 combos × 5 seeds = 3,600 simulations
- Should complete in ~5-10 minutes
"""

# Import the full engine
import sys
sys.path.insert(0, '/root/canonic_backtest')
from regime_switching_backtest import *

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  QUICK REGIME-SWITCHING TEST")
    print("  Reduced grid for fast validation")
    print("="*80 + "\n")

    # Reduced parameters
    gammas = [0.05, 0.1, 0.3, 0.5, 1.0]  # 5 instead of 10
    ks = [1.5, 3.0, 7.5]  # 3 instead of 7
    spread_modes = ["as_optimal", "fib_adaptive", "regime_switching"]  # Keep all 3
    q_maxs = [5, 10]  # 2 instead of 5
    rebalance_thresholds = [0.85, 1.0]  # 2 instead of 5
    micro_pulse_options = [False, True]  # Keep both

    vol_regimes = {
        "low": 0.02,
        "normal": 0.055,
        "high": 0.12,
        "breakout": 0.25,
    }

    n_ticks = 7200  # 2 hours instead of 4
    n_seeds = 5  # 5 instead of 10

    combos_per_regime = (len(gammas) * len(ks) * len(spread_modes) * len(q_maxs) *
                         len(rebalance_thresholds) * len(micro_pulse_options))
    total_sims = combos_per_regime * len(vol_regimes) * n_seeds

    print(f"  Combos per regime: {combos_per_regime}")
    print(f"  Total simulations: {total_sims:,}")
    print(f"  Ticks per run: {n_ticks} ({n_ticks / 3600:.1f} hours)")
    print("="*80 + "\n")

    # Results storage
    all_results = {}

    # Run sweep for each vol regime
    for regime_name, sigma_daily in vol_regimes.items():
        print(f"\n{'='*80}")
        print(f"  VOL REGIME: {regime_name.upper()} (σ_daily = {sigma_daily:.1%})")
        print(f"{'='*80}")

        regime_results = []
        best_sharpe = -np.inf

        # Pre-generate price paths
        print(f"  Generating {n_seeds} price paths...")
        price_paths = []
        for seed in range(n_seeds):
            prices = simulate_price_path(WETH_USDM, n_ticks, sigma_override=sigma_daily, seed=seed)
            price_paths.append(prices)

        # Generate all combinations
        combos = list(itertools.product(
            gammas, ks, spread_modes, q_maxs, rebalance_thresholds, micro_pulse_options
        ))

        print(f"  Running {len(combos)} combinations × {n_seeds} seeds...")
        start_time = time.time()

        for idx, (gamma, k, sm, qm, rt, mp) in enumerate(combos):
            combo_pnls = []
            combo_sharpes = []
            combo_trades = []

            # Run across all MC seeds
            for prices in price_paths:
                result = run_backtest(
                    pair=WETH_USDM, prices=prices,
                    gamma=gamma, k=k, spread_mode=sm,
                    q_max=qm, rebalance_threshold=rt,
                    micro_pulse_enabled=mp,
                )
                combo_pnls.append(result["total_pnl"])
                combo_sharpes.append(result["sharpe_ratio"])
                combo_trades.append(result["num_trades"])

            # Aggregate
            avg_pnl = np.mean(combo_pnls)
            avg_sharpe = np.mean(combo_sharpes)
            avg_trades = np.mean(combo_trades)

            row = {
                "regime": regime_name,
                "sigma_daily": sigma_daily,
                "gamma": gamma,
                "k": k,
                "spread_mode": sm,
                "q_max": qm,
                "rebalance_threshold": rt,
                "micro_pulse": mp,
                "avg_pnl": round(avg_pnl, 6),
                "avg_sharpe": round(avg_sharpe, 4),
                "avg_trades": round(avg_trades, 1),
                "pnl_per_trade": round(avg_pnl / max(avg_trades, 1), 8),
            }
            regime_results.append(row)

            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_config = row

            # Progress every 30 combos
            if (idx + 1) % 30 == 0 or idx == len(combos) - 1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(combos) - idx - 1) / rate if rate > 0 else 0
                print(f"    {idx+1}/{len(combos)} ({(idx+1)/len(combos)*100:.0f}%) | "
                      f"Best Sharpe: {best_sharpe:.4f} | "
                      f"{rate:.1f} combos/sec | ETA: {remaining:.0f}s")

        all_results[regime_name] = regime_results

        # Print best for this regime
        print(f"\n  BEST CONFIG: γ={best_config['gamma']:.2f}, k={best_config['k']:.1f}, "
              f"mode={best_config['spread_mode']}, Sharpe={best_config['avg_sharpe']:.4f}")

    # Save results
    print(f"\n{'='*80}")
    print(f"  Saving results...")
    with open("quick_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print top 10 per regime
    print_top_configs(all_results, top_n=10)

    # Find robust regime-switching config
    robust_config = find_robust_config(all_results)

    print(f"\n{'='*80}")
    print(f"  ✓ QUICK TEST COMPLETE")
    print(f"  Results saved to: quick_test_results.json")
    print(f"{'='*80}\n")
