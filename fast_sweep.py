"""
Fast A-S Backtest Sweep — Vectorized
Calibrated to Canonic WETH/USDm live orderbook
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict

# ============================================================
# FIBONACCI
# ============================================================
def fib_spread_tiers(base_bps):
    return {
        "tight": base_bps * 0.382,
        "mid": base_bps * 0.618,
        "wide": base_bps * 1.0,
        "very_wide": base_bps * 1.618,
    }

# ============================================================
# PAIR PROFILES (Canonic live data)
# ============================================================
PAIRS = {
    "WETH/USDm": {
        "base_price": 2052.36,
        "sigma_daily": 0.055,
        "base_spread_bps": 3.2,
        "tick_sec": 1.0,
        "liquidity": 0.45,
        "maker_fee_bps": 0.0,
        "taker_fee_bps": 0.3,
        "bid_boost": 1.3,   # thin top-of-book bid
        "ask_dampen": 0.9,  # thicker ask side
    },
    "CANONIC/USDm": {
        "base_price": 0.50,
        "sigma_daily": 0.12,
        "base_spread_bps": 15.0,
        "tick_sec": 1.0,
        "liquidity": 0.20,
        "maker_fee_bps": 0.0,
        "taker_fee_bps": 0.3,
        "bid_boost": 1.0,
        "ask_dampen": 1.0,
    },
    "CANONIC/WETH": {
        "base_price": 0.000244,
        "sigma_daily": 0.14,
        "base_spread_bps": 20.0,
        "tick_sec": 1.0,
        "liquidity": 0.15,
        "maker_fee_bps": 0.0,
        "taker_fee_bps": 0.3,
        "bid_boost": 1.0,
        "ask_dampen": 1.0,
    },
}


def gen_prices(base, sigma_daily, tick_sec, n_ticks, seed=42):
    rng = np.random.RandomState(seed)
    ticks_per_day = 86400 / tick_sec
    sigma_tick = sigma_daily / np.sqrt(ticks_per_day)

    # Pre-generate all random numbers
    z = rng.randn(n_ticks)
    jumps = rng.rand(n_ticks)
    jump_dir = rng.rand(n_ticks)

    log_returns = np.zeros(n_ticks)
    log_returns[0] = 0.0

    for i in range(1, n_ticks):
        r = -0.5 * sigma_tick**2 + sigma_tick * z[i]
        if jumps[i] < 0.002:
            r += sigma_tick * 5 * (1 if jump_dir[i] > 0.5 else -1)
        log_returns[i] = r

    prices = base * np.exp(np.cumsum(log_returns))
    return prices


def run_backtest(prices, gamma, k, spread_mode, q_max, rebal_thresh,
                 micro_pulse, pair_cfg, order_size=1.0):
    n = len(prices)
    sigma_daily = pair_cfg["sigma_daily"]
    tick_sec = pair_cfg["tick_sec"]
    ticks_per_day = 86400 / tick_sec
    sigma_tick = sigma_daily / np.sqrt(ticks_per_day)
    base_bps = pair_cfg["base_spread_bps"]
    liquidity = pair_cfg["liquidity"]
    bid_boost = pair_cfg["bid_boost"]
    ask_dampen = pair_cfg["ask_dampen"]

    fib = fib_spread_tiers(base_bps)

    inventory = 0
    cash = 0.0
    n_trades = 0
    spreads_captured = []
    max_inv = 0
    inv_sum = 0

    # Pre-generate fill randomness
    rng = np.random.RandomState(int(gamma*1000 + k*100))
    fill_rands = rng.rand(n, 2)  # bid, ask

    pnl_curve = np.zeros(n)

    # Micro-pulse state
    run_len = 0
    last_dir = 0

    for t in range(n):
        mid = prices[t]
        t_rem = max(0.001, 1.0 - t / n)

        # Reservation price
        r = mid - inventory * gamma * sigma_tick**2 * t_rem

        # Spread
        if spread_mode == "as_optimal":
            hs = (gamma * sigma_tick**2 * t_rem + (2/gamma) * np.log(1 + gamma/k)) / 2
            bid_p = r - hs
            ask_p = r + hs
        else:  # fib_adaptive
            inv_ratio = abs(inventory) / max(q_max, 1)
            if inv_ratio < 0.2:
                tier = fib["tight"]
            elif inv_ratio < 0.5:
                tier = fib["mid"]
            elif inv_ratio < 0.8:
                tier = fib["wide"]
            else:
                tier = fib["very_wide"]

            # Micro-pulse adjustment
            if micro_pulse and t > 0:
                chg = prices[t] - prices[t-1]
                new_dir = 1 if chg > 0 else -1
                if new_dir == last_dir:
                    run_len += 1
                else:
                    # Check shock/bounce
                    if run_len >= 3:
                        if new_dir > 0 and inventory < 0:
                            tier *= 0.7
                        elif new_dir < 0 and inventory > 0:
                            tier *= 0.7
                    run_len = 1
                last_dir = new_dir

                if run_len >= 4:
                    tier *= 1.4  # Widen during sustained runs

            hs_abs = mid * (tier / 10000) / 2
            bid_p = r - hs_abs
            ask_p = r + hs_abs

        # Inventory skew
        if inventory > 0:
            skew = 0.0002 * inventory * mid
            bid_p -= skew
            ask_p -= skew * 0.5
        elif inventory < 0:
            skew = 0.0002 * abs(inventory) * mid
            bid_p += skew * 0.5
            ask_p += skew

        # Fills
        bid_dist = max(0, (mid - bid_p) / mid)
        ask_dist = max(0, (ask_p - mid) / mid)

        base_fp = liquidity * 0.08
        bid_fp = min(0.5, base_fp * bid_boost * np.exp(-k * bid_dist * 60))
        ask_fp = min(0.5, base_fp * ask_dampen * np.exp(-k * ask_dist * 60))

        if fill_rands[t, 0] < bid_fp and inventory < q_max:
            inventory += 1
            cash -= bid_p * order_size
            n_trades += 1

        if fill_rands[t, 1] < ask_fp and inventory > -q_max:
            inventory -= 1
            cash += ask_p * order_size
            n_trades += 1
            if n_trades >= 2:
                spreads_captured.append(ask_p - bid_p)

        # Rebalance
        if rebal_thresh < 1.0 and abs(inventory) >= q_max * rebal_thresh:
            reduce = abs(inventory) // 2
            if reduce > 0:
                slip = mid * 0.001
                if inventory > 0:
                    cash += (mid - slip) * reduce * order_size
                    inventory -= reduce
                else:
                    cash -= (mid + slip) * reduce * order_size
                    inventory += reduce

        max_inv = max(max_inv, abs(inventory))
        inv_sum += abs(inventory)
        pnl_curve[t] = cash + inventory * mid * order_size

    # Metrics
    final_pnl = pnl_curve[-1]
    peak = np.maximum.accumulate(pnl_curve)
    dd = peak - pnl_curve
    max_dd = np.max(dd) if len(dd) > 0 else 0

    rets = np.diff(pnl_curve)
    if len(rets) > 0 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(86400 * 365)
    else:
        sharpe = 0.0

    avg_spread = np.mean(spreads_captured) if spreads_captured else 0.0
    fill_rate = n_trades / n
    inv_risk = inv_sum / n / max(q_max, 1)

    return {
        "total_pnl": round(float(final_pnl), 6),
        "sharpe": round(float(sharpe), 4),
        "max_dd": round(float(max_dd), 6),
        "trades": n_trades,
        "fill_rate": round(float(fill_rate), 4),
        "avg_spread": round(float(avg_spread), 6),
        "max_inv": max_inv,
        "inv_risk": round(float(inv_risk), 4),
        "pnl_per_trade": round(float(final_pnl / max(n_trades, 1)), 6),
    }


def main():
    # Parameter grid
    gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    ks = [1.5, 3.0, 5.0, 10.0]
    spread_modes = ["as_optimal", "fib_adaptive"]
    q_maxs = [5, 10]
    rebal_thresholds = [0.85, 1.0]
    micro_options = [False, True]

    N_TICKS = 7200   # 2 hours on MegaETH
    N_SEEDS = 3

    all_results = {}

    for pair_name, pcfg in PAIRS.items():
        print(f"\n{'='*65}")
        print(f"  SWEEPING: {pair_name}")
        print(f"  Base: ${pcfg['base_price']} | Spread: {pcfg['base_spread_bps']} bps | "
              f"σ_daily: {pcfg['sigma_daily']*100:.1f}%")
        print(f"  Maker fee: {pcfg['maker_fee_bps']} bps | Taker fee: {pcfg['taker_fee_bps']} bps")
        print(f"{'='*65}")

        # Pre-gen price paths
        paths = [gen_prices(pcfg["base_price"], pcfg["sigma_daily"],
                           pcfg["tick_sec"], N_TICKS, seed=s) for s in range(N_SEEDS)]

        results = []
        best_sharpe = -1e9
        best_cfg = None
        combo_count = 0

        from itertools import product
        combos = list(product(gammas, ks, spread_modes, q_maxs, rebal_thresholds, micro_options))
        total = len(combos)

        for gamma, k, sm, qm, rt, mp in combos:
            pnls, sharpes, trades_list = [], [], []

            for prices in paths:
                res = run_backtest(prices, gamma, k, sm, qm, rt, mp, pcfg)
                pnls.append(res["total_pnl"])
                sharpes.append(res["sharpe"])
                trades_list.append(res["trades"])

            row = {
                "pair": pair_name,
                "gamma": gamma, "k": k, "spread_mode": sm,
                "q_max": qm, "rebalance": rt, "micro_pulse": mp,
                "avg_pnl": round(np.mean(pnls), 6),
                "std_pnl": round(np.std(pnls), 6),
                "avg_sharpe": round(np.mean(sharpes), 4),
                "avg_trades": round(np.mean(trades_list), 1),
                "pnl_per_trade": round(np.mean(pnls) / max(np.mean(trades_list), 1), 8),
            }
            results.append(row)

            if row["avg_sharpe"] > best_sharpe:
                best_sharpe = row["avg_sharpe"]
                best_cfg = row

            combo_count += 1
            if combo_count % 40 == 0:
                print(f"  {combo_count}/{total} | Best Sharpe: {best_sharpe:.4f}")

        all_results[pair_name] = results

        # Print top 10
        sorted_r = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)
        print(f"\n  TOP 10 CONFIGS:")
        print(f"  {'#':>3} {'γ':>5} {'k':>5} {'Mode':>13} {'qM':>3} {'Reb':>5} "
              f"{'Puls':>4} {'AvgPnL':>12} {'Sharpe':>8} {'Trd':>5} {'PnL/Trd':>12}")
        print(f"  {'-'*85}")
        for i, r in enumerate(sorted_r[:10]):
            print(f"  {i+1:>3} {r['gamma']:>5.2f} {r['k']:>5.1f} {r['spread_mode']:>13} "
                  f"{r['q_max']:>3} {r['rebalance']:>5.2f} "
                  f"{'Y' if r['micro_pulse'] else 'N':>4} "
                  f"{r['avg_pnl']:>12.6f} {r['avg_sharpe']:>8.4f} "
                  f"{r['avg_trades']:>5.0f} {r['pnl_per_trade']:>12.8f}")

        # Bottom 5 (worst)
        print(f"\n  WORST 5 (avoid these):")
        for i, r in enumerate(sorted_r[-5:]):
            print(f"  {i+1:>3} {r['gamma']:>5.2f} {r['k']:>5.1f} {r['spread_mode']:>13} "
                  f"{r['q_max']:>3} {'Y' if r['micro_pulse'] else 'N':>4} "
                  f"{r['avg_pnl']:>12.6f} {r['avg_sharpe']:>8.4f}")

    # Save
    with open("backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ALL RESULTS SAVED → backtest_results.json")
    print(f"{'='*65}")

    # Cross-pair comparison
    print(f"\n{'='*65}")
    print(f"  CROSS-PAIR BEST CONFIGS COMPARISON")
    print(f"{'='*65}")
    for pname, results in all_results.items():
        best = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)[0]
        print(f"\n  {pname}:")
        print(f"    γ={best['gamma']}, k={best['k']}, mode={best['spread_mode']}")
        print(f"    q_max={best['q_max']}, rebalance={best['rebalance']}, pulse={'Yes' if best['micro_pulse'] else 'No'}")
        print(f"    Sharpe: {best['avg_sharpe']:.4f} | PnL: {best['avg_pnl']:.6f} | Trades: {best['avg_trades']:.0f}")


if __name__ == "__main__":
    main()
