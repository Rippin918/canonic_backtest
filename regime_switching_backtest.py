"""
Regime-Switching Avellaneda-Stoikov Backtest Engine
====================================================
Comprehensive parameter sweep with:
- Regime switching (consolidation vs breakout modes)
- Volatility-based regime detection
- Fibonacci spread tiers with inventory adaptation
- Micro-pulse timing (Coulter-style)
- Asymmetric fill model calibrated to Canonic orderbook
- Volume-capped reality check
- Full parameter grid: 10 gammas × 7 ks × 3 spread modes × 5 q_maxs × 5 rebalance × 2 micro × 4 vol regimes
- Monte Carlo: 10 seeds × 14,400 ticks (4 hours MegaETH)

Reference: Avellaneda & Stoikov (2008) + Canonic live orderbook data
"""

import numpy as np
import json
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# FIBONACCI UTILITIES
# ============================================================
def fib_ratios() -> Dict[str, float]:
    """Fibonacci ratios for spread tiers."""
    return {
        "fib_382": 0.382,   # Tight
        "fib_618": 0.618,   # Golden ratio
        "fib_1000": 1.0,    # Full spread
        "fib_1618": 1.618,  # Max defensive
    }

def fib_spread_tier(base_spread_bps: float, inv_ratio: float, regime: str = "consolidation") -> float:
    """
    Calculate Fibonacci spread tier based on inventory pressure.

    Args:
        base_spread_bps: Base market spread in bps
        inv_ratio: Absolute inventory / q_max (0 to 1)
        regime: "consolidation" or "breakout"

    Returns:
        Spread in bps
    """
    ratios = fib_ratios()

    # Select tier based on inventory pressure
    if inv_ratio < 0.2:
        tier = ratios["fib_382"]     # Tight when flat
    elif inv_ratio < 0.5:
        tier = ratios["fib_618"]     # Golden ratio
    elif inv_ratio < 0.8:
        tier = ratios["fib_1000"]    # Full spread
    else:
        tier = ratios["fib_1618"]    # Max defensive

    spread_bps = base_spread_bps * tier

    # Breakout mode: multiply by 1.5× additional
    if regime == "breakout":
        spread_bps *= 1.5

    return spread_bps


# ============================================================
# MICRO-PULSE DETECTION (Coulter-style)
# ============================================================
@dataclass
class MicroPulse:
    """
    Tracks micro-pulse rhythm of price movements.
    Detects shock, bounce, and Fibonacci retracement patterns.
    """
    tick_history: List[float] = field(default_factory=list)
    run_length: int = 0
    last_direction: int = 0  # +1 for up, -1 for down, 0 for start

    def update(self, price_change: float):
        """Update with new price change."""
        self.tick_history.append(price_change)

        if price_change == 0:
            return

        current_dir = 1 if price_change > 0 else -1

        if self.last_direction == 0:
            self.last_direction = current_dir
            self.run_length = 1
        elif current_dir == self.last_direction:
            self.run_length += 1
        else:
            # Direction changed
            self.last_direction = current_dir
            self.run_length = 1

    def detect_shock_bounce(self) -> str:
        """
        Detect shock & bounce patterns.

        Returns:
            - "shock": 4+ consecutive ticks same direction
            - "bounce_up": Large down run followed by up reversal
            - "bounce_down": Large up run followed by down reversal
            - "neutral": No pattern
        """
        if self.run_length >= 4:
            return "shock"

        # Check for bounce (need at least 2 ticks)
        if len(self.tick_history) >= 2:
            prev_change = self.tick_history[-2]
            curr_change = self.tick_history[-1]

            # Look for reversal after strong move
            if abs(prev_change) > np.std(self.tick_history[-20:]) * 1.5:
                if prev_change < 0 and curr_change > 0:
                    return "bounce_up"
                elif prev_change > 0 and curr_change < 0:
                    return "bounce_down"

        return "neutral"

    def fib_retracement_signal(self, lookback: int = 20) -> Optional[float]:
        """
        Check if recent price move is near Fibonacci retracement level.

        Returns:
            Retracement ratio if near 0.382 or 0.618, else None
        """
        if len(self.tick_history) < lookback:
            return None

        recent = self.tick_history[-lookback:]
        cumsum = np.cumsum(recent)
        high = np.max(cumsum)
        low = np.min(cumsum)
        rng = high - low

        if rng == 0:
            return None

        current_pos = cumsum[-1] - low
        ratio = current_pos / rng

        # Check if near key Fibonacci levels
        for fib_level in [0.382, 0.618]:
            if abs(ratio - fib_level) < 0.05:
                return fib_level

        return None


# ============================================================
# REGIME DETECTION
# ============================================================
@dataclass
class RegimeDetector:
    """
    Detects market regime based on realized volatility.
    Switches between consolidation and breakout modes.
    """
    baseline_vol: float = 0.055  # Normal vol (daily)
    current_regime: str = "consolidation"
    regime_history: List[str] = field(default_factory=list)
    low_vol_counter: int = 0

    def detect_regime(self, prices: np.ndarray, window: int = 20) -> str:
        """
        Detect current market regime based on realized volatility.

        Args:
            prices: Recent price history
            window: Lookback window for vol calculation

        Returns:
            "consolidation" or "breakout"
        """
        if len(prices) < window:
            return "consolidation"

        # Calculate realized vol over window
        window_prices = prices[-window:]
        returns = np.diff(window_prices) / window_prices[:-1]
        realized_vol = np.std(returns) * np.sqrt(86400)  # Annualize to daily

        # Price level check
        current_price = prices[-1]

        # Regime switching logic
        if current_price >= 5000:
            # Price crossed $5K → maximum defensive
            self.current_regime = "breakout"
            self.low_vol_counter = 0
        elif realized_vol > 2.0 * self.baseline_vol:
            # Vol exceeds 2× baseline → breakout
            self.current_regime = "breakout"
            self.low_vol_counter = 0
        elif realized_vol < 1.5 * self.baseline_vol:
            # Vol drops below 1.5× baseline
            self.low_vol_counter += 1
            if self.low_vol_counter >= 100:
                # Sustained low vol for 100+ ticks → consolidation
                self.current_regime = "consolidation"
        else:
            # In between thresholds - maintain current regime
            self.low_vol_counter = 0

        self.regime_history.append(self.current_regime)
        return self.current_regime


# ============================================================
# PAIR PROFILE
# ============================================================
@dataclass
class PairProfile:
    """Trading pair profile with Canonic live data."""
    name: str
    base_price: float
    sigma_daily: float          # Daily volatility
    base_spread_bps: float      # Market spread in bps
    tick_interval_sec: float    # MegaETH: ~1 sec
    liquidity_score: float      # 0-1
    maker_fee_bps: float = 0.0  # Canonic: 0% maker
    taker_fee_bps: float = 0.3  # Canonic: 0.03% taker
    bid_boost: float = 1.3      # Thin bid top-of-book
    ask_dampen: float = 0.9     # Thick ask side
    daily_volume_base: float = 1.12  # WETH per day observed

    @property
    def sigma_per_tick(self) -> float:
        """Volatility per tick interval."""
        ticks_per_day = 86400 / self.tick_interval_sec
        return self.sigma_daily / np.sqrt(ticks_per_day)


# Canonic WETH/USDm (live orderbook data)
WETH_USDM = PairProfile(
    name="WETH/USDm",
    base_price=2052.36,
    sigma_daily=0.055,           # Normal vol
    base_spread_bps=3.2,         # 0.032% - extremely tight
    tick_interval_sec=1.0,
    liquidity_score=0.45,
    daily_volume_base=1.12,      # 1.12 WETH / $2,234 USDm per day
)


# ============================================================
# AVELLANEDA-STOIKOV PARAMETERS
# ============================================================
@dataclass
class ASParams:
    """A-S model parameters with regime support."""
    gamma: float                # Risk aversion
    k: float                    # Order intensity
    sigma: float                # Volatility per tick
    q_max: int                  # Max inventory
    spread_mode: str            # "as_optimal", "fib_adaptive", "regime_switching"
    rebalance_threshold: float  # Fraction of q_max to trigger rebalance
    micro_pulse_enabled: bool   # Use micro-pulse signals

    def reservation_price(self, mid: float, q: int, t_remaining: float) -> float:
        """A-S reservation price."""
        return mid - q * self.gamma * (self.sigma ** 2) * t_remaining

    def optimal_spread(self, t_remaining: float) -> float:
        """A-S optimal spread."""
        return (self.gamma * (self.sigma ** 2) * t_remaining +
                (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k))


# ============================================================
# PRICE SIMULATION
# ============================================================
def simulate_price_path(pair: PairProfile, n_ticks: int, sigma_override: float = None,
                       seed: int = 42) -> np.ndarray:
    """
    Generate GBM price path with jumps and mean reversion.

    Args:
        pair: Pair profile
        n_ticks: Number of ticks to simulate
        sigma_override: Override daily vol (for regime testing)
        seed: Random seed

    Returns:
        Price path array
    """
    rng = np.random.RandomState(seed)
    sigma_daily = sigma_override if sigma_override is not None else pair.sigma_daily
    ticks_per_day = 86400 / pair.tick_interval_sec
    sigma_tick = sigma_daily / np.sqrt(ticks_per_day)

    prices = np.zeros(n_ticks)
    prices[0] = pair.base_price

    mean_revert_speed = 0.001

    for i in range(1, n_ticks):
        # GBM
        gbm_return = -0.5 * sigma_tick**2 + sigma_tick * rng.randn()

        # Mean reversion
        mr_component = mean_revert_speed * (pair.base_price - prices[i-1]) / pair.base_price

        # Jumps (shock events)
        jump = 0.0
        if rng.rand() < 0.002:  # 0.2% chance per tick
            jump = sigma_tick * 5 * (1 if rng.rand() > 0.5 else -1)

        prices[i] = prices[i-1] * np.exp(gbm_return + mr_component + jump)

    return prices


# ============================================================
# BACKTEST ENGINE
# ============================================================
@dataclass
class TradeRecord:
    tick: int
    side: str
    price: float
    size: float
    inventory_after: int
    regime: str


def run_backtest(
    pair: PairProfile,
    prices: np.ndarray,
    gamma: float,
    k: float,
    spread_mode: str,
    q_max: int,
    rebalance_threshold: float,
    micro_pulse_enabled: bool,
    order_size: float = 1.0,
    volume_cap_per_side: Optional[float] = None,
) -> Dict:
    """
    Run single backtest with regime switching.

    Args:
        pair: Trading pair profile
        prices: Simulated price path
        gamma: Risk aversion
        k: Order intensity
        spread_mode: "as_optimal", "fib_adaptive", or "regime_switching"
        q_max: Max inventory
        rebalance_threshold: Rebalance trigger (fraction of q_max)
        micro_pulse_enabled: Enable micro-pulse signals
        order_size: Size per order
        volume_cap_per_side: Max volume per side (for reality check)

    Returns:
        Backtest results dict
    """
    n_ticks = len(prices)
    sigma_tick = pair.sigma_per_tick

    params = ASParams(
        gamma=gamma, k=k, sigma=sigma_tick, q_max=q_max,
        spread_mode=spread_mode, rebalance_threshold=rebalance_threshold,
        micro_pulse_enabled=micro_pulse_enabled,
    )

    # State
    inventory = 0
    cash = 0.0
    trades: List[TradeRecord] = []
    pnl_curve = []

    # Regime & micro-pulse
    regime_detector = RegimeDetector(baseline_vol=pair.sigma_daily)
    pulse = MicroPulse()

    # Volume tracking
    volume_bought = 0.0
    volume_sold = 0.0

    # RNG for fills
    rng = np.random.RandomState(int(gamma * 1000 + k * 100))

    for tick in range(n_ticks):
        mid = prices[tick]
        t_remaining = max(0.001, 1.0 - tick / n_ticks)

        # Update regime
        regime = regime_detector.detect_regime(prices[:tick+1])

        # Update micro-pulse
        if tick > 0:
            pulse.update(prices[tick] - prices[tick - 1])

        # --- Compute reservation price ---
        r = params.reservation_price(mid, inventory, t_remaining)

        # --- Compute spread based on mode ---
        if spread_mode == "as_optimal":
            # Standard A-S optimal spread
            half_spread = params.optimal_spread(t_remaining) / 2.0
            bid = r - half_spread
            ask = r + half_spread

        elif spread_mode == "fib_adaptive":
            # Fibonacci adaptive (no regime switching)
            inv_ratio = abs(inventory) / max(q_max, 1)
            tier_bps = fib_spread_tier(pair.base_spread_bps, inv_ratio, regime="consolidation")

            # Micro-pulse adjustment
            if micro_pulse_enabled:
                signal = pulse.detect_shock_bounce()
                fib_retr = pulse.fib_retracement_signal()

                if signal == "bounce_up" and inventory < 0:
                    tier_bps *= 0.7  # Tighten ask to capture bounce
                elif signal == "bounce_down" and inventory > 0:
                    tier_bps *= 0.7  # Tighten bid
                elif signal == "shock":
                    tier_bps *= 1.4  # Widen during shock

                if fib_retr is not None:
                    tier_bps *= 0.8  # Tighten at Fib retracement levels

            half_spread_abs = mid * (tier_bps / 10000.0) / 2.0
            bid = r - half_spread_abs
            ask = r + half_spread_abs

        elif spread_mode == "regime_switching":
            # Regime-switching mode
            inv_ratio = abs(inventory) / max(q_max, 1)
            tier_bps = fib_spread_tier(pair.base_spread_bps, inv_ratio, regime=regime)

            # Micro-pulse adjustment (only in breakout mode)
            if micro_pulse_enabled and regime == "breakout":
                signal = pulse.detect_shock_bounce()
                fib_retr = pulse.fib_retracement_signal()

                if signal == "bounce_up" and inventory < 0:
                    tier_bps *= 0.7
                elif signal == "bounce_down" and inventory > 0:
                    tier_bps *= 0.7
                elif signal == "shock":
                    tier_bps *= 1.4

                if fib_retr is not None:
                    tier_bps *= 0.8

            half_spread_abs = mid * (tier_bps / 10000.0) / 2.0
            bid = r - half_spread_abs
            ask = r + half_spread_abs

        else:
            raise ValueError(f"Unknown spread_mode: {spread_mode}")

        # --- Inventory skew ---
        if inventory > 0:
            skew = 0.0002 * inventory * mid
            bid -= skew
            ask -= skew * 0.5
        elif inventory < 0:
            skew = 0.0002 * abs(inventory) * mid
            bid += skew * 0.5
            ask += skew

        # --- Fill probability (asymmetric model) ---
        bid_dist = max(0, (mid - bid) / mid)
        ask_dist = max(0, (ask - mid) / mid)

        base_fill_prob = pair.liquidity_score * 0.08

        # Asymmetric fills: thin bid top-of-book, thick ask
        bid_fill_prob = min(0.5, base_fill_prob * pair.bid_boost * np.exp(-k * bid_dist * 60))
        ask_fill_prob = min(0.5, base_fill_prob * pair.ask_dampen * np.exp(-k * ask_dist * 60))

        # --- Execute fills (with volume cap check) ---
        if rng.rand() < bid_fill_prob and inventory < q_max:
            # Check volume cap
            if volume_cap_per_side is None or volume_bought < volume_cap_per_side:
                inventory += 1
                cash -= bid * order_size
                volume_bought += order_size
                trades.append(TradeRecord(
                    tick=tick, side="buy", price=bid, size=order_size,
                    inventory_after=inventory, regime=regime
                ))

        if rng.rand() < ask_fill_prob and inventory > -q_max:
            if volume_cap_per_side is None or volume_sold < volume_cap_per_side:
                inventory -= 1
                cash += ask * order_size
                volume_sold += order_size
                trades.append(TradeRecord(
                    tick=tick, side="sell", price=ask, size=order_size,
                    inventory_after=inventory, regime=regime
                ))

        # --- Rebalance ---
        if abs(inventory) >= q_max * rebalance_threshold:
            reduce_qty = abs(inventory) // 2
            if reduce_qty > 0:
                slippage = mid * 0.001  # 10 bps
                if inventory > 0:
                    cash += (mid - slippage) * reduce_qty * order_size
                    inventory -= reduce_qty
                else:
                    cash -= (mid + slippage) * reduce_qty * order_size
                    inventory += reduce_qty

        # Mark-to-market
        mtm = cash + inventory * mid * order_size
        pnl_curve.append(mtm)

    # --- Compute metrics ---
    pnl_arr = np.array(pnl_curve)
    final_pnl = pnl_arr[-1] if len(pnl_arr) > 0 else 0.0

    # Realized PnL from round-trips
    realized = 0.0
    buy_stack = []
    for t in trades:
        if t.side == "buy":
            buy_stack.append(t.price)
        elif t.side == "sell" and buy_stack:
            buy_price = buy_stack.pop(0)
            realized += (t.price - buy_price) * order_size

    unrealized = final_pnl - realized

    # Max drawdown
    peak = np.maximum.accumulate(pnl_arr) if len(pnl_arr) > 0 else np.array([0])
    drawdowns = peak - pnl_arr if len(pnl_arr) > 0 else np.array([0])
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # Sharpe (annualized)
    if len(pnl_arr) > 1:
        returns = np.diff(pnl_arr)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(86400 * 365)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Trades & fills
    num_trades = len(trades)
    fill_rate = num_trades / max(n_ticks, 1)

    # Regime stats
    regime_counts = {}
    for t in trades:
        regime_counts[t.regime] = regime_counts.get(t.regime, 0) + 1

    return {
        "total_pnl": float(final_pnl),
        "realized_pnl": float(realized),
        "unrealized_pnl": float(unrealized),
        "max_drawdown": float(max_dd),
        "sharpe_ratio": float(sharpe),
        "num_trades": num_trades,
        "fill_rate": float(fill_rate),
        "regime_counts": regime_counts,
        "pnl_per_trade": float(final_pnl / max(num_trades, 1)),
    }


# ============================================================
# PARAMETER SWEEP
# ============================================================
def run_full_sweep(
    pair: PairProfile,
    n_ticks: int = 14400,  # 4 hours on MegaETH
    n_seeds: int = 10,
    output_file: str = "backtest_results_full.json",
) -> Dict:
    """
    Run comprehensive parameter sweep across all combinations.

    Full grid:
    - 10 gammas × 7 ks × 3 spread modes × 5 q_maxs × 5 rebalance × 2 micro × 4 vol regimes
    - = 21,000 combinations × 10 seeds = 210,000 simulations

    Returns:
        Results dict organized by vol regime
    """

    # Parameter grid (FULL as specified)
    gammas = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    ks = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
    spread_modes = ["as_optimal", "fib_adaptive", "regime_switching"]
    q_maxs = [3, 5, 8, 10, 15]
    rebalance_thresholds = [0.7, 0.8, 0.85, 0.9, 1.0]
    micro_pulse_options = [False, True]

    # Vol regimes
    vol_regimes = {
        "low": 0.02,
        "normal": 0.055,
        "high": 0.12,
        "breakout": 0.25,
    }

    print("=" * 80)
    print("  REGIME-SWITCHING A-S BACKTEST ENGINE - FULL PARAMETER SWEEP")
    print("=" * 80)
    print(f"  Pair: {pair.name}")
    print(f"  Base price: ${pair.base_price}")
    print(f"  Spread: {pair.base_spread_bps} bps | Maker fee: {pair.maker_fee_bps} bps")
    print(f"  Ticks per run: {n_ticks} ({n_ticks * pair.tick_interval_sec / 3600:.1f} hours)")
    print(f"  Monte Carlo seeds: {n_seeds}")
    print("=" * 80)
    print(f"  Parameter combinations per vol regime:")
    print(f"    Gammas: {len(gammas)}")
    print(f"    Ks: {len(ks)}")
    print(f"    Spread modes: {len(spread_modes)}")
    print(f"    Q_maxs: {len(q_maxs)}")
    print(f"    Rebalance thresholds: {len(rebalance_thresholds)}")
    print(f"    Micro-pulse: {len(micro_pulse_options)}")
    print(f"    Vol regimes: {len(vol_regimes)}")

    combos_per_regime = (len(gammas) * len(ks) * len(spread_modes) * len(q_maxs) *
                         len(rebalance_thresholds) * len(micro_pulse_options))
    total_combos = combos_per_regime * len(vol_regimes)
    total_sims = total_combos * n_seeds

    print(f"    = {combos_per_regime} combos/regime × {len(vol_regimes)} regimes = {total_combos}")
    print(f"  Total simulations: {total_sims:,}")
    print("=" * 80)

    # Results storage
    all_results = {}

    # Run sweep for each vol regime
    for regime_name, sigma_daily in vol_regimes.items():
        print(f"\n{'='*80}")
        print(f"  VOL REGIME: {regime_name.upper()} (σ_daily = {sigma_daily:.1%})")
        print(f"{'='*80}")

        regime_results = []
        best_sharpe = -np.inf
        best_config = None

        # Pre-generate price paths for this regime
        print(f"  Generating {n_seeds} price paths...")
        price_paths = []
        for seed in range(n_seeds):
            prices = simulate_price_path(pair, n_ticks, sigma_override=sigma_daily, seed=seed)
            price_paths.append(prices)

        # Generate all combinations
        combos = list(itertools.product(
            gammas, ks, spread_modes, q_maxs, rebalance_thresholds, micro_pulse_options
        ))

        print(f"  Running {len(combos)} combinations × {n_seeds} seeds = {len(combos) * n_seeds} sims...")
        start_time = time.time()

        for idx, (gamma, k, sm, qm, rt, mp) in enumerate(combos):
            combo_pnls = []
            combo_sharpes = []
            combo_trades = []
            combo_regimes = []

            # Run across all MC seeds
            for prices in price_paths:
                result = run_backtest(
                    pair=pair, prices=prices,
                    gamma=gamma, k=k, spread_mode=sm,
                    q_max=qm, rebalance_threshold=rt,
                    micro_pulse_enabled=mp,
                )
                combo_pnls.append(result["total_pnl"])
                combo_sharpes.append(result["sharpe_ratio"])
                combo_trades.append(result["num_trades"])
                combo_regimes.append(result["regime_counts"])

            # Aggregate
            avg_pnl = np.mean(combo_pnls)
            std_pnl = np.std(combo_pnls)
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
                "std_pnl": round(std_pnl, 6),
                "avg_sharpe": round(avg_sharpe, 4),
                "avg_trades": round(avg_trades, 1),
                "pnl_per_trade": round(avg_pnl / max(avg_trades, 1), 8),
            }
            regime_results.append(row)

            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_config = row

            # Progress
            if (idx + 1) % 200 == 0 or idx == len(combos) - 1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(combos) - idx - 1) / rate if rate > 0 else 0
                print(f"    Progress: {idx+1}/{len(combos)} ({(idx+1)/len(combos)*100:.1f}%) | "
                      f"Best Sharpe: {best_sharpe:.4f} | "
                      f"Rate: {rate:.1f} combos/sec | "
                      f"ETA: {remaining/60:.1f} min")

        all_results[regime_name] = regime_results

        # Print best config for this regime
        print(f"\n  {'='*80}")
        print(f"  BEST CONFIG FOR {regime_name.upper()}:")
        print(f"  {'='*80}")
        for key in ["gamma", "k", "spread_mode", "q_max", "rebalance_threshold",
                    "micro_pulse", "avg_pnl", "avg_sharpe", "avg_trades", "pnl_per_trade"]:
            print(f"    {key:25s}: {best_config[key]}")

    # Save intermediate results
    print(f"\n{'='*80}")
    print(f"  Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Saved")

    return all_results


# ============================================================
# VOLUME-CAPPED REALITY CHECK
# ============================================================
def run_volume_capped_check(
    pair: PairProfile,
    top_configs: List[Dict],
    n_ticks: int = 14400,
    n_seeds: int = 10,
) -> List[Dict]:
    """
    Re-run top configs with realistic volume cap.

    Args:
        pair: Trading pair
        top_configs: List of top config dicts
        n_ticks: Ticks per run
        n_seeds: MC seeds

    Returns:
        Results with realistic PnL
    """
    print(f"\n{'='*80}")
    print(f"  VOLUME-CAPPED REALITY CHECK")
    print(f"  Max daily volume: {pair.daily_volume_base} {pair.name.split('/')[0]} per side")
    print(f"{'='*80}")

    # Calculate volume cap per run
    hours_per_run = n_ticks * pair.tick_interval_sec / 3600
    volume_cap_per_side = pair.daily_volume_base * (hours_per_run / 24.0)

    print(f"  Run duration: {hours_per_run:.1f} hours")
    print(f"  Volume cap per side: {volume_cap_per_side:.4f} {pair.name.split('/')[0]}")
    print(f"{'='*80}\n")

    results = []

    for i, config in enumerate(top_configs):
        print(f"  Testing config {i+1}/{len(top_configs)}: "
              f"γ={config['gamma']:.2f}, k={config['k']:.1f}, "
              f"mode={config['spread_mode']}, q_max={config['q_max']}")

        # Generate price paths
        sigma_daily = config.get("sigma_daily", pair.sigma_daily)
        price_paths = [simulate_price_path(pair, n_ticks, sigma_override=sigma_daily, seed=s)
                      for s in range(n_seeds)]

        # Run with volume cap
        capped_pnls = []
        capped_trades = []

        for prices in price_paths:
            result = run_backtest(
                pair=pair, prices=prices,
                gamma=config["gamma"],
                k=config["k"],
                spread_mode=config["spread_mode"],
                q_max=config["q_max"],
                rebalance_threshold=config["rebalance_threshold"],
                micro_pulse_enabled=config["micro_pulse"],
                volume_cap_per_side=volume_cap_per_side,
            )
            capped_pnls.append(result["total_pnl"])
            capped_trades.append(result["num_trades"])

        # Scale to daily
        daily_pnl_theoretical = config["avg_pnl"] * (24.0 / hours_per_run)
        daily_pnl_realistic = np.mean(capped_pnls) * (24.0 / hours_per_run)

        results.append({
            **config,
            "realistic_avg_pnl": round(np.mean(capped_pnls), 6),
            "realistic_avg_trades": round(np.mean(capped_trades), 1),
            "daily_pnl_theoretical": round(daily_pnl_theoretical, 6),
            "daily_pnl_realistic": round(daily_pnl_realistic, 6),
            "volume_cap_impact": round((daily_pnl_realistic / max(daily_pnl_theoretical, 0.0001) - 1) * 100, 2),
        })

        print(f"    Theoretical daily PnL: ${daily_pnl_theoretical:.4f}")
        print(f"    Realistic daily PnL:   ${daily_pnl_realistic:.4f}")
        print(f"    Impact: {results[-1]['volume_cap_impact']:.1f}%\n")

    return results


# ============================================================
# ANALYSIS & REPORTING
# ============================================================
def print_top_configs(all_results: Dict, top_n: int = 20):
    """Print top N configs per vol regime."""
    print(f"\n{'='*80}")
    print(f"  TOP {top_n} CONFIGS PER VOL REGIME")
    print(f"{'='*80}")

    for regime_name, results in all_results.items():
        sorted_results = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)

        print(f"\n  {regime_name.upper()} REGIME (σ = {sorted_results[0]['sigma_daily']:.1%}):")
        print(f"  {'#':>3} {'γ':>5} {'k':>5} {'Mode':>16} {'qM':>3} {'Reb':>5} "
              f"{'μP':>3} {'AvgPnL':>12} {'Sharpe':>8} {'Trd':>5} {'$/Trd':>12}")
        print(f"  {'-'*90}")

        for i, r in enumerate(sorted_results[:top_n]):
            print(f"  {i+1:>3} {r['gamma']:>5.2f} {r['k']:>5.1f} {r['spread_mode']:>16} "
                  f"{r['q_max']:>3} {r['rebalance_threshold']:>5.2f} "
                  f"{'Y' if r['micro_pulse'] else 'N':>3} "
                  f"{r['avg_pnl']:>12.6f} {r['avg_sharpe']:>8.4f} "
                  f"{r['avg_trades']:>5.0f} {r['pnl_per_trade']:>12.8f}")


def find_robust_config(all_results: Dict) -> Dict:
    """
    Find single best regime-switching config that maximizes Sharpe across ALL vol regimes.
    """
    print(f"\n{'='*80}")
    print(f"  FINDING ROBUST REGIME-SWITCHING CONFIG")
    print(f"  (Best Sharpe across all vol regimes)")
    print(f"{'='*80}")

    # Only consider regime_switching configs
    regime_switching_configs = {}

    for regime_name, results in all_results.items():
        rs_results = [r for r in results if r["spread_mode"] == "regime_switching"]
        regime_switching_configs[regime_name] = rs_results

    # Find config that appears in top performers across all regimes
    # Strategy: average Sharpe rank across regimes
    config_scores = {}

    for regime_name, results in regime_switching_configs.items():
        sorted_results = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)

        for rank, r in enumerate(sorted_results):
            config_key = (r["gamma"], r["k"], r["q_max"], r["rebalance_threshold"], r["micro_pulse"])

            if config_key not in config_scores:
                config_scores[config_key] = {
                    "config": r,
                    "sharpes": {},
                    "ranks": {},
                }

            config_scores[config_key]["sharpes"][regime_name] = r["avg_sharpe"]
            config_scores[config_key]["ranks"][regime_name] = rank

    # Score by average Sharpe across all regimes
    for config_key, data in config_scores.items():
        data["avg_sharpe_across_regimes"] = np.mean(list(data["sharpes"].values()))
        data["min_sharpe"] = min(data["sharpes"].values())
        data["max_rank"] = max(data["ranks"].values())

    # Best = highest avg Sharpe across regimes
    best_config_key = max(config_scores.keys(),
                         key=lambda k: config_scores[k]["avg_sharpe_across_regimes"])
    best = config_scores[best_config_key]

    print(f"\n  ROBUST CONFIG:")
    print(f"    Gamma: {best['config']['gamma']:.2f}")
    print(f"    K: {best['config']['k']:.1f}")
    print(f"    Spread mode: {best['config']['spread_mode']}")
    print(f"    Q_max: {best['config']['q_max']}")
    print(f"    Rebalance threshold: {best['config']['rebalance_threshold']:.2f}")
    print(f"    Micro-pulse: {'Yes' if best['config']['micro_pulse'] else 'No'}")
    print(f"\n  PERFORMANCE ACROSS VOL REGIMES:")
    for regime_name in ["low", "normal", "high", "breakout"]:
        sharpe = best["sharpes"].get(regime_name, 0)
        rank = best["ranks"].get(regime_name, 999)
        print(f"    {regime_name:12s}: Sharpe = {sharpe:7.4f}  (Rank #{rank+1})")
    print(f"\n    Avg Sharpe across regimes: {best['avg_sharpe_across_regimes']:.4f}")
    print(f"    Min Sharpe: {best['min_sharpe']:.4f}")

    return best["config"]


def print_anti_patterns(all_results: Dict):
    """Print configs that lose money (anti-patterns to avoid)."""
    print(f"\n{'='*80}")
    print(f"  ANTI-PATTERNS (Configs that lose money)")
    print(f"{'='*80}")

    for regime_name, results in all_results.items():
        losers = [r for r in results if r["avg_pnl"] < 0]
        losers_sorted = sorted(losers, key=lambda x: x["avg_pnl"])

        if losers_sorted:
            print(f"\n  {regime_name.upper()} REGIME: {len(losers)} losing configs")
            print(f"    Worst offenders:")
            for i, r in enumerate(losers_sorted[:5]):
                print(f"      {i+1}. γ={r['gamma']:.2f}, k={r['k']:.1f}, "
                      f"mode={r['spread_mode']}, q_max={r['q_max']}, "
                      f"PnL={r['avg_pnl']:.6f}, Sharpe={r['avg_sharpe']:.4f}")


def generate_summary_markdown(all_results: Dict, robust_config: Dict,
                              volume_capped_results: List[Dict],
                              output_file: str = "BACKTEST_SUMMARY.md"):
    """Generate summary markdown with optimal Hummingbot-ready parameters."""

    with open(output_file, "w") as f:
        f.write("# Regime-Switching A-S Backtest Summary\n\n")
        f.write("## Optimal Configuration for Canonic WETH/USDm\n\n")

        f.write("### Robust Regime-Switching Config\n")
        f.write("*Best Sharpe across all volatility regimes*\n\n")
        f.write(f"- **Gamma (γ)**: {robust_config['gamma']}\n")
        f.write(f"- **Order intensity (k)**: {robust_config['k']}\n")
        f.write(f"- **Spread mode**: {robust_config['spread_mode']}\n")
        f.write(f"- **Max inventory (q_max)**: {robust_config['q_max']} WETH\n")
        f.write(f"- **Rebalance threshold**: {robust_config['rebalance_threshold']:.0%} of q_max\n")
        f.write(f"- **Micro-pulse enabled**: {'Yes' if robust_config['micro_pulse'] else 'No'}\n\n")

        f.write("### Performance Across Vol Regimes\n\n")
        f.write("| Regime | Sigma (daily) | Avg Sharpe | Avg PnL | Avg Trades |\n")
        f.write("|--------|---------------|------------|---------|------------|\n")
        for regime_name in ["low", "normal", "high", "breakout"]:
            regime_results = all_results[regime_name]
            matching = [r for r in regime_results if
                       r["gamma"] == robust_config["gamma"] and
                       r["k"] == robust_config["k"] and
                       r["spread_mode"] == robust_config["spread_mode"] and
                       r["q_max"] == robust_config["q_max"]]
            if matching:
                r = matching[0]
                f.write(f"| {regime_name.capitalize():8s} | {r['sigma_daily']:.1%} | "
                       f"{r['avg_sharpe']:10.4f} | ${r['avg_pnl']:8.4f} | "
                       f"{r['avg_trades']:10.1f} |\n")

        f.write("\n### Volume-Capped Reality Check\n\n")
        f.write("Based on observed daily volume of 1.12 WETH:\n\n")
        f.write("| Config | Theoretical Daily PnL | Realistic Daily PnL | Impact |\n")
        f.write("|--------|----------------------|-----------------------|--------|\n")
        for r in volume_capped_results[:5]:
            f.write(f"| γ={r['gamma']:.2f}, k={r['k']:.1f} | "
                   f"${r['daily_pnl_theoretical']:8.4f} | "
                   f"${r['daily_pnl_realistic']:8.4f} | "
                   f"{r['volume_cap_impact']:+6.1f}% |\n")

        f.write("\n## Hummingbot Configuration\n\n")
        f.write("```yaml\n")
        f.write("# Avellaneda-Stoikov Pure Market Making Strategy\n")
        f.write("exchange: canonic_megaeth\n")
        f.write("market: WETH-USDm\n")
        f.write("execution_timeframe: infinite_run\n\n")
        f.write(f"risk_factor: {robust_config['gamma']}\n")
        f.write(f"order_amount: 1.0  # WETH\n")
        f.write(f"inventory_target_base_pct: 50  # Neutral target\n")
        f.write(f"order_refresh_time: 10.0  # seconds\n")
        f.write(f"max_order_age: 300.0  # 5 minutes\n")
        f.write(f"filled_order_delay: 60.0\n\n")
        f.write(f"# Inventory management\n")
        f.write(f"inventory_range_multiplier: {robust_config['q_max']}\n")
        f.write(f"hanging_orders_enabled: false\n")
        f.write("```\n\n")

        f.write("## Top Configurations by Vol Regime\n\n")
        for regime_name in ["low", "normal", "high", "breakout"]:
            results = all_results[regime_name]
            sorted_results = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)

            f.write(f"### {regime_name.capitalize()} Volatility (σ = {sorted_results[0]['sigma_daily']:.1%})\n\n")
            f.write("| Rank | γ | k | Mode | q_max | Sharpe | Avg PnL | Trades |\n")
            f.write("|------|---|---|------|-------|--------|---------|--------|\n")
            for i, r in enumerate(sorted_results[:10]):
                f.write(f"| {i+1} | {r['gamma']:.2f} | {r['k']:.1f} | "
                       f"{r['spread_mode']:16s} | {r['q_max']:2d} | "
                       f"{r['avg_sharpe']:7.4f} | ${r['avg_pnl']:8.4f} | "
                       f"{r['avg_trades']:5.0f} |\n")
            f.write("\n")

        f.write("## Notes\n\n")
        f.write("- All results based on 10 Monte Carlo paths × 14,400 ticks (4 hours)\n")
        f.write("- Asymmetric fill model: 1.3× bid boost (thin top-of-book), 0.9× ask dampen\n")
        f.write("- 0% maker fee, 0.03% taker fee (Canonic)\n")
        f.write("- Regime switching triggers on 2× baseline vol or price > $5,000\n")
        f.write("- Fibonacci spread tiers: 0.382× (tight) → 1.618× (defensive)\n")

    print(f"\n{'='*80}")
    print(f"  ✓ Summary markdown saved to {output_file}")
    print(f"{'='*80}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("  REGIME-SWITCHING AVELLANEDA-STOIKOV BACKTEST ENGINE")
    print("  Full Parameter Sweep for Canonic WETH/USDm")
    print("="*80 + "\n")

    # Run full sweep
    all_results = run_full_sweep(
        pair=WETH_USDM,
        n_ticks=14400,  # 4 hours
        n_seeds=10,
        output_file="backtest_results_full.json",
    )

    # Print top 20 per regime
    print_top_configs(all_results, top_n=20)

    # Find robust regime-switching config
    robust_config = find_robust_config(all_results)

    # Print anti-patterns
    print_anti_patterns(all_results)

    # Volume-capped reality check (top 10 configs)
    top_10_configs = []
    for regime_name, results in all_results.items():
        sorted_results = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)
        for r in sorted_results[:3]:  # Top 3 per regime
            if r not in top_10_configs:
                top_10_configs.append(r)

    top_10_configs = sorted(top_10_configs, key=lambda x: x["avg_sharpe"], reverse=True)[:10]

    volume_capped_results = run_volume_capped_check(
        pair=WETH_USDM,
        top_configs=top_10_configs,
        n_ticks=14400,
        n_seeds=10,
    )

    # Generate summary markdown
    generate_summary_markdown(
        all_results=all_results,
        robust_config=robust_config,
        volume_capped_results=volume_capped_results,
        output_file="BACKTEST_SUMMARY.md",
    )

    print("\n" + "="*80)
    print("  ✓ BACKTEST COMPLETE")
    print("="*80)
    print(f"  Full results: backtest_results_full.json")
    print(f"  Summary: BACKTEST_SUMMARY.md")
    print("="*80 + "\n")
