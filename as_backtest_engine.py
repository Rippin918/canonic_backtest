"""
Avellaneda-Stoikov Market Making Backtest Engine
=================================================
Full parameter sweep with:
- Gamma (risk aversion), k (order intensity), sigma (volatility)
- Fibonacci-based spread tiers
- Micro-pulse timing windows (Coulter-style counting)
- Inventory limits & rebalance intervals
- Multi-pair support: WETH/USDm + Canonic coin

Reference: Avellaneda & Stoikov (2008) - "High-frequency trading in a limit order book"
"""

import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FIBONACCI UTILITIES
# ============================================================
def fib_sequence(n: int) -> List[int]:
    """Generate first n Fibonacci numbers."""
    fibs = [1, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:n]

def fib_ratios() -> List[float]:
    """Golden ratio spread tiers: 0.236, 0.382, 0.500, 0.618, 0.786, 1.0, 1.618"""
    return [0.236, 0.382, 0.500, 0.618, 0.786, 1.0, 1.618]

def fib_spread_tiers(base_spread_bps: float) -> Dict[str, float]:
    """Map Fibonacci ratios to spread levels in bps."""
    ratios = fib_ratios()
    labels = ["fib_236", "fib_382", "fib_500", "fib_618", "fib_786", "fib_1000", "fib_1618"]
    return {label: base_spread_bps * ratio for label, ratio in zip(labels, ratios)}


# ============================================================
# MICRO-PULSE ANALYSIS (Coulter-inspired counting)
# ============================================================
@dataclass
class MicroPulse:
    """
    Tracks micro-pulse rhythm of price movements.
    Counts consecutive up/down ticks and detects
    shock & bounce patterns in simulated orderbook flow.
    """
    tick_history: List[float] = field(default_factory=list)
    pulse_counts: List[int] = field(default_factory=list)  # +N for up runs, -N for down runs

    def update(self, price_change: float):
        self.tick_history.append(price_change)
        if len(self.pulse_counts) == 0:
            self.pulse_counts.append(1 if price_change > 0 else -1)
        else:
            last = self.pulse_counts[-1]
            same_dir = (price_change > 0 and last > 0) or (price_change < 0 and last < 0)
            if same_dir:
                self.pulse_counts[-1] += (1 if last > 0 else -1)
            else:
                self.pulse_counts.append(1 if price_change > 0 else -1)

    def current_run(self) -> int:
        """Current consecutive run length (signed)."""
        return self.pulse_counts[-1] if self.pulse_counts else 0

    def detect_shock_bounce(self, threshold: int = 3) -> str:
        """Detect shock & bounce pattern: large run followed by reversal."""
        if len(self.pulse_counts) < 2:
            return "neutral"
        prev, curr = self.pulse_counts[-2], self.pulse_counts[-1]
        if abs(prev) >= threshold and np.sign(prev) != np.sign(curr):
            return "bounce_up" if curr > 0 else "bounce_down"
        if abs(curr) >= threshold:
            return "shock_up" if curr > 0 else "shock_down"
        return "neutral"

    def fib_retracement_signal(self, lookback: int = 20) -> Optional[str]:
        """Check if recent price move is near a Fibonacci retracement level."""
        if len(self.tick_history) < lookback:
            return None
        recent = self.tick_history[-lookback:]
        high = max(np.cumsum(recent))
        low = min(np.cumsum(recent))
        rng = high - low
        if rng == 0:
            return None
        current_pos = np.cumsum(recent)[-1] - low
        ratio = current_pos / rng
        for fib_level, label in [(0.236, "fib_236"), (0.382, "fib_382"),
                                  (0.500, "fib_500"), (0.618, "fib_618")]:
            if abs(ratio - fib_level) < 0.05:
                return label
        return None


# ============================================================
# TOKEN PAIR PROFILES
# ============================================================
@dataclass
class PairProfile:
    name: str
    base_price: float          # Starting mid-price
    sigma_daily: float         # Daily volatility (decimal)
    base_spread_bps: float     # Typical market spread in bps
    avg_trade_size: float      # Average trade size in base units
    tick_interval_sec: float   # Seconds between price updates
    liquidity_score: float     # 0-1, higher = more liquid

    @property
    def sigma_per_tick(self) -> float:
        """Volatility per tick interval."""
        ticks_per_day = 86400 / self.tick_interval_sec
        return self.sigma_daily / np.sqrt(ticks_per_day)


# Pre-configured pair profiles
# ============================================================
# REAL CANONIC ORDERBOOK SNAPSHOT (from live canonic.trade)
# Chain: MegaETH | Fee: 0% maker / 0.03% taker
# ============================================================
# WETH/USDm orderbook structure:
#   ASK side (selling WETH):
#     2,048.52 | +3.0 bps  | 0.619521 WETH
#     2,048.93 | +5.0 bps  | 0.602058 WETH
#     2,049.95 | +10.0 bps | 0.602058 WETH
#     2,052.00 | +20.0 bps | 0.602058 WETH
#     2,054.05 | +30.0 bps | 0.602058 WETH
#     2,058.14 | +50.0 bps | 0.602058 WETH
#     2,068.38 | +100 bps  | 0.602058 WETH
#   MID: 2,047.90 | Spread: 0.66 (0.032%)
#   BID side (buying WETH):
#     2,047.86 | -0.2 bps  | 0.0556847 WETH  <-- THIN!
#     2,047.84 | -0.3 bps  | 0.00001826 WETH <-- DUST
#     2,047.29 | -3.0 bps  | 0.597575 WETH
#     2,046.88 | -5.0 bps  | 0.597575 WETH
#     ... symmetric down to -100 bps
#
# KEY INSIGHT: Bid top-of-book is paper-thin (0.055 WETH)
#   before the real depth at -3 bps. Ask side is thicker.
#   This asymmetry = opportunity.
#
# CLP Vault TVL: $30,894 (7.55673 WETH / 15,417.4 USDm)
# 24h Vol: 1.12 WETH / 2,234.57 USDm
# ============================================================

CANONIC_BOOK_LEVELS = {
    "asks": [  # (bps_from_mid, size_weth)
        (3.0, 0.619521), (5.0, 0.602058), (10.0, 0.602058),
        (20.0, 0.602058), (30.0, 0.602058), (50.0, 0.602058),
        (100.0, 0.602058),
    ],
    "bids": [  # (bps_from_mid, size_weth) -- negative = below mid
        (0.2, 0.0556847), (0.3, 0.00001826), (3.0, 0.597575),
        (5.0, 0.597575), (10.0, 0.597575), (20.0, 0.597575),
        (30.0, 0.597575), (50.0, 0.597575), (100.0, 0.597575),
    ],
    "spread_bps": 3.2,    # 0.032%
    "maker_fee_bps": 0.0,
    "taker_fee_bps": 0.3,  # 0.03%
}

WETH_USDM = PairProfile(
    name="WETH/USDm",
    base_price=2052.36,       # Live price from Canonic
    sigma_daily=0.055,        # +5.58% in 24h = elevated vol
    base_spread_bps=3.2,      # REAL spread: 0.032% — extremely tight!
    avg_trade_size=0.6,       # ~0.6 WETH per level (from book)
    tick_interval_sec=1.0,    # MegaETH: sub-second blocks
    liquidity_score=0.45,     # Moderate — $30K TVL, thin top-of-book
)

# Canonic native token pair (estimated from vault/points mechanics)
CANONIC_USDM = PairProfile(
    name="CANONIC/USDm",
    base_price=0.50,          # Placeholder — adjust when token is live
    sigma_daily=0.12,         # Higher vol for native token
    base_spread_bps=15.0,     # Likely wider than WETH pair
    avg_trade_size=500.0,     # Estimated
    tick_interval_sec=1.0,    # MegaETH
    liquidity_score=0.2,      # Lower liquidity initially
)

CANONIC_WETH = PairProfile(
    name="CANONIC/WETH",
    base_price=0.000244,      # ~0.50 / 2052
    sigma_daily=0.14,         # Combined vol
    base_spread_bps=20.0,
    avg_trade_size=500.0,
    tick_interval_sec=1.0,
    liquidity_score=0.15,
)


# ============================================================
# AVELLANEDA-STOIKOV MODEL
# ============================================================
@dataclass
class ASParams:
    """Avellaneda-Stoikov model parameters."""
    gamma: float          # Risk aversion (higher = more conservative)
    k: float              # Order intensity parameter
    sigma: float          # Volatility (per tick)
    T: float              # Time horizon (in ticks remaining, normalized 0-1)
    q_max: int            # Max inventory
    spread_mode: str      # "as_optimal", "fixed", "fib_adaptive"
    fib_base_bps: float   # Base spread for Fibonacci tiers
    rebalance_threshold: float  # Fraction of q_max to trigger rebalance
    micro_pulse_enabled: bool   # Use micro-pulse signals

    def reservation_price(self, mid: float, q: int, t_remaining: float) -> float:
        """AS reservation price: r = s - q * gamma * sigma^2 * (T - t)"""
        return mid - q * self.gamma * (self.sigma ** 2) * t_remaining

    def optimal_spread(self, t_remaining: float) -> float:
        """AS optimal spread: delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)"""
        return (self.gamma * (self.sigma ** 2) * t_remaining +
                (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k))


# ============================================================
# BACKTEST ENGINE
# ============================================================
@dataclass
class TradeRecord:
    tick: int
    side: str           # "buy" or "sell"
    price: float
    size: float
    inventory_after: int
    pnl_realized: float
    mid_at_trade: float

@dataclass
class BacktestResult:
    params: Dict
    pair_name: str
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    num_trades: int
    avg_spread_captured: float
    max_inventory: int
    fill_rate: float
    inventory_risk_score: float
    pnl_curve: List[float]
    trades: List[TradeRecord]

    def summary_dict(self) -> Dict:
        return {
            "pair": self.pair_name,
            "gamma": self.params.get("gamma"),
            "k": self.params.get("k"),
            "spread_mode": self.params.get("spread_mode"),
            "total_pnl": round(self.total_pnl, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "sharpe": round(self.sharpe_ratio, 4),
            "max_dd": round(self.max_drawdown, 4),
            "trades": self.num_trades,
            "fill_rate": round(self.fill_rate, 4),
            "avg_spread": round(self.avg_spread_captured, 4),
            "max_inv": self.max_inventory,
            "inv_risk": round(self.inventory_risk_score, 4),
        }


def simulate_price_path(pair: PairProfile, n_ticks: int, seed: int = 42) -> np.ndarray:
    """Generate GBM price path with mean-reverting microstructure noise."""
    rng = np.random.RandomState(seed)
    sigma = pair.sigma_per_tick

    # GBM + mean-reverting component
    prices = np.zeros(n_ticks)
    prices[0] = pair.base_price

    mean_revert_speed = 0.001  # Slow mean reversion

    for i in range(1, n_ticks):
        # GBM component
        gbm_return = -0.5 * sigma**2 + sigma * rng.randn()
        # Mean-revert noise (microstructure)
        mr_component = mean_revert_speed * (pair.base_price - prices[i-1]) / pair.base_price
        # Occasional jumps (shock events)
        jump = 0.0
        if rng.rand() < 0.002:  # 0.2% chance per tick
            jump = sigma * 5 * (1 if rng.rand() > 0.5 else -1)

        prices[i] = prices[i-1] * np.exp(gbm_return + mr_component + jump)

    return prices


def run_single_backtest(
    pair: PairProfile,
    prices: np.ndarray,
    gamma: float,
    k: float,
    spread_mode: str,
    q_max: int,
    rebalance_threshold: float,
    micro_pulse_enabled: bool,
    order_size: float = 1.0,
) -> BacktestResult:
    """Run a single backtest with given parameters."""

    n_ticks = len(prices)
    sigma = pair.sigma_per_tick

    params = ASParams(
        gamma=gamma, k=k, sigma=sigma, T=1.0, q_max=q_max,
        spread_mode=spread_mode, fib_base_bps=pair.base_spread_bps,
        rebalance_threshold=rebalance_threshold,
        micro_pulse_enabled=micro_pulse_enabled,
    )

    # State
    inventory = 0
    cash = 0.0
    trades: List[TradeRecord] = []
    pnl_curve = []
    pulse = MicroPulse()
    fib_tiers = fib_spread_tiers(pair.base_spread_bps)

    for tick in range(n_ticks):
        mid = prices[tick]
        t_remaining = max(0.001, 1.0 - tick / n_ticks)

        # Update micro-pulse
        if tick > 0:
            pulse.update(prices[tick] - prices[tick - 1])

        # --- Compute reservation price ---
        r = params.reservation_price(mid, inventory, t_remaining)

        # --- Compute spread ---
        if spread_mode == "as_optimal":
            half_spread = params.optimal_spread(t_remaining) / 2.0
            bid = r - half_spread
            ask = r + half_spread

        elif spread_mode == "fixed":
            half_spread_abs = mid * (pair.base_spread_bps / 10000.0) / 2.0
            bid = r - half_spread_abs
            ask = r + half_spread_abs

        elif spread_mode == "fib_adaptive":
            # Use Fibonacci tiers based on inventory & micro-pulse
            inv_ratio = abs(inventory) / max(q_max, 1)

            # Select Fibonacci tier based on inventory pressure
            if inv_ratio < 0.2:
                tier_bps = fib_tiers["fib_382"]   # Tight when flat
            elif inv_ratio < 0.5:
                tier_bps = fib_tiers["fib_618"]   # Medium
            elif inv_ratio < 0.8:
                tier_bps = fib_tiers["fib_786"]   # Wider
            else:
                tier_bps = fib_tiers["fib_1618"]  # Very wide near limits

            # Micro-pulse adjustment
            if micro_pulse_enabled:
                signal = pulse.detect_shock_bounce()
                fib_signal = pulse.fib_retracement_signal()

                if signal == "bounce_up" and inventory < 0:
                    tier_bps *= 0.7   # Tighten ask to capture bounce
                elif signal == "bounce_down" and inventory > 0:
                    tier_bps *= 0.7   # Tighten bid
                elif signal in ("shock_up", "shock_down"):
                    tier_bps *= 1.5   # Widen during shock

                # Fibonacci retracement: tighten at key levels
                if fib_signal in ("fib_382", "fib_618"):
                    tier_bps *= 0.8

            half_spread_abs = mid * (tier_bps / 10000.0) / 2.0
            bid = r - half_spread_abs
            ask = r + half_spread_abs
        else:
            raise ValueError(f"Unknown spread_mode: {spread_mode}")

        # --- Skew quotes based on inventory ---
        if inventory > 0:
            # Long: lower ask to reduce, raise bid to discourage buys
            skew = 0.0002 * inventory * mid
            bid -= skew
            ask -= skew * 0.5
        elif inventory < 0:
            skew = 0.0002 * abs(inventory) * mid
            bid += skew * 0.5
            ask += skew

        # --- Simulate fills (calibrated to Canonic book) ---
        # Real book shows thin bid top (0.055 WETH at -0.2 bps)
        # vs thick ask layers (0.6 WETH at +3 bps).
        # Bid fills are easier to get (less competition at top).
        # 0% maker fee means all spread captured is pure edge.
        bid_dist = (mid - bid) / mid
        ask_dist = (ask - mid) / mid

        # Asymmetric fill model reflecting thin bid / thick ask
        base_fill_prob = pair.liquidity_score * 0.12

        rng = np.random.RandomState(tick * 1000 + int(gamma * 100))

        # Bid fills: boosted because top-of-book bid is thin
        bid_fill_prob = base_fill_prob * 1.3 * np.exp(-k * bid_dist * 80)
        # Ask fills: slightly harder, more competition
        ask_fill_prob = base_fill_prob * 0.9 * np.exp(-k * ask_dist * 80)

        # Fee adjustment: 0% maker means we keep everything
        # Taker fee only hits if we cross (which we don't as MM)
        maker_fee = 0.0  # Canonic: 0% maker

        # Execute fills
        if rng.rand() < bid_fill_prob and inventory < q_max:
            # Bid filled: we buy
            inventory += 1
            cash -= bid * order_size
            trades.append(TradeRecord(
                tick=tick, side="buy", price=bid, size=order_size,
                inventory_after=inventory,
                pnl_realized=0.0,
                mid_at_trade=mid,
            ))

        if rng.rand() < ask_fill_prob and inventory > -q_max:
            # Ask filled: we sell
            inventory -= 1
            cash += ask * order_size
            trades.append(TradeRecord(
                tick=tick, side="sell", price=ask, size=order_size,
                inventory_after=inventory,
                pnl_realized=0.0,
                mid_at_trade=mid,
            ))

        # --- Rebalance if needed ---
        if abs(inventory) >= q_max * rebalance_threshold:
            # Force reduce by half at mid price (slippage)
            reduce_qty = abs(inventory) // 2
            if reduce_qty > 0:
                slippage = mid * 0.001  # 10 bps slippage
                if inventory > 0:
                    sell_price = mid - slippage
                    cash += sell_price * reduce_qty * order_size
                    inventory -= reduce_qty
                else:
                    buy_price = mid + slippage
                    cash -= buy_price * reduce_qty * order_size
                    inventory += reduce_qty

        # Mark-to-market P&L
        mtm = cash + inventory * mid * order_size
        pnl_curve.append(mtm)

    # --- Compute metrics ---
    pnl_arr = np.array(pnl_curve)
    final_pnl = pnl_arr[-1] if len(pnl_arr) > 0 else 0.0

    # Realized P&L from round-trip trades
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

    # Sharpe (annualized, assuming 7200 ticks/day)
    if len(pnl_arr) > 1:
        returns = np.diff(pnl_arr)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(7200 * 365)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Average spread captured
    spreads = []
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            if trades[i].side == "buy" and trades[i+1].side == "sell":
                spreads.append(trades[i+1].price - trades[i].price)
    avg_spread = np.mean(spreads) if spreads else 0.0

    # Fill rate (trades per tick)
    fill_rate = len(trades) / max(n_ticks, 1)

    # Inventory risk (time-weighted absolute inventory)
    inv_risk = np.mean(np.abs([t.inventory_after for t in trades])) / max(q_max, 1) if trades else 0.0

    # Max inventory held
    max_inv = max(abs(t.inventory_after) for t in trades) if trades else 0

    return BacktestResult(
        params={
            "gamma": gamma, "k": k, "spread_mode": spread_mode,
            "q_max": q_max, "rebalance_threshold": rebalance_threshold,
            "micro_pulse": micro_pulse_enabled,
        },
        pair_name=pair.name,
        total_pnl=final_pnl,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        num_trades=len(trades),
        avg_spread_captured=avg_spread,
        max_inventory=max_inv,
        fill_rate=fill_rate,
        inventory_risk_score=inv_risk,
        pnl_curve=pnl_curve,
        trades=trades,
    )


# ============================================================
# PARAMETER SWEEP
# ============================================================
def run_sweep(
    pair: PairProfile,
    n_ticks: int = 7200,   # 1 day at 12s ticks
    n_seeds: int = 5,      # Monte Carlo paths
    gammas: List[float] = None,
    ks: List[float] = None,
    spread_modes: List[str] = None,
    q_maxs: List[int] = None,
    rebalance_thresholds: List[float] = None,
    micro_pulse_options: List[bool] = None,
) -> List[Dict]:
    """Run full parameter sweep across all combinations."""

    if gammas is None:
        gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    if ks is None:
        ks = [1.0, 1.5, 2.0, 5.0, 10.0]
    if spread_modes is None:
        spread_modes = ["as_optimal", "fixed", "fib_adaptive"]
    if q_maxs is None:
        q_maxs = [5, 10, 20]
    if rebalance_thresholds is None:
        rebalance_thresholds = [0.8, 0.9, 1.0]  # 1.0 = no forced rebalance
    if micro_pulse_options is None:
        micro_pulse_options = [False, True]

    combos = list(itertools.product(
        gammas, ks, spread_modes, q_maxs, rebalance_thresholds, micro_pulse_options
    ))

    print(f"\n{'='*60}")
    print(f"BACKTEST SWEEP: {pair.name}")
    print(f"{'='*60}")
    print(f"Parameter combinations: {len(combos)}")
    print(f"Monte Carlo paths per combo: {n_seeds}")
    print(f"Total simulations: {len(combos) * n_seeds}")
    print(f"Ticks per sim: {n_ticks} (~{n_ticks * pair.tick_interval_sec / 3600:.1f} hours)")
    print(f"{'='*60}\n")

    # Pre-generate price paths
    price_paths = [simulate_price_path(pair, n_ticks, seed=s) for s in range(n_seeds)]

    all_results = []
    best_sharpe = -np.inf
    best_config = None

    for idx, (gamma, k, sm, qm, rt, mp) in enumerate(combos):
        combo_pnls = []
        combo_sharpes = []
        combo_trades = []

        for seed_idx, prices in enumerate(price_paths):
            result = run_single_backtest(
                pair=pair, prices=prices,
                gamma=gamma, k=k, spread_mode=sm,
                q_max=qm, rebalance_threshold=rt,
                micro_pulse_enabled=mp,
            )
            combo_pnls.append(result.total_pnl)
            combo_sharpes.append(result.sharpe_ratio)
            combo_trades.append(result.num_trades)

        avg_pnl = np.mean(combo_pnls)
        avg_sharpe = np.mean(combo_sharpes)
        std_pnl = np.std(combo_pnls)
        avg_trades = np.mean(combo_trades)

        row = {
            "pair": pair.name,
            "gamma": gamma,
            "k": k,
            "spread_mode": sm,
            "q_max": qm,
            "rebalance_threshold": rt,
            "micro_pulse": mp,
            "avg_pnl": round(avg_pnl, 4),
            "std_pnl": round(std_pnl, 4),
            "avg_sharpe": round(avg_sharpe, 4),
            "avg_trades": round(avg_trades, 1),
            "pnl_per_trade": round(avg_pnl / max(avg_trades, 1), 6),
        }
        all_results.append(row)

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_config = row

        if (idx + 1) % 50 == 0 or idx == len(combos) - 1:
            print(f"  Progress: {idx+1}/{len(combos)} combos | "
                  f"Best Sharpe so far: {best_sharpe:.4f}")

    print(f"\n{'='*60}")
    print(f"BEST CONFIG for {pair.name}:")
    print(f"{'='*60}")
    for key, val in best_config.items():
        print(f"  {key:25s}: {val}")
    print(f"{'='*60}\n")

    return all_results


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    # --- Sweep parameters (focused grid) ---
    # Core A-S
    gammas = [0.01, 0.1, 0.5, 1.0]
    ks = [1.5, 5.0, 10.0]

    # Spread strategies
    spread_modes = ["as_optimal", "fib_adaptive"]

    # Inventory
    q_maxs = [5, 10]
    rebalance_thresholds = [0.85, 1.0]

    # Micro-pulse
    micro_pulse_options = [False, True]

    # Simulation params
    N_TICKS = 3600      # ~1 hour on MegaETH (1s ticks)
    N_SEEDS = 3         # Monte Carlo paths

    print("=" * 70)
    print("  AVELLANEDA-STOIKOV BACKTEST SWEEP")
    print("  Full parameter grid + Fibonacci tiers + Micro-pulse timing")
    print("=" * 70)

    # ---- WETH/USDm ----
    weth_results = run_sweep(
        pair=WETH_USDM,
        n_ticks=N_TICKS, n_seeds=N_SEEDS,
        gammas=gammas, ks=ks, spread_modes=spread_modes,
        q_maxs=q_maxs, rebalance_thresholds=rebalance_thresholds,
        micro_pulse_options=micro_pulse_options,
    )

    # ---- CANONIC/USDm ----
    canonic_results = run_sweep(
        pair=CANONIC_USDM,
        n_ticks=N_TICKS, n_seeds=N_SEEDS,
        gammas=gammas, ks=ks, spread_modes=spread_modes,
        q_maxs=q_maxs, rebalance_thresholds=rebalance_thresholds,
        micro_pulse_options=micro_pulse_options,
    )

    # ---- CANONIC/WETH ----
    canonic_weth_results = run_sweep(
        pair=CANONIC_WETH,
        n_ticks=N_TICKS, n_seeds=N_SEEDS,
        gammas=gammas, ks=ks, spread_modes=spread_modes,
        q_maxs=q_maxs, rebalance_thresholds=rebalance_thresholds,
        micro_pulse_options=micro_pulse_options,
    )

    # ---- Save all results ----
    all_data = {
        "weth_usdm": weth_results,
        "canonic_usdm": canonic_results,
        "canonic_weth": canonic_weth_results,
    }

    with open("backtest_results.json", "w") as f:
        json.dump(all_data, f, indent=2)

    # ---- Top 10 configs per pair ----
    for name, results in all_data.items():
        sorted_r = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)
        print(f"\n{'='*70}")
        print(f"  TOP 10 CONFIGS: {name.upper()}")
        print(f"{'='*70}")
        print(f"{'Rank':>4} {'Gamma':>6} {'k':>5} {'Mode':>14} {'qMax':>5} "
              f"{'Rebal':>6} {'Pulse':>6} {'AvgPnL':>10} {'Sharpe':>8} {'Trades':>7}")
        print("-" * 80)
        for i, r in enumerate(sorted_r[:10]):
            print(f"{i+1:>4} {r['gamma']:>6.2f} {r['k']:>5.1f} {r['spread_mode']:>14} "
                  f"{r['q_max']:>5} {r['rebalance_threshold']:>6.2f} "
                  f"{'Yes' if r['micro_pulse'] else 'No':>6} "
                  f"{r['avg_pnl']:>10.4f} {r['avg_sharpe']:>8.4f} {r['avg_trades']:>7.1f}")

    print(f"\n\nResults saved to backtest_results.json")
    print("Run the dashboard generator next for visual comparison.")
