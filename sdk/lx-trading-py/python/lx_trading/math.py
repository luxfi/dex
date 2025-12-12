"""
Financial mathematics for accurate market pricing.

Includes:
- Options pricing (Black-Scholes, Greeks)
- AMM pricing (Constant Product, Concentrated Liquidity)
- Risk metrics (VaR, CVaR, Sharpe, Sortino)
- Statistical measures (volatility, drawdown)
"""

from decimal import Decimal
from typing import List, Optional, Tuple, Union
import math

# Use numpy for performance if available
try:
    import numpy as np
    from scipy import stats
    from scipy.optimize import brentq
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# Options Pricing
# =============================================================================

def black_scholes(
    S: float,  # Spot price
    K: float,  # Strike price
    T: float,  # Time to expiry (years)
    r: float,  # Risk-free rate
    sigma: float,  # Volatility
    option_type: str = "call",  # "call" or "put"
) -> float:
    """
    Black-Scholes option pricing.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: "call" or "put"

    Returns:
        Option price

    Example:
        >>> price = black_scholes(100, 100, 1, 0.05, 0.2, "call")
        >>> print(f"Call price: ${price:.2f}")
    """
    if T <= 0:
        # At expiry
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)

    if HAS_NUMPY:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    # Pure Python fallback
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Calculate implied volatility from option price using Newton-Raphson.

    Args:
        price: Observed option price
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        option_type: "call" or "put"

    Returns:
        Implied volatility
    """
    if HAS_NUMPY:
        try:
            return brentq(
                lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - price,
                0.001, 5.0
            )
        except ValueError:
            return float('nan')

    # Newton-Raphson fallback
    sigma = 0.2  # Initial guess

    for _ in range(max_iter):
        bs_price = black_scholes(S, K, T, r, sigma, option_type)
        vega = _vega(S, K, T, r, sigma)

        if abs(vega) < 1e-10:
            break

        diff = bs_price - price
        if abs(diff) < tol:
            return sigma

        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 5.0))

    return sigma


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """
    Calculate option Greeks.

    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    if T <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if HAS_NUMPY:
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        cdf_d2 = stats.norm.cdf(d2)
        cdf_neg_d1 = stats.norm.cdf(-d1)
        cdf_neg_d2 = stats.norm.cdf(-d2)
    else:
        pdf_d1 = _norm_pdf(d1)
        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)
        cdf_neg_d1 = _norm_cdf(-d1)
        cdf_neg_d2 = _norm_cdf(-d2)

    if option_type == "call":
        delta = cdf_d1
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * cdf_d2
        )
        rho = K * T * math.exp(-r * T) * cdf_d2
    else:
        delta = cdf_d1 - 1
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * cdf_neg_d2
        )
        rho = -K * T * math.exp(-r * T) * cdf_neg_d2

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # Per 1% change in vol
    theta = theta / 365  # Daily theta

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


def _vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate vega for Newton-Raphson."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return S * _norm_pdf(d1) * sqrt_T


# =============================================================================
# AMM Pricing
# =============================================================================

def constant_product_price(
    reserve_x: Union[float, Decimal],
    reserve_y: Union[float, Decimal],
    amount_in: Union[float, Decimal],
    fee_rate: float = 0.003,  # 0.3%
    is_x_to_y: bool = True,
) -> Tuple[float, float]:
    """
    Calculate output amount for Uniswap V2 style constant product AMM.

    Formula: x * y = k (invariant)

    Args:
        reserve_x: Reserve of token X
        reserve_y: Reserve of token Y
        amount_in: Amount of input token
        fee_rate: Trading fee (e.g., 0.003 = 0.3%)
        is_x_to_y: True if swapping X for Y

    Returns:
        Tuple of (output_amount, effective_price)

    Example:
        >>> out, price = constant_product_price(1000, 1000, 10, fee_rate=0.003)
        >>> print(f"Output: {out:.4f}, Price: {price:.4f}")
    """
    reserve_x = float(reserve_x)
    reserve_y = float(reserve_y)
    amount_in = float(amount_in)

    amount_in_with_fee = amount_in * (1 - fee_rate)

    if is_x_to_y:
        amount_out = (reserve_y * amount_in_with_fee) / (reserve_x + amount_in_with_fee)
    else:
        amount_out = (reserve_x * amount_in_with_fee) / (reserve_y + amount_in_with_fee)

    effective_price = amount_out / amount_in if amount_in > 0 else 0

    return amount_out, effective_price


def concentrated_liquidity_price(
    liquidity: float,
    sqrt_price_current: float,
    sqrt_price_lower: float,
    sqrt_price_upper: float,
    amount_in: Union[float, Decimal],
    fee_rate: float = 0.003,
    is_token0_in: bool = True,
) -> Tuple[float, float, float]:
    """
    Calculate output for Uniswap V3 style concentrated liquidity.

    Args:
        liquidity: L value (sqrt(x * y))
        sqrt_price_current: Current sqrt(P) = sqrt(y/x)
        sqrt_price_lower: Lower tick sqrt price
        sqrt_price_upper: Upper tick sqrt price
        amount_in: Input amount
        fee_rate: Trading fee
        is_token0_in: True if swapping token0 for token1

    Returns:
        Tuple of (output_amount, new_sqrt_price, price_impact)
    """
    amount_in = float(amount_in)
    amount_in_with_fee = amount_in * (1 - fee_rate)

    if is_token0_in:
        # Swapping X for Y (price goes up)
        # ΔY = L * (sqrt_P_new - sqrt_P_current)
        # ΔX = L * (1/sqrt_P_current - 1/sqrt_P_new)

        delta_inv_sqrt_p = amount_in_with_fee / liquidity
        new_inv_sqrt_p = 1 / sqrt_price_current - delta_inv_sqrt_p

        if new_inv_sqrt_p <= 0:
            new_sqrt_p = sqrt_price_upper
        else:
            new_sqrt_p = 1 / new_inv_sqrt_p

        new_sqrt_p = min(new_sqrt_p, sqrt_price_upper)
        amount_out = liquidity * (new_sqrt_p - sqrt_price_current)
    else:
        # Swapping Y for X (price goes down)
        delta_sqrt_p = amount_in_with_fee / liquidity
        new_sqrt_p = sqrt_price_current - delta_sqrt_p

        new_sqrt_p = max(new_sqrt_p, sqrt_price_lower)
        amount_out = liquidity * (1 / new_sqrt_p - 1 / sqrt_price_current)

    # Price impact
    old_price = sqrt_price_current ** 2
    new_price = new_sqrt_p ** 2
    price_impact = abs(new_price - old_price) / old_price

    return max(amount_out, 0), new_sqrt_p, price_impact


def calculate_liquidity(
    amount_x: float,
    amount_y: float,
    sqrt_price_current: float,
    sqrt_price_lower: float,
    sqrt_price_upper: float,
) -> float:
    """
    Calculate liquidity (L) for a concentrated liquidity position.

    Args:
        amount_x: Amount of token X to provide
        amount_y: Amount of token Y to provide
        sqrt_price_current: Current sqrt price
        sqrt_price_lower: Lower tick sqrt price
        sqrt_price_upper: Upper tick sqrt price

    Returns:
        Liquidity value L
    """
    if sqrt_price_current <= sqrt_price_lower:
        # Only token X
        return amount_x * sqrt_price_lower * sqrt_price_upper / (sqrt_price_upper - sqrt_price_lower)
    elif sqrt_price_current >= sqrt_price_upper:
        # Only token Y
        return amount_y / (sqrt_price_upper - sqrt_price_lower)
    else:
        # Both tokens
        l_x = amount_x * sqrt_price_current * sqrt_price_upper / (sqrt_price_upper - sqrt_price_current)
        l_y = amount_y / (sqrt_price_current - sqrt_price_lower)
        return min(l_x, l_y)


# =============================================================================
# Risk Metrics
# =============================================================================

def volatility(
    returns: List[float],
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate historical volatility.

    Args:
        returns: List of period returns
        annualize: Whether to annualize the result
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Volatility (standard deviation of returns)
    """
    if len(returns) < 2:
        return 0.0

    if HAS_NUMPY:
        std = np.std(returns, ddof=1)
    else:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)

    if annualize:
        std *= math.sqrt(periods_per_year)

    return std


def sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    if HAS_NUMPY:
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
    else:
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_return = math.sqrt(variance)

    if std_return == 0:
        return 0.0

    # Annualize
    period_rf = risk_free_rate / periods_per_year
    excess_return = mean_return - period_rf

    return (excess_return * periods_per_year) / (std_return * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    # Calculate downside deviation
    downside_returns = [min(r - target_return, 0) ** 2 for r in returns]

    if HAS_NUMPY:
        downside_std = np.sqrt(np.mean(downside_returns))
        mean_return = np.mean(returns)
    else:
        downside_std = math.sqrt(sum(downside_returns) / len(downside_returns))
        mean_return = sum(returns) / len(returns)

    if downside_std == 0:
        return float('inf') if mean_return > risk_free_rate / periods_per_year else 0.0

    period_rf = risk_free_rate / periods_per_year
    excess_return = mean_return - period_rf

    return (excess_return * periods_per_year) / (downside_std * math.sqrt(periods_per_year))


def max_drawdown(prices: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.

    Args:
        prices: List of prices/equity values

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if len(prices) < 2:
        return 0.0, 0, 0

    peak = prices[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak = 0
    max_dd_trough = 0

    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            peak_idx = i

        dd = (peak - price) / peak if peak > 0 else 0

        if dd > max_dd:
            max_dd = dd
            max_dd_peak = peak_idx
            max_dd_trough = i

    return max_dd, max_dd_peak, max_dd_trough


def var(
    returns: List[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: List of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: "historical" or "parametric"

    Returns:
        VaR as a positive number (potential loss)
    """
    if len(returns) < 10:
        return 0.0

    if method == "historical":
        if HAS_NUMPY:
            return -np.percentile(returns, (1 - confidence) * 100)
        sorted_returns = sorted(returns)
        idx = int(len(sorted_returns) * (1 - confidence))
        return -sorted_returns[idx]

    # Parametric (assumes normal distribution)
    if HAS_NUMPY:
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z = stats.norm.ppf(1 - confidence)
    else:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        z = -1.645 if confidence == 0.95 else -2.326  # Approximate

    return -(mean + z * std)


def cvar(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: List of returns
        confidence: Confidence level

    Returns:
        CVaR as a positive number (expected loss beyond VaR)
    """
    if len(returns) < 10:
        return 0.0

    var_value = var(returns, confidence, "historical")

    # Average of returns worse than VaR
    tail_returns = [r for r in returns if r <= -var_value]

    if not tail_returns:
        return var_value

    if HAS_NUMPY:
        return -np.mean(tail_returns)
    return -sum(tail_returns) / len(tail_returns)


# =============================================================================
# Helper Functions
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF (pure Python)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF (pure Python)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def price_to_sqrt_price(price: float) -> float:
    """Convert price to sqrt price for concentrated liquidity."""
    return math.sqrt(price)


def sqrt_price_to_price(sqrt_price: float) -> float:
    """Convert sqrt price to regular price."""
    return sqrt_price ** 2


def tick_to_sqrt_price(tick: int, tick_spacing: int = 60) -> float:
    """Convert tick to sqrt price (Uniswap V3 style)."""
    return 1.0001 ** (tick / 2)


def sqrt_price_to_tick(sqrt_price: float, tick_spacing: int = 60) -> int:
    """Convert sqrt price to nearest tick."""
    tick = int(2 * math.log(sqrt_price) / math.log(1.0001))
    return (tick // tick_spacing) * tick_spacing
