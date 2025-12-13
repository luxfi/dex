//! Financial mathematics.
//!
//! Options pricing (Black-Scholes), Greeks, AMM pricing, and risk metrics.

use std::f64::consts::{PI, SQRT_2};

/// Square root of 2*PI.
pub const SQRT_2PI: f64 = 2.506628274631000502;

/// Greeks for options.
#[derive(Debug, Clone, Copy, Default)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / SQRT_2;

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Standard normal PDF.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

/// Black-Scholes option price.
///
/// # Arguments
/// * `s` - Spot price
/// * `k` - Strike price
/// * `t` - Time to expiry (years)
/// * `r` - Risk-free rate
/// * `sigma` - Volatility
/// * `is_call` - True for call, false for put
pub fn black_scholes(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
    if t <= 0.0 {
        return if is_call {
            (s - k).max(0.0)
        } else {
            (k - s).max(0.0)
        };
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    if is_call {
        s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
    } else {
        k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
    }
}

/// Implied volatility from option price (Newton-Raphson).
///
/// # Arguments
/// * `price` - Market price of option
/// * `s` - Spot price
/// * `k` - Strike price
/// * `t` - Time to expiry (years)
/// * `r` - Risk-free rate
/// * `is_call` - True for call, false for put
/// * `tol` - Tolerance (default 1e-6)
/// * `max_iter` - Maximum iterations (default 100)
pub fn implied_volatility(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    is_call: bool,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> f64 {
    let tol = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(100);

    let mut sigma = 0.2;

    for _ in 0..max_iter {
        let bs_price = black_scholes(s, k, t, r, sigma, is_call);

        // Vega
        let sqrt_t = t.sqrt();
        let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
        let vega = s * norm_pdf(d1) * sqrt_t;

        if vega.abs() < 1e-10 {
            break;
        }

        let diff = bs_price - price;
        if diff.abs() < tol {
            return sigma;
        }

        sigma -= diff / vega;
        sigma = sigma.clamp(0.001, 5.0);
    }

    sigma
}

/// Calculate all Greeks.
///
/// # Arguments
/// * `s` - Spot price
/// * `k` - Strike price
/// * `t` - Time to expiry (years)
/// * `r` - Risk-free rate
/// * `sigma` - Volatility
/// * `is_call` - True for call, false for put
pub fn greeks(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> Greeks {
    if t <= 0.0 {
        return Greeks::default();
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let pdf_d1 = norm_pdf(d1);
    let cdf_d1 = norm_cdf(d1);
    let cdf_d2 = norm_cdf(d2);
    let cdf_neg_d1 = norm_cdf(-d1);
    let cdf_neg_d2 = norm_cdf(-d2);
    let exp_neg_rt = (-r * t).exp();

    let (delta, theta, rho) = if is_call {
        let delta = cdf_d1;
        let theta = (-s * pdf_d1 * sigma / (2.0 * sqrt_t) - r * k * exp_neg_rt * cdf_d2) / 365.0;
        let rho = k * t * exp_neg_rt * cdf_d2;
        (delta, theta, rho)
    } else {
        let delta = cdf_d1 - 1.0;
        let theta = (-s * pdf_d1 * sigma / (2.0 * sqrt_t) + r * k * exp_neg_rt * cdf_neg_d2) / 365.0;
        let rho = -k * t * exp_neg_rt * cdf_neg_d2;
        (delta, theta, rho)
    };

    let gamma = pdf_d1 / (s * sigma * sqrt_t);
    let vega = s * pdf_d1 * sqrt_t / 100.0; // Per 1% vol change

    Greeks {
        delta,
        gamma,
        theta,
        vega,
        rho,
    }
}

/// Constant product AMM price (Uniswap V2 style).
///
/// Returns (output_amount, effective_price).
///
/// # Arguments
/// * `reserve_x` - Reserve of token X
/// * `reserve_y` - Reserve of token Y
/// * `amount_in` - Amount of input token
/// * `fee_rate` - Fee rate (e.g., 0.003 for 0.3%)
/// * `is_x_to_y` - True if swapping X for Y
pub fn constant_product_price(
    reserve_x: f64,
    reserve_y: f64,
    amount_in: f64,
    fee_rate: f64,
    is_x_to_y: bool,
) -> (f64, f64) {
    let amount_in_with_fee = amount_in * (1.0 - fee_rate);

    let amount_out = if is_x_to_y {
        (reserve_y * amount_in_with_fee) / (reserve_x + amount_in_with_fee)
    } else {
        (reserve_x * amount_in_with_fee) / (reserve_y + amount_in_with_fee)
    };

    let effective_price = if amount_in > 0.0 {
        amount_out / amount_in
    } else {
        0.0
    };

    (amount_out, effective_price)
}

/// Concentrated liquidity price (Uniswap V3 style).
///
/// Returns (output_amount, new_sqrt_price, price_impact).
///
/// # Arguments
/// * `liquidity` - Position liquidity
/// * `sqrt_price_current` - Current sqrt price
/// * `sqrt_price_lower` - Lower bound sqrt price
/// * `sqrt_price_upper` - Upper bound sqrt price
/// * `amount_in` - Amount of input token
/// * `fee_rate` - Fee rate
/// * `is_token0_in` - True if token0 is input
pub fn concentrated_liquidity_price(
    liquidity: f64,
    sqrt_price_current: f64,
    sqrt_price_lower: f64,
    sqrt_price_upper: f64,
    amount_in: f64,
    fee_rate: f64,
    is_token0_in: bool,
) -> (f64, f64, f64) {
    let amount_in_with_fee = amount_in * (1.0 - fee_rate);

    let (new_sqrt_p, amount_out) = if is_token0_in {
        // Swapping X for Y (price goes up)
        let delta_inv_sqrt_p = amount_in_with_fee / liquidity;
        let new_inv_sqrt_p = 1.0 / sqrt_price_current - delta_inv_sqrt_p;

        let new_sqrt_p = if new_inv_sqrt_p <= 0.0 {
            sqrt_price_upper
        } else {
            (1.0 / new_inv_sqrt_p).min(sqrt_price_upper)
        };

        let amount_out = liquidity * (new_sqrt_p - sqrt_price_current);
        (new_sqrt_p, amount_out)
    } else {
        // Swapping Y for X (price goes down)
        let delta_sqrt_p = amount_in_with_fee / liquidity;
        let new_sqrt_p = (sqrt_price_current - delta_sqrt_p).max(sqrt_price_lower);

        let amount_out = liquidity * (1.0 / new_sqrt_p - 1.0 / sqrt_price_current);
        (new_sqrt_p, amount_out)
    };

    let old_price = sqrt_price_current * sqrt_price_current;
    let new_price = new_sqrt_p * new_sqrt_p;
    let price_impact = ((new_price - old_price).abs()) / old_price;

    (amount_out.max(0.0), new_sqrt_p, price_impact)
}

/// Calculate liquidity for concentrated position.
///
/// # Arguments
/// * `amount_x` - Amount of token X
/// * `amount_y` - Amount of token Y
/// * `sqrt_price_current` - Current sqrt price
/// * `sqrt_price_lower` - Lower bound sqrt price
/// * `sqrt_price_upper` - Upper bound sqrt price
pub fn calculate_liquidity(
    amount_x: f64,
    amount_y: f64,
    sqrt_price_current: f64,
    sqrt_price_lower: f64,
    sqrt_price_upper: f64,
) -> f64 {
    if sqrt_price_current <= sqrt_price_lower {
        // Only token X
        amount_x * sqrt_price_lower * sqrt_price_upper / (sqrt_price_upper - sqrt_price_lower)
    } else if sqrt_price_current >= sqrt_price_upper {
        // Only token Y
        amount_y / (sqrt_price_upper - sqrt_price_lower)
    } else {
        // Both tokens
        let l_x =
            amount_x * sqrt_price_current * sqrt_price_upper / (sqrt_price_upper - sqrt_price_current);
        let l_y = amount_y / (sqrt_price_current - sqrt_price_lower);
        l_x.min(l_y)
    }
}

/// Convert price to sqrt price.
#[inline]
pub fn price_to_sqrt_price(price: f64) -> f64 {
    price.sqrt()
}

/// Convert sqrt price to price.
#[inline]
pub fn sqrt_price_to_price(sqrt_price: f64) -> f64 {
    sqrt_price * sqrt_price
}

/// Convert tick to sqrt price.
#[inline]
pub fn tick_to_sqrt_price(tick: i32) -> f64 {
    1.0001_f64.powf(tick as f64 / 2.0)
}

/// Convert sqrt price to tick.
#[inline]
pub fn sqrt_price_to_tick(sqrt_price: f64, tick_spacing: i32) -> i32 {
    let tick = (2.0 * sqrt_price.ln() / 1.0001_f64.ln()) as i32;
    (tick / tick_spacing) * tick_spacing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, margin: f64) -> bool {
        (a - b).abs() < margin
    }

    #[test]
    fn test_black_scholes_atm_call() {
        let price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(approx_eq(price, 10.45, 0.1));
    }

    #[test]
    fn test_black_scholes_itm_call() {
        let price = black_scholes(110.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(approx_eq(price, 17.68, 0.1));
    }

    #[test]
    fn test_black_scholes_otm_put() {
        let price = black_scholes(110.0, 100.0, 1.0, 0.05, 0.2, false);
        assert!(approx_eq(price, 2.80, 0.1));
    }

    #[test]
    fn test_black_scholes_zero_time() {
        let call_price = black_scholes(110.0, 100.0, 0.0, 0.05, 0.2, true);
        assert!(approx_eq(call_price, 10.0, 0.001));

        let put_price = black_scholes(90.0, 100.0, 0.0, 0.05, 0.2, false);
        assert!(approx_eq(put_price, 10.0, 0.001));
    }

    #[test]
    fn test_greeks_delta() {
        let g = greeks(100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(approx_eq(g.delta, 0.64, 0.02));
    }

    #[test]
    fn test_greeks_gamma() {
        let g = greeks(100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(approx_eq(g.gamma, 0.019, 0.002));
    }

    #[test]
    fn test_greeks_vega() {
        let g = greeks(100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(approx_eq(g.vega, 0.38, 0.02));
    }

    #[test]
    fn test_greeks_theta() {
        let g = greeks(100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(g.theta < 0.0); // Time decay is negative
    }

    #[test]
    fn test_implied_volatility() {
        let true_vol = 0.25;
        let price = black_scholes(100.0, 100.0, 0.5, 0.05, true_vol, true);
        let iv = implied_volatility(price, 100.0, 100.0, 0.5, 0.05, true, None, None);
        assert!(approx_eq(iv, true_vol, 0.01));
    }

    #[test]
    fn test_constant_product_basic() {
        let (out, price) = constant_product_price(1000.0, 1000.0, 10.0, 0.003, true);
        assert!(approx_eq(out, 9.88, 0.02));
        assert!(approx_eq(price, 0.988, 0.002));
    }

    #[test]
    fn test_constant_product_large_swap() {
        let (out, price) = constant_product_price(1000.0, 1000.0, 100.0, 0.003, true);
        // Larger trade = more slippage
        assert!(out < 100.0);
        assert!(price < 1.0);
    }

    #[test]
    fn test_constant_product_symmetric() {
        let (out1, _) = constant_product_price(1000.0, 1000.0, 50.0, 0.003, true);
        let (out2, _) = constant_product_price(1000.0, 1000.0, 50.0, 0.003, false);
        assert!(approx_eq(out1, out2, 0.01));
    }

    #[test]
    fn test_concentrated_liquidity() {
        let sqrt_price = 10.0;
        let sqrt_lower = 90.0_f64.sqrt();
        let sqrt_upper = 110.0_f64.sqrt();
        let liquidity = 1000.0;

        let (out, new_sqrt_p, impact) =
            concentrated_liquidity_price(liquidity, sqrt_price, sqrt_lower, sqrt_upper, 10.0, 0.003, true);

        assert!(out > 0.0);
        assert!(new_sqrt_p > sqrt_price); // Price increased
        assert!(impact >= 0.0);
    }

    #[test]
    fn test_price_conversions() {
        let sqrt_p = price_to_sqrt_price(100.0);
        assert!(approx_eq(sqrt_p, 10.0, 0.0001));

        let p = sqrt_price_to_price(10.0);
        assert!(approx_eq(p, 100.0, 0.0001));

        // Round trip
        let original = 12345.67;
        let sqrt_p = price_to_sqrt_price(original);
        let back = sqrt_price_to_price(sqrt_p);
        assert!(approx_eq(back, original, 0.01));
    }
}
