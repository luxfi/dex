//! Statistical functions for trading.
//!
//! Volatility, Sharpe ratio, VaR, max drawdown, and more.

/// Historical volatility of returns.
///
/// # Arguments
/// * `returns` - Vector of periodic returns
/// * `annualize` - Whether to annualize the result
/// * `periods_per_year` - Number of periods per year (default 252 for daily)
pub fn volatility(returns: &[f64], annualize: bool, periods_per_year: Option<i32>) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let periods = periods_per_year.unwrap_or(252);
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;

    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
        / (returns.len() - 1) as f64;

    let std_dev = variance.sqrt();

    if annualize {
        std_dev * (periods as f64).sqrt()
    } else {
        std_dev
    }
}

/// Sharpe ratio.
///
/// # Arguments
/// * `returns` - Vector of periodic returns
/// * `risk_free_rate` - Annual risk-free rate (default 0.0)
/// * `periods_per_year` - Number of periods per year (default 252)
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: Option<f64>, periods_per_year: Option<i32>) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let rf = risk_free_rate.unwrap_or(0.0);
    let periods = periods_per_year.unwrap_or(252);

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;

    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
        / (returns.len() - 1) as f64;

    let std_dev = variance.sqrt();
    if std_dev == 0.0 {
        return 0.0;
    }

    let period_rf = rf / periods as f64;
    let excess_return = mean - period_rf;

    (excess_return * periods as f64) / (std_dev * (periods as f64).sqrt())
}

/// Sortino ratio (uses downside deviation).
///
/// # Arguments
/// * `returns` - Vector of periodic returns
/// * `risk_free_rate` - Annual risk-free rate (default 0.0)
/// * `target_return` - Target return for downside calculation (default 0.0)
/// * `periods_per_year` - Number of periods per year (default 252)
pub fn sortino_ratio(
    returns: &[f64],
    risk_free_rate: Option<f64>,
    target_return: Option<f64>,
    periods_per_year: Option<i32>,
) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let rf = risk_free_rate.unwrap_or(0.0);
    let target = target_return.unwrap_or(0.0);
    let periods = periods_per_year.unwrap_or(252);

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;

    // Downside deviation
    let downside_sum: f64 = returns
        .iter()
        .map(|r| (r - target).min(0.0).powi(2))
        .sum();
    let downside_std = (downside_sum / returns.len() as f64).sqrt();

    if downside_std == 0.0 {
        return if mean > rf / periods as f64 {
            f64::INFINITY
        } else {
            0.0
        };
    }

    let period_rf = rf / periods as f64;
    let excess_return = mean - period_rf;

    (excess_return * periods as f64) / (downside_std * (periods as f64).sqrt())
}

/// Maximum drawdown.
///
/// Returns (max_drawdown, peak_index, trough_index).
///
/// # Arguments
/// * `prices` - Vector of prices
pub fn max_drawdown(prices: &[f64]) -> (f64, usize, usize) {
    if prices.len() < 2 {
        return (0.0, 0, 0);
    }

    let mut peak = prices[0];
    let mut peak_idx = 0;
    let mut max_dd = 0.0;
    let mut max_dd_peak = 0;
    let mut max_dd_trough = 0;

    for (i, &price) in prices.iter().enumerate() {
        if price > peak {
            peak = price;
            peak_idx = i;
        }

        let dd = if peak > 0.0 {
            (peak - price) / peak
        } else {
            0.0
        };

        if dd > max_dd {
            max_dd = dd;
            max_dd_peak = peak_idx;
            max_dd_trough = i;
        }
    }

    (max_dd, max_dd_peak, max_dd_trough)
}

/// Value at Risk (Historical or Parametric).
///
/// # Arguments
/// * `returns` - Vector of returns
/// * `confidence` - Confidence level (default 0.95)
/// * `parametric` - Use parametric (normal) VaR instead of historical
pub fn var(returns: &[f64], confidence: Option<f64>, parametric: Option<bool>) -> f64 {
    if returns.len() < 10 {
        return 0.0;
    }

    let conf = confidence.unwrap_or(0.95);
    let is_parametric = parametric.unwrap_or(false);

    if !is_parametric {
        // Historical VaR
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f64) * (1.0 - conf)) as usize;
        -sorted[idx]
    } else {
        // Parametric VaR (assumes normal distribution)
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;

        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        // Z-score approximation
        let z = if (conf - 0.95).abs() < 0.001 {
            -1.645
        } else {
            -2.326 // 99%
        };

        -(mean + z * std_dev)
    }
}

/// Conditional VaR (Expected Shortfall).
///
/// # Arguments
/// * `returns` - Vector of returns
/// * `confidence` - Confidence level (default 0.95)
pub fn cvar(returns: &[f64], confidence: Option<f64>) -> f64 {
    if returns.len() < 10 {
        return 0.0;
    }

    let conf = confidence.unwrap_or(0.95);
    let var_value = var(returns, Some(conf), Some(false));

    // Average of returns worse than VaR
    let tail: Vec<f64> = returns.iter().filter(|&&r| r <= -var_value).copied().collect();

    if tail.is_empty() {
        var_value
    } else {
        -tail.iter().sum::<f64>() / tail.len() as f64
    }
}

/// Calculate returns from prices.
pub fn calculate_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .filter_map(|w| {
            if w[0] > 0.0 {
                Some((w[1] - w[0]) / w[0])
            } else {
                None
            }
        })
        .collect()
}

/// Rolling mean.
///
/// # Arguments
/// * `data` - Input data
/// * `window` - Window size
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len() - window + 1);

    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);

    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }

    result
}

/// Rolling standard deviation.
///
/// # Arguments
/// * `data` - Input data
/// * `window` - Window size
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len() - window + 1);

    for i in 0..=(data.len() - window) {
        let slice = &data[i..i + window];
        let mean = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (window - 1) as f64;
        result.push(variance.sqrt());
    }

    result
}

/// Exponential moving average.
///
/// # Arguments
/// * `data` - Input data
/// * `alpha` - Smoothing factor (0 < alpha <= 1)
pub fn ema(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);

    for &x in data.iter().skip(1) {
        let prev = *result.last().unwrap();
        result.push(alpha * x + (1.0 - alpha) * prev);
    }

    result
}

/// Correlation coefficient.
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

/// Covariance.
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    cov / (n - 1.0)
}

/// Beta coefficient (asset vs market).
pub fn beta(asset_returns: &[f64], market_returns: &[f64]) -> f64 {
    let cov = covariance(asset_returns, market_returns);

    let n = market_returns.len() as f64;
    let mean_m = market_returns.iter().sum::<f64>() / n;

    let var_m: f64 = market_returns.iter().map(|r| (r - mean_m).powi(2)).sum::<f64>()
        / (n - 1.0);

    if var_m > 0.0 {
        cov / var_m
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, margin: f64) -> bool {
        (a - b).abs() < margin
    }

    #[test]
    fn test_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, 0.005];

        // Non-annualized
        let vol = volatility(&returns, false, None);
        assert!(vol > 0.0);
        assert!(vol < 0.1);

        // Annualized
        let vol_daily = volatility(&returns, false, None);
        let vol_annual = volatility(&returns, true, Some(252));
        assert!(approx_eq(vol_annual, vol_daily * 252.0_f64.sqrt(), 0.001));
    }

    #[test]
    fn test_sharpe_ratio() {
        let positive_returns = vec![0.01, 0.02, 0.015, 0.01, 0.02];
        let mixed_returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];

        let sharpe_pos = sharpe_ratio(&positive_returns, None, None);
        assert!(sharpe_pos > 0.0);

        let sharpe_mix = sharpe_ratio(&mixed_returns, None, None);
        assert!(sharpe_mix < sharpe_pos);
    }

    #[test]
    fn test_max_drawdown() {
        let prices = vec![100.0, 110.0, 105.0, 95.0, 90.0, 100.0, 85.0];

        let (dd, peak_idx, trough_idx) = max_drawdown(&prices);

        // Max DD from 110 to 85 = 22.7%
        assert!(approx_eq(dd, 0.227, 0.01));
        assert_eq!(peak_idx, 1); // 110
        assert_eq!(trough_idx, 6); // 85
    }

    #[test]
    fn test_var() {
        let returns = vec![
            -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, -0.04, -0.02, -0.01,
            0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
        ];

        // Historical VaR 95%
        let var95 = var(&returns, Some(0.95), Some(false));
        assert!(var95 > 0.0);
        assert!(var95 < 0.1);

        // Parametric VaR 95%
        let var95_p = var(&returns, Some(0.95), Some(true));
        assert!(var95_p > 0.0);
    }

    #[test]
    fn test_cvar() {
        let returns = vec![
            -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
        ];

        let cvar95 = cvar(&returns, Some(0.95));
        let var95 = var(&returns, Some(0.95), Some(false));

        // CVaR should be >= VaR
        assert!(cvar95 >= var95);
    }

    #[test]
    fn test_calculate_returns() {
        let prices = vec![100.0, 105.0, 102.0, 110.0];
        let returns = calculate_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!(approx_eq(returns[0], 0.05, 0.0001));
        assert!(approx_eq(returns[1], -0.0286, 0.001));
        assert!(approx_eq(returns[2], 0.0784, 0.001));
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let rm = rolling_mean(&data, 3);

        assert_eq!(rm.len(), 5);
        assert!(approx_eq(rm[0], 2.0, 0.0001)); // (1+2+3)/3
        assert!(approx_eq(rm[1], 3.0, 0.0001)); // (2+3+4)/3
    }

    #[test]
    fn test_ema() {
        let data = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0];
        let e = ema(&data, 0.3);

        assert_eq!(e.len(), 6);
        assert!(approx_eq(e[0], 10.0, 0.0001));
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = correlation(&x, &y);
        assert!(approx_eq(corr, 1.0, 0.0001)); // Perfect positive
    }

    #[test]
    fn test_beta() {
        let asset = vec![0.02, 0.03, -0.01, 0.02, 0.01];
        let market = vec![0.01, 0.015, -0.005, 0.01, 0.005];

        let b = beta(&asset, &market);
        assert!(b > 1.0); // Asset more volatile than market
    }
}
