//! Financial mathematics for trading operations.
//!
//! Provides:
//! - Price calculations (VWAP, TWAP, mid-price)
//! - Volatility and risk metrics
//! - Position sizing
//! - PnL calculations
//! - Order book analytics

use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use std::collections::VecDeque;

use crate::types::{PriceLevel, Side};

/// Calculate mid-price from bid and ask
pub fn mid_price(bid: Decimal, ask: Decimal) -> Decimal {
    (bid + ask) / Decimal::from(2)
}

/// Calculate spread in absolute terms
pub fn spread(bid: Decimal, ask: Decimal) -> Decimal {
    ask - bid
}

/// Calculate spread as basis points
pub fn spread_bps(bid: Decimal, ask: Decimal) -> Decimal {
    if bid.is_zero() {
        return Decimal::ZERO;
    }
    ((ask - bid) / bid) * Decimal::from(10_000)
}

/// Calculate spread as percentage
pub fn spread_percent(bid: Decimal, ask: Decimal) -> Decimal {
    if bid.is_zero() {
        return Decimal::ZERO;
    }
    ((ask - bid) / bid) * Decimal::from(100)
}

/// Calculate Volume-Weighted Average Price (VWAP)
pub fn vwap(trades: &[(Decimal, Decimal)]) -> Option<Decimal> {
    if trades.is_empty() {
        return None;
    }

    let mut total_value = Decimal::ZERO;
    let mut total_volume = Decimal::ZERO;

    for (price, volume) in trades {
        total_value += *price * *volume;
        total_volume += *volume;
    }

    if total_volume.is_zero() {
        None
    } else {
        Some(total_value / total_volume)
    }
}

/// Calculate Time-Weighted Average Price (TWAP)
pub fn twap(prices: &[(Decimal, i64)]) -> Option<Decimal> {
    if prices.len() < 2 {
        return prices.first().map(|(p, _)| *p);
    }

    let mut weighted_sum = Decimal::ZERO;
    let mut total_time = 0i64;

    for window in prices.windows(2) {
        let (price, t1) = window[0];
        let (_, t2) = window[1];
        let duration = t2 - t1;

        weighted_sum += price * Decimal::from(duration);
        total_time += duration;
    }

    // Add last price for remaining time
    if let Some((last_price, _)) = prices.last() {
        // Assume last price holds for 1 unit
        weighted_sum += *last_price;
        total_time += 1;
    }

    if total_time == 0 {
        None
    } else {
        Some(weighted_sum / Decimal::from(total_time))
    }
}

/// Calculate execution VWAP from order book levels
pub fn execution_vwap(levels: &[PriceLevel], quantity: Decimal) -> Option<Decimal> {
    if levels.is_empty() || quantity.is_zero() {
        return None;
    }

    let mut remaining = quantity;
    let mut total_value = Decimal::ZERO;
    let mut filled_qty = Decimal::ZERO;

    for level in levels {
        if remaining <= Decimal::ZERO {
            break;
        }

        let fill = remaining.min(level.quantity);
        total_value += fill * level.price;
        filled_qty += fill;
        remaining -= fill;
    }

    if filled_qty.is_zero() {
        None
    } else {
        Some(total_value / filled_qty)
    }
}

/// Calculate price impact for a given order size
pub fn price_impact(levels: &[PriceLevel], quantity: Decimal) -> Option<Decimal> {
    let best_price = levels.first()?.price;
    let avg_price = execution_vwap(levels, quantity)?;

    if best_price.is_zero() {
        return None;
    }

    Some(((avg_price - best_price).abs() / best_price) * Decimal::from(100))
}

/// Calculate slippage from expected to actual price
pub fn slippage(expected_price: Decimal, actual_price: Decimal, side: Side) -> Decimal {
    if expected_price.is_zero() {
        return Decimal::ZERO;
    }

    let diff = match side {
        Side::Buy => actual_price - expected_price, // Higher is worse for buys
        Side::Sell => expected_price - actual_price, // Lower is worse for sells
    };

    (diff / expected_price) * Decimal::from(100)
}

/// Calculate position PnL
pub fn position_pnl(
    entry_price: Decimal,
    current_price: Decimal,
    quantity: Decimal,
    side: Side,
) -> Decimal {
    let price_diff = match side {
        Side::Buy => current_price - entry_price,
        Side::Sell => entry_price - current_price,
    };
    price_diff * quantity
}

/// Calculate position PnL percentage
pub fn position_pnl_percent(entry_price: Decimal, current_price: Decimal, side: Side) -> Decimal {
    if entry_price.is_zero() {
        return Decimal::ZERO;
    }

    match side {
        Side::Buy => ((current_price - entry_price) / entry_price) * Decimal::from(100),
        Side::Sell => ((entry_price - current_price) / entry_price) * Decimal::from(100),
    }
}

/// Calculate unrealized PnL with fees
pub fn unrealized_pnl(
    entry_price: Decimal,
    current_price: Decimal,
    quantity: Decimal,
    side: Side,
    entry_fee: Decimal,
    exit_fee_rate: Decimal,
) -> Decimal {
    let gross_pnl = position_pnl(entry_price, current_price, quantity, side);
    let exit_value = current_price * quantity;
    let exit_fee = exit_value * exit_fee_rate;
    gross_pnl - entry_fee - exit_fee
}

/// Calculate break-even price after fees
pub fn break_even_price(
    entry_price: Decimal,
    entry_fee_rate: Decimal,
    exit_fee_rate: Decimal,
    side: Side,
) -> Decimal {
    let total_fee_rate = entry_fee_rate + exit_fee_rate;

    match side {
        Side::Buy => entry_price * (Decimal::ONE + total_fee_rate),
        Side::Sell => entry_price * (Decimal::ONE - total_fee_rate),
    }
}

/// Calculate position size based on risk
pub fn position_size_by_risk(
    account_balance: Decimal,
    risk_percent: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
) -> Decimal {
    let risk_amount = account_balance * (risk_percent / Decimal::from(100));
    let price_risk = (entry_price - stop_loss_price).abs();

    if price_risk.is_zero() {
        return Decimal::ZERO;
    }

    risk_amount / price_risk
}

/// Calculate Kelly Criterion optimal bet size
pub fn kelly_criterion(win_rate: Decimal, avg_win: Decimal, avg_loss: Decimal) -> Decimal {
    if avg_loss.is_zero() {
        return Decimal::ZERO;
    }

    let loss_rate = Decimal::ONE - win_rate;
    let win_loss_ratio = avg_win / avg_loss;

    // Kelly formula: f* = (p * b - q) / b
    // where p = win_rate, q = loss_rate, b = win/loss ratio
    let kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio;

    // Clamp to [0, 1]
    kelly.max(Decimal::ZERO).min(Decimal::ONE)
}

/// Simple moving average
pub fn sma(prices: &[Decimal]) -> Option<Decimal> {
    if prices.is_empty() {
        return None;
    }
    let sum: Decimal = prices.iter().copied().sum();
    Some(sum / Decimal::from(prices.len()))
}

/// Exponential moving average
pub fn ema(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.is_empty() || period == 0 {
        return None;
    }

    let multiplier = Decimal::from(2) / Decimal::from(period + 1);
    let mut ema = *prices.first()?;

    for price in prices.iter().skip(1) {
        ema = (*price - ema) * multiplier + ema;
    }

    Some(ema)
}

/// Calculate standard deviation
pub fn std_dev(values: &[Decimal]) -> Option<Decimal> {
    if values.len() < 2 {
        return None;
    }
    let mean = sma(values)?;
    let variance: Decimal = values
        .iter()
        .map(|v| {
            let diff = *v - mean;
            diff * diff
        })
        .sum::<Decimal>()
        / Decimal::from(values.len());

    // Approximate square root using Newton's method
    sqrt(variance)
}

/// Calculate historical volatility (annualized)
pub fn historical_volatility(returns: &[Decimal], trading_days_per_year: u32) -> Option<Decimal> {
    let std = std_dev(returns)?;
    Some(std * sqrt(Decimal::from(trading_days_per_year))?)
}

/// Calculate logarithmic return
pub fn log_return(price_start: Decimal, price_end: Decimal) -> Option<Decimal> {
    if price_start <= Decimal::ZERO || price_end <= Decimal::ZERO {
        return None;
    }

    // ln(price_end / price_start) = ln(price_end) - ln(price_start)
    // Approximate using Taylor series for small changes
    let ratio = price_end / price_start;

    // For ratios close to 1, use Taylor approximation: ln(1+x) ~ x - x^2/2 + x^3/3
    let x = ratio - Decimal::ONE;
    if x.abs() < Decimal::from_str("0.5").unwrap() {
        let x2 = x * x;
        let x3 = x2 * x;
        Some(x - x2 / Decimal::from(2) + x3 / Decimal::from(3))
    } else {
        // For larger changes, use simple approximation
        Some((ratio - Decimal::ONE) / ((ratio + Decimal::ONE) / Decimal::from(2)))
    }
}

/// Calculate simple return
pub fn simple_return(price_start: Decimal, price_end: Decimal) -> Decimal {
    if price_start.is_zero() {
        return Decimal::ZERO;
    }
    (price_end - price_start) / price_start
}

/// Calculate returns from price series
pub fn returns_from_prices(prices: &[Decimal]) -> Vec<Decimal> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| simple_return(w[0], w[1]))
        .collect()
}

/// Sharpe ratio calculation
pub fn sharpe_ratio(returns: &[Decimal], risk_free_rate: Decimal) -> Option<Decimal> {
    let mean_return = sma(returns)?;
    let std = std_dev(returns)?;

    if std.is_zero() {
        return None;
    }

    Some((mean_return - risk_free_rate) / std)
}

/// Sortino ratio (downside deviation only)
pub fn sortino_ratio(returns: &[Decimal], risk_free_rate: Decimal) -> Option<Decimal> {
    let mean_return = sma(returns)?;

    // Calculate downside deviation
    let negative_returns: Vec<Decimal> = returns
        .iter()
        .filter(|r| **r < Decimal::ZERO)
        .copied()
        .collect();

    let downside_dev = std_dev(&negative_returns)?;

    if downside_dev.is_zero() {
        return None;
    }

    Some((mean_return - risk_free_rate) / downside_dev)
}

/// Maximum drawdown calculation
pub fn max_drawdown(equity_curve: &[Decimal]) -> Decimal {
    if equity_curve.is_empty() {
        return Decimal::ZERO;
    }

    let mut peak = equity_curve[0];
    let mut max_dd = Decimal::ZERO;

    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let drawdown = (peak - value) / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    max_dd * Decimal::from(100) // Return as percentage
}

/// Calculate order book imbalance
pub fn order_book_imbalance(bid_depth: Decimal, ask_depth: Decimal) -> Decimal {
    let total = bid_depth + ask_depth;
    if total.is_zero() {
        return Decimal::ZERO;
    }
    (bid_depth - ask_depth) / total
}

/// Calculate weighted order book imbalance (closer to mid = higher weight)
/// Uses exponential decay based on distance from mid price
pub fn weighted_order_book_imbalance(
    bids: &[PriceLevel],
    asks: &[PriceLevel],
    mid_price: Decimal,
    decay: Decimal,
) -> Decimal {
    let calc_weighted_depth = |levels: &[PriceLevel]| -> Decimal {
        levels
            .iter()
            .map(|l| {
                if mid_price.is_zero() {
                    return l.quantity;
                }
                let distance = (l.price - mid_price).abs() / mid_price;
                // Exponential decay approximation: e^(-x) ~ 1/(1+x) for small x
                let weight = if distance * decay < Decimal::ONE {
                    Decimal::ONE - distance * decay / Decimal::from(2)
                } else {
                    Decimal::ONE / (Decimal::ONE + distance * decay)
                };
                l.quantity * weight
            })
            .sum()
    };

    let weighted_bids = calc_weighted_depth(bids);
    let weighted_asks = calc_weighted_depth(asks);
    let total = weighted_bids + weighted_asks;

    if total.is_zero() {
        Decimal::ZERO
    } else {
        (weighted_bids - weighted_asks) / total
    }
}

/// Newton's method square root approximation
fn sqrt(n: Decimal) -> Option<Decimal> {
    if n < Decimal::ZERO {
        return None;
    }
    if n.is_zero() {
        return Some(Decimal::ZERO);
    }

    let mut guess = n / Decimal::from(2);
    let tolerance = Decimal::from_str("0.0000000001").unwrap();

    for _ in 0..100 {
        let next = (guess + n / guess) / Decimal::from(2);
        if (next - guess).abs() < tolerance {
            return Some(next);
        }
        guess = next;
    }

    Some(guess)
}

/// Rolling statistics calculator
pub struct RollingStats {
    window_size: usize,
    values: VecDeque<Decimal>,
    sum: Decimal,
    sum_sq: Decimal,
}

impl RollingStats {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
            sum: Decimal::ZERO,
            sum_sq: Decimal::ZERO,
        }
    }

    pub fn push(&mut self, value: Decimal) {
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    pub fn mean(&self) -> Option<Decimal> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.sum / Decimal::from(self.values.len()))
    }

    pub fn variance(&self) -> Option<Decimal> {
        if self.values.len() < 2 {
            return None;
        }
        let n = Decimal::from(self.values.len());
        let mean = self.mean()?;
        Some((self.sum_sq / n) - (mean * mean))
    }

    pub fn std_dev(&self) -> Option<Decimal> {
        sqrt(self.variance()?)
    }

    pub fn count(&self) -> usize {
        self.values.len()
    }

    pub fn min(&self) -> Option<Decimal> {
        self.values.iter().copied().min()
    }

    pub fn max(&self) -> Option<Decimal> {
        self.values.iter().copied().max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mid_price() {
        let mid = mid_price(Decimal::from(100), Decimal::from(102));
        assert_eq!(mid, Decimal::from(101));
    }

    #[test]
    fn test_spread_calculations() {
        let bid = Decimal::from(100);
        let ask = Decimal::from(101);

        assert_eq!(spread(bid, ask), Decimal::from(1));
        assert_eq!(spread_percent(bid, ask), Decimal::from(1));
        assert_eq!(spread_bps(bid, ask), Decimal::from(100));
    }

    #[test]
    fn test_vwap() {
        let trades = vec![
            (Decimal::from(100), Decimal::from(10)),
            (Decimal::from(102), Decimal::from(20)),
            (Decimal::from(101), Decimal::from(10)),
        ];

        let result = vwap(&trades).unwrap();
        // (100*10 + 102*20 + 101*10) / (10+20+10) = 4050/40 = 101.25
        assert_eq!(result, Decimal::from_str("101.25").unwrap());
    }

    #[test]
    fn test_position_pnl() {
        let pnl = position_pnl(
            Decimal::from(100),
            Decimal::from(110),
            Decimal::from(10),
            Side::Buy,
        );
        assert_eq!(pnl, Decimal::from(100)); // (110-100) * 10 = 100

        let pnl = position_pnl(
            Decimal::from(100),
            Decimal::from(90),
            Decimal::from(10),
            Side::Sell,
        );
        assert_eq!(pnl, Decimal::from(100)); // (100-90) * 10 = 100
    }

    #[test]
    fn test_position_size_by_risk() {
        let size = position_size_by_risk(
            Decimal::from(10000), // $10,000 account
            Decimal::from(1),     // 1% risk
            Decimal::from(100),   // entry at $100
            Decimal::from(95),    // stop at $95
        );
        // Risk: $100, Price risk: $5, Position: 100/5 = 20
        assert_eq!(size, Decimal::from(20));
    }

    #[test]
    fn test_kelly_criterion() {
        // 60% win rate, 2:1 risk/reward
        let kelly = kelly_criterion(
            Decimal::from_str("0.6").unwrap(),
            Decimal::from(2),
            Decimal::from(1),
        );
        // f* = (0.6*2 - 0.4)/2 = (1.2 - 0.4)/2 = 0.4
        assert!(
            (kelly - Decimal::from_str("0.4").unwrap()).abs() < Decimal::from_str("0.01").unwrap()
        );
    }

    #[test]
    fn test_sma() {
        let prices: Vec<Decimal> = vec![1, 2, 3, 4, 5].into_iter().map(Decimal::from).collect();
        let avg = sma(&prices).unwrap();
        assert_eq!(avg, Decimal::from(3));
    }

    #[test]
    fn test_simple_return() {
        let ret = simple_return(Decimal::from(100), Decimal::from(110));
        assert_eq!(ret, Decimal::from_str("0.1").unwrap());
    }

    #[test]
    fn test_max_drawdown() {
        let curve: Vec<Decimal> = vec![100, 120, 100, 80, 90, 110]
            .into_iter()
            .map(Decimal::from)
            .collect();

        let dd = max_drawdown(&curve);
        // Peak was 120, trough was 80, drawdown = (120-80)/120 = 33.33%
        assert!(
            (dd - Decimal::from_str("33.333333").unwrap()).abs()
                < Decimal::from_str("0.001").unwrap()
        );
    }

    #[test]
    fn test_order_book_imbalance() {
        let imbalance = order_book_imbalance(Decimal::from(100), Decimal::from(50));
        // (100-50)/(100+50) = 50/150 = 0.333...
        assert!(
            (imbalance - Decimal::from_str("0.333333").unwrap()).abs()
                < Decimal::from_str("0.001").unwrap()
        );
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(3);

        stats.push(Decimal::from(1));
        stats.push(Decimal::from(2));
        stats.push(Decimal::from(3));

        assert_eq!(stats.mean(), Some(Decimal::from(2)));
        assert_eq!(stats.count(), 3);
        assert_eq!(stats.min(), Some(Decimal::from(1)));
        assert_eq!(stats.max(), Some(Decimal::from(3)));

        stats.push(Decimal::from(4)); // Removes 1
        assert_eq!(stats.mean(), Some(Decimal::from(3)));
    }

    #[test]
    fn test_sqrt() {
        let result = sqrt(Decimal::from(4)).unwrap();
        assert!((result - Decimal::from(2)).abs() < Decimal::from_str("0.0001").unwrap());

        let result = sqrt(Decimal::from(2)).unwrap();
        assert!(
            (result - Decimal::from_str("1.41421").unwrap()).abs()
                < Decimal::from_str("0.001").unwrap()
        );
    }

    #[test]
    fn test_execution_vwap() {
        let levels = vec![
            PriceLevel::new(Decimal::from(100), Decimal::from(10)),
            PriceLevel::new(Decimal::from(101), Decimal::from(20)),
            PriceLevel::new(Decimal::from(102), Decimal::from(30)),
        ];

        // Buy 15 units - fills 10@100 + 5@101 = 1505/15 = 100.333...
        let vwap = execution_vwap(&levels, Decimal::from(15)).unwrap();
        assert!(
            (vwap - Decimal::from_str("100.333333").unwrap()).abs()
                < Decimal::from_str("0.001").unwrap()
        );
    }

    #[test]
    fn test_price_impact() {
        let levels = vec![
            PriceLevel::new(Decimal::from(100), Decimal::from(10)),
            PriceLevel::new(Decimal::from(101), Decimal::from(10)),
            PriceLevel::new(Decimal::from(102), Decimal::from(10)),
        ];

        // Large order should have more impact
        let impact = price_impact(&levels, Decimal::from(25)).unwrap();
        assert!(impact > Decimal::ZERO);
    }
}
