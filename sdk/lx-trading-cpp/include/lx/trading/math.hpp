// LX Trading SDK - Financial Mathematics
// SIMD-optimized options pricing, AMM math, and risk metrics

#pragma once

#include <lx/trading/types.hpp>
#include <cmath>
#include <tuple>
#include <vector>

// SIMD support detection
#if defined(__AVX2__)
    #include <immintrin.h>
    #define LX_SIMD_AVX2 1
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
    #define LX_SIMD_SSE4 1
#endif

namespace lx::trading::math {

// =============================================================================
// Constants
// =============================================================================

constexpr double PI = 3.14159265358979323846;
constexpr double SQRT_2PI = 2.506628274631000502;
constexpr double SQRT_2 = 1.41421356237309504880;

// =============================================================================
// Options Pricing - Black-Scholes
// =============================================================================

// Greeks structure
struct Greeks {
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
};

// Standard normal CDF (approximation - Abramowitz & Stegun)
inline double norm_cdf(double x) noexcept {
    constexpr double a1 = 0.254829592;
    constexpr double a2 = -0.284496736;
    constexpr double a3 = 1.421413741;
    constexpr double a4 = -1.453152027;
    constexpr double a5 = 1.061405429;
    constexpr double p = 0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = std::abs(x) / SQRT_2;

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// Standard normal PDF
inline double norm_pdf(double x) noexcept {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

// Black-Scholes option price
// S: spot, K: strike, T: time (years), r: rate, sigma: vol
double black_scholes(
    double S, double K, double T, double r, double sigma,
    bool is_call = true) noexcept;

// Implied volatility from option price (Newton-Raphson)
double implied_volatility(
    double price, double S, double K, double T, double r,
    bool is_call = true, double tol = 1e-6, int max_iter = 100) noexcept;

// Calculate all Greeks
Greeks greeks(
    double S, double K, double T, double r, double sigma,
    bool is_call = true) noexcept;

// =============================================================================
// AMM Pricing
// =============================================================================

// Constant product AMM (Uniswap V2 style)
// Returns: (output_amount, effective_price)
std::pair<double, double> constant_product_price(
    double reserve_x, double reserve_y,
    double amount_in, double fee_rate = 0.003,
    bool is_x_to_y = true) noexcept;

// Concentrated liquidity (Uniswap V3 style)
// Returns: (output_amount, new_sqrt_price, price_impact)
std::tuple<double, double, double> concentrated_liquidity_price(
    double liquidity,
    double sqrt_price_current,
    double sqrt_price_lower,
    double sqrt_price_upper,
    double amount_in,
    double fee_rate = 0.003,
    bool is_token0_in = true) noexcept;

// Calculate liquidity for concentrated position
double calculate_liquidity(
    double amount_x, double amount_y,
    double sqrt_price_current,
    double sqrt_price_lower,
    double sqrt_price_upper) noexcept;

// Price conversions
inline double price_to_sqrt_price(double price) noexcept {
    return std::sqrt(price);
}

inline double sqrt_price_to_price(double sqrt_price) noexcept {
    return sqrt_price * sqrt_price;
}

inline double tick_to_sqrt_price(int tick) noexcept {
    return std::pow(1.0001, tick / 2.0);
}

inline int sqrt_price_to_tick(double sqrt_price, int tick_spacing = 60) noexcept {
    int tick = static_cast<int>(2.0 * std::log(sqrt_price) / std::log(1.0001));
    return (tick / tick_spacing) * tick_spacing;
}

// =============================================================================
// Risk Metrics
// =============================================================================

// Historical volatility
double volatility(
    const std::vector<double>& returns,
    bool annualize = true,
    int periods_per_year = 252) noexcept;

// Sharpe ratio
double sharpe_ratio(
    const std::vector<double>& returns,
    double risk_free_rate = 0.0,
    int periods_per_year = 252) noexcept;

// Sortino ratio (downside deviation)
double sortino_ratio(
    const std::vector<double>& returns,
    double risk_free_rate = 0.0,
    double target_return = 0.0,
    int periods_per_year = 252) noexcept;

// Maximum drawdown
// Returns: (max_drawdown, peak_index, trough_index)
std::tuple<double, size_t, size_t> max_drawdown(
    const std::vector<double>& prices) noexcept;

// Value at Risk (Historical or Parametric)
double var(
    const std::vector<double>& returns,
    double confidence = 0.95,
    bool parametric = false) noexcept;

// Conditional VaR (Expected Shortfall)
double cvar(
    const std::vector<double>& returns,
    double confidence = 0.95) noexcept;

// =============================================================================
// SIMD Optimized Operations
// =============================================================================

#ifdef LX_SIMD_AVX2

// Batch Black-Scholes pricing (AVX2)
void black_scholes_batch(
    const double* S, const double* K, const double* T,
    const double* r, const double* sigma,
    double* prices, size_t count, bool is_call = true) noexcept;

// Batch norm_cdf (AVX2)
void norm_cdf_batch(
    const double* x, double* result, size_t count) noexcept;

// Vectorized sum (AVX2)
double sum_avx2(const double* data, size_t count) noexcept;

// Vectorized mean (AVX2)
double mean_avx2(const double* data, size_t count) noexcept;

// Vectorized variance (AVX2)
double variance_avx2(const double* data, size_t count, double mean) noexcept;

#endif  // LX_SIMD_AVX2

// =============================================================================
// Statistical Utilities
// =============================================================================

// Calculate returns from prices
std::vector<double> calculate_returns(const std::vector<double>& prices);

// Rolling mean
std::vector<double> rolling_mean(const std::vector<double>& data, size_t window);

// Rolling std deviation
std::vector<double> rolling_std(const std::vector<double>& data, size_t window);

// Exponential moving average
std::vector<double> ema(const std::vector<double>& data, double alpha);

// Correlation coefficient
double correlation(
    const std::vector<double>& x,
    const std::vector<double>& y) noexcept;

// Covariance
double covariance(
    const std::vector<double>& x,
    const std::vector<double>& y) noexcept;

// Beta coefficient
double beta(
    const std::vector<double>& asset_returns,
    const std::vector<double>& market_returns) noexcept;

}  // namespace lx::trading::math
