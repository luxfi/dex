// LX Trading SDK - Financial Mathematics Implementation
// SIMD-optimized where beneficial

#include <lx/trading/math.hpp>
#include <algorithm>
#include <numeric>

namespace lx::trading::math {

// =============================================================================
// Black-Scholes Implementation
// =============================================================================

double black_scholes(
    double S, double K, double T, double r, double sigma, bool is_call) noexcept {
    if (T <= 0) {
        return is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }

    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    if (is_call) {
        return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    }
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

double implied_volatility(
    double price, double S, double K, double T, double r,
    bool is_call, double tol, int max_iter) noexcept {
    // Newton-Raphson
    double sigma = 0.2;

    for (int i = 0; i < max_iter; ++i) {
        double bs_price = black_scholes(S, K, T, r, sigma, is_call);

        // Vega
        double sqrt_T = std::sqrt(T);
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
        double vega = S * norm_pdf(d1) * sqrt_T;

        if (std::abs(vega) < 1e-10) break;

        double diff = bs_price - price;
        if (std::abs(diff) < tol) return sigma;

        sigma -= diff / vega;
        sigma = std::clamp(sigma, 0.001, 5.0);
    }

    return sigma;
}

Greeks greeks(
    double S, double K, double T, double r, double sigma, bool is_call) noexcept {
    Greeks g{0, 0, 0, 0, 0};

    if (T <= 0) return g;

    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    double pdf_d1 = norm_pdf(d1);
    double cdf_d1 = norm_cdf(d1);
    double cdf_d2 = norm_cdf(d2);
    double cdf_neg_d1 = norm_cdf(-d1);
    double cdf_neg_d2 = norm_cdf(-d2);
    double exp_neg_rT = std::exp(-r * T);

    if (is_call) {
        g.delta = cdf_d1;
        g.theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) -
                   r * K * exp_neg_rT * cdf_d2);
        g.rho = K * T * exp_neg_rT * cdf_d2;
    } else {
        g.delta = cdf_d1 - 1;
        g.theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) +
                   r * K * exp_neg_rT * cdf_neg_d2);
        g.rho = -K * T * exp_neg_rT * cdf_neg_d2;
    }

    g.gamma = pdf_d1 / (S * sigma * sqrt_T);
    g.vega = S * pdf_d1 * sqrt_T / 100;  // Per 1% vol change
    g.theta /= 365;  // Daily theta

    return g;
}

// =============================================================================
// AMM Pricing
// =============================================================================

std::pair<double, double> constant_product_price(
    double reserve_x, double reserve_y,
    double amount_in, double fee_rate, bool is_x_to_y) noexcept {
    double amount_in_with_fee = amount_in * (1.0 - fee_rate);

    double amount_out;
    if (is_x_to_y) {
        amount_out = (reserve_y * amount_in_with_fee) /
                     (reserve_x + amount_in_with_fee);
    } else {
        amount_out = (reserve_x * amount_in_with_fee) /
                     (reserve_y + amount_in_with_fee);
    }

    double effective_price = (amount_in > 0) ? amount_out / amount_in : 0;
    return {amount_out, effective_price};
}

std::tuple<double, double, double> concentrated_liquidity_price(
    double liquidity, double sqrt_price_current,
    double sqrt_price_lower, double sqrt_price_upper,
    double amount_in, double fee_rate, bool is_token0_in) noexcept {
    double amount_in_with_fee = amount_in * (1.0 - fee_rate);
    double new_sqrt_p;
    double amount_out;

    if (is_token0_in) {
        // Swapping X for Y (price goes up)
        double delta_inv_sqrt_p = amount_in_with_fee / liquidity;
        double new_inv_sqrt_p = 1.0 / sqrt_price_current - delta_inv_sqrt_p;

        if (new_inv_sqrt_p <= 0) {
            new_sqrt_p = sqrt_price_upper;
        } else {
            new_sqrt_p = 1.0 / new_inv_sqrt_p;
        }

        new_sqrt_p = std::min(new_sqrt_p, sqrt_price_upper);
        amount_out = liquidity * (new_sqrt_p - sqrt_price_current);
    } else {
        // Swapping Y for X (price goes down)
        double delta_sqrt_p = amount_in_with_fee / liquidity;
        new_sqrt_p = sqrt_price_current - delta_sqrt_p;

        new_sqrt_p = std::max(new_sqrt_p, sqrt_price_lower);
        amount_out = liquidity * (1.0 / new_sqrt_p - 1.0 / sqrt_price_current);
    }

    double old_price = sqrt_price_current * sqrt_price_current;
    double new_price = new_sqrt_p * new_sqrt_p;
    double price_impact = std::abs(new_price - old_price) / old_price;

    return {std::max(amount_out, 0.0), new_sqrt_p, price_impact};
}

double calculate_liquidity(
    double amount_x, double amount_y,
    double sqrt_price_current,
    double sqrt_price_lower, double sqrt_price_upper) noexcept {
    if (sqrt_price_current <= sqrt_price_lower) {
        // Only token X
        return amount_x * sqrt_price_lower * sqrt_price_upper /
               (sqrt_price_upper - sqrt_price_lower);
    } else if (sqrt_price_current >= sqrt_price_upper) {
        // Only token Y
        return amount_y / (sqrt_price_upper - sqrt_price_lower);
    } else {
        // Both tokens
        double l_x = amount_x * sqrt_price_current * sqrt_price_upper /
                     (sqrt_price_upper - sqrt_price_current);
        double l_y = amount_y / (sqrt_price_current - sqrt_price_lower);
        return std::min(l_x, l_y);
    }
}

// =============================================================================
// Risk Metrics
// =============================================================================

double volatility(
    const std::vector<double>& returns, bool annualize, int periods_per_year) noexcept {
    if (returns.size() < 2) return 0.0;

    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    double variance = 0.0;
    for (double r : returns) {
        double diff = r - mean;
        variance += diff * diff;
    }
    variance /= (returns.size() - 1);

    double std_dev = std::sqrt(variance);

    if (annualize) {
        std_dev *= std::sqrt(periods_per_year);
    }

    return std_dev;
}

double sharpe_ratio(
    const std::vector<double>& returns, double risk_free_rate, int periods_per_year) noexcept {
    if (returns.size() < 2) return 0.0;

    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    double variance = 0.0;
    for (double r : returns) {
        double diff = r - mean;
        variance += diff * diff;
    }
    variance /= (returns.size() - 1);

    double std_dev = std::sqrt(variance);
    if (std_dev == 0.0) return 0.0;

    double period_rf = risk_free_rate / periods_per_year;
    double excess_return = mean - period_rf;

    return (excess_return * periods_per_year) / (std_dev * std::sqrt(periods_per_year));
}

double sortino_ratio(
    const std::vector<double>& returns,
    double risk_free_rate, double target_return, int periods_per_year) noexcept {
    if (returns.size() < 2) return 0.0;

    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    // Downside deviation
    double downside_sum = 0.0;
    for (double r : returns) {
        double diff = std::min(r - target_return, 0.0);
        downside_sum += diff * diff;
    }
    double downside_std = std::sqrt(downside_sum / returns.size());

    if (downside_std == 0.0) {
        return (mean > risk_free_rate / periods_per_year) ? INFINITY : 0.0;
    }

    double period_rf = risk_free_rate / periods_per_year;
    double excess_return = mean - period_rf;

    return (excess_return * periods_per_year) / (downside_std * std::sqrt(periods_per_year));
}

std::tuple<double, size_t, size_t> max_drawdown(
    const std::vector<double>& prices) noexcept {
    if (prices.size() < 2) return {0.0, 0, 0};

    double peak = prices[0];
    size_t peak_idx = 0;
    double max_dd = 0.0;
    size_t max_dd_peak = 0;
    size_t max_dd_trough = 0;

    for (size_t i = 0; i < prices.size(); ++i) {
        if (prices[i] > peak) {
            peak = prices[i];
            peak_idx = i;
        }

        double dd = (peak > 0) ? (peak - prices[i]) / peak : 0.0;

        if (dd > max_dd) {
            max_dd = dd;
            max_dd_peak = peak_idx;
            max_dd_trough = i;
        }
    }

    return {max_dd, max_dd_peak, max_dd_trough};
}

double var(
    const std::vector<double>& returns, double confidence, bool parametric) noexcept {
    if (returns.size() < 10) return 0.0;

    if (!parametric) {
        // Historical VaR
        std::vector<double> sorted = returns;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(sorted.size() * (1.0 - confidence));
        return -sorted[idx];
    }

    // Parametric VaR (assumes normal)
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    double variance = 0.0;
    for (double r : returns) {
        double diff = r - mean;
        variance += diff * diff;
    }
    variance /= (returns.size() - 1);
    double std_dev = std::sqrt(variance);

    // Z-score approximation
    double z = (confidence == 0.95) ? -1.645 : -2.326;

    return -(mean + z * std_dev);
}

double cvar(
    const std::vector<double>& returns, double confidence) noexcept {
    if (returns.size() < 10) return 0.0;

    double var_value = var(returns, confidence, false);

    // Average of returns worse than VaR
    std::vector<double> tail;
    for (double r : returns) {
        if (r <= -var_value) {
            tail.push_back(r);
        }
    }

    if (tail.empty()) return var_value;

    return -std::accumulate(tail.begin(), tail.end(), 0.0) / tail.size();
}

// =============================================================================
// SIMD Operations (AVX2)
// =============================================================================

#ifdef LX_SIMD_AVX2

void black_scholes_batch(
    const double* S, const double* K, const double* T,
    const double* r, const double* sigma,
    double* prices, size_t count, bool is_call) noexcept {
    // Process 4 at a time with AVX2
    size_t i = 0;

    for (; i + 4 <= count; i += 4) {
        __m256d vS = _mm256_loadu_pd(&S[i]);
        __m256d vK = _mm256_loadu_pd(&K[i]);
        __m256d vT = _mm256_loadu_pd(&T[i]);
        __m256d vr = _mm256_loadu_pd(&r[i]);
        __m256d vsigma = _mm256_loadu_pd(&sigma[i]);

        // sqrt(T)
        __m256d sqrt_T = _mm256_sqrt_pd(vT);

        // d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        __m256d log_SK = _mm256_div_pd(vS, vK);
        // Note: Would need log intrinsics or loop here

        // Simplified: fall back to scalar for now
        for (size_t j = 0; j < 4; ++j) {
            prices[i + j] = black_scholes(S[i+j], K[i+j], T[i+j], r[i+j], sigma[i+j], is_call);
        }
    }

    // Handle remainder
    for (; i < count; ++i) {
        prices[i] = black_scholes(S[i], K[i], T[i], r[i], sigma[i], is_call);
    }
}

double sum_avx2(const double* data, size_t count) noexcept {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 4 <= count; i += 4) {
        __m256d v = _mm256_loadu_pd(&data[i]);
        sum = _mm256_add_pd(sum, v);
    }

    // Horizontal sum
    double result[4];
    _mm256_storeu_pd(result, sum);
    double total = result[0] + result[1] + result[2] + result[3];

    // Remainder
    for (; i < count; ++i) {
        total += data[i];
    }

    return total;
}

double mean_avx2(const double* data, size_t count) noexcept {
    if (count == 0) return 0.0;
    return sum_avx2(data, count) / count;
}

double variance_avx2(const double* data, size_t count, double mean) noexcept {
    if (count < 2) return 0.0;

    __m256d vmean = _mm256_set1_pd(mean);
    __m256d sum_sq = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 4 <= count; i += 4) {
        __m256d v = _mm256_loadu_pd(&data[i]);
        __m256d diff = _mm256_sub_pd(v, vmean);
        sum_sq = _mm256_fmadd_pd(diff, diff, sum_sq);
    }

    double result[4];
    _mm256_storeu_pd(result, sum_sq);
    double total = result[0] + result[1] + result[2] + result[3];

    // Remainder
    for (; i < count; ++i) {
        double diff = data[i] - mean;
        total += diff * diff;
    }

    return total / (count - 1);
}

#endif  // LX_SIMD_AVX2

// =============================================================================
// Statistical Utilities
// =============================================================================

std::vector<double> calculate_returns(const std::vector<double>& prices) {
    if (prices.size() < 2) return {};

    std::vector<double> returns;
    returns.reserve(prices.size() - 1);

    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i-1] > 0) {
            returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
        }
    }

    return returns;
}

std::vector<double> rolling_mean(const std::vector<double>& data, size_t window) {
    if (data.size() < window) return {};

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    double sum = std::accumulate(data.begin(), data.begin() + window, 0.0);
    result.push_back(sum / window);

    for (size_t i = window; i < data.size(); ++i) {
        sum += data[i] - data[i - window];
        result.push_back(sum / window);
    }

    return result;
}

std::vector<double> rolling_std(const std::vector<double>& data, size_t window) {
    if (data.size() < window) return {};

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    for (size_t i = 0; i <= data.size() - window; ++i) {
        double mean = 0.0;
        for (size_t j = 0; j < window; ++j) {
            mean += data[i + j];
        }
        mean /= window;

        double variance = 0.0;
        for (size_t j = 0; j < window; ++j) {
            double diff = data[i + j] - mean;
            variance += diff * diff;
        }
        variance /= (window - 1);

        result.push_back(std::sqrt(variance));
    }

    return result;
}

std::vector<double> ema(const std::vector<double>& data, double alpha) {
    if (data.empty()) return {};

    std::vector<double> result;
    result.reserve(data.size());
    result.push_back(data[0]);

    for (size_t i = 1; i < data.size(); ++i) {
        result.push_back(alpha * data[i] + (1.0 - alpha) * result.back());
    }

    return result;
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) noexcept {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    size_t n = x.size();

    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    double denom = std::sqrt(var_x * var_y);
    return (denom > 0) ? cov / denom : 0.0;
}

double covariance(const std::vector<double>& x, const std::vector<double>& y) noexcept {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    size_t n = x.size();

    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;

    double cov = 0.0;
    for (size_t i = 0; i < n; ++i) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }

    return cov / (n - 1);
}

double beta(
    const std::vector<double>& asset_returns,
    const std::vector<double>& market_returns) noexcept {
    double cov = covariance(asset_returns, market_returns);

    double mean_m = std::accumulate(market_returns.begin(), market_returns.end(), 0.0) /
                    market_returns.size();

    double var_m = 0.0;
    for (double r : market_returns) {
        double diff = r - mean_m;
        var_m += diff * diff;
    }
    var_m /= (market_returns.size() - 1);

    return (var_m > 0) ? cov / var_m : 0.0;
}

}  // namespace lx::trading::math
