/**
 * LX Trading SDK - Financial Mathematics Implementation
 */

#include "lx_trading/math.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Statistical Functions
 * ============================================================================ */

double lx_norm_cdf(double x) {
    /* Abramowitz & Stegun approximation */
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;
    const double p = 0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = fabs(x) / LX_MATH_SQRT_2;

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

double lx_norm_pdf(double x) {
    return exp(-0.5 * x * x) / LX_MATH_SQRT_2PI;
}

/* ============================================================================
 * Black-Scholes Options Pricing
 * ============================================================================ */

double lx_black_scholes(double S, double K, double T, double r, double sigma, bool is_call) {
    if (T <= 0) {
        /* At expiry */
        if (is_call) {
            return (S > K) ? S - K : 0.0;
        } else {
            return (K > S) ? K - S : 0.0;
        }
    }

    if (sigma <= 0 || S <= 0 || K <= 0) {
        return 0.0;
    }

    double sqrt_T = sqrt(T);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    double discount = exp(-r * T);

    if (is_call) {
        return S * lx_norm_cdf(d1) - K * discount * lx_norm_cdf(d2);
    } else {
        return K * discount * lx_norm_cdf(-d2) - S * lx_norm_cdf(-d1);
    }
}

double lx_implied_volatility(double price, double S, double K, double T, double r,
                             bool is_call, double tol, int max_iter) {
    if (price <= 0 || S <= 0 || K <= 0 || T <= 0) {
        return -1.0;
    }

    /* Newton-Raphson starting point */
    double sigma = 0.2;  /* Initial guess */

    for (int i = 0; i < max_iter; i++) {
        double calc_price = lx_black_scholes(S, K, T, r, sigma, is_call);
        double diff = calc_price - price;

        if (fabs(diff) < tol) {
            return sigma;
        }

        /* Calculate vega */
        double sqrt_T = sqrt(T);
        double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
        double vega = S * lx_norm_pdf(d1) * sqrt_T;

        if (vega < 1e-10) {
            /* Vega too small, can't converge */
            break;
        }

        sigma -= diff / vega;

        /* Bound sigma */
        if (sigma < 0.001) sigma = 0.001;
        if (sigma > 10.0) sigma = 10.0;
    }

    return -1.0;  /* Did not converge */
}

LxGreeks lx_greeks(double S, double K, double T, double r, double sigma, bool is_call) {
    LxGreeks g = {0};

    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) {
        return g;
    }

    double sqrt_T = sqrt(T);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    double nd1 = lx_norm_cdf(d1);
    double nd2 = lx_norm_cdf(d2);
    double pdf_d1 = lx_norm_pdf(d1);
    double discount = exp(-r * T);

    /* Delta */
    if (is_call) {
        g.delta = nd1;
    } else {
        g.delta = nd1 - 1.0;
    }

    /* Gamma */
    g.gamma = pdf_d1 / (S * sigma * sqrt_T);

    /* Vega (per 1% change in volatility) */
    g.vega = S * pdf_d1 * sqrt_T * 0.01;

    /* Theta (daily) */
    double theta_term1 = -S * pdf_d1 * sigma / (2.0 * sqrt_T);
    if (is_call) {
        double theta_term2 = -r * K * discount * nd2;
        g.theta = (theta_term1 + theta_term2) / 365.0;
    } else {
        double theta_term2 = r * K * discount * lx_norm_cdf(-d2);
        g.theta = (theta_term1 + theta_term2) / 365.0;
    }

    /* Rho (per 1% change in rate) */
    if (is_call) {
        g.rho = K * T * discount * nd2 * 0.01;
    } else {
        g.rho = -K * T * discount * lx_norm_cdf(-d2) * 0.01;
    }

    return g;
}

/* ============================================================================
 * AMM Pricing - Constant Product
 * ============================================================================ */

void lx_constant_product_price(double reserve_x, double reserve_y,
                               double amount_in, double fee_rate, bool is_x_to_y,
                               double* output_amount, double* effective_price) {
    if (!output_amount || !effective_price) return;

    if (reserve_x <= 0 || reserve_y <= 0 || amount_in <= 0) {
        *output_amount = 0;
        *effective_price = 0;
        return;
    }

    double amount_in_with_fee = amount_in * (1.0 - fee_rate);

    double input_reserve, output_reserve;
    if (is_x_to_y) {
        input_reserve = reserve_x;
        output_reserve = reserve_y;
    } else {
        input_reserve = reserve_y;
        output_reserve = reserve_x;
    }

    /* xy = k, so output = output_reserve - k / (input_reserve + amount_in) */
    double k = input_reserve * output_reserve;
    double new_input_reserve = input_reserve + amount_in_with_fee;
    *output_amount = output_reserve - k / new_input_reserve;

    if (*output_amount > 0) {
        *effective_price = *output_amount / amount_in;
    } else {
        *effective_price = 0;
    }
}

/* ============================================================================
 * AMM Pricing - Concentrated Liquidity
 * ============================================================================ */

void lx_concentrated_liquidity_price(double liquidity,
                                     double sqrt_price_current,
                                     double sqrt_price_lower,
                                     double sqrt_price_upper,
                                     double amount_in,
                                     double fee_rate,
                                     bool is_token0_in,
                                     double* output_amount,
                                     double* new_sqrt_price,
                                     double* price_impact) {
    if (!output_amount || !new_sqrt_price || !price_impact) return;

    if (liquidity <= 0 || amount_in <= 0 || sqrt_price_current <= 0) {
        *output_amount = 0;
        *new_sqrt_price = sqrt_price_current;
        *price_impact = 0;
        return;
    }

    double amount_in_with_fee = amount_in * (1.0 - fee_rate);
    double initial_price = sqrt_price_current * sqrt_price_current;

    if (is_token0_in) {
        /* Token0 in, price increases */
        double delta_sqrt_price = amount_in_with_fee / liquidity;
        double new_sp = sqrt_price_current + delta_sqrt_price;

        /* Clamp to range */
        if (new_sp > sqrt_price_upper) new_sp = sqrt_price_upper;

        /* Calculate token1 output */
        *output_amount = liquidity * (new_sp - sqrt_price_current);
        *new_sqrt_price = new_sp;
    } else {
        /* Token1 in, price decreases */
        double delta_1_sqrt_price = amount_in_with_fee / liquidity;
        double new_sp = sqrt_price_current - delta_1_sqrt_price / sqrt_price_current;

        /* Clamp to range */
        if (new_sp < sqrt_price_lower) new_sp = sqrt_price_lower;

        /* Calculate token0 output */
        *output_amount = liquidity * (1.0 / new_sp - 1.0 / sqrt_price_current);
        if (*output_amount < 0) *output_amount = -*output_amount;
        *new_sqrt_price = new_sp;
    }

    /* Calculate price impact */
    double final_price = (*new_sqrt_price) * (*new_sqrt_price);
    if (initial_price > 0) {
        *price_impact = fabs(final_price - initial_price) / initial_price * 100.0;
    } else {
        *price_impact = 0;
    }
}

double lx_calculate_liquidity(double amount_x, double amount_y,
                              double sqrt_price_current,
                              double sqrt_price_lower,
                              double sqrt_price_upper) {
    if (sqrt_price_current <= 0 || sqrt_price_lower <= 0 || sqrt_price_upper <= 0) {
        return 0;
    }

    if (sqrt_price_lower >= sqrt_price_upper) {
        return 0;
    }

    double liquidity_x = 0;
    double liquidity_y = 0;

    if (sqrt_price_current <= sqrt_price_lower) {
        /* All token0 */
        if (amount_x > 0) {
            liquidity_x = amount_x * sqrt_price_lower * sqrt_price_upper /
                         (sqrt_price_upper - sqrt_price_lower);
        }
        return liquidity_x;
    } else if (sqrt_price_current >= sqrt_price_upper) {
        /* All token1 */
        if (amount_y > 0) {
            liquidity_y = amount_y / (sqrt_price_upper - sqrt_price_lower);
        }
        return liquidity_y;
    } else {
        /* Both tokens */
        if (amount_x > 0) {
            liquidity_x = amount_x * sqrt_price_current * sqrt_price_upper /
                         (sqrt_price_upper - sqrt_price_current);
        }
        if (amount_y > 0) {
            liquidity_y = amount_y / (sqrt_price_current - sqrt_price_lower);
        }
        return (liquidity_x < liquidity_y) ? liquidity_x : liquidity_y;
    }
}

/* ============================================================================
 * Price Conversions
 * ============================================================================ */

double lx_price_to_sqrt_price(double price) {
    return sqrt(price);
}

double lx_sqrt_price_to_price(double sqrt_price) {
    return sqrt_price * sqrt_price;
}

double lx_tick_to_sqrt_price(int tick) {
    return pow(1.0001, tick / 2.0);
}

int lx_sqrt_price_to_tick(double sqrt_price, int tick_spacing) {
    if (sqrt_price <= 0 || tick_spacing <= 0) return 0;
    int tick = (int)(2.0 * log(sqrt_price) / log(1.0001));
    return (tick / tick_spacing) * tick_spacing;
}

/* ============================================================================
 * Basic Statistics
 * ============================================================================ */

double lx_sum(const double* data, size_t count) {
    if (!data || count == 0) return 0;

    double sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += data[i];
    }
    return sum;
}

double lx_mean(const double* data, size_t count) {
    if (!data || count == 0) return 0;
    return lx_sum(data, count) / (double)count;
}

double lx_variance(const double* data, size_t count) {
    if (!data || count < 2) return 0;

    double m = lx_mean(data, count);
    double sum_sq = 0;

    for (size_t i = 0; i < count; i++) {
        double diff = data[i] - m;
        sum_sq += diff * diff;
    }

    return sum_sq / (double)(count - 1);  /* Sample variance */
}

double lx_std_dev(const double* data, size_t count) {
    return sqrt(lx_variance(data, count));
}

/* ============================================================================
 * Risk Metrics
 * ============================================================================ */

double lx_volatility(const double* returns, size_t count,
                     bool annualize, int periods_per_year) {
    if (!returns || count < 2) return 0;

    double vol = lx_std_dev(returns, count);

    if (annualize && periods_per_year > 0) {
        vol *= sqrt((double)periods_per_year);
    }

    return vol;
}

double lx_sharpe_ratio(const double* returns, size_t count,
                       double risk_free_rate, int periods_per_year) {
    if (!returns || count < 2) return 0;

    double mean_return = lx_mean(returns, count);
    double vol = lx_std_dev(returns, count);

    if (vol < 1e-10) return 0;

    double excess_return = mean_return - risk_free_rate / (double)periods_per_year;
    double sharpe = excess_return / vol;

    /* Annualize */
    if (periods_per_year > 0) {
        sharpe *= sqrt((double)periods_per_year);
    }

    return sharpe;
}

double lx_sortino_ratio(const double* returns, size_t count,
                        double risk_free_rate, double target_return,
                        int periods_per_year) {
    if (!returns || count < 2) return 0;

    double mean_return = lx_mean(returns, count);

    /* Calculate downside deviation */
    double sum_sq = 0;
    size_t downside_count = 0;

    for (size_t i = 0; i < count; i++) {
        if (returns[i] < target_return) {
            double diff = returns[i] - target_return;
            sum_sq += diff * diff;
            downside_count++;
        }
    }

    if (downside_count == 0) {
        return INFINITY;  /* No downside */
    }

    double downside_dev = sqrt(sum_sq / (double)downside_count);

    if (downside_dev < 1e-10) return 0;

    double excess_return = mean_return - risk_free_rate / (double)periods_per_year;
    double sortino = excess_return / downside_dev;

    /* Annualize */
    if (periods_per_year > 0) {
        sortino *= sqrt((double)periods_per_year);
    }

    return sortino;
}

double lx_max_drawdown(const double* prices, size_t count,
                       size_t* peak_idx, size_t* trough_idx) {
    if (!prices || count < 2) {
        if (peak_idx) *peak_idx = 0;
        if (trough_idx) *trough_idx = 0;
        return 0;
    }

    double max_dd = 0;
    double peak = prices[0];
    size_t peak_i = 0;
    size_t best_peak_i = 0;
    size_t best_trough_i = 0;

    for (size_t i = 1; i < count; i++) {
        if (prices[i] > peak) {
            peak = prices[i];
            peak_i = i;
        } else {
            double dd = (peak - prices[i]) / peak;
            if (dd > max_dd) {
                max_dd = dd;
                best_peak_i = peak_i;
                best_trough_i = i;
            }
        }
    }

    if (peak_idx) *peak_idx = best_peak_i;
    if (trough_idx) *trough_idx = best_trough_i;

    return max_dd;
}

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

double lx_var(const double* returns, size_t count,
              double confidence, bool parametric) {
    if (!returns || count < 2 || confidence <= 0 || confidence >= 1) return 0;

    if (parametric) {
        /* Parametric VaR using normal distribution */
        double m = lx_mean(returns, count);
        double s = lx_std_dev(returns, count);

        /* Z-score for confidence level (one-tailed) */
        /* For 95%, z ~ 1.645; for 99%, z ~ 2.326 */
        /* Using inverse of norm_cdf approximation */
        double z;
        if (confidence >= 0.99) {
            z = 2.326;
        } else if (confidence >= 0.95) {
            z = 1.645;
        } else {
            /* Rough approximation */
            z = sqrt(2.0) * (confidence - 0.5) * 3.0;
        }

        return -(m - z * s);
    } else {
        /* Historical VaR */
        double* sorted = malloc(count * sizeof(double));
        if (!sorted) return 0;

        memcpy(sorted, returns, count * sizeof(double));
        qsort(sorted, count, sizeof(double), compare_doubles);

        size_t idx = (size_t)((1.0 - confidence) * (double)count);
        if (idx >= count) idx = count - 1;

        double var_val = -sorted[idx];
        free(sorted);

        return var_val;
    }
}

double lx_cvar(const double* returns, size_t count, double confidence) {
    if (!returns || count < 2 || confidence <= 0 || confidence >= 1) return 0;

    double* sorted = malloc(count * sizeof(double));
    if (!sorted) return 0;

    memcpy(sorted, returns, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_doubles);

    size_t cutoff = (size_t)((1.0 - confidence) * (double)count);
    if (cutoff == 0) cutoff = 1;
    if (cutoff > count) cutoff = count;

    double sum = 0;
    for (size_t i = 0; i < cutoff; i++) {
        sum += sorted[i];
    }

    double cvar_val = -sum / (double)cutoff;
    free(sorted);

    return cvar_val;
}

/* ============================================================================
 * Statistical Utilities
 * ============================================================================ */

size_t lx_calculate_returns(const double* prices, size_t price_count, double* returns) {
    if (!prices || !returns || price_count < 2) return 0;

    for (size_t i = 1; i < price_count; i++) {
        if (prices[i - 1] != 0) {
            returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
        } else {
            returns[i - 1] = 0;
        }
    }

    return price_count - 1;
}

size_t lx_rolling_mean(const double* data, size_t count, size_t window, double* result) {
    if (!data || !result || count < window || window == 0) return 0;

    size_t result_count = count - window + 1;

    for (size_t i = 0; i < result_count; i++) {
        double sum = 0;
        for (size_t j = 0; j < window; j++) {
            sum += data[i + j];
        }
        result[i] = sum / (double)window;
    }

    return result_count;
}

size_t lx_rolling_std(const double* data, size_t count, size_t window, double* result) {
    if (!data || !result || count < window || window < 2) return 0;

    size_t result_count = count - window + 1;

    for (size_t i = 0; i < result_count; i++) {
        /* Calculate mean */
        double sum = 0;
        for (size_t j = 0; j < window; j++) {
            sum += data[i + j];
        }
        double mean = sum / (double)window;

        /* Calculate variance */
        double sum_sq = 0;
        for (size_t j = 0; j < window; j++) {
            double diff = data[i + j] - mean;
            sum_sq += diff * diff;
        }

        result[i] = sqrt(sum_sq / (double)(window - 1));
    }

    return result_count;
}

void lx_ema(const double* data, size_t count, double alpha, double* result) {
    if (!data || !result || count == 0 || alpha <= 0 || alpha > 1) return;

    result[0] = data[0];

    for (size_t i = 1; i < count; i++) {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }
}

double lx_correlation(const double* x, const double* y, size_t count) {
    if (!x || !y || count < 2) return 0;

    double mean_x = lx_mean(x, count);
    double mean_y = lx_mean(y, count);

    double cov = 0;
    double var_x = 0;
    double var_y = 0;

    for (size_t i = 0; i < count; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    double denom = sqrt(var_x * var_y);
    if (denom < 1e-10) return 0;

    return cov / denom;
}

double lx_covariance(const double* x, const double* y, size_t count) {
    if (!x || !y || count < 2) return 0;

    double mean_x = lx_mean(x, count);
    double mean_y = lx_mean(y, count);

    double cov = 0;
    for (size_t i = 0; i < count; i++) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }

    return cov / (double)(count - 1);
}

double lx_beta(const double* asset_returns, const double* market_returns, size_t count) {
    if (!asset_returns || !market_returns || count < 2) return 0;

    double cov = lx_covariance(asset_returns, market_returns, count);
    double var_market = lx_variance(market_returns, count);

    if (var_market < 1e-10) return 0;

    return cov / var_market;
}
