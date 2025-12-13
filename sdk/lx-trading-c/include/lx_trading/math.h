/**
 * LX Trading SDK - Financial Mathematics
 * Options pricing, AMM math, and risk metrics
 */

#ifndef LX_TRADING_MATH_H
#define LX_TRADING_MATH_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

#define LX_MATH_PI 3.14159265358979323846
#define LX_MATH_SQRT_2PI 2.506628274631000502
#define LX_MATH_SQRT_2 1.41421356237309504880

/* ============================================================================
 * Greeks Structure
 * ============================================================================ */

typedef struct {
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
} LxGreeks;

/* ============================================================================
 * Statistical Functions
 * ============================================================================ */

/* Standard normal CDF (Abramowitz & Stegun approximation) */
double lx_norm_cdf(double x);

/* Standard normal PDF */
double lx_norm_pdf(double x);

/* ============================================================================
 * Black-Scholes Options Pricing
 * ============================================================================ */

/**
 * Calculate Black-Scholes option price
 * @param S       Spot price
 * @param K       Strike price
 * @param T       Time to expiration (years)
 * @param r       Risk-free rate
 * @param sigma   Volatility
 * @param is_call true for call, false for put
 * @return Option price
 */
double lx_black_scholes(double S, double K, double T, double r, double sigma, bool is_call);

/**
 * Calculate implied volatility from option price (Newton-Raphson)
 * @param price   Option price
 * @param S       Spot price
 * @param K       Strike price
 * @param T       Time to expiration (years)
 * @param r       Risk-free rate
 * @param is_call true for call, false for put
 * @param tol     Tolerance (default 1e-6)
 * @param max_iter Maximum iterations (default 100)
 * @return Implied volatility, or -1 if not found
 */
double lx_implied_volatility(double price, double S, double K, double T, double r,
                             bool is_call, double tol, int max_iter);

/**
 * Calculate all Greeks
 * @param S       Spot price
 * @param K       Strike price
 * @param T       Time to expiration (years)
 * @param r       Risk-free rate
 * @param sigma   Volatility
 * @param is_call true for call, false for put
 * @return Greeks structure
 */
LxGreeks lx_greeks(double S, double K, double T, double r, double sigma, bool is_call);

/* ============================================================================
 * AMM Pricing - Constant Product (Uniswap V2)
 * ============================================================================ */

/**
 * Calculate constant product AMM swap output
 * @param reserve_x  Reserve of token X
 * @param reserve_y  Reserve of token Y
 * @param amount_in  Amount of input token
 * @param fee_rate   Fee rate (e.g., 0.003 for 0.3%)
 * @param is_x_to_y  true if swapping X for Y, false for Y to X
 * @param output_amount Output: amount of output token
 * @param effective_price Output: effective price
 */
void lx_constant_product_price(double reserve_x, double reserve_y,
                               double amount_in, double fee_rate, bool is_x_to_y,
                               double* output_amount, double* effective_price);

/* ============================================================================
 * AMM Pricing - Concentrated Liquidity (Uniswap V3)
 * ============================================================================ */

/**
 * Calculate concentrated liquidity swap output
 * @param liquidity        Liquidity in the range
 * @param sqrt_price_current Current sqrt price
 * @param sqrt_price_lower   Lower sqrt price bound
 * @param sqrt_price_upper   Upper sqrt price bound
 * @param amount_in          Amount of input token
 * @param fee_rate           Fee rate
 * @param is_token0_in       true if token0 is input
 * @param output_amount      Output: amount of output token
 * @param new_sqrt_price     Output: new sqrt price after swap
 * @param price_impact       Output: price impact percentage
 */
void lx_concentrated_liquidity_price(double liquidity,
                                     double sqrt_price_current,
                                     double sqrt_price_lower,
                                     double sqrt_price_upper,
                                     double amount_in,
                                     double fee_rate,
                                     bool is_token0_in,
                                     double* output_amount,
                                     double* new_sqrt_price,
                                     double* price_impact);

/**
 * Calculate liquidity for a concentrated position
 */
double lx_calculate_liquidity(double amount_x, double amount_y,
                              double sqrt_price_current,
                              double sqrt_price_lower,
                              double sqrt_price_upper);

/* ============================================================================
 * Price Conversions
 * ============================================================================ */

double lx_price_to_sqrt_price(double price);
double lx_sqrt_price_to_price(double sqrt_price);
double lx_tick_to_sqrt_price(int tick);
int lx_sqrt_price_to_tick(double sqrt_price, int tick_spacing);

/* ============================================================================
 * Risk Metrics
 * ============================================================================ */

/**
 * Calculate historical volatility
 * @param returns      Array of returns
 * @param count        Number of returns
 * @param annualize    Whether to annualize
 * @param periods_per_year Periods per year (e.g., 252 for daily)
 * @return Volatility
 */
double lx_volatility(const double* returns, size_t count,
                     bool annualize, int periods_per_year);

/**
 * Calculate Sharpe ratio
 * @param returns           Array of returns
 * @param count             Number of returns
 * @param risk_free_rate    Risk-free rate (period, not annual)
 * @param periods_per_year  Periods per year
 * @return Sharpe ratio
 */
double lx_sharpe_ratio(const double* returns, size_t count,
                       double risk_free_rate, int periods_per_year);

/**
 * Calculate Sortino ratio (uses downside deviation)
 * @param returns           Array of returns
 * @param count             Number of returns
 * @param risk_free_rate    Risk-free rate
 * @param target_return     Target return (for downside calculation)
 * @param periods_per_year  Periods per year
 * @return Sortino ratio
 */
double lx_sortino_ratio(const double* returns, size_t count,
                        double risk_free_rate, double target_return,
                        int periods_per_year);

/**
 * Calculate maximum drawdown
 * @param prices       Array of prices
 * @param count        Number of prices
 * @param peak_idx     Output: index of peak
 * @param trough_idx   Output: index of trough
 * @return Maximum drawdown as a fraction (0.0 to 1.0)
 */
double lx_max_drawdown(const double* prices, size_t count,
                       size_t* peak_idx, size_t* trough_idx);

/**
 * Calculate Value at Risk
 * @param returns      Array of returns
 * @param count        Number of returns
 * @param confidence   Confidence level (e.g., 0.95)
 * @param parametric   Use parametric (true) or historical (false)
 * @return VaR as positive number (loss)
 */
double lx_var(const double* returns, size_t count,
              double confidence, bool parametric);

/**
 * Calculate Conditional VaR (Expected Shortfall)
 * @param returns      Array of returns
 * @param count        Number of returns
 * @param confidence   Confidence level
 * @return CVaR as positive number
 */
double lx_cvar(const double* returns, size_t count, double confidence);

/* ============================================================================
 * Statistical Utilities
 * ============================================================================ */

/**
 * Calculate returns from prices
 * @param prices       Array of prices
 * @param price_count  Number of prices
 * @param returns      Output: array of returns (must have space for price_count - 1)
 * @return Number of returns calculated
 */
size_t lx_calculate_returns(const double* prices, size_t price_count, double* returns);

/**
 * Calculate rolling mean
 * @param data         Input data
 * @param count        Number of data points
 * @param window       Window size
 * @param result       Output: rolling means (must have space for count - window + 1)
 * @return Number of results
 */
size_t lx_rolling_mean(const double* data, size_t count, size_t window, double* result);

/**
 * Calculate rolling standard deviation
 * @param data         Input data
 * @param count        Number of data points
 * @param window       Window size
 * @param result       Output: rolling stds (must have space for count - window + 1)
 * @return Number of results
 */
size_t lx_rolling_std(const double* data, size_t count, size_t window, double* result);

/**
 * Calculate exponential moving average
 * @param data         Input data
 * @param count        Number of data points
 * @param alpha        Smoothing factor (0 < alpha <= 1)
 * @param result       Output: EMA values (must have space for count)
 */
void lx_ema(const double* data, size_t count, double alpha, double* result);

/**
 * Calculate correlation coefficient
 * @param x            First array
 * @param y            Second array
 * @param count        Number of elements
 * @return Correlation coefficient (-1 to 1)
 */
double lx_correlation(const double* x, const double* y, size_t count);

/**
 * Calculate covariance
 * @param x            First array
 * @param y            Second array
 * @param count        Number of elements
 * @return Covariance
 */
double lx_covariance(const double* x, const double* y, size_t count);

/**
 * Calculate beta coefficient
 * @param asset_returns  Asset returns
 * @param market_returns Market returns
 * @param count          Number of returns
 * @return Beta coefficient
 */
double lx_beta(const double* asset_returns, const double* market_returns, size_t count);

/* ============================================================================
 * Basic Statistics
 * ============================================================================ */

double lx_mean(const double* data, size_t count);
double lx_variance(const double* data, size_t count);
double lx_std_dev(const double* data, size_t count);
double lx_sum(const double* data, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* LX_TRADING_MATH_H */
