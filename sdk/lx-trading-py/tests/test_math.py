"""Tests for LX Trading SDK math module."""

import pytest
import math
from decimal import Decimal

from lx_trading.math import (
    black_scholes,
    implied_volatility,
    greeks,
    constant_product_price,
    concentrated_liquidity_price,
    calculate_liquidity,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    var,
    cvar,
    price_to_sqrt_price,
    sqrt_price_to_price,
    tick_to_sqrt_price,
    sqrt_price_to_tick,
    _norm_cdf,
    _norm_pdf,
)


class TestBlackScholes:
    def test_atm_call_price(self):
        """ATM call: S=100, K=100, T=1yr, r=5%, vol=20%"""
        price = black_scholes(100, 100, 1, 0.05, 0.2, "call")
        # Expected ~10.45 based on standard B-S
        assert 10 < price < 11

    def test_atm_put_price(self):
        price = black_scholes(100, 100, 1, 0.05, 0.2, "put")
        # Put-call parity: Put ≈ 5.57
        assert 5 < price < 6

    def test_intrinsic_value_at_expiry(self):
        # At expiry (T=0), call = max(S-K, 0)
        call_itm = black_scholes(110, 100, 0, 0.05, 0.2, "call")
        assert call_itm == 10

        call_otm = black_scholes(90, 100, 0, 0.05, 0.2, "call")
        assert call_otm == 0

        put_itm = black_scholes(90, 100, 0, 0.05, 0.2, "put")
        assert put_itm == 10

    def test_deep_itm_call(self):
        price = black_scholes(150, 100, 0.1, 0.05, 0.2, "call")
        assert price > 49  # At least intrinsic
        assert price < 52  # Not too much time value

    def test_deep_otm_call(self):
        price = black_scholes(50, 100, 0.1, 0.05, 0.2, "call")
        assert price < 0.01


class TestImpliedVolatility:
    def test_recover_volatility(self):
        true_vol = 0.25
        price = black_scholes(100, 100, 1, 0.05, true_vol, "call")
        iv = implied_volatility(price, 100, 100, 1, 0.05, "call")
        assert abs(iv - true_vol) < 0.02

    def test_works_for_puts(self):
        true_vol = 0.30
        price = black_scholes(100, 105, 0.5, 0.03, true_vol, "put")
        iv = implied_volatility(price, 100, 105, 0.5, 0.03, "put")
        assert abs(iv - true_vol) < 0.02


class TestGreeks:
    def test_delta_atm_call(self):
        g = greeks(100, 100, 1, 0.05, 0.2, "call")
        # ATM call delta should be around 0.5-0.6
        assert 0.5 < g["delta"] < 0.7

    def test_delta_atm_put(self):
        g = greeks(100, 100, 1, 0.05, 0.2, "put")
        # Put delta is negative
        assert -0.6 < g["delta"] < -0.3

    def test_gamma_positive(self):
        g = greeks(100, 100, 1, 0.05, 0.2, "call")
        assert g["gamma"] > 0

    def test_theta_negative_for_long_call(self):
        g = greeks(100, 100, 1, 0.05, 0.2, "call")
        assert g["theta"] < 0  # Time decay

    def test_vega_positive(self):
        g = greeks(100, 100, 1, 0.05, 0.2, "call")
        assert g["vega"] > 0

    def test_zeros_at_expiry(self):
        g = greeks(100, 100, 0, 0.05, 0.2, "call")
        assert g["delta"] == 0
        assert g["gamma"] == 0
        assert g["vega"] == 0


class TestConstantProductAMM:
    def test_balanced_pool(self):
        out, price = constant_product_price(1000, 1000, 10, 0.003, True)
        # Without fee: dy = 1000 * 10 / (1000 + 10) ≈ 9.9
        # With fee: dy ≈ 9.87
        assert 9.8 < out < 10

    def test_effective_price(self):
        out, price = constant_product_price(1000, 2000, 10, 0.003, True)
        assert price > 0
        assert price < 2  # Less than spot due to slippage

    def test_symmetric(self):
        x_to_y, _ = constant_product_price(1000, 1000, 10, 0.003, True)
        y_to_x, _ = constant_product_price(1000, 1000, 10, 0.003, False)
        assert abs(x_to_y - y_to_x) < 0.1

    def test_zero_input(self):
        out, price = constant_product_price(1000, 1000, 0, 0.003, True)
        assert out == 0
        assert price == 0

    def test_fee_impact(self):
        no_fee, _ = constant_product_price(1000, 1000, 100, 0, True)
        with_fee, _ = constant_product_price(1000, 1000, 100, 0.01, True)
        assert with_fee < no_fee


class TestConcentratedLiquidity:
    def test_output_within_range(self):
        out, new_sqrt_p, impact = concentrated_liquidity_price(
            1000, 10, 9, 11, 10, 0.003, True
        )
        assert out > 0
        assert new_sqrt_p > 10  # Price goes up when buying Y
        assert new_sqrt_p <= 11  # Capped at upper bound

    def test_price_impact(self):
        _, _, impact = concentrated_liquidity_price(
            1000, 10, 9, 11, 100, 0.003, True
        )
        assert impact > 0
        assert impact < 1

    def test_swap_y_for_x(self):
        out, new_sqrt_p, _ = concentrated_liquidity_price(
            1000, 10, 9, 11, 10, 0.003, False
        )
        assert out > 0
        assert new_sqrt_p < 10  # Price goes down
        assert new_sqrt_p >= 9  # Capped at lower bound


class TestCalculateLiquidity:
    def test_in_range_position(self):
        L = calculate_liquidity(100, 1000, 10, 9, 11)
        assert L > 0

    def test_below_range_only_x(self):
        L = calculate_liquidity(100, 0, 8, 9, 11)
        assert L > 0

    def test_above_range_only_y(self):
        L = calculate_liquidity(0, 100, 12, 9, 11)
        assert L > 0


class TestVolatility:
    def test_calculate_volatility(self):
        returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005]
        vol = volatility(returns, annualize=False)
        assert vol > 0
        assert vol < 1

    def test_annualize(self):
        returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005]
        daily_vol = volatility(returns, annualize=False)
        annual_vol = volatility(returns, annualize=True, periods_per_year=252)
        assert abs(annual_vol - daily_vol * math.sqrt(252)) < 0.01

    def test_insufficient_data(self):
        assert volatility([0.01]) == 0
        assert volatility([]) == 0


class TestSharpeRatio:
    def test_positive_sharpe(self):
        returns = [0.01, 0.02, 0.015, 0.005, 0.02, 0.01, 0.015]
        sharpe = sharpe_ratio(returns, 0, 252)
        assert sharpe > 0

    def test_negative_sharpe(self):
        returns = [-0.01, -0.02, -0.015, -0.005, -0.02, -0.01, -0.015]
        sharpe = sharpe_ratio(returns, 0, 252)
        assert sharpe < 0

    def test_insufficient_data(self):
        assert sharpe_ratio([0.01]) == 0

    def test_risk_free_rate_impact(self):
        returns = [0.0001, 0.0002, 0.0001, 0.0002, 0.0001]
        sharpe_no_rf = sharpe_ratio(returns, 0, 252)
        sharpe_with_rf = sharpe_ratio(returns, 0.05, 252)
        assert sharpe_with_rf < sharpe_no_rf


class TestSortinoRatio:
    def test_calculate_sortino(self):
        returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.015]
        sortino = sortino_ratio(returns, 0, 0, 252)
        assert not math.isnan(sortino)

    def test_higher_than_sharpe_when_downside_limited(self):
        returns = [0.02, 0.01, 0.015, -0.002, 0.02, 0.01, 0.015]
        sharpe = sharpe_ratio(returns, 0, 252)
        sortino = sortino_ratio(returns, 0, 0, 252)
        assert sortino >= sharpe


class TestMaxDrawdown:
    def test_find_max_drawdown(self):
        prices = [100, 110, 105, 120, 90, 95, 100]
        dd, peak_idx, trough_idx = max_drawdown(prices)
        # Peak at 120, trough at 90: (120-90)/120 = 0.25
        assert abs(dd - 0.25) < 0.01

    def test_track_indices(self):
        prices = [100, 110, 105, 120, 90, 95, 100]
        _, peak_idx, trough_idx = max_drawdown(prices)
        assert peak_idx == 3  # Index of 120
        assert trough_idx == 4  # Index of 90

    def test_monotonic_increase(self):
        prices = [100, 110, 120, 130, 140]
        dd, _, _ = max_drawdown(prices)
        assert dd == 0

    def test_insufficient_data(self):
        dd, _, _ = max_drawdown([100])
        assert dd == 0


class TestValueAtRisk:
    def test_historical_var(self):
        returns = [(i - 50) / 1000 for i in range(100)]
        var_95 = var(returns, 0.95, "historical")
        assert var_95 > 0

    def test_parametric_var(self):
        returns = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, -0.015, 0.015, -0.025]
        var_95 = var(returns, 0.95, "parametric")
        assert var_95 > 0

    def test_insufficient_data(self):
        var_95 = var([0.01, 0.02], 0.95)
        assert var_95 == 0

    def test_higher_confidence_higher_var(self):
        import random
        returns = [(random.random() - 0.5) * 0.1 for _ in range(100)]
        var_95 = var(returns, 0.95)
        var_99 = var(returns, 0.99)
        assert var_99 >= var_95


class TestConditionalVaR:
    def test_cvar_greater_than_var(self):
        import random
        returns = [(random.random() - 0.5) * 0.1 for _ in range(100)]
        var_95 = var(returns, 0.95)
        cvar_95 = cvar(returns, 0.95)
        assert cvar_95 >= var_95 * 0.99  # Allow tolerance

    def test_insufficient_data(self):
        assert cvar([0.01, 0.02], 0.95) == 0


class TestNormalDistribution:
    def test_cdf(self):
        assert abs(_norm_cdf(0) - 0.5) < 0.001
        assert abs(_norm_cdf(-1.96) - 0.025) < 0.01
        assert abs(_norm_cdf(1.96) - 0.975) < 0.01

    def test_pdf(self):
        # PDF at 0 = 1/sqrt(2*pi) ≈ 0.3989
        assert abs(_norm_pdf(0) - 0.3989) < 0.001
        # Symmetric
        assert abs(_norm_pdf(-1) - _norm_pdf(1)) < 0.0001


class TestPriceTickConversions:
    def test_price_to_sqrt_price(self):
        assert price_to_sqrt_price(100) == 10
        assert price_to_sqrt_price(1) == 1

    def test_sqrt_price_to_price(self):
        assert sqrt_price_to_price(10) == 100
        assert sqrt_price_to_price(1) == 1

    def test_inverse_operations(self):
        price = 150
        recovered = sqrt_price_to_price(price_to_sqrt_price(price))
        assert abs(recovered - price) < 0.0001

    def test_tick_to_sqrt_price(self):
        # tick = 0 => sqrt(1.0001^0) = 1
        assert abs(tick_to_sqrt_price(0) - 1) < 0.0001

    def test_sqrt_price_to_tick(self):
        sqrt_p = tick_to_sqrt_price(1000)
        tick = sqrt_price_to_tick(sqrt_p, tick_spacing=1)
        assert abs(tick - 1000) < 1
