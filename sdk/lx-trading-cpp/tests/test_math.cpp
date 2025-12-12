// LX Trading SDK - Math Tests

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <lx/trading/math.hpp>

using namespace lx::trading::math;
using Catch::Approx;

TEST_CASE("Black-Scholes pricing", "[math]") {
    SECTION("At-the-money call") {
        double price = black_scholes(100, 100, 1.0, 0.05, 0.2, true);
        REQUIRE(price == Approx(10.45).margin(0.1));
    }

    SECTION("In-the-money call") {
        double price = black_scholes(110, 100, 1.0, 0.05, 0.2, true);
        REQUIRE(price == Approx(17.68).margin(0.1));
    }

    SECTION("Out-of-the-money put") {
        double price = black_scholes(110, 100, 1.0, 0.05, 0.2, false);
        REQUIRE(price == Approx(2.80).margin(0.1));
    }

    SECTION("Zero time to expiry") {
        double call_price = black_scholes(110, 100, 0, 0.05, 0.2, true);
        REQUIRE(call_price == Approx(10.0));

        double put_price = black_scholes(90, 100, 0, 0.05, 0.2, false);
        REQUIRE(put_price == Approx(10.0));
    }
}

TEST_CASE("Greeks calculation", "[math]") {
    Greeks g = greeks(100, 100, 1.0, 0.05, 0.2, true);

    SECTION("Delta") {
        REQUIRE(g.delta == Approx(0.64).margin(0.02));
    }

    SECTION("Gamma") {
        REQUIRE(g.gamma == Approx(0.019).margin(0.002));
    }

    SECTION("Vega") {
        REQUIRE(g.vega == Approx(0.38).margin(0.02));
    }

    SECTION("Theta (daily)") {
        REQUIRE(g.theta < 0);  // Time decay
    }
}

TEST_CASE("Implied volatility", "[math]") {
    double true_vol = 0.25;
    double price = black_scholes(100, 100, 0.5, 0.05, true_vol, true);

    double iv = implied_volatility(price, 100, 100, 0.5, 0.05, true);
    REQUIRE(iv == Approx(true_vol).margin(0.01));
}

TEST_CASE("Constant product AMM", "[math]") {
    SECTION("Basic swap") {
        auto [out, price] = constant_product_price(1000, 1000, 10, 0.003, true);
        REQUIRE(out == Approx(9.88).margin(0.02));
        REQUIRE(price == Approx(0.988).margin(0.002));
    }

    SECTION("Large swap with slippage") {
        auto [out, price] = constant_product_price(1000, 1000, 100, 0.003, true);
        // Larger trade = more slippage
        REQUIRE(out < 100);
        REQUIRE(price < 1.0);
    }

    SECTION("Symmetric reserves") {
        auto [out1, p1] = constant_product_price(1000, 1000, 50, 0.003, true);
        auto [out2, p2] = constant_product_price(1000, 1000, 50, 0.003, false);
        REQUIRE(out1 == Approx(out2).margin(0.01));
    }
}

TEST_CASE("Concentrated liquidity", "[math]") {
    // Price range 90-110, current sqrt price = 10 (price = 100)
    double sqrt_price = 10.0;
    double sqrt_lower = std::sqrt(90.0);
    double sqrt_upper = std::sqrt(110.0);
    double liquidity = 1000.0;

    SECTION("Swap within range") {
        auto [out, new_sqrt_p, impact] = concentrated_liquidity_price(
            liquidity, sqrt_price, sqrt_lower, sqrt_upper, 10, 0.003, true);

        REQUIRE(out > 0);
        REQUIRE(new_sqrt_p > sqrt_price);  // Price increased
        REQUIRE(impact >= 0);
    }
}

TEST_CASE("Volatility calculation", "[math]") {
    std::vector<double> returns = {0.01, -0.02, 0.015, -0.01, 0.02, 0.005};

    SECTION("Non-annualized") {
        double vol = volatility(returns, false);
        REQUIRE(vol > 0);
        REQUIRE(vol < 0.1);  // Daily returns typically small
    }

    SECTION("Annualized") {
        double vol_daily = volatility(returns, false);
        double vol_annual = volatility(returns, true, 252);
        REQUIRE(vol_annual == Approx(vol_daily * std::sqrt(252)).margin(0.001));
    }
}

TEST_CASE("Sharpe ratio", "[math]") {
    std::vector<double> positive_returns = {0.01, 0.02, 0.015, 0.01, 0.02};
    std::vector<double> mixed_returns = {0.01, -0.02, 0.015, -0.01, 0.02};

    SECTION("Positive returns") {
        double sharpe = sharpe_ratio(positive_returns, 0.0);
        REQUIRE(sharpe > 0);
    }

    SECTION("Mixed returns") {
        double sharpe = sharpe_ratio(mixed_returns, 0.0);
        // Lower Sharpe due to more volatility
        REQUIRE(sharpe < sharpe_ratio(positive_returns, 0.0));
    }
}

TEST_CASE("Maximum drawdown", "[math]") {
    std::vector<double> prices = {100, 110, 105, 95, 90, 100, 85};

    auto [dd, peak_idx, trough_idx] = max_drawdown(prices);

    // Max DD from 110 to 85 = 22.7%
    REQUIRE(dd == Approx(0.227).margin(0.01));
    REQUIRE(peak_idx == 1);  // 110
    REQUIRE(trough_idx == 6);  // 85
}

TEST_CASE("Value at Risk", "[math]") {
    std::vector<double> returns = {
        -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
        -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
    };

    SECTION("Historical VaR 95%") {
        double var95 = var(returns, 0.95, false);
        REQUIRE(var95 > 0);
        REQUIRE(var95 < 0.1);
    }

    SECTION("Parametric VaR 95%") {
        double var95 = var(returns, 0.95, true);
        REQUIRE(var95 > 0);
    }
}

TEST_CASE("Conditional VaR", "[math]") {
    std::vector<double> returns = {
        -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02,
        0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12
    };

    double cvar95 = cvar(returns, 0.95);
    double var95 = var(returns, 0.95, false);

    // CVaR should be >= VaR
    REQUIRE(cvar95 >= var95);
}

TEST_CASE("Statistical utilities", "[math]") {
    SECTION("Calculate returns") {
        std::vector<double> prices = {100, 105, 102, 110};
        auto returns = calculate_returns(prices);

        REQUIRE(returns.size() == 3);
        REQUIRE(returns[0] == Approx(0.05));  // 5%
        REQUIRE(returns[1] == Approx(-0.0286).margin(0.001));  // -2.86%
        REQUIRE(returns[2] == Approx(0.0784).margin(0.001));  // 7.84%
    }

    SECTION("Rolling mean") {
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7};
        auto rm = rolling_mean(data, 3);

        REQUIRE(rm.size() == 5);
        REQUIRE(rm[0] == Approx(2.0));  // (1+2+3)/3
        REQUIRE(rm[1] == Approx(3.0));  // (2+3+4)/3
    }

    SECTION("EMA") {
        std::vector<double> data = {10, 12, 11, 13, 12, 14};
        auto e = ema(data, 0.3);

        REQUIRE(e.size() == 6);
        REQUIRE(e[0] == Approx(10.0));
        // Each subsequent value is weighted average
    }

    SECTION("Correlation") {
        std::vector<double> x = {1, 2, 3, 4, 5};
        std::vector<double> y = {2, 4, 6, 8, 10};

        double corr = correlation(x, y);
        REQUIRE(corr == Approx(1.0));  // Perfect positive
    }

    SECTION("Beta") {
        std::vector<double> asset = {0.02, 0.03, -0.01, 0.02, 0.01};
        std::vector<double> market = {0.01, 0.015, -0.005, 0.01, 0.005};

        double b = beta(asset, market);
        REQUIRE(b > 1.0);  // Asset more volatile than market
    }
}

TEST_CASE("Price conversions", "[math]") {
    SECTION("Price to sqrt price") {
        double sqrt_p = price_to_sqrt_price(100);
        REQUIRE(sqrt_p == Approx(10.0));
    }

    SECTION("Sqrt price to price") {
        double p = sqrt_price_to_price(10.0);
        REQUIRE(p == Approx(100.0));
    }

    SECTION("Round trip") {
        double original = 12345.67;
        double sqrt_p = price_to_sqrt_price(original);
        double back = sqrt_price_to_price(sqrt_p);
        REQUIRE(back == Approx(original));
    }
}
