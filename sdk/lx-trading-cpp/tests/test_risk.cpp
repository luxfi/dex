// LX Trading SDK - Risk Manager Tests

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <lx/trading/risk.hpp>

using namespace lx::trading;
using Catch::Approx;

TEST_CASE("RiskManager basic validation", "[risk]") {
    RiskConfig config;
    config.enabled = true;
    config.max_order_size = Decimal::from_double(100.0);
    config.max_position_size = Decimal::from_double(1000.0);
    config.max_open_orders = 10;

    RiskManager rm(config);

    SECTION("Valid order passes") {
        OrderRequest req = OrderRequest::market("BTC-USDC", Side::Buy,
            Decimal::from_double(50.0));

        REQUIRE_NOTHROW(rm.validate_order(req));
    }

    SECTION("Order size exceeded") {
        OrderRequest req = OrderRequest::market("BTC-USDC", Side::Buy,
            Decimal::from_double(150.0));

        REQUIRE_THROWS_AS(rm.validate_order(req), RiskError);
    }
}

TEST_CASE("RiskManager position limits", "[risk]") {
    RiskConfig config;
    config.enabled = true;
    config.max_position_size = Decimal::from_double(100.0);
    config.position_limits["BTC"] = Decimal::from_double(50.0);

    RiskManager rm(config);

    SECTION("Within global limit") {
        rm.update_position("BTC", Decimal::from_double(30.0), Side::Buy);
        REQUIRE(rm.position("BTC").to_double() == Approx(30.0));

        OrderRequest req = OrderRequest::market("BTC-USDC", Side::Buy,
            Decimal::from_double(10.0));
        REQUIRE_NOTHROW(rm.validate_order(req));
    }

    SECTION("Exceeds asset-specific limit") {
        rm.update_position("BTC", Decimal::from_double(45.0), Side::Buy);

        OrderRequest req = OrderRequest::market("BTC-USDC", Side::Buy,
            Decimal::from_double(10.0));
        REQUIRE_THROWS_AS(rm.validate_order(req), RiskError);
    }

    SECTION("Exceeds global limit") {
        rm.update_position("ETH", Decimal::from_double(90.0), Side::Buy);

        OrderRequest req = OrderRequest::market("ETH-USDC", Side::Buy,
            Decimal::from_double(20.0));
        REQUIRE_THROWS_AS(rm.validate_order(req), RiskError);
    }
}

TEST_CASE("RiskManager PnL tracking", "[risk]") {
    RiskConfig config;
    config.enabled = true;
    config.max_daily_loss = Decimal::from_double(1000.0);
    config.kill_switch_enabled = true;

    RiskManager rm(config);

    SECTION("Track PnL") {
        rm.update_pnl(Decimal::from_double(100.0));
        rm.update_pnl(Decimal::from_double(-50.0));
        REQUIRE(rm.daily_pnl().to_double() == Approx(50.0));
    }

    SECTION("Daily loss limit triggers rejection") {
        rm.update_pnl(Decimal::from_double(-1001.0));

        OrderRequest req = OrderRequest::market("BTC-USDC", Side::Buy,
            Decimal::from_double(1.0));
        REQUIRE_THROWS_AS(rm.validate_order(req), RiskError);
    }

    SECTION("Kill switch activation") {
        REQUIRE_FALSE(rm.is_killed());

        rm.update_pnl(Decimal::from_double(-1001.0));
        REQUIRE(rm.is_killed());
    }

    SECTION("Reset PnL") {
        rm.update_pnl(Decimal::from_double(-500.0));
        rm.reset_daily_pnl();
        REQUIRE(rm.daily_pnl().is_zero());
    }
}

TEST_CASE("RiskManager order tracking", "[risk]") {
    RiskConfig config;
    config.enabled = true;
    config.max_open_orders = 5;

    RiskManager rm(config);

    SECTION("Track open orders") {
        rm.order_opened("BTC-USDC");
        rm.order_opened("BTC-USDC");
        REQUIRE(rm.open_orders("BTC-USDC") == 2);

        rm.order_closed("BTC-USDC");
        REQUIRE(rm.open_orders("BTC-USDC") == 1);
    }

    SECTION("Max open orders limit") {
        for (int i = 0; i < 5; ++i) {
            rm.order_opened("ETH-USDC");
        }

        OrderRequest req = OrderRequest::market("ETH-USDC", Side::Buy,
            Decimal::from_double(1.0));
        REQUIRE_THROWS_AS(rm.validate_order(req), RiskError);
    }
}

TEST_CASE("RiskManager kill switch", "[risk]") {
    RiskConfig config;
    config.enabled = true;

    RiskManager rm(config);

    SECTION("Manual kill") {
        REQUIRE_FALSE(rm.is_killed());

        rm.kill();
        REQUIRE(rm.is_killed());

        OrderRequest req = OrderRequest::market("BTC-USDC", Side::Buy,
            Decimal::from_double(1.0));
        REQUIRE_THROWS_AS(rm.validate_order(req), RiskError);
    }

    SECTION("Reset kill switch") {
        rm.kill();
        rm.reset();
        REQUIRE_FALSE(rm.is_killed());
    }
}

TEST_CASE("RiskManager disabled", "[risk]") {
    RiskConfig config;
    config.enabled = false;
    config.max_order_size = Decimal::from_double(10.0);

    RiskManager rm(config);

    SECTION("Validation passes when disabled") {
        OrderRequest req = OrderRequest::market("BTC-USDC", Side::Buy,
            Decimal::from_double(1000.0));  // Way over limit

        REQUIRE_NOTHROW(rm.validate_order(req));
    }
}

TEST_CASE("RiskManager pre-trade checks", "[risk]") {
    RiskConfig config;
    config.enabled = true;
    config.max_order_size = Decimal::from_double(100.0);
    config.max_position_size = Decimal::from_double(500.0);
    config.max_daily_loss = Decimal::from_double(1000.0);
    config.max_open_orders = 5;

    RiskManager rm(config);

    SECTION("Check order size") {
        REQUIRE(rm.check_order_size(Decimal::from_double(50.0)));
        REQUIRE_FALSE(rm.check_order_size(Decimal::from_double(150.0)));
    }

    SECTION("Check position limit") {
        REQUIRE(rm.check_position_limit("BTC", Decimal::from_double(100.0)));
        REQUIRE_FALSE(rm.check_position_limit("BTC", Decimal::from_double(600.0)));
    }

    SECTION("Check daily loss") {
        REQUIRE(rm.check_daily_loss());
        rm.update_pnl(Decimal::from_double(-1001.0));
        REQUIRE_FALSE(rm.check_daily_loss());
    }

    SECTION("Check open orders") {
        REQUIRE(rm.check_open_orders("BTC-USDC"));
        for (int i = 0; i < 5; ++i) {
            rm.order_opened("BTC-USDC");
        }
        REQUIRE_FALSE(rm.check_open_orders("BTC-USDC"));
    }
}

TEST_CASE("OrderTracker RAII", "[risk]") {
    RiskConfig config;
    config.enabled = true;
    config.max_open_orders = 10;

    RiskManager rm(config);

    SECTION("Auto decrement on destruction") {
        {
            OrderTracker tracker(rm, "BTC-USDC");
            REQUIRE(rm.open_orders("BTC-USDC") == 1);
        }
        REQUIRE(rm.open_orders("BTC-USDC") == 0);
    }

    SECTION("Release prevents decrement") {
        {
            OrderTracker tracker(rm, "BTC-USDC");
            tracker.release();
        }
        REQUIRE(rm.open_orders("BTC-USDC") == 1);
    }
}
