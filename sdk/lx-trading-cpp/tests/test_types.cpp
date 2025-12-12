// LX Trading SDK - Types Tests

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <lx/trading/types.hpp>

using namespace lx::trading;
using Catch::Approx;

TEST_CASE("Decimal arithmetic", "[types]") {
    SECTION("Basic operations") {
        Decimal a = Decimal::from_double(100.5);
        Decimal b = Decimal::from_double(50.25);

        REQUIRE((a + b).to_double() == Approx(150.75));
        REQUIRE((a - b).to_double() == Approx(50.25));
        REQUIRE((a * Decimal::from_double(2.0)).to_double() == Approx(201.0));
        REQUIRE((a / Decimal::from_double(2.0)).to_double() == Approx(50.25));
    }

    SECTION("String conversion") {
        auto d = Decimal::from_string("123.456");
        REQUIRE(d.to_double() == Approx(123.456));

        auto d2 = Decimal::from_string("-99.99");
        REQUIRE(d2.to_double() == Approx(-99.99));
        REQUIRE(d2.is_negative());
    }

    SECTION("Comparison") {
        Decimal a = Decimal::from_double(10.0);
        Decimal b = Decimal::from_double(20.0);

        REQUIRE(a < b);
        REQUIRE(b > a);
        REQUIRE(a <= a);
        REQUIRE(a == a);
        REQUIRE(a != b);
    }

    SECTION("Zero and one") {
        REQUIRE(Decimal::zero().is_zero());
        REQUIRE(Decimal::one().to_double() == Approx(1.0));
    }
}

TEST_CASE("TradingPair parsing", "[types]") {
    SECTION("Hyphen separator") {
        auto pair = TradingPair::from_symbol("BTC-USDC");
        REQUIRE(pair.has_value());
        REQUIRE(std::string(pair->base.data()) == "BTC");
        REQUIRE(std::string(pair->quote.data()) == "USDC");
    }

    SECTION("Slash separator") {
        auto pair = TradingPair::from_symbol("ETH/USD");
        REQUIRE(pair.has_value());
        REQUIRE(std::string(pair->base.data()) == "ETH");
        REQUIRE(std::string(pair->quote.data()) == "USD");
    }

    SECTION("Underscore separator") {
        auto pair = TradingPair::from_symbol("LUX_USDT");
        REQUIRE(pair.has_value());
        REQUIRE(std::string(pair->base.data()) == "LUX");
        REQUIRE(std::string(pair->quote.data()) == "USDT");
    }

    SECTION("Invalid symbol") {
        auto pair = TradingPair::from_symbol("INVALID");
        REQUIRE_FALSE(pair.has_value());
    }

    SECTION("Format conversions") {
        auto pair = TradingPair::from_symbol("BTC-USDC");
        REQUIRE(pair->to_hummingbot() == "BTC-USDC");
        REQUIRE(pair->to_ccxt() == "BTC/USDC");
    }
}

TEST_CASE("OrderRequest factory methods", "[types]") {
    SECTION("Market order") {
        auto req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_double(1.5));
        REQUIRE(req.symbol == "BTC-USDC");
        REQUIRE(req.side == Side::Buy);
        REQUIRE(req.order_type == OrderType::Market);
        REQUIRE(req.quantity.to_double() == Approx(1.5));
        REQUIRE(req.time_in_force == TimeInForce::IOC);
    }

    SECTION("Limit order") {
        auto req = OrderRequest::limit("ETH-USDC", Side::Sell,
            Decimal::from_double(10.0), Decimal::from_double(2000.0));
        REQUIRE(req.symbol == "ETH-USDC");
        REQUIRE(req.side == Side::Sell);
        REQUIRE(req.order_type == OrderType::Limit);
        REQUIRE(req.price.value().to_double() == Approx(2000.0));
        REQUIRE(req.time_in_force == TimeInForce::GTC);
    }

    SECTION("Builder methods") {
        auto req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_double(1.0))
            .with_venue("lx_dex")
            .with_post_only()
            .with_client_id("my-order-123");

        REQUIRE(req.venue.value() == "lx_dex");
        REQUIRE(req.post_only == true);
        REQUIRE(req.time_in_force == TimeInForce::PostOnly);
        REQUIRE(req.client_order_id == "my-order-123");
    }
}

TEST_CASE("Order status checks", "[types]") {
    Order order;
    order.quantity = Decimal::from_double(100.0);
    order.filled_quantity = Decimal::from_double(50.0);
    order.remaining_quantity = Decimal::from_double(50.0);

    SECTION("Partially filled") {
        order.status = OrderStatus::PartiallyFilled;
        REQUIRE(order.is_open());
        REQUIRE_FALSE(order.is_done());
        REQUIRE(order.fill_percent().to_double() == Approx(50.0));
    }

    SECTION("Filled") {
        order.status = OrderStatus::Filled;
        order.filled_quantity = order.quantity;
        REQUIRE_FALSE(order.is_open());
        REQUIRE(order.is_done());
    }

    SECTION("Cancelled") {
        order.status = OrderStatus::Cancelled;
        REQUIRE_FALSE(order.is_open());
        REQUIRE(order.is_done());
    }
}

TEST_CASE("Ticker calculations", "[types]") {
    Ticker ticker;
    ticker.bid = Decimal::from_double(100.0);
    ticker.ask = Decimal::from_double(101.0);

    SECTION("Mid price") {
        auto mid = ticker.mid_price();
        REQUIRE(mid.has_value());
        REQUIRE(mid->to_double() == Approx(100.5));
    }

    SECTION("Spread") {
        auto spread = ticker.spread();
        REQUIRE(spread.has_value());
        REQUIRE(spread->to_double() == Approx(1.0));
    }

    SECTION("Spread percent") {
        auto spread_pct = ticker.spread_percent();
        REQUIRE(spread_pct.has_value());
        REQUIRE(spread_pct->to_double() == Approx(1.0));
    }
}

TEST_CASE("Enum to_string", "[types]") {
    REQUIRE(std::string(to_string(Side::Buy)) == "buy");
    REQUIRE(std::string(to_string(Side::Sell)) == "sell");
    REQUIRE(std::string(to_string(OrderType::Market)) == "market");
    REQUIRE(std::string(to_string(OrderType::Limit)) == "limit");
    REQUIRE(std::string(to_string(TimeInForce::GTC)) == "GTC");
    REQUIRE(std::string(to_string(OrderStatus::Filled)) == "filled");
}
