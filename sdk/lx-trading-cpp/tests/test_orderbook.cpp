// LX Trading SDK - Orderbook Tests

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <lx/trading/orderbook.hpp>

using namespace lx::trading;
using Catch::Approx;

TEST_CASE("Orderbook basic operations", "[orderbook]") {
    Orderbook book("BTC-USDC", "test_venue");

    SECTION("Add and retrieve levels") {
        book.add_bid(Decimal::from_double(100.0), Decimal::from_double(1.0));
        book.add_bid(Decimal::from_double(99.0), Decimal::from_double(2.0));
        book.add_ask(Decimal::from_double(101.0), Decimal::from_double(1.5));
        book.add_ask(Decimal::from_double(102.0), Decimal::from_double(2.5));
        book.sort();

        auto bids = book.bids();
        auto asks = book.asks();

        REQUIRE(bids.size() == 2);
        REQUIRE(asks.size() == 2);

        // Bids sorted descending
        REQUIRE(bids[0].price.to_double() == Approx(100.0));
        REQUIRE(bids[1].price.to_double() == Approx(99.0));

        // Asks sorted ascending
        REQUIRE(asks[0].price.to_double() == Approx(101.0));
        REQUIRE(asks[1].price.to_double() == Approx(102.0));
    }

    SECTION("Best bid/ask") {
        book.add_bid(Decimal::from_double(100.0), Decimal::from_double(1.0));
        book.add_ask(Decimal::from_double(101.0), Decimal::from_double(1.0));
        book.sort();

        REQUIRE(book.best_bid().value().to_double() == Approx(100.0));
        REQUIRE(book.best_ask().value().to_double() == Approx(101.0));
    }

    SECTION("Mid price and spread") {
        book.add_bid(Decimal::from_double(100.0), Decimal::from_double(1.0));
        book.add_ask(Decimal::from_double(102.0), Decimal::from_double(1.0));
        book.sort();

        REQUIRE(book.mid_price().value().to_double() == Approx(101.0));
        REQUIRE(book.spread().value().to_double() == Approx(2.0));
        REQUIRE(book.spread_percent().value().to_double() == Approx(1.98).margin(0.01));
    }
}

TEST_CASE("Orderbook VWAP", "[orderbook]") {
    Orderbook book("BTC-USDC", "test_venue");

    book.add_ask(Decimal::from_double(100.0), Decimal::from_double(1.0));
    book.add_ask(Decimal::from_double(101.0), Decimal::from_double(2.0));
    book.add_ask(Decimal::from_double(102.0), Decimal::from_double(3.0));
    book.sort();

    SECTION("VWAP for small amount") {
        auto vwap = book.vwap_buy(Decimal::from_double(0.5));
        REQUIRE(vwap.has_value());
        REQUIRE(vwap->to_double() == Approx(100.0));
    }

    SECTION("VWAP across multiple levels") {
        auto vwap = book.vwap_buy(Decimal::from_double(2.5));
        REQUIRE(vwap.has_value());
        // (1.0 * 100 + 1.5 * 101) / 2.5 = 100.6
        REQUIRE(vwap->to_double() == Approx(100.6));
    }

    SECTION("VWAP for full book") {
        auto vwap = book.vwap_buy(Decimal::from_double(6.0));
        REQUIRE(vwap.has_value());
        // (1*100 + 2*101 + 3*102) / 6 = 101.333...
        REQUIRE(vwap->to_double() == Approx(101.333).margin(0.01));
    }
}

TEST_CASE("Orderbook liquidity", "[orderbook]") {
    Orderbook book("BTC-USDC", "test_venue");

    book.add_bid(Decimal::from_double(100.0), Decimal::from_double(1.0));
    book.add_bid(Decimal::from_double(99.0), Decimal::from_double(2.0));
    book.add_ask(Decimal::from_double(101.0), Decimal::from_double(1.5));
    book.add_ask(Decimal::from_double(102.0), Decimal::from_double(2.5));
    book.sort();

    SECTION("Bid liquidity") {
        // 1.0 * 100 + 2.0 * 99 = 298
        REQUIRE(book.bid_liquidity().to_double() == Approx(298.0));
    }

    SECTION("Ask liquidity") {
        // 1.5 * 101 + 2.5 * 102 = 406.5
        REQUIRE(book.ask_liquidity().to_double() == Approx(406.5));
    }

    SECTION("Depth for specific levels") {
        REQUIRE(book.bid_depth(1).to_double() == Approx(100.0));
        REQUIRE(book.ask_depth(1).to_double() == Approx(151.5));
    }

    SECTION("Has liquidity") {
        REQUIRE(book.has_liquidity(Side::Buy, Decimal::from_double(3.0)));
        REQUIRE_FALSE(book.has_liquidity(Side::Buy, Decimal::from_double(10.0)));
    }
}

TEST_CASE("AggregatedOrderbook", "[orderbook]") {
    AggregatedOrderbook agg("BTC-USDC");

    // Create first venue orderbook
    Orderbook book1("BTC-USDC", "venue1");
    book1.add_bid(Decimal::from_double(100.0), Decimal::from_double(1.0));
    book1.add_ask(Decimal::from_double(102.0), Decimal::from_double(1.0));
    book1.sort();

    // Create second venue orderbook
    Orderbook book2("BTC-USDC", "venue2");
    book2.add_bid(Decimal::from_double(99.0), Decimal::from_double(2.0));
    book2.add_ask(Decimal::from_double(101.0), Decimal::from_double(1.5));
    book2.sort();

    agg.add_orderbook(book1);
    agg.add_orderbook(book2);

    SECTION("Best bid across venues") {
        auto best = agg.best_bid();
        REQUIRE(best.has_value());
        auto [price, venue, qty] = *best;
        REQUIRE(price.to_double() == Approx(100.0));
        REQUIRE(venue == "venue1");
    }

    SECTION("Best ask across venues") {
        auto best = agg.best_ask();
        REQUIRE(best.has_value());
        auto [price, venue, qty] = *best;
        REQUIRE(price.to_double() == Approx(101.0));
        REQUIRE(venue == "venue2");
    }

    SECTION("Aggregated levels") {
        auto asks = agg.aggregated_asks();
        REQUIRE(asks.size() == 2);
        // Sorted by price ascending
        REQUIRE(asks[0].price.to_double() == Approx(101.0));
        REQUIRE(asks[1].price.to_double() == Approx(102.0));
    }

    SECTION("Best venue for buying") {
        auto best = agg.best_venue_buy(Decimal::from_double(1.0));
        REQUIRE(best.has_value());
        auto [venue, price] = *best;
        REQUIRE(venue == "venue2");  // Lower ask price
        REQUIRE(price.to_double() == Approx(101.0));
    }

    SECTION("Best venue for selling") {
        auto best = agg.best_venue_sell(Decimal::from_double(0.5));
        REQUIRE(best.has_value());
        auto [venue, price] = *best;
        REQUIRE(venue == "venue1");  // Higher bid price
        REQUIRE(price.to_double() == Approx(100.0));
    }
}
