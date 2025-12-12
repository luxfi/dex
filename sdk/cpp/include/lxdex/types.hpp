// LX C++ SDK - Type Definitions
// Copyright (c) 2025 Lux Partners Limited
// SPDX-License-Identifier: MIT

#ifndef LXDEX_TYPES_HPP
#define LXDEX_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <nlohmann/json.hpp>

namespace lxdex {

/// Order side
enum class Side : int32_t {
    Buy = 0,
    Sell = 1
};

/// Order type
enum class OrderType : int32_t {
    Limit = 0,
    Market = 1,
    Stop = 2,
    StopLimit = 3,
    Iceberg = 4,
    Peg = 5
};

/// Order status
enum class OrderStatus {
    Open,
    Partial,
    Filled,
    Cancelled,
    Rejected
};

/// Time in force
enum class TimeInForce {
    GTC,  // Good Till Cancelled
    IOC,  // Immediate Or Cancel
    FOK,  // Fill Or Kill
    DAY   // Day Order
};

/// Trading order
struct Order {
    uint64_t order_id = 0;
    std::string symbol;
    OrderType type = OrderType::Limit;
    Side side = Side::Buy;
    double price = 0.0;
    double size = 0.0;
    double filled = 0.0;
    double remaining = 0.0;
    OrderStatus status = OrderStatus::Open;
    std::string user_id;
    std::string client_id;
    int64_t timestamp = 0;
    TimeInForce time_in_force = TimeInForce::GTC;
    bool post_only = false;
    bool reduce_only = false;

    /// Check if order is open
    [[nodiscard]] bool is_open() const noexcept {
        return status == OrderStatus::Open || status == OrderStatus::Partial;
    }

    /// Check if order is closed
    [[nodiscard]] bool is_closed() const noexcept {
        return status == OrderStatus::Filled ||
               status == OrderStatus::Cancelled ||
               status == OrderStatus::Rejected;
    }

    /// Get fill rate [0.0, 1.0]
    [[nodiscard]] double fill_rate() const noexcept {
        return (size > 0.0) ? filled / size : 0.0;
    }
};

/// Order response from server
struct OrderResponse {
    uint64_t order_id = 0;
    std::string status;
    std::string message;
};

/// Executed trade
struct Trade {
    uint64_t trade_id = 0;
    std::string symbol;
    double price = 0.0;
    double size = 0.0;
    Side side = Side::Buy;
    uint64_t buy_order_id = 0;
    uint64_t sell_order_id = 0;
    std::string buyer_id;
    std::string seller_id;
    int64_t timestamp = 0;

    /// Get total trade value
    [[nodiscard]] double total_value() const noexcept {
        return price * size;
    }

    /// Get timestamp as time_point
    [[nodiscard]] std::chrono::system_clock::time_point timestamp_time() const noexcept {
        return std::chrono::system_clock::from_time_t(timestamp);
    }
};

/// Price level in orderbook
struct PriceLevel {
    double price = 0.0;
    double size = 0.0;
    int32_t count = 0;
};

/// Orderbook snapshot
struct OrderBook {
    std::string symbol;
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    int64_t timestamp = 0;

    /// Get best bid price (0 if empty)
    [[nodiscard]] double best_bid() const noexcept {
        return bids.empty() ? 0.0 : bids.front().price;
    }

    /// Get best ask price (0 if empty)
    [[nodiscard]] double best_ask() const noexcept {
        return asks.empty() ? 0.0 : asks.front().price;
    }

    /// Get bid-ask spread
    [[nodiscard]] double spread() const noexcept {
        double bid = best_bid();
        double ask = best_ask();
        return (bid > 0.0 && ask > 0.0) ? (ask - bid) : 0.0;
    }

    /// Get mid price
    [[nodiscard]] double mid_price() const noexcept {
        double bid = best_bid();
        double ask = best_ask();
        return (bid > 0.0 && ask > 0.0) ? (bid + ask) / 2.0 : 0.0;
    }

    /// Get spread as percentage of mid price
    [[nodiscard]] double spread_percentage() const noexcept {
        double mid = mid_price();
        return (mid > 0.0) ? (spread() / mid) * 100.0 : 0.0;
    }
};

/// Node/server information
struct NodeInfo {
    std::string version;
    std::string network;
    int64_t order_count = 0;
    int64_t trade_count = 0;
    int64_t timestamp = 0;
    int64_t block_height = 0;
    bool syncing = false;
    int64_t uptime = 0;
};

/// Account balance
struct Balance {
    std::string asset;
    double available = 0.0;
    double locked = 0.0;
    double total = 0.0;

    /// Get utilization ratio [0.0, 1.0]
    [[nodiscard]] double utilization() const noexcept {
        return (total > 0.0) ? locked / total : 0.0;
    }
};

/// Trading position
struct Position {
    std::string symbol;
    double size = 0.0;
    double entry_price = 0.0;
    double mark_price = 0.0;
    double pnl = 0.0;
    double margin = 0.0;

    /// Calculate unrealized PnL
    [[nodiscard]] double unrealized_pnl() const noexcept {
        return (mark_price - entry_price) * size;
    }

    /// Calculate PnL percentage
    [[nodiscard]] double pnl_percentage() const noexcept {
        return (entry_price > 0.0)
            ? ((mark_price - entry_price) / entry_price) * 100.0
            : 0.0;
    }
};

/// WebSocket message
struct Message {
    std::string type;
    nlohmann::json data;
    std::string error;
    std::string request_id;
    int64_t timestamp = 0;
};

// JSON serialization

NLOHMANN_JSON_SERIALIZE_ENUM(Side, {
    {Side::Buy, "buy"},
    {Side::Sell, "sell"}
})

NLOHMANN_JSON_SERIALIZE_ENUM(OrderType, {
    {OrderType::Limit, "limit"},
    {OrderType::Market, "market"},
    {OrderType::Stop, "stop"},
    {OrderType::StopLimit, "stop_limit"},
    {OrderType::Iceberg, "iceberg"},
    {OrderType::Peg, "peg"}
})

NLOHMANN_JSON_SERIALIZE_ENUM(OrderStatus, {
    {OrderStatus::Open, "open"},
    {OrderStatus::Partial, "partial"},
    {OrderStatus::Filled, "filled"},
    {OrderStatus::Cancelled, "cancelled"},
    {OrderStatus::Rejected, "rejected"}
})

NLOHMANN_JSON_SERIALIZE_ENUM(TimeInForce, {
    {TimeInForce::GTC, "GTC"},
    {TimeInForce::IOC, "IOC"},
    {TimeInForce::FOK, "FOK"},
    {TimeInForce::DAY, "DAY"}
})

inline void to_json(nlohmann::json& j, const Order& o) {
    j = nlohmann::json{
        {"orderId", o.order_id},
        {"symbol", o.symbol},
        {"type", o.type},
        {"side", o.side},
        {"price", o.price},
        {"size", o.size},
        {"filled", o.filled},
        {"remaining", o.remaining},
        {"status", o.status},
        {"userId", o.user_id},
        {"clientId", o.client_id},
        {"timestamp", o.timestamp},
        {"timeInForce", o.time_in_force},
        {"postOnly", o.post_only},
        {"reduceOnly", o.reduce_only}
    };
}

inline void from_json(const nlohmann::json& j, Order& o) {
    if (j.contains("orderId")) j.at("orderId").get_to(o.order_id);
    if (j.contains("symbol")) j.at("symbol").get_to(o.symbol);
    if (j.contains("type")) j.at("type").get_to(o.type);
    if (j.contains("side")) j.at("side").get_to(o.side);
    if (j.contains("price")) j.at("price").get_to(o.price);
    if (j.contains("size")) j.at("size").get_to(o.size);
    if (j.contains("filled")) j.at("filled").get_to(o.filled);
    if (j.contains("remaining")) j.at("remaining").get_to(o.remaining);
    if (j.contains("status")) j.at("status").get_to(o.status);
    if (j.contains("userId")) j.at("userId").get_to(o.user_id);
    if (j.contains("clientId")) j.at("clientId").get_to(o.client_id);
    if (j.contains("timestamp")) j.at("timestamp").get_to(o.timestamp);
    if (j.contains("timeInForce")) j.at("timeInForce").get_to(o.time_in_force);
    if (j.contains("postOnly")) j.at("postOnly").get_to(o.post_only);
    if (j.contains("reduceOnly")) j.at("reduceOnly").get_to(o.reduce_only);
}

inline void to_json(nlohmann::json& j, const Trade& t) {
    j = nlohmann::json{
        {"tradeId", t.trade_id},
        {"symbol", t.symbol},
        {"price", t.price},
        {"size", t.size},
        {"side", t.side},
        {"buyOrderId", t.buy_order_id},
        {"sellOrderId", t.sell_order_id},
        {"buyerId", t.buyer_id},
        {"sellerId", t.seller_id},
        {"timestamp", t.timestamp}
    };
}

inline void from_json(const nlohmann::json& j, Trade& t) {
    if (j.contains("tradeId")) j.at("tradeId").get_to(t.trade_id);
    if (j.contains("symbol")) j.at("symbol").get_to(t.symbol);
    if (j.contains("price")) j.at("price").get_to(t.price);
    if (j.contains("size")) j.at("size").get_to(t.size);
    if (j.contains("side")) j.at("side").get_to(t.side);
    if (j.contains("buyOrderId")) j.at("buyOrderId").get_to(t.buy_order_id);
    if (j.contains("sellOrderId")) j.at("sellOrderId").get_to(t.sell_order_id);
    if (j.contains("buyerId")) j.at("buyerId").get_to(t.buyer_id);
    if (j.contains("sellerId")) j.at("sellerId").get_to(t.seller_id);
    if (j.contains("timestamp")) j.at("timestamp").get_to(t.timestamp);
}

inline void to_json(nlohmann::json& j, const PriceLevel& p) {
    j = nlohmann::json{{"price", p.price}, {"size", p.size}, {"count", p.count}};
}

inline void from_json(const nlohmann::json& j, PriceLevel& p) {
    if (j.contains("price")) j.at("price").get_to(p.price);
    if (j.contains("size")) j.at("size").get_to(p.size);
    if (j.contains("count")) j.at("count").get_to(p.count);
}

inline void to_json(nlohmann::json& j, const OrderBook& ob) {
    j = nlohmann::json{
        {"symbol", ob.symbol},
        {"bids", ob.bids},
        {"asks", ob.asks},
        {"timestamp", ob.timestamp}
    };
}

inline void from_json(const nlohmann::json& j, OrderBook& ob) {
    if (j.contains("symbol")) j.at("symbol").get_to(ob.symbol);
    if (j.contains("bids")) j.at("bids").get_to(ob.bids);
    if (j.contains("asks")) j.at("asks").get_to(ob.asks);
    if (j.contains("timestamp")) j.at("timestamp").get_to(ob.timestamp);
}

inline void from_json(const nlohmann::json& j, OrderResponse& r) {
    if (j.contains("orderId")) j.at("orderId").get_to(r.order_id);
    if (j.contains("status")) j.at("status").get_to(r.status);
    if (j.contains("message")) j.at("message").get_to(r.message);
}

inline void from_json(const nlohmann::json& j, Balance& b) {
    if (j.contains("asset")) j.at("asset").get_to(b.asset);
    if (j.contains("available")) j.at("available").get_to(b.available);
    if (j.contains("locked")) j.at("locked").get_to(b.locked);
    if (j.contains("total")) j.at("total").get_to(b.total);
}

inline void from_json(const nlohmann::json& j, Position& p) {
    if (j.contains("symbol")) j.at("symbol").get_to(p.symbol);
    if (j.contains("size")) j.at("size").get_to(p.size);
    if (j.contains("entryPrice")) j.at("entryPrice").get_to(p.entry_price);
    if (j.contains("markPrice")) j.at("markPrice").get_to(p.mark_price);
    if (j.contains("pnl")) j.at("pnl").get_to(p.pnl);
    if (j.contains("margin")) j.at("margin").get_to(p.margin);
}

inline void from_json(const nlohmann::json& j, NodeInfo& n) {
    if (j.contains("version")) j.at("version").get_to(n.version);
    if (j.contains("network")) j.at("network").get_to(n.network);
    if (j.contains("orderCount")) j.at("orderCount").get_to(n.order_count);
    if (j.contains("tradeCount")) j.at("tradeCount").get_to(n.trade_count);
    if (j.contains("timestamp")) j.at("timestamp").get_to(n.timestamp);
    if (j.contains("blockHeight")) j.at("blockHeight").get_to(n.block_height);
    if (j.contains("syncing")) j.at("syncing").get_to(n.syncing);
    if (j.contains("uptime")) j.at("uptime").get_to(n.uptime);
}

} // namespace lxdex

#endif // LXDEX_TYPES_HPP
