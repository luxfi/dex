// LX Trading SDK - Core Types
// Zero-copy, cache-friendly structures for HFT

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lx::trading {

// Fixed-point decimal for exact financial arithmetic
// Stores value as integer * 10^(-precision)
class Decimal {
public:
    static constexpr int PRECISION = 8;
    static constexpr int64_t SCALE = 100000000LL;

    constexpr Decimal() noexcept : value_(0) {}
    constexpr explicit Decimal(int64_t scaled) noexcept : value_(scaled) {}

    static Decimal from_double(double d) noexcept {
        return Decimal(static_cast<int64_t>(d * SCALE));
    }

    static Decimal from_string(std::string_view s);

    [[nodiscard]] double to_double() const noexcept {
        return static_cast<double>(value_) / SCALE;
    }

    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] int64_t scaled_value() const noexcept { return value_; }

    constexpr Decimal operator+(Decimal rhs) const noexcept {
        return Decimal(value_ + rhs.value_);
    }
    constexpr Decimal operator-(Decimal rhs) const noexcept {
        return Decimal(value_ - rhs.value_);
    }
    constexpr Decimal operator*(Decimal rhs) const noexcept {
        return Decimal((value_ * rhs.value_) / SCALE);
    }
    constexpr Decimal operator/(Decimal rhs) const noexcept {
        return Decimal((value_ * SCALE) / rhs.value_);
    }

    constexpr bool operator==(Decimal rhs) const noexcept { return value_ == rhs.value_; }
    constexpr bool operator!=(Decimal rhs) const noexcept { return value_ != rhs.value_; }
    constexpr bool operator<(Decimal rhs) const noexcept { return value_ < rhs.value_; }
    constexpr bool operator<=(Decimal rhs) const noexcept { return value_ <= rhs.value_; }
    constexpr bool operator>(Decimal rhs) const noexcept { return value_ > rhs.value_; }
    constexpr bool operator>=(Decimal rhs) const noexcept { return value_ >= rhs.value_; }

    constexpr Decimal abs() const noexcept { return Decimal(value_ < 0 ? -value_ : value_); }
    constexpr bool is_zero() const noexcept { return value_ == 0; }
    constexpr bool is_positive() const noexcept { return value_ > 0; }
    constexpr bool is_negative() const noexcept { return value_ < 0; }

    static constexpr Decimal zero() noexcept { return Decimal(0); }
    static constexpr Decimal one() noexcept { return Decimal(SCALE); }

private:
    int64_t value_;
};

// Trading side
enum class Side : uint8_t {
    Buy = 0,
    Sell = 1
};

inline constexpr const char* to_string(Side s) noexcept {
    return s == Side::Buy ? "buy" : "sell";
}

// Order types
enum class OrderType : uint8_t {
    Market = 0,
    Limit = 1,
    LimitMaker = 2,
    StopLoss = 3,
    StopLossLimit = 4,
    TakeProfit = 5,
    TakeProfitLimit = 6
};

inline constexpr const char* to_string(OrderType t) noexcept {
    switch (t) {
        case OrderType::Market: return "market";
        case OrderType::Limit: return "limit";
        case OrderType::LimitMaker: return "limit_maker";
        case OrderType::StopLoss: return "stop_loss";
        case OrderType::StopLossLimit: return "stop_loss_limit";
        case OrderType::TakeProfit: return "take_profit";
        case OrderType::TakeProfitLimit: return "take_profit_limit";
    }
    return "unknown";
}

// Time in force
enum class TimeInForce : uint8_t {
    GTC = 0,  // Good till cancelled
    IOC = 1,  // Immediate or cancel
    FOK = 2,  // Fill or kill
    GTD = 3,  // Good till date
    PostOnly = 4
};

inline constexpr const char* to_string(TimeInForce tif) noexcept {
    switch (tif) {
        case TimeInForce::GTC: return "GTC";
        case TimeInForce::IOC: return "IOC";
        case TimeInForce::FOK: return "FOK";
        case TimeInForce::GTD: return "GTD";
        case TimeInForce::PostOnly: return "POST_ONLY";
    }
    return "unknown";
}

// Order status
enum class OrderStatus : uint8_t {
    Pending = 0,
    Open = 1,
    PartiallyFilled = 2,
    Filled = 3,
    Cancelled = 4,
    Rejected = 5,
    Expired = 6
};

inline constexpr const char* to_string(OrderStatus s) noexcept {
    switch (s) {
        case OrderStatus::Pending: return "pending";
        case OrderStatus::Open: return "open";
        case OrderStatus::PartiallyFilled: return "partially_filled";
        case OrderStatus::Filled: return "filled";
        case OrderStatus::Cancelled: return "cancelled";
        case OrderStatus::Rejected: return "rejected";
        case OrderStatus::Expired: return "expired";
    }
    return "unknown";
}

// Venue type
enum class VenueType : uint8_t {
    Native = 0,
    Ccxt = 1,
    Hummingbot = 2,
    Custom = 3
};

// Trading pair - inline storage for performance
struct TradingPair {
    std::array<char, 16> base{};
    std::array<char, 16> quote{};

    static std::optional<TradingPair> from_symbol(std::string_view symbol);
    std::string to_hummingbot() const;
    std::string to_ccxt() const;
    std::string to_string() const { return to_hummingbot(); }
};

// Fee information
struct Fee {
    std::string asset;
    Decimal amount;
    std::optional<Decimal> rate;
};

// Balance
struct Balance {
    std::string asset;
    std::string venue;
    Decimal free;
    Decimal locked;

    [[nodiscard]] Decimal total() const noexcept { return free + locked; }
};

// Aggregated balance across venues
struct AggregatedBalance {
    std::string asset;
    Decimal total_free;
    Decimal total_locked;
    std::vector<Balance> by_venue;

    [[nodiscard]] Decimal total() const noexcept { return total_free + total_locked; }
};

// Price level in orderbook
struct PriceLevel {
    Decimal price;
    Decimal quantity;

    [[nodiscard]] Decimal value() const noexcept { return price * quantity; }
};

// Order request - builder pattern
class OrderRequest {
public:
    std::string symbol;
    Side side = Side::Buy;
    OrderType order_type = OrderType::Market;
    Decimal quantity;
    std::optional<Decimal> price;
    std::optional<Decimal> stop_price;
    TimeInForce time_in_force = TimeInForce::GTC;
    bool reduce_only = false;
    bool post_only = false;
    std::optional<std::string> venue;
    std::string client_order_id;

    OrderRequest() = default;

    static OrderRequest market(std::string_view symbol, Side side, Decimal quantity);
    static OrderRequest limit(std::string_view symbol, Side side, Decimal quantity, Decimal price);

    OrderRequest& with_venue(std::string_view v) {
        venue = std::string(v);
        return *this;
    }

    OrderRequest& with_post_only() {
        post_only = true;
        time_in_force = TimeInForce::PostOnly;
        return *this;
    }

    OrderRequest& with_client_id(std::string_view id) {
        client_order_id = std::string(id);
        return *this;
    }
};

// Order
struct Order {
    std::string order_id;
    std::string client_order_id;
    std::string symbol;
    std::string venue;
    Side side = Side::Buy;
    OrderType order_type = OrderType::Limit;
    OrderStatus status = OrderStatus::Pending;
    Decimal quantity;
    Decimal filled_quantity;
    Decimal remaining_quantity;
    std::optional<Decimal> price;
    std::optional<Decimal> average_price;
    int64_t created_at = 0;
    int64_t updated_at = 0;
    std::vector<Fee> fees;

    [[nodiscard]] bool is_open() const noexcept {
        return status == OrderStatus::Open ||
               status == OrderStatus::PartiallyFilled ||
               status == OrderStatus::Pending;
    }

    [[nodiscard]] bool is_done() const noexcept {
        return status == OrderStatus::Filled ||
               status == OrderStatus::Cancelled ||
               status == OrderStatus::Rejected ||
               status == OrderStatus::Expired;
    }

    [[nodiscard]] Decimal fill_percent() const noexcept {
        if (quantity.is_zero()) return Decimal::zero();
        return (filled_quantity / quantity) * Decimal::from_double(100.0);
    }
};

// Trade/fill
struct Trade {
    std::string trade_id;
    std::string order_id;
    std::string symbol;
    std::string venue;
    Side side = Side::Buy;
    Decimal price;
    Decimal quantity;
    Fee fee;
    int64_t timestamp = 0;
    bool is_maker = false;

    [[nodiscard]] Decimal value() const noexcept { return price * quantity; }
};

// Ticker
struct Ticker {
    std::string symbol;
    std::string venue;
    std::optional<Decimal> bid;
    std::optional<Decimal> ask;
    std::optional<Decimal> last;
    std::optional<Decimal> volume_24h;
    std::optional<Decimal> high_24h;
    std::optional<Decimal> low_24h;
    std::optional<Decimal> change_24h;
    int64_t timestamp = 0;

    [[nodiscard]] std::optional<Decimal> mid_price() const noexcept {
        if (bid && ask) {
            return (*bid + *ask) / Decimal::from_double(2.0);
        }
        return last;
    }

    [[nodiscard]] std::optional<Decimal> spread() const noexcept {
        if (bid && ask) {
            return *ask - *bid;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<Decimal> spread_percent() const noexcept {
        if (bid && ask && bid->is_positive()) {
            return ((*ask - *bid) / *bid) * Decimal::from_double(100.0);
        }
        return std::nullopt;
    }
};

// AMM swap quote
struct SwapQuote {
    std::string base_token;
    std::string quote_token;
    Decimal input_amount;
    Decimal output_amount;
    Decimal price;
    Decimal price_impact;
    Decimal fee;
    std::vector<std::string> route;
    int64_t expires_at = 0;
};

// Pool information
struct PoolInfo {
    std::string address;
    std::string base_token;
    std::string quote_token;
    Decimal base_reserve;
    Decimal quote_reserve;
    Decimal total_liquidity;
    Decimal fee_rate;
    std::optional<Decimal> apy;
};

// LP position
struct LpPosition {
    std::string pool_address;
    std::string base_token;
    std::string quote_token;
    Decimal lp_tokens;
    Decimal base_amount;
    Decimal quote_amount;
    Decimal share_percent;
    std::optional<Decimal> unrealized_pnl;
};

// Liquidity operation result
struct LiquidityResult {
    std::string tx_hash;
    std::string pool_address;
    Decimal base_amount;
    Decimal quote_amount;
    Decimal lp_tokens;
    Decimal share_percent;
};

// Venue information
struct VenueInfo {
    std::string name;
    VenueType venue_type = VenueType::Native;
    bool connected = false;
    std::optional<int> latency_ms;
    std::vector<std::string> supported_pairs;
    Decimal maker_fee;
    Decimal taker_fee;
};

// Market information
struct MarketInfo {
    std::string symbol;
    std::string base;
    std::string quote;
    int price_precision = 8;
    int quantity_precision = 8;
    Decimal min_quantity;
    std::optional<Decimal> max_quantity;
    std::optional<Decimal> min_notional;
    Decimal tick_size;
    Decimal lot_size;
};

// Timestamp utilities
inline int64_t now_ms() noexcept {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

inline int64_t now_us() noexcept {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

inline int64_t now_ns() noexcept {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

}  // namespace lx::trading
