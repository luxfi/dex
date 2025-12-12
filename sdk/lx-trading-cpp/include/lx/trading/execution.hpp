// LX Trading SDK - Execution Algorithms
// TWAP, VWAP, Iceberg, Sniper execution strategies

#pragma once

#include <lx/trading/types.hpp>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lx::trading {

// Forward declaration
class Client;

// Execution result
struct ExecutionResult {
    std::vector<Order> orders;
    Decimal total_quantity;
    Decimal total_filled;
    std::optional<Decimal> average_price;
    int64_t start_time;
    int64_t end_time;
    bool completed;
    std::string error;

    [[nodiscard]] Decimal fill_rate() const {
        if (total_quantity.is_zero()) return Decimal::zero();
        return total_filled / total_quantity;
    }

    [[nodiscard]] int64_t duration_ms() const {
        return end_time - start_time;
    }
};

// Progress callback
using ExecutionCallback = std::function<void(const Order&, Decimal remaining)>;

// Base executor interface
class Executor {
public:
    virtual ~Executor() = default;
    virtual std::future<ExecutionResult> execute() = 0;
    virtual void cancel() = 0;
    virtual void set_callback(ExecutionCallback cb) { callback_ = std::move(cb); }

protected:
    ExecutionCallback callback_;
};

// TWAP - Time-Weighted Average Price
class TwapExecutor : public Executor {
public:
    TwapExecutor(
        Client& client,
        std::string symbol,
        Side side,
        Decimal total_quantity,
        std::chrono::seconds duration,
        int num_slices);

    std::future<ExecutionResult> execute() override;
    void cancel() override;

private:
    Client& client_;
    std::string symbol_;
    Side side_;
    Decimal total_quantity_;
    std::chrono::seconds duration_;
    int num_slices_;
    std::atomic<bool> cancelled_{false};
};

// VWAP - Volume-Weighted Average Price
class VwapExecutor : public Executor {
public:
    VwapExecutor(
        Client& client,
        std::string symbol,
        Side side,
        Decimal total_quantity,
        Decimal participation_rate,  // e.g., 0.1 = 10% of volume
        std::chrono::seconds max_duration);

    std::future<ExecutionResult> execute() override;
    void cancel() override;

private:
    Client& client_;
    std::string symbol_;
    Side side_;
    Decimal total_quantity_;
    Decimal participation_rate_;
    std::chrono::seconds max_duration_;
    std::atomic<bool> cancelled_{false};
};

// Iceberg - Hidden large order
class IcebergExecutor : public Executor {
public:
    IcebergExecutor(
        Client& client,
        std::string symbol,
        Side side,
        Decimal total_quantity,
        Decimal visible_quantity,
        Decimal price,
        std::optional<std::string> venue = std::nullopt);

    std::future<ExecutionResult> execute() override;
    void cancel() override;

private:
    Client& client_;
    std::string symbol_;
    Side side_;
    Decimal total_quantity_;
    Decimal visible_quantity_;
    Decimal price_;
    std::optional<std::string> venue_;
    std::atomic<bool> cancelled_{false};
};

// Sniper - Wait for price target then execute
class SniperExecutor : public Executor {
public:
    SniperExecutor(
        Client& client,
        std::string symbol,
        Side side,
        Decimal quantity,
        Decimal target_price,
        std::chrono::seconds timeout);

    std::future<ExecutionResult> execute() override;
    void cancel() override;

private:
    Client& client_;
    std::string symbol_;
    Side side_;
    Decimal quantity_;
    Decimal target_price_;
    std::chrono::seconds timeout_;
    std::atomic<bool> cancelled_{false};
};

// POV - Percentage of Volume
class PovExecutor : public Executor {
public:
    PovExecutor(
        Client& client,
        std::string symbol,
        Side side,
        Decimal total_quantity,
        Decimal target_participation,  // e.g., 0.15 = 15% of volume
        std::chrono::seconds max_duration,
        std::optional<Decimal> price_limit = std::nullopt);

    std::future<ExecutionResult> execute() override;
    void cancel() override;

private:
    Client& client_;
    std::string symbol_;
    Side side_;
    Decimal total_quantity_;
    Decimal target_participation_;
    std::chrono::seconds max_duration_;
    std::optional<Decimal> price_limit_;
    std::atomic<bool> cancelled_{false};
};

// Factory functions for convenience
inline std::unique_ptr<TwapExecutor> make_twap(
    Client& client,
    std::string symbol,
    Side side,
    Decimal quantity,
    std::chrono::seconds duration,
    int slices = 10) {
    return std::make_unique<TwapExecutor>(
        client, std::move(symbol), side, quantity, duration, slices);
}

inline std::unique_ptr<VwapExecutor> make_vwap(
    Client& client,
    std::string symbol,
    Side side,
    Decimal quantity,
    Decimal participation = Decimal::from_double(0.1),
    std::chrono::seconds max_duration = std::chrono::seconds(3600)) {
    return std::make_unique<VwapExecutor>(
        client, std::move(symbol), side, quantity, participation, max_duration);
}

inline std::unique_ptr<IcebergExecutor> make_iceberg(
    Client& client,
    std::string symbol,
    Side side,
    Decimal total_quantity,
    Decimal visible_quantity,
    Decimal price) {
    return std::make_unique<IcebergExecutor>(
        client, std::move(symbol), side, total_quantity, visible_quantity, price);
}

inline std::unique_ptr<SniperExecutor> make_sniper(
    Client& client,
    std::string symbol,
    Side side,
    Decimal quantity,
    Decimal target_price,
    std::chrono::seconds timeout = std::chrono::seconds(60)) {
    return std::make_unique<SniperExecutor>(
        client, std::move(symbol), side, quantity, target_price, timeout);
}

}  // namespace lx::trading
