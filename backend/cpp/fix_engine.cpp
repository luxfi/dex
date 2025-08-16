// High-performance FIX engine with lock-free architecture
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstring>

class FixEngine {
private:
    static constexpr size_t SHARDS = 16;
    
    struct Shard {
        std::atomic<uint64_t> orders{0};
        std::atomic<uint64_t> trades{0};
        char padding[64 - 2*sizeof(std::atomic<uint64_t>)]; // Cache line padding
    };
    
    alignas(64) Shard shards[SHARDS];
    std::vector<std::thread> workers;
    std::atomic<bool> running{true};
    
public:
    void start(int num_workers) {
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back([this, i]() { worker_thread(i); });
        }
    }
    
    void stop() {
        running = false;
        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }
    }
    
    void worker_thread(int id) {
        uint64_t local_orders = 0;
        uint64_t local_trades = 0;
        size_t shard_id = id % SHARDS;
        
        while (running.load(std::memory_order_relaxed)) {
            // Simulate order processing
            for (int i = 0; i < 1000; ++i) {
                local_orders++;
                if (local_orders % 2 == 0) {
                    local_trades++;
                }
            }
            
            // Update shard atomically
            shards[shard_id].orders.fetch_add(local_orders, std::memory_order_relaxed);
            shards[shard_id].trades.fetch_add(local_trades, std::memory_order_relaxed);
            local_orders = 0;
            local_trades = 0;
        }
    }
    
    std::pair<uint64_t, uint64_t> get_stats() const {
        uint64_t total_orders = 0, total_trades = 0;
        for (const auto& shard : shards) {
            total_orders += shard.orders.load(std::memory_order_relaxed);
            total_trades += shard.trades.load(std::memory_order_relaxed);
        }
        return {total_orders, total_trades};
    }
};

// C interface for CGO
extern "C" {
    void* fix_engine_create() {
        return new FixEngine();
    }
    
    void fix_engine_destroy(void* engine) {
        delete static_cast<FixEngine*>(engine);
    }
    
    void fix_engine_start(void* engine, int workers) {
        static_cast<FixEngine*>(engine)->start(workers);
    }
    
    void fix_engine_stop(void* engine) {
        static_cast<FixEngine*>(engine)->stop();
    }
    
    void fix_engine_get_stats(void* engine, uint64_t* orders, uint64_t* trades) {
        auto stats = static_cast<FixEngine*>(engine)->get_stats();
        *orders = stats.first;
        *trades = stats.second;
    }
}