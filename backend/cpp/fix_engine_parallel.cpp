// Thread-safe, parallel, high-performance FIX engine
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstring>
#include <mutex>
#include <immintrin.h>  // For SIMD optimizations
#include <sched.h>      // For CPU affinity

// Cache line size for padding
constexpr size_t CACHE_LINE_SIZE = 64;

// Align to cache line to prevent false sharing
template<typename T>
struct alignas(CACHE_LINE_SIZE) CacheAligned {
    T value;
    char padding[CACHE_LINE_SIZE - sizeof(T)];
};

class ParallelFixEngine {
private:
    static constexpr size_t NUM_SHARDS = 32;  // More shards for better parallelism
    static constexpr size_t ORDERS_PER_BATCH = 1024;
    
    // Per-shard data structure with proper alignment
    struct alignas(CACHE_LINE_SIZE) Shard {
        // Separate cache lines for each counter
        CacheAligned<std::atomic<uint64_t>> orders{0};
        CacheAligned<std::atomic<uint64_t>> trades{0};
        CacheAligned<std::atomic<uint64_t>> volume{0};
        CacheAligned<std::atomic<uint64_t>> bid_count{0};
        CacheAligned<std::atomic<uint64_t>> ask_count{0};
        
        // Spinlock for this shard (rarely needed)
        alignas(CACHE_LINE_SIZE) std::atomic_flag lock = ATOMIC_FLAG_INIT;
        
        // Lock-free order pool
        struct Order {
            uint64_t id;
            uint64_t price;  // Fixed point
            uint64_t quantity;
            uint8_t side;    // 0=buy, 1=sell
            uint8_t type;    // 0=market, 1=limit
        };
        
        alignas(CACHE_LINE_SIZE) Order order_pool[1024];
        CacheAligned<std::atomic<uint32_t>> pool_head{0};
        CacheAligned<std::atomic<uint32_t>> pool_tail{0};
        
        void acquire_lock() {
            while (lock.test_and_set(std::memory_order_acquire)) {
                _mm_pause(); // CPU yield hint
            }
        }
        
        void release_lock() {
            lock.clear(std::memory_order_release);
        }
    };
    
    // Sharded data structure
    alignas(CACHE_LINE_SIZE) Shard shards[NUM_SHARDS];
    
    // Thread management
    std::vector<std::thread> worker_threads;
    std::atomic<bool> running{true};
    std::atomic<uint64_t> total_orders{0};
    std::atomic<uint64_t> total_trades{0};
    
    // Performance monitoring
    struct alignas(CACHE_LINE_SIZE) ThreadStats {
        uint64_t orders_processed{0};
        uint64_t trades_executed{0};
        uint64_t latency_sum{0};
        uint64_t latency_count{0};
        char padding[CACHE_LINE_SIZE - 4*sizeof(uint64_t)];
    };
    
    std::vector<ThreadStats> thread_stats;
    
public:
    ParallelFixEngine() : thread_stats(std::thread::hardware_concurrency()) {}
    
    // Start worker threads with CPU affinity
    void start() {
        size_t num_cores = std::thread::hardware_concurrency();
        worker_threads.reserve(num_cores);
        
        for (size_t i = 0; i < num_cores; ++i) {
            worker_threads.emplace_back([this, i]() {
                // Set CPU affinity for better cache locality
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                
                worker_thread(i);
            });
        }
    }
    
    void stop() {
        running.store(false, std::memory_order_relaxed);
        for (auto& t : worker_threads) {
            if (t.joinable()) t.join();
        }
    }
    
    // Process FIX message with minimal contention
    uint64_t process_fix_message(const char* fix_msg, size_t len) {
        static std::atomic<uint64_t> order_id_gen{1};
        uint64_t order_id = order_id_gen.fetch_add(1, std::memory_order_relaxed);
        
        // Hash order ID to shard for load balancing
        size_t shard_idx = (order_id * 0x9E3779B97F4A7C15ULL) % NUM_SHARDS;
        auto& shard = shards[shard_idx];
        
        // Parse FIX message (simplified)
        uint64_t price = parse_price(fix_msg);
        uint64_t quantity = parse_quantity(fix_msg);
        uint8_t side = parse_side(fix_msg);
        
        // Lock-free order insertion
        uint32_t tail = shard.pool_tail.value.load(std::memory_order_acquire);
        uint32_t next_tail = (tail + 1) % 1024;
        
        // Try to reserve slot
        if (shard.pool_tail.value.compare_exchange_strong(tail, next_tail, 
                                                          std::memory_order_release)) {
            auto& order = shard.order_pool[tail];
            order.id = order_id;
            order.price = price;
            order.quantity = quantity;
            order.side = side;
            
            // Update counters
            shard.orders.value.fetch_add(1, std::memory_order_relaxed);
            if (side == 0) {
                shard.bid_count.value.fetch_add(1, std::memory_order_relaxed);
            } else {
                shard.ask_count.value.fetch_add(1, std::memory_order_relaxed);
            }
        }
        
        return order_id;
    }
    
    // Worker thread for matching
    void worker_thread(size_t thread_id) {
        auto& stats = thread_stats[thread_id];
        
        // Each thread handles specific shards
        size_t shards_per_thread = NUM_SHARDS / worker_threads.size();
        size_t start_shard = thread_id * shards_per_thread;
        size_t end_shard = std::min(start_shard + shards_per_thread, NUM_SHARDS);
        
        while (running.load(std::memory_order_relaxed)) {
            for (size_t s = start_shard; s < end_shard; ++s) {
                process_shard(shards[s], stats);
            }
            
            // Yield to prevent spinning
            std::this_thread::yield();
        }
    }
    
    // Process orders in a shard
    void process_shard(Shard& shard, ThreadStats& stats) {
        uint32_t head = shard.pool_head.value.load(std::memory_order_acquire);
        uint32_t tail = shard.pool_tail.value.load(std::memory_order_acquire);
        
        // Process pending orders
        while (head != tail) {
            auto& order = shard.order_pool[head];
            
            // Simulate matching logic
            auto start = std::chrono::high_resolution_clock::now();
            
            // Match order (simplified)
            if (try_match_order(order)) {
                shard.trades.value.fetch_add(1, std::memory_order_relaxed);
                stats.trades_executed++;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            
            stats.orders_processed++;
            stats.latency_sum += latency;
            stats.latency_count++;
            
            head = (head + 1) % 1024;
            shard.pool_head.value.store(head, std::memory_order_release);
        }
    }
    
    // Simplified matching logic
    bool try_match_order(const Shard::Order& order) {
        // Simulate matching with 50% probability
        return (order.id % 2) == 0;
    }
    
    // Helper functions for FIX parsing
    uint64_t parse_price(const char* msg) {
        // Convert to fixed point (7 decimal places)
        return 50000'0000000ULL; // $50,000.00
    }
    
    uint64_t parse_quantity(const char* msg) {
        return 1'0000000ULL; // 1.0
    }
    
    uint8_t parse_side(const char* msg) {
        return 0; // Buy
    }
    
    // Get statistics
    void get_stats(uint64_t& orders, uint64_t& trades, double& avg_latency_ns) {
        orders = 0;
        trades = 0;
        uint64_t total_latency = 0;
        uint64_t total_count = 0;
        
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            orders += shards[i].orders.value.load(std::memory_order_relaxed);
            trades += shards[i].trades.value.load(std::memory_order_relaxed);
        }
        
        for (const auto& stats : thread_stats) {
            total_latency += stats.latency_sum;
            total_count += stats.latency_count;
        }
        
        avg_latency_ns = total_count > 0 ? 
                        static_cast<double>(total_latency) / total_count : 0;
    }
};

// C interface for CGO integration
extern "C" {
    void* create_parallel_engine() {
        return new ParallelFixEngine();
    }
    
    void destroy_parallel_engine(void* engine) {
        delete static_cast<ParallelFixEngine*>(engine);
    }
    
    void start_parallel_engine(void* engine) {
        static_cast<ParallelFixEngine*>(engine)->start();
    }
    
    void stop_parallel_engine(void* engine) {
        static_cast<ParallelFixEngine*>(engine)->stop();
    }
    
    uint64_t process_fix_parallel(void* engine, const char* msg, size_t len) {
        return static_cast<ParallelFixEngine*>(engine)->process_fix_message(msg, len);
    }
    
    void get_parallel_stats(void* engine, uint64_t* orders, uint64_t* trades, double* latency) {
        static_cast<ParallelFixEngine*>(engine)->get_stats(*orders, *trades, *latency);
    }
}

// Benchmark for testing
#ifdef TEST_MAIN
int main() {
    ParallelFixEngine engine;
    engine.start();
    
    std::cout << "Running parallel FIX engine benchmark..." << std::endl;
    std::cout << "Cores: " << std::thread::hardware_concurrency() << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate load
    const size_t num_orders = 10'000'000;
    std::vector<std::thread> clients;
    
    for (size_t i = 0; i < 4; ++i) {
        clients.emplace_back([&engine, num_orders, i]() {
            char fix_msg[256];
            for (size_t j = 0; j < num_orders/4; ++j) {
                snprintf(fix_msg, sizeof(fix_msg), 
                        "35=D|55=BTC|54=1|38=%zu|44=50000|", j);
                engine.process_fix_message(fix_msg, strlen(fix_msg));
            }
        });
    }
    
    for (auto& t : clients) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    uint64_t orders, trades;
    double avg_latency;
    engine.get_stats(orders, trades, avg_latency);
    
    engine.stop();
    
    std::cout << "Results:" << std::endl;
    std::cout << "  Orders: " << orders << std::endl;
    std::cout << "  Trades: " << trades << std::endl;
    std::cout << "  Duration: " << duration << " ms" << std::endl;
    std::cout << "  Throughput: " << (orders * 1000 / duration) << " orders/sec" << std::endl;
    std::cout << "  Avg Latency: " << avg_latency << " ns" << std::endl;
    
    return 0;
}
#endif