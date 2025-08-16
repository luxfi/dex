// Standalone C++ orderbook benchmark
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <random>
#include <atomic>
#include <thread>
#include <mutex>

class FastOrderBook {
private:
    struct Order {
        uint64_t id;
        double price;
        double quantity;
        bool is_buy;
    };
    
    std::multimap<double, Order*, std::greater<double>> bids;
    std::multimap<double, Order*> asks;
    std::atomic<uint64_t> order_id{1};
    std::atomic<uint64_t> trades_executed{0};
    std::mutex book_mutex;
    
public:
    void addOrder(double price, double quantity, bool is_buy) {
        std::lock_guard<std::mutex> lock(book_mutex);
        
        auto* order = new Order{order_id.fetch_add(1), price, quantity, is_buy};
        
        if (is_buy) {
            bids.insert({price, order});
        } else {
            asks.insert({price, order});
        }
        
        // Try to match immediately
        matchOrders();
    }
    
    void matchOrders() {
        while (!bids.empty() && !asks.empty()) {
            auto best_bid = bids.begin();
            auto best_ask = asks.begin();
            
            if (best_bid->first >= best_ask->first) {
                // Execute trade
                double qty = std::min(best_bid->second->quantity, best_ask->second->quantity);
                
                best_bid->second->quantity -= qty;
                best_ask->second->quantity -= qty;
                
                trades_executed.fetch_add(1);
                
                // Remove filled orders
                if (best_bid->second->quantity <= 0) {
                    delete best_bid->second;
                    bids.erase(best_bid);
                }
                if (best_ask->second->quantity <= 0) {
                    delete best_ask->second;
                    asks.erase(best_ask);
                }
            } else {
                break;
            }
        }
    }
    
    uint64_t getTrades() const { return trades_executed.load(); }
    
    ~FastOrderBook() {
        for (auto& [price, order] : bids) delete order;
        for (auto& [price, order] : asks) delete order;
    }
};

void runBenchmark(int num_threads, int orders_per_thread, int duration_sec) {
    FastOrderBook book;
    std::atomic<uint64_t> total_orders{0};
    std::atomic<bool> stop{false};
    
    auto worker = [&](int id) {
        std::mt19937 rng(id);
        std::uniform_real_distribution<> price_dist(99.0, 101.0);
        std::uniform_real_distribution<> qty_dist(0.1, 10.0);
        std::uniform_int_distribution<> side_dist(0, 1);
        
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t local_orders = 0;
        
        while (!stop.load()) {
            book.addOrder(
                price_dist(rng),
                qty_dist(rng),
                side_dist(rng) == 0
            );
            local_orders++;
            
            // Check time
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= duration_sec) {
                break;
            }
        }
        
        total_orders.fetch_add(local_orders);
    };
    
    // Start threads
    std::vector<std::thread> threads;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for duration
    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
    stop.store(true);
    
    // Join threads
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    
    uint64_t orders = total_orders.load();
    uint64_t trades = book.getTrades();
    
    std::cout << "C++ Standalone Benchmark Results:\n";
    std::cout << "  Threads: " << num_threads << "\n";
    std::cout << "  Duration: " << elapsed << " seconds\n";
    std::cout << "  Total Orders: " << orders << "\n";
    std::cout << "  Total Trades: " << trades << "\n";
    std::cout << "  Throughput: " << (orders / elapsed) << " orders/sec\n";
    std::cout << "  Trade Rate: " << (trades / elapsed) << " trades/sec\n";
}

int main(int argc, char* argv[]) {
    int threads = 100;
    int duration = 10;
    
    if (argc > 1) threads = std::stoi(argv[1]);
    if (argc > 2) duration = std::stoi(argv[2]);
    
    std::cout << "=== C++ OrderBook Benchmark ===\n";
    std::cout << "Testing with " << threads << " threads for " << duration << " seconds\n\n";
    
    // Warmup
    std::cout << "Warming up...\n";
    runBenchmark(10, 1000, 2);
    
    // Test different thread counts
    for (int t : {10, 50, 100, 200, 500, 1000}) {
        if (t > threads) break;
        std::cout << "\n--- Test with " << t << " threads ---\n";
        runBenchmark(t, 0, duration);
    }
    
    return 0;
}