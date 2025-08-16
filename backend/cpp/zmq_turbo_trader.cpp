// Ultra-Fast C++ LX Trader
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <iomanip>
#include <zmq.hpp>
#include <sstream>

constexpr size_t MSG_SIZE = 256;

struct Order {
    char symbol[16];
    char side[8];
    double price;
    double quantity;
    uint64_t timestamp;
};

class TurboZMQTrader {
private:
    std::atomic<uint64_t> orders_sent{0};
    std::atomic<uint64_t> responses_received{0};
    std::atomic<bool> running{true};
    std::vector<std::thread> traders;
    std::string server_address;
    int num_traders;
    int orders_per_second;
    
public:
    TurboZMQTrader(const std::string& server, int traders, int rate) 
        : server_address(server), num_traders(traders), orders_per_second(rate) {
        
        std::cout << "🚀 C++ LX Turbo Trader\n";
        std::cout << "Server: " << server << "\n";
        std::cout << "Traders: " << traders << "\n";
        std::cout << "Rate: " << rate << " orders/sec per trader\n";
        std::cout << "Total target: " << (traders * rate) << " orders/sec\n";
        std::cout << "========================================\n\n";
    }
    
    ~TurboZMQTrader() {
        stop();
    }
    
    void start() {
        for (int i = 0; i < num_traders; ++i) {
            traders.emplace_back([this, i]() {
                trader_thread(i);
            });
        }
    }
    
    void stop() {
        running = false;
        for (auto& t : traders) {
            if (t.joinable()) t.join();
        }
    }
    
    void run_for(int seconds) {
        start();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Monitor progress
        for (int i = 0; i < seconds; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            uint64_t sent = orders_sent.load();
            uint64_t received = responses_received.load();
            
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            
            double send_rate = sent / elapsed;
            double recv_rate = received / elapsed;
            
            std::cout << "\r⚡ Sent: " << sent 
                      << " (" << std::fixed << std::setprecision(0) << send_rate << "/sec)"
                      << " | Received: " << received
                      << " (" << recv_rate << "/sec)" << std::flush;
        }
        
        running = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Final stats
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        uint64_t final_sent = orders_sent.load();
        uint64_t final_received = responses_received.load();
        
        std::cout << "\n\n========================================\n";
        std::cout << "📊 FINAL RESULTS\n";
        std::cout << "========================================\n";
        std::cout << "Duration: " << total_elapsed << " seconds\n";
        std::cout << "Orders Sent: " << final_sent << "\n";
        std::cout << "Responses: " << final_received << "\n";
        std::cout << "Send Rate: " << (final_sent / total_elapsed) << " orders/sec\n";
        std::cout << "Recv Rate: " << (final_received / total_elapsed) << " resp/sec\n";
        
        double million_rate = (final_sent / total_elapsed) / 1000000.0;
        std::cout << "\n🎯 THROUGHPUT: " << std::fixed << std::setprecision(3) 
                  << million_rate << "M orders/sec\n";
        
        if (million_rate >= 1.0) {
            std::cout << "✅ ACHIEVED 1M+ orders/sec!\n";
        } else if (million_rate >= 0.5) {
            std::cout << "🔥 Excellent! Over 500K orders/sec!\n";
        } else if (million_rate >= 0.1) {
            std::cout << "👍 Good! Over 100K orders/sec!\n";
        }
    }
    
private:
    void trader_thread(int id) {
        // Create ZMQ context and socket
        zmq::context_t context(1);
        zmq::socket_t socket(context, zmq::socket_type::dealer);
        
        // Set socket options for performance
        int linger = 0;
        socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
        
        int sndhwm = 100000;
        socket.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        
        int rcvhwm = 100000;
        socket.setsockopt(ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
        
        // Connect
        socket.connect(server_address);
        
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(50000, 60000);
        std::uniform_real_distribution<> qty_dist(0.1, 10.0);
        std::uniform_int_distribution<> side_dist(0, 1);
        
        // Calculate sleep time between orders
        auto sleep_ns = std::chrono::nanoseconds(1000000000 / orders_per_second);
        
        // Order buffer
        char buffer[MSG_SIZE];
        
        while (running) {
            // Generate order
            Order order;
            strcpy(order.symbol, "BTC/USD");
            strcpy(order.side, side_dist(gen) ? "BUY" : "SELL");
            order.price = price_dist(gen);
            order.quantity = qty_dist(gen);
            order.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            
            // Format as JSON-like message
            std::stringstream ss;
            ss << "{\"symbol\":\"" << order.symbol << "\","
               << "\"side\":\"" << order.side << "\","
               << "\"price\":" << order.price << ","
               << "\"quantity\":" << order.quantity << ","
               << "\"timestamp\":" << order.timestamp << "}";
            
            std::string msg = ss.str();
            
            // Send order
            zmq::message_t request(msg.data(), msg.size());
            socket.send(request, zmq::send_flags::dontwait);
            orders_sent.fetch_add(1);
            
            // Try to receive response (non-blocking)
            zmq::message_t reply;
            auto result = socket.recv(reply, zmq::recv_flags::dontwait);
            if (result) {
                responses_received.fetch_add(1);
            }
            
            // Rate limiting
            if (orders_per_second < 10000) {
                std::this_thread::sleep_for(sleep_ns);
            }
        }
        
        socket.close();
    }
};

int main(int argc, char* argv[]) {
    std::string server = "tcp://localhost:5555";
    int traders = std::thread::hardware_concurrency() * 2;
    int rate = 10000;
    int duration = 30;
    
    if (argc > 1) server = argv[1];
    if (argc > 2) traders = std::atoi(argv[2]);
    if (argc > 3) rate = std::atoi(argv[3]);
    if (argc > 4) duration = std::atoi(argv[4]);
    
    TurboZMQTrader trader(server, traders, rate);
    trader.run_for(duration);
    
    return 0;
}