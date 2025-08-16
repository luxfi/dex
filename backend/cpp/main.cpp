// Pure C++ LX Engine Implementation
#include <iostream>
#include <map>
#include <vector>
#include <queue>
#include <memory>
#include <atomic>
#include <chrono>
#include <thread>
#include <grpcpp/grpcpp.h>
#include "lx_engine.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using namespace lx_engine;

// High-performance lock-free order book
class OrderBook {
private:
    struct Order {
        uint64_t id;
        double price;
        double quantity;
        OrderSide side;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    std::multimap<double, std::shared_ptr<Order>, std::greater<double>> bids;
    std::multimap<double, std::shared_ptr<Order>> asks;
    std::atomic<uint64_t> order_counter{1};
    std::atomic<uint64_t> trade_counter{1};
    std::mutex book_mutex;
    
public:
    uint64_t addOrder(double price, double quantity, OrderSide side) {
        std::lock_guard<std::mutex> lock(book_mutex);
        
        auto order = std::make_shared<Order>();
        order->id = order_counter.fetch_add(1);
        order->price = price;
        order->quantity = quantity;
        order->side = side;
        order->timestamp = std::chrono::high_resolution_clock::now();
        
        if (side == OrderSide::ORDER_SIDE_BUY) {
            bids.insert({price, order});
        } else {
            asks.insert({price, order});
        }
        
        return order->id;
    }
    
    bool cancelOrder(uint64_t order_id) {
        std::lock_guard<std::mutex> lock(book_mutex);
        
        // Search in bids
        for (auto it = bids.begin(); it != bids.end(); ++it) {
            if (it->second->id == order_id) {
                bids.erase(it);
                return true;
            }
        }
        
        // Search in asks
        for (auto it = asks.begin(); it != asks.end(); ++it) {
            if (it->second->id == order_id) {
                asks.erase(it);
                return true;
            }
        }
        
        return false;
    }
    
    std::vector<std::pair<uint64_t, uint64_t>> matchOrders() {
        std::lock_guard<std::mutex> lock(book_mutex);
        std::vector<std::pair<uint64_t, uint64_t>> trades;
        
        while (!bids.empty() && !asks.empty()) {
            auto best_bid = bids.begin();
            auto best_ask = asks.begin();
            
            if (best_bid->first >= best_ask->first) {
                // Execute trade
                double trade_qty = std::min(best_bid->second->quantity, best_ask->second->quantity);
                
                trades.push_back({best_bid->second->id, best_ask->second->id});
                
                // Update quantities
                best_bid->second->quantity -= trade_qty;
                best_ask->second->quantity -= trade_qty;
                
                // Remove filled orders
                if (best_bid->second->quantity <= 0) {
                    bids.erase(best_bid);
                }
                if (best_ask->second->quantity <= 0) {
                    asks.erase(best_ask);
                }
            } else {
                break;
            }
        }
        
        return trades;
    }
    
    void getDepth(GetOrderBookResponse* response, int levels) {
        std::lock_guard<std::mutex> lock(book_mutex);
        
        int count = 0;
        double last_price = -1;
        PriceLevel* current_level = nullptr;
        
        // Aggregate bids
        for (const auto& [price, order] : bids) {
            if (price != last_price) {
                if (count >= levels) break;
                current_level = response->add_bids();
                current_level->set_price(price);
                current_level->set_quantity(0);
                last_price = price;
                count++;
            }
            current_level->set_quantity(current_level->quantity() + order->quantity);
        }
        
        // Aggregate asks
        count = 0;
        last_price = -1;
        for (const auto& [price, order] : asks) {
            if (price != last_price) {
                if (count >= levels) break;
                current_level = response->add_asks();
                current_level->set_price(price);
                current_level->set_quantity(0);
                last_price = price;
                count++;
            }
            current_level->set_quantity(current_level->quantity() + order->quantity);
        }
    }
};

// gRPC Service Implementation
class EngineServiceImpl final : public EngineService::Service {
private:
    std::map<std::string, std::unique_ptr<OrderBook>> order_books;
    std::mutex books_mutex;
    std::atomic<uint64_t> total_orders{0};
    std::atomic<uint64_t> total_cancels{0};
    
    OrderBook* getOrCreateOrderBook(const std::string& symbol) {
        std::lock_guard<std::mutex> lock(books_mutex);
        if (order_books.find(symbol) == order_books.end()) {
            order_books[symbol] = std::make_unique<OrderBook>();
        }
        return order_books[symbol].get();
    }
    
public:
    Status SubmitOrder(ServerContext* context, 
                       const SubmitOrderRequest* request,
                       SubmitOrderResponse* response) override {
        auto* book = getOrCreateOrderBook(request->symbol());
        
        uint64_t order_id = book->addOrder(
            request->price(),
            request->quantity(),
            request->side()
        );
        
        // Match orders immediately
        auto trades = book->matchOrders();
        
        response->set_order_id("order-" + request->symbol());
        response->set_status(OrderStatus::ORDER_STATUS_NEW);
        response->set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());
        
        total_orders.fetch_add(1);
        
        return Status::OK;
    }
    
    Status CancelOrder(ServerContext* context,
                      const CancelOrderRequest* request,
                      CancelOrderResponse* response) override {
        // For simplicity, extract symbol from order_id
        bool success = false;
        for (auto& [symbol, book] : order_books) {
            if (book->cancelOrder(std::hash<std::string>{}(request->order_id()))) {
                success = true;
                break;
            }
        }
        
        response->set_success(success);
        total_cancels.fetch_add(1);
        
        return Status::OK;
    }
    
    Status GetOrderBook(ServerContext* context,
                       const GetOrderBookRequest* request,
                       GetOrderBookResponse* response) override {
        auto* book = getOrCreateOrderBook(request->symbol());
        
        response->set_symbol(request->symbol());
        response->set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());
        
        book->getDepth(response, request->depth());
        
        return Status::OK;
    }
    
    Status StreamOrderBook(ServerContext* context,
                          const StreamOrderBookRequest* request,
                          grpc::ServerWriter<OrderBookUpdate>* writer) override {
        // Simplified streaming - send updates every 100ms
        for (int i = 0; i < 10; ++i) {
            OrderBookUpdate update;
            update.set_symbol(request->symbol());
            update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());
            
            writer->Write(update);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        return Status::OK;
    }
    
    void PrintStats() {
        std::cout << "Total Orders: " << total_orders.load() 
                  << ", Total Cancels: " << total_cancels.load() << std::endl;
    }
};

void RunServer(int port) {
    std::string server_address = "0.0.0.0:" + std::to_string(port);
    EngineServiceImpl service;
    
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    // Optimize for performance
    builder.SetMaxReceiveMessageSize(50 * 1024 * 1024);
    builder.SetMaxSendMessageSize(50 * 1024 * 1024);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 5000);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "C++ Engine listening on " << server_address << std::endl;
    
    // Stats printer thread
    std::thread stats_thread([&service]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            service.PrintStats();
        }
    });
    
    server->Wait();
}