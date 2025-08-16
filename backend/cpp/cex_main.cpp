#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "lx_engine.grpc.pb.h"
#include "../bridge/orderbook_bridge.h"
#include "../bridge/fix_bridge.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using lx::engine::v1::LXEngine;
using lx::engine::v1::CreateSessionRequest;
using lx::engine::v1::CreateSessionResponse;
using lx::engine::v1::CreateOrderRequest;
using lx::engine::v1::CreateOrderResponse;
using lx::engine::v1::GetEngineStatusRequest;
using lx::engine::v1::GetEngineStatusResponse;
using lx::engine::v1::GetEngineStatsRequest;
using lx::engine::v1::GetEngineStatsResponse;

// Pure C++ implementation of the LX CEX Engine
class LXCexEngine final : public LXEngine::Service {
private:
    struct Session {
        std::string id;
        std::string user_id;
        std::unordered_map<uint64_t, OrderC> orders;
        std::chrono::steady_clock::time_point last_activity;
    };
    
    std::unordered_map<std::string, std::unique_ptr<Session>> sessions_;
    std::unordered_map<std::string, OrderBookHandle> order_books_;
    std::mutex sessions_mutex_;
    std::mutex books_mutex_;
    
    // Performance counters
    std::atomic<uint64_t> orders_processed_{0};
    std::atomic<uint64_t> trades_executed_{0};
    std::atomic<uint64_t> sessions_created_{0};
    
    // Engine start time
    std::chrono::steady_clock::time_point start_time_;
    
public:
    LXCexEngine() : start_time_(std::chrono::steady_clock::now()) {
        std::cout << "LX CEX Engine (Pure C++) initialized" << std::endl;
    }
    
    ~LXCexEngine() {
        // Clean up order books
        for (auto& [symbol, handle] : order_books_) {
            orderbook_destroy(handle);
        }
    }
    
    // Session management
    Status CreateSession(ServerContext* context, 
                        const CreateSessionRequest* request,
                        CreateSessionResponse* response) override {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        
        auto session = std::make_unique<Session>();
        session->id = "session_" + request->user_id() + "_" + 
                     std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        session->user_id = request->user_id();
        session->last_activity = std::chrono::steady_clock::now();
        
        response->set_session_id(session->id);
        response->set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());
        
        sessions_[session->id] = std::move(session);
        sessions_created_++;
        
        return Status::OK;
    }
    
    // Order management
    Status CreateOrder(ServerContext* context,
                      const CreateOrderRequest* request,
                      CreateOrderResponse* response) override {
        // Get or create order book
        OrderBookHandle book = getOrCreateOrderBook(request->symbol());
        
        // Create order
        OrderC order = {};
        order.order_id = ++orders_processed_;
        order.user_id = std::hash<std::string>{}(request->session_id());
        strncpy(order.symbol, request->symbol().c_str(), 31);
        order.price = request->price();
        order.quantity = request->quantity();
        order.filled_quantity = 0;
        order.side = request->side() == lx::engine::v1::ORDER_SIDE_BUY ? 0 : 1;
        order.order_type = request->order_type() == lx::engine::v1::ORDER_TYPE_MARKET ? 0 : 1;
        order.status = 0; // pending
        order.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        
        // Add to order book
        uint64_t order_id = orderbook_add_order(book, &order);
        
        // Match orders
        TradeC trades[100];
        int num_trades = orderbook_match_orders(book, trades, 100);
        
        // Update response
        response->set_order_id(order_id);
        response->set_status(lx::engine::v1::ORDER_STATUS_PENDING);
        
        // Add executed trades to response
        for (int i = 0; i < num_trades; i++) {
            auto* trade = response->add_trades();
            trade->set_id(trades[i].trade_id);
            trade->set_price(trades[i].price);
            trade->set_quantity(trades[i].quantity);
            trade->set_buy_order_id(trades[i].buy_order_id);
            trade->set_sell_order_id(trades[i].sell_order_id);
            trade->set_timestamp(trades[i].timestamp);
            
            trades_executed_++;
        }
        
        return Status::OK;
    }
    
    // Engine status
    Status GetEngineStatus(ServerContext* context,
                          const GetEngineStatusRequest* request,
                          GetEngineStatusResponse* response) override {
        response->set_is_running(true);
        response->set_implementation(lx::engine::v1::ENGINE_IMPL_CPP);
        response->set_version("1.0.0");
        
        auto now = std::chrono::steady_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
        response->set_uptime_seconds(uptime.count());
        
        return Status::OK;
    }
    
    // Engine statistics
    Status GetEngineStats(ServerContext* context,
                         const GetEngineStatsRequest* request,
                         GetEngineStatsResponse* response) override {
        response->set_orders_processed(orders_processed_.load());
        response->set_trades_executed(trades_executed_.load());
        response->set_active_sessions(sessions_.size());
        response->set_active_order_books(order_books_.size());
        
        // Calculate rates
        auto now = std::chrono::steady_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
        if (uptime.count() > 0) {
            response->set_messages_per_second(
                static_cast<double>(orders_processed_.load()) / uptime.count());
            response->set_trades_per_second(
                static_cast<double>(trades_executed_.load()) / uptime.count());
        }
        
        // Add order book volumes
        for (const auto& [symbol, book] : order_books_) {
            (*response->mutable_order_book_volumes())[symbol] = orderbook_get_volume(book);
        }
        
        return Status::OK;
    }
    
private:
    OrderBookHandle getOrCreateOrderBook(const std::string& symbol) {
        std::lock_guard<std::mutex> lock(books_mutex_);
        
        auto it = order_books_.find(symbol);
        if (it != order_books_.end()) {
            return it->second;
        }
        
        OrderBookHandle book = orderbook_create(symbol.c_str());
        order_books_[symbol] = book;
        return book;
    }
};

// Main function for the CEX server
int main(int argc, char** argv) {
    std::string server_address("0.0.0.0:50051");
    
    if (argc > 1) {
        server_address = argv[1];
    }
    
    // Enable gRPC reflection for debugging
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    
    // Create and configure server
    LXCexEngine service;
    ServerBuilder builder;
    
    // Configure server
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    // Add health check service
    grpc::EnableDefaultHealthCheckService(true);
    
    // Set max message size (100MB)
    builder.SetMaxReceiveMessageSize(100 * 1024 * 1024);
    builder.SetMaxSendMessageSize(100 * 1024 * 1024);
    
    // Build and start server
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "LX CEX Server (Pure C++) listening on " << server_address << std::endl;
    
    // Wait for shutdown
    server->Wait();
    
    return 0;
}