#include "orderbook_bridge.h"
#include <map>
#include <queue>
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <chrono>

// Internal C++ Order Book implementation
class OrderBook {
private:
    struct Order {
        uint64_t id;
        uint64_t user_id;
        double price;
        double quantity;
        double filled_quantity;
        uint8_t side;
        uint8_t type;
        uint8_t status;
        uint64_t timestamp;
    };

    std::string symbol;
    std::map<uint64_t, Order> orders;
    
    // Price-time priority order books
    std::multimap<double, uint64_t> bids; // descending price
    std::multimap<double, uint64_t, std::greater<double>> asks; // ascending price
    
    uint64_t next_trade_id = 1;
    uint64_t total_volume = 0;
    
public:
    OrderBook(const std::string& sym) : symbol(sym) {}
    
    uint64_t addOrder(const OrderC* order_c) {
        Order order;
        order.id = order_c->order_id;
        order.user_id = order_c->user_id;
        order.price = order_c->price;
        order.quantity = order_c->quantity;
        order.filled_quantity = order_c->filled_quantity;
        order.side = order_c->side;
        order.type = order_c->order_type;
        order.status = order_c->status;
        order.timestamp = order_c->timestamp;
        
        orders[order.id] = order;
        
        if (order.type == 1) { // Limit order
            if (order.side == 0) { // Buy
                bids.emplace(order.price, order.id);
            } else { // Sell
                asks.emplace(order.price, order.id);
            }
        }
        
        return order.id;
    }
    
    bool cancelOrder(uint64_t order_id) {
        auto it = orders.find(order_id);
        if (it == orders.end()) return false;
        
        const Order& order = it->second;
        
        // Remove from price books
        if (order.type == 1) {
            if (order.side == 0) {
                auto range = bids.equal_range(order.price);
                for (auto itr = range.first; itr != range.second; ++itr) {
                    if (itr->second == order_id) {
                        bids.erase(itr);
                        break;
                    }
                }
            } else {
                auto range = asks.equal_range(order.price);
                for (auto itr = range.first; itr != range.second; ++itr) {
                    if (itr->second == order_id) {
                        asks.erase(itr);
                        break;
                    }
                }
            }
        }
        
        orders.erase(it);
        return true;
    }
    
    int matchOrders(TradeC* trades, int max_trades) {
        int trade_count = 0;
        
        while (!bids.empty() && !asks.empty() && trade_count < max_trades) {
            auto best_bid = bids.begin();
            auto best_ask = asks.begin();
            
            if (best_bid->first >= best_ask->first) {
                // Match found
                uint64_t bid_id = best_bid->second;
                uint64_t ask_id = best_ask->second;
                
                Order& bid_order = orders[bid_id];
                Order& ask_order = orders[ask_id];
                
                double match_quantity = std::min(
                    bid_order.quantity - bid_order.filled_quantity,
                    ask_order.quantity - ask_order.filled_quantity
                );
                
                double match_price = (bid_order.timestamp < ask_order.timestamp) 
                    ? bid_order.price : ask_order.price;
                
                // Record trade
                trades[trade_count].trade_id = next_trade_id++;
                trades[trade_count].buy_order_id = bid_id;
                trades[trade_count].sell_order_id = ask_id;
                trades[trade_count].price = match_price;
                trades[trade_count].quantity = match_quantity;
                trades[trade_count].timestamp = std::chrono::system_clock::now().time_since_epoch().count();
                
                // Update orders
                bid_order.filled_quantity += match_quantity;
                ask_order.filled_quantity += match_quantity;
                total_volume += match_quantity;
                
                // Update status
                if (bid_order.filled_quantity >= bid_order.quantity) {
                    bid_order.status = 2; // filled
                    bids.erase(best_bid);
                } else {
                    bid_order.status = 1; // partial
                }
                
                if (ask_order.filled_quantity >= ask_order.quantity) {
                    ask_order.status = 2; // filled
                    asks.erase(best_ask);
                } else {
                    ask_order.status = 1; // partial
                }
                
                trade_count++;
            } else {
                break; // No more matches possible
            }
        }
        
        return trade_count;
    }
    
    double getBestBid() const {
        return bids.empty() ? 0.0 : bids.begin()->first;
    }
    
    double getBestAsk() const {
        return asks.empty() ? 0.0 : asks.begin()->first;
    }
    
    int getDepth(int level, double* bid_prices, double* ask_prices, 
                 double* bid_sizes, double* ask_sizes) const {
        int depth = 0;
        
        // Aggregate bids by price level
        std::map<double, double, std::greater<double>> bid_levels;
        for (const auto& [price, order_id] : bids) {
            const Order& order = orders.at(order_id);
            bid_levels[price] += order.quantity - order.filled_quantity;
        }
        
        // Aggregate asks by price level
        std::map<double, double> ask_levels;
        for (const auto& [price, order_id] : asks) {
            const Order& order = orders.at(order_id);
            ask_levels[price] += order.quantity - order.filled_quantity;
        }
        
        // Fill bid arrays
        auto bid_it = bid_levels.begin();
        for (int i = 0; i < level && bid_it != bid_levels.end(); ++i, ++bid_it) {
            bid_prices[i] = bid_it->first;
            bid_sizes[i] = bid_it->second;
            depth = i + 1;
        }
        
        // Fill ask arrays
        auto ask_it = ask_levels.begin();
        for (int i = 0; i < level && ask_it != ask_levels.end(); ++i, ++ask_it) {
            ask_prices[i] = ask_it->first;
            ask_sizes[i] = ask_it->second;
        }
        
        return depth;
    }
    
    uint64_t getVolume() const {
        return total_volume;
    }
};

// C Bridge Implementation
extern "C" {

OrderBookHandle orderbook_create(const char* symbol) {
    return new OrderBook(symbol);
}

void orderbook_destroy(OrderBookHandle handle) {
    delete static_cast<OrderBook*>(handle);
}

uint64_t orderbook_add_order(OrderBookHandle handle, const OrderC* order) {
    return static_cast<OrderBook*>(handle)->addOrder(order);
}

bool orderbook_cancel_order(OrderBookHandle handle, uint64_t order_id) {
    return static_cast<OrderBook*>(handle)->cancelOrder(order_id);
}

bool orderbook_modify_order(OrderBookHandle handle, uint64_t order_id, 
                           double new_price, double new_quantity) {
    auto book = static_cast<OrderBook*>(handle);
    // Cancel and re-add for simplicity
    if (book->cancelOrder(order_id)) {
        OrderC new_order = {};
        new_order.order_id = order_id;
        new_order.price = new_price;
        new_order.quantity = new_quantity;
        new_order.order_type = 1; // limit
        book->addOrder(&new_order);
        return true;
    }
    return false;
}

int orderbook_match_orders(OrderBookHandle handle, TradeC* trades, int max_trades) {
    return static_cast<OrderBook*>(handle)->matchOrders(trades, max_trades);
}

double orderbook_get_best_bid(OrderBookHandle handle) {
    return static_cast<OrderBook*>(handle)->getBestBid();
}

double orderbook_get_best_ask(OrderBookHandle handle) {
    return static_cast<OrderBook*>(handle)->getBestAsk();
}

int orderbook_get_depth(OrderBookHandle handle, int level, double* bids, double* asks, 
                        double* bid_sizes, double* ask_sizes) {
    return static_cast<OrderBook*>(handle)->getDepth(level, bids, asks, bid_sizes, ask_sizes);
}

uint64_t orderbook_get_volume(OrderBookHandle handle) {
    return static_cast<OrderBook*>(handle)->getVolume();
}

int orderbook_get_snapshot(OrderBookHandle handle, char* buffer, int buffer_size) {
    // Simple binary serialization (would use protobuf in production)
    // For now, return 0 (not implemented)
    return 0;
}

bool orderbook_restore_snapshot(OrderBookHandle handle, const char* buffer, int buffer_size) {
    // Simple binary deserialization (would use protobuf in production)
    // For now, return false (not implemented)
    return false;
}

} // extern "C"