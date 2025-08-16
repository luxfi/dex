// Simple high-performance C++ orderbook for CGO integration
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>
#include <atomic>

extern "C" {

struct OrderC {
    uint64_t id;
    double price;
    double quantity;
    uint8_t side; // 0=buy, 1=sell
};

struct TradeC {
    uint64_t id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    double price;
    double quantity;
};

class SimpleOrderBook {
private:
    std::multimap<double, OrderC, std::greater<double>> bids; // descending
    std::multimap<double, OrderC> asks; // ascending
    std::atomic<uint64_t> trade_counter{1};
    std::vector<TradeC> pending_trades;
    
public:
    void addOrder(const OrderC& order) {
        if (order.side == 0) { // buy
            bids.insert({order.price, order});
        } else { // sell
            asks.insert({order.price, order});
        }
    }
    
    bool cancelOrder(uint64_t order_id) {
        // Search in bids
        for (auto it = bids.begin(); it != bids.end(); ++it) {
            if (it->second.id == order_id) {
                bids.erase(it);
                return true;
            }
        }
        // Search in asks
        for (auto it = asks.begin(); it != asks.end(); ++it) {
            if (it->second.id == order_id) {
                asks.erase(it);
                return true;
            }
        }
        return false;
    }
    
    std::vector<TradeC> matchOrders() {
        pending_trades.clear();
        
        while (!bids.empty() && !asks.empty()) {
            auto best_bid = bids.begin();
            auto best_ask = asks.begin();
            
            // Check if orders cross
            if (best_bid->first >= best_ask->first) {
                // Execute trade
                double trade_price = best_ask->first; // Price-time priority
                double trade_qty = std::min(best_bid->second.quantity, best_ask->second.quantity);
                
                TradeC trade;
                trade.id = trade_counter.fetch_add(1);
                trade.buy_order_id = best_bid->second.id;
                trade.sell_order_id = best_ask->second.id;
                trade.price = trade_price;
                trade.quantity = trade_qty;
                
                pending_trades.push_back(trade);
                
                // Update quantities
                if (best_bid->second.quantity > trade_qty) {
                    // Partial fill - update bid
                    OrderC updated_bid = best_bid->second;
                    updated_bid.quantity -= trade_qty;
                    bids.erase(best_bid);
                    bids.insert({updated_bid.price, updated_bid});
                } else {
                    // Full fill - remove bid
                    bids.erase(best_bid);
                }
                
                if (best_ask->second.quantity > trade_qty) {
                    // Partial fill - update ask
                    OrderC updated_ask = best_ask->second;
                    updated_ask.quantity -= trade_qty;
                    asks.erase(best_ask);
                    asks.insert({updated_ask.price, updated_ask});
                } else {
                    // Full fill - remove ask
                    asks.erase(best_ask);
                }
            } else {
                break; // No more crosses
            }
        }
        
        return pending_trades;
    }
    
    double getBestBid() const {
        return bids.empty() ? 0.0 : bids.begin()->first;
    }
    
    double getBestAsk() const {
        return asks.empty() ? 0.0 : asks.begin()->first;
    }
    
    size_t getDepth(uint8_t side) const {
        return side == 0 ? bids.size() : asks.size();
    }
};

// C interface for CGO
void* orderbook_create() {
    return new SimpleOrderBook();
}

void orderbook_destroy(void* ob) {
    delete static_cast<SimpleOrderBook*>(ob);
}

void orderbook_add_order(void* ob, uint64_t id, double price, double quantity, uint8_t side) {
    OrderC order = {id, price, quantity, side};
    static_cast<SimpleOrderBook*>(ob)->addOrder(order);
}

int orderbook_cancel_order(void* ob, uint64_t order_id) {
    return static_cast<SimpleOrderBook*>(ob)->cancelOrder(order_id) ? 1 : 0;
}

int orderbook_match_orders(void* ob, TradeC* trades_out, int max_trades) {
    auto trades = static_cast<SimpleOrderBook*>(ob)->matchOrders();
    int count = std::min(static_cast<int>(trades.size()), max_trades);
    if (count > 0 && trades_out != nullptr) {
        memcpy(trades_out, trades.data(), count * sizeof(TradeC));
    }
    return count;
}

double orderbook_get_best_bid(void* ob) {
    return static_cast<SimpleOrderBook*>(ob)->getBestBid();
}

double orderbook_get_best_ask(void* ob) {
    return static_cast<SimpleOrderBook*>(ob)->getBestAsk();
}

int orderbook_get_depth(void* ob, uint8_t side) {
    return static_cast<SimpleOrderBook*>(ob)->getDepth(side);
}

} // extern "C"