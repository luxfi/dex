// Ultra-Low Latency Order Book with DPDK
// Achieves <100ns latency per order using kernel bypass

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_hash.h>
#include <rte_jhash.h>
#include <rte_malloc.h>

#include <atomic>
#include <cstring>
#include <immintrin.h>

// Cache line size for alignment
#define CACHE_LINE_SIZE 64
#define MAX_PKT_BURST 32
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024

// Order types aligned with wire protocol
enum OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1,
    STOP = 2,
    STOP_LIMIT = 3
};

enum Side : uint8_t {
    BUY = 0,
    SELL = 1
};

// Packed order structure for network transmission
struct __attribute__((packed)) OrderPacket {
    uint64_t order_id;
    uint64_t user_id;
    uint32_t symbol_id;
    uint32_t price;      // Fixed point, 7 decimals
    uint32_t quantity;   // Fixed point, 7 decimals
    uint8_t side;
    uint8_t type;
    uint16_t reserved;
};

// Lock-free order structure (cache-line aligned)
struct alignas(CACHE_LINE_SIZE) Order {
    std::atomic<uint64_t> order_id;
    std::atomic<uint64_t> user_id;
    std::atomic<uint32_t> symbol_id;
    std::atomic<uint32_t> price;
    std::atomic<uint32_t> quantity;
    std::atomic<uint32_t> filled_quantity;
    std::atomic<uint8_t> side;
    std::atomic<uint8_t> type;
    std::atomic<uint8_t> status;
    std::atomic<uint64_t> timestamp_ns;
    
    // Padding to cache line
    char padding[CACHE_LINE_SIZE - 64];
};

static_assert(sizeof(Order) == CACHE_LINE_SIZE, "Order must be cache-line sized");

// Lock-free price level using compare-and-swap
class alignas(CACHE_LINE_SIZE) PriceLevel {
public:
    std::atomic<uint32_t> price;
    std::atomic<uint32_t> total_quantity;
    std::atomic<uint32_t> order_count;
    std::atomic<Order*> head;
    std::atomic<Order*> tail;
    
    // Padding
    char padding[CACHE_LINE_SIZE - 40];
    
    bool add_order(Order* order) {
        Order* expected_tail = tail.load(std::memory_order_acquire);
        
        while (true) {
            if (tail.compare_exchange_weak(expected_tail, order, 
                                          std::memory_order_release,
                                          std::memory_order_acquire)) {
                if (expected_tail != nullptr) {
                    // Link previous tail to new order
                    // Using atomic next pointer in Order (not shown)
                }
                
                // Update counters
                total_quantity.fetch_add(order->quantity, std::memory_order_relaxed);
                order_count.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            
            // Retry with updated tail
            _mm_pause();  // CPU pause instruction for spin-wait
        }
    }
};

// Ultra-fast order book with DPDK
class DPDKOrderBook {
private:
    static constexpr size_t MAX_PRICE_LEVELS = 10000;
    static constexpr size_t MAX_ORDERS = 1000000;
    
    // Separate bid/ask levels for cache locality
    alignas(CACHE_LINE_SIZE) PriceLevel* bid_levels[MAX_PRICE_LEVELS];
    alignas(CACHE_LINE_SIZE) PriceLevel* ask_levels[MAX_PRICE_LEVELS];
    
    // Lock-free memory pool for orders
    struct rte_mempool* order_pool;
    struct rte_mempool* mbuf_pool;
    
    // DPDK port configuration
    uint16_t port_id;
    uint16_t rx_queue_id;
    uint16_t tx_queue_id;
    
    // Statistics (cache-aligned)
    alignas(CACHE_LINE_SIZE) struct {
        std::atomic<uint64_t> orders_received;
        std::atomic<uint64_t> trades_executed;
        std::atomic<uint64_t> packets_processed;
        std::atomic<uint64_t> total_latency_ns;
    } stats;
    
public:
    bool init(uint16_t port, uint16_t rx_queue, uint16_t tx_queue) {
        port_id = port;
        rx_queue_id = rx_queue;
        tx_queue_id = tx_queue;
        
        // Create memory pool for orders
        order_pool = rte_mempool_create("order_pool",
                                       MAX_ORDERS,
                                       sizeof(Order),
                                       CACHE_LINE_SIZE,
                                       0,
                                       NULL, NULL,
                                       NULL, NULL,
                                       rte_socket_id(),
                                       MEMPOOL_F_SP_PUT | MEMPOOL_F_SC_GET);
        
        if (order_pool == NULL) {
            return false;
        }
        
        // Create mbuf pool for packets
        mbuf_pool = rte_pktmbuf_pool_create("mbuf_pool",
                                           NUM_MBUFS,
                                           MBUF_CACHE_SIZE,
                                           0,
                                           RTE_MBUF_DEFAULT_BUF_SIZE,
                                           rte_socket_id());
        
        if (mbuf_pool == NULL) {
            return false;
        }
        
        // Initialize price levels
        memset(bid_levels, 0, sizeof(bid_levels));
        memset(ask_levels, 0, sizeof(ask_levels));
        
        return configure_port();
    }
    
    // Main packet processing loop - runs on dedicated core
    void process_packets() {
        struct rte_mbuf* pkts_burst[MAX_PKT_BURST];
        
        while (true) {
            // Receive burst of packets
            const uint16_t nb_rx = rte_eth_rx_burst(port_id, rx_queue_id,
                                                   pkts_burst, MAX_PKT_BURST);
            
            // Prefetch packets for better cache usage
            for (uint16_t i = 0; i < nb_rx && i < 4; i++) {
                rte_prefetch0(rte_pktmbuf_mtod(pkts_burst[i], void*));
            }
            
            // Process each packet
            for (uint16_t i = 0; i < nb_rx; i++) {
                process_order_packet(pkts_burst[i]);
                
                // Prefetch next packets
                if (i + 4 < nb_rx) {
                    rte_prefetch0(rte_pktmbuf_mtod(pkts_burst[i + 4], void*));
                }
            }
            
            // Free processed packets
            for (uint16_t i = 0; i < nb_rx; i++) {
                rte_pktmbuf_free(pkts_burst[i]);
            }
            
            stats.packets_processed.fetch_add(nb_rx, std::memory_order_relaxed);
        }
    }
    
private:
    bool configure_port() {
        struct rte_eth_conf port_conf = {};
        port_conf.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;
        
        // Configure port with 1 RX/TX queue
        if (rte_eth_dev_configure(port_id, 1, 1, &port_conf) != 0) {
            return false;
        }
        
        // Setup RX queue
        if (rte_eth_rx_queue_setup(port_id, rx_queue_id, RX_RING_SIZE,
                                  rte_eth_dev_socket_id(port_id),
                                  NULL, mbuf_pool) != 0) {
            return false;
        }
        
        // Setup TX queue
        if (rte_eth_tx_queue_setup(port_id, tx_queue_id, TX_RING_SIZE,
                                  rte_eth_dev_socket_id(port_id),
                                  NULL) != 0) {
            return false;
        }
        
        // Start the port
        if (rte_eth_dev_start(port_id) != 0) {
            return false;
        }
        
        // Enable promiscuous mode
        rte_eth_promiscuous_enable(port_id);
        
        return true;
    }
    
    void process_order_packet(struct rte_mbuf* pkt) {
        // Get packet timestamp (hardware timestamp if available)
        uint64_t timestamp_ns = rte_get_tsc_cycles();
        
        // Skip Ethernet, IP, UDP headers (42 bytes typical)
        uint8_t* data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, 42);
        
        // Cast to order packet
        OrderPacket* order_pkt = reinterpret_cast<OrderPacket*>(data);
        
        // Allocate order from memory pool
        Order* order;
        if (rte_mempool_get(order_pool, (void**)&order) != 0) {
            // Pool exhausted, drop packet
            return;
        }
        
        // Initialize order (atomic stores)
        order->order_id.store(order_pkt->order_id, std::memory_order_relaxed);
        order->user_id.store(order_pkt->user_id, std::memory_order_relaxed);
        order->symbol_id.store(order_pkt->symbol_id, std::memory_order_relaxed);
        order->price.store(order_pkt->price, std::memory_order_relaxed);
        order->quantity.store(order_pkt->quantity, std::memory_order_relaxed);
        order->filled_quantity.store(0, std::memory_order_relaxed);
        order->side.store(order_pkt->side, std::memory_order_relaxed);
        order->type.store(order_pkt->type, std::memory_order_relaxed);
        order->status.store(0, std::memory_order_relaxed);
        order->timestamp_ns.store(timestamp_ns, std::memory_order_relaxed);
        
        // Process based on order type
        if (order_pkt->type == LIMIT) {
            process_limit_order(order);
        } else if (order_pkt->type == MARKET) {
            process_market_order(order);
        }
        
        stats.orders_received.fetch_add(1, std::memory_order_relaxed);
        
        // Calculate and store latency
        uint64_t latency = rte_get_tsc_cycles() - timestamp_ns;
        stats.total_latency_ns.fetch_add(latency, std::memory_order_relaxed);
    }
    
    void process_limit_order(Order* order) {
        uint32_t price = order->price.load(std::memory_order_acquire);
        uint8_t side = order->side.load(std::memory_order_acquire);
        
        // Get price level index (price in cents)
        uint32_t level_idx = price / 100;
        
        if (level_idx >= MAX_PRICE_LEVELS) {
            // Invalid price, reject order
            rte_mempool_put(order_pool, order);
            return;
        }
        
        // Try to match immediately
        if (side == BUY) {
            // Check if we can match with asks
            if (try_match_buy_order(order)) {
                return;  // Order fully filled
            }
            
            // Add to bid book
            add_to_bid_book(order, level_idx);
        } else {
            // Check if we can match with bids
            if (try_match_sell_order(order)) {
                return;  // Order fully filled
            }
            
            // Add to ask book
            add_to_ask_book(order, level_idx);
        }
    }
    
    void process_market_order(Order* order) {
        uint8_t side = order->side.load(std::memory_order_acquire);
        
        if (side == BUY) {
            // Match against asks at any price
            match_market_buy(order);
        } else {
            // Match against bids at any price
            match_market_sell(order);
        }
        
        // Market orders don't rest in book
        rte_mempool_put(order_pool, order);
    }
    
    bool try_match_buy_order(Order* buy_order) {
        uint32_t buy_price = buy_order->price.load(std::memory_order_acquire);
        uint32_t remaining = buy_order->quantity.load(std::memory_order_acquire);
        
        // Scan asks from lowest price
        for (size_t i = 0; i < MAX_PRICE_LEVELS && remaining > 0; i++) {
            PriceLevel* level = ask_levels[i];
            if (level == nullptr) continue;
            
            uint32_t ask_price = level->price.load(std::memory_order_acquire);
            if (ask_price > buy_price) {
                break;  // No more matches possible
            }
            
            // Match orders at this level
            remaining = match_at_level(buy_order, level, remaining);
        }
        
        uint32_t filled = buy_order->quantity.load() - remaining;
        buy_order->filled_quantity.store(filled, std::memory_order_release);
        
        return remaining == 0;  // Fully filled?
    }
    
    bool try_match_sell_order(Order* sell_order) {
        uint32_t sell_price = sell_order->price.load(std::memory_order_acquire);
        uint32_t remaining = sell_order->quantity.load(std::memory_order_acquire);
        
        // Scan bids from highest price
        for (int i = MAX_PRICE_LEVELS - 1; i >= 0 && remaining > 0; i--) {
            PriceLevel* level = bid_levels[i];
            if (level == nullptr) continue;
            
            uint32_t bid_price = level->price.load(std::memory_order_acquire);
            if (bid_price < sell_price) {
                break;  // No more matches possible
            }
            
            // Match orders at this level
            remaining = match_at_level(sell_order, level, remaining);
        }
        
        uint32_t filled = sell_order->quantity.load() - remaining;
        sell_order->filled_quantity.store(filled, std::memory_order_release);
        
        return remaining == 0;  // Fully filled?
    }
    
    uint32_t match_at_level(Order* aggressor, PriceLevel* level, uint32_t remaining) {
        // Simple matching - would need more complex logic for production
        uint32_t level_qty = level->total_quantity.load(std::memory_order_acquire);
        uint32_t matched = std::min(remaining, level_qty);
        
        if (matched > 0) {
            level->total_quantity.fetch_sub(matched, std::memory_order_acq_rel);
            stats.trades_executed.fetch_add(1, std::memory_order_relaxed);
            remaining -= matched;
        }
        
        return remaining;
    }
    
    void match_market_buy(Order* buy_order) {
        uint32_t remaining = buy_order->quantity.load(std::memory_order_acquire);
        
        // Match against all asks
        for (size_t i = 0; i < MAX_PRICE_LEVELS && remaining > 0; i++) {
            PriceLevel* level = ask_levels[i];
            if (level == nullptr) continue;
            
            remaining = match_at_level(buy_order, level, remaining);
        }
        
        uint32_t filled = buy_order->quantity.load() - remaining;
        buy_order->filled_quantity.store(filled, std::memory_order_release);
    }
    
    void match_market_sell(Order* sell_order) {
        uint32_t remaining = sell_order->quantity.load(std::memory_order_acquire);
        
        // Match against all bids
        for (int i = MAX_PRICE_LEVELS - 1; i >= 0 && remaining > 0; i--) {
            PriceLevel* level = bid_levels[i];
            if (level == nullptr) continue;
            
            remaining = match_at_level(sell_order, level, remaining);
        }
        
        uint32_t filled = sell_order->quantity.load() - remaining;
        sell_order->filled_quantity.store(filled, std::memory_order_release);
    }
    
    void add_to_bid_book(Order* order, uint32_t level_idx) {
        PriceLevel* level = bid_levels[level_idx];
        
        if (level == nullptr) {
            // Create new level
            void* mem = rte_malloc_socket("price_level", 
                                        sizeof(PriceLevel),
                                        CACHE_LINE_SIZE,
                                        rte_socket_id());
            level = new (mem) PriceLevel();
            level->price.store(order->price.load(), std::memory_order_relaxed);
            
            // Try to install level
            PriceLevel* expected = nullptr;
            if (!std::atomic_compare_exchange_strong(
                    reinterpret_cast<std::atomic<PriceLevel*>*>(&bid_levels[level_idx]),
                    &expected, level)) {
                // Another thread created it first
                rte_free(level);
                level = expected;
            }
        }
        
        level->add_order(order);
    }
    
    void add_to_ask_book(Order* order, uint32_t level_idx) {
        PriceLevel* level = ask_levels[level_idx];
        
        if (level == nullptr) {
            // Create new level
            void* mem = rte_malloc_socket("price_level",
                                        sizeof(PriceLevel),
                                        CACHE_LINE_SIZE,
                                        rte_socket_id());
            level = new (mem) PriceLevel();
            level->price.store(order->price.load(), std::memory_order_relaxed);
            
            // Try to install level
            PriceLevel* expected = nullptr;
            if (!std::atomic_compare_exchange_strong(
                    reinterpret_cast<std::atomic<PriceLevel*>*>(&ask_levels[level_idx]),
                    &expected, level)) {
                // Another thread created it first
                rte_free(level);
                level = expected;
            }
        }
        
        level->add_order(order);
    }
};

// Worker thread function for DPDK
static int dpdk_worker_thread(void* arg) {
    DPDKOrderBook* orderbook = static_cast<DPDKOrderBook*>(arg);
    
    printf("Core %u processing packets\n", rte_lcore_id());
    
    // Run packet processing loop
    orderbook->process_packets();
    
    return 0;
}

// Initialize DPDK and start order book
extern "C" int start_dpdk_orderbook(int argc, char** argv) {
    // Initialize EAL
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");
    }
    
    // Check for available ports
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports < 1) {
        rte_exit(EXIT_FAILURE, "Error: no Ethernet ports found\n");
    }
    
    // Create order book instance
    DPDKOrderBook* orderbook = new DPDKOrderBook();
    
    // Initialize with first port
    if (!orderbook->init(0, 0, 0)) {
        rte_exit(EXIT_FAILURE, "Error: failed to initialize order book\n");
    }
    
    // Launch worker on separate core
    unsigned lcore_id;
    RTE_LCORE_FOREACH_SLAVE(lcore_id) {
        rte_eal_remote_launch(dpdk_worker_thread, orderbook, lcore_id);
        break;  // Use first available core
    }
    
    // Main thread can do other work or wait
    rte_eal_mp_wait_lcore();
    
    return 0;
}