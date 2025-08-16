// Cross-Platform GPU Acceleration for Order Matching
// Supports both Apple MLX (Metal) and NVIDIA CUDA

#include <cstdint>
#include <cstring>
#include <atomic>
#include <vector>
#include <algorithm>

#ifdef __APPLE__
    #define USE_MLX
    #include <Metal/Metal.hpp>
    #include <MetalPerformanceShaders/MetalPerformanceShaders.hpp>
#else
    #define USE_CUDA
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif

// Order structure for GPU processing (must match host structure)
struct GPUOrder {
    uint64_t order_id;
    uint32_t price;      // Fixed point, 7 decimals
    uint32_t quantity;   // Fixed point, 7 decimals
    uint32_t timestamp;
    uint8_t side;        // 0=buy, 1=sell
    uint8_t status;      // 0=active, 1=filled, 2=cancelled
    uint16_t pad;
};

// Trade structure for GPU output
struct GPUTrade {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint32_t price;
    uint32_t quantity;
    uint32_t timestamp;
};

// Statistics structure
struct GPUStats {
    uint64_t orders_processed;
    uint64_t trades_executed;
    uint64_t total_volume;
    uint32_t min_latency_ns;
    uint32_t max_latency_ns;
    uint32_t avg_latency_ns;
};

#ifdef USE_CUDA
// CUDA kernel for parallel order matching
__global__ void match_orders_kernel(
    GPUOrder* bids, int bid_count,
    GPUOrder* asks, int ask_count,
    GPUTrade* trades, int* trade_count,
    int max_trades) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one bid
    if (tid >= bid_count) return;
    
    GPUOrder* bid = &bids[tid];
    if (bid->status != 0) return;  // Skip non-active orders
    
    // Try to match with asks
    for (int i = 0; i < ask_count; i++) {
        GPUOrder* ask = &asks[i];
        
        // Check if orders can match
        if (ask->status != 0) continue;
        if (bid->price < ask->price) continue;
        
        // Calculate trade quantity
        uint32_t trade_qty = min(bid->quantity, ask->quantity);
        if (trade_qty == 0) continue;
        
        // Atomically get trade slot
        int trade_idx = atomicAdd(trade_count, 1);
        if (trade_idx >= max_trades) {
            atomicSub(trade_count, 1);
            break;
        }
        
        // Record trade
        GPUTrade* trade = &trades[trade_idx];
        trade->trade_id = ((uint64_t)bid->order_id << 32) | ask->order_id;
        trade->buy_order_id = bid->order_id;
        trade->sell_order_id = ask->order_id;
        trade->price = ask->price;  // Trade at maker price
        trade->quantity = trade_qty;
        trade->timestamp = max(bid->timestamp, ask->timestamp);
        
        // Update order quantities atomically
        atomicSub(&bid->quantity, trade_qty);
        atomicSub(&ask->quantity, trade_qty);
        
        // Mark as filled if fully matched
        if (bid->quantity == 0) {
            bid->status = 1;
            break;
        }
        if (ask->quantity == 0) {
            ask->status = 1;
        }
    }
}

// CUDA kernel for order book aggregation
__global__ void aggregate_orderbook_kernel(
    GPUOrder* orders, int order_count,
    uint32_t* price_levels, uint32_t* quantities,
    int* level_count, int max_levels) {
    
    __shared__ uint32_t shared_prices[256];
    __shared__ uint32_t shared_quantities[256];
    __shared__ int shared_count;
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid == 0) {
        shared_count = 0;
    }
    __syncthreads();
    
    // Process orders
    if (gid < order_count) {
        GPUOrder* order = &orders[gid];
        if (order->status == 0 && order->quantity > 0) {
            // Find or create price level
            int level_idx = -1;
            for (int i = 0; i < shared_count; i++) {
                if (shared_prices[i] == order->price) {
                    level_idx = i;
                    break;
                }
            }
            
            if (level_idx == -1) {
                level_idx = atomicAdd(&shared_count, 1);
                if (level_idx < 256) {
                    shared_prices[level_idx] = order->price;
                    shared_quantities[level_idx] = 0;
                }
            }
            
            if (level_idx < 256) {
                atomicAdd(&shared_quantities[level_idx], order->quantity);
            }
        }
    }
    __syncthreads();
    
    // Write results to global memory
    if (tid < shared_count && tid < max_levels) {
        int global_idx = atomicAdd(level_count, 1);
        if (global_idx < max_levels) {
            price_levels[global_idx] = shared_prices[tid];
            quantities[global_idx] = shared_quantities[tid];
        }
    }
}
#endif // USE_CUDA

#ifdef USE_MLX
// Metal shader source for order matching
const char* metal_match_shader = R"(
#include <metal_stdlib>
using namespace metal;

struct Order {
    uint64_t order_id;
    uint32_t price;
    uint32_t quantity;
    uint32_t timestamp;
    uint8_t side;
    uint8_t status;
    uint16_t pad;
};

struct Trade {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint32_t price;
    uint32_t quantity;
    uint32_t timestamp;
};

kernel void match_orders(
    device Order* bids [[buffer(0)]],
    device Order* asks [[buffer(1)]],
    device Trade* trades [[buffer(2)]],
    device atomic_int* trade_count [[buffer(3)]],
    constant int& bid_count [[buffer(4)]],
    constant int& ask_count [[buffer(5)]],
    constant int& max_trades [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    
    if (tid >= bid_count) return;
    
    device Order* bid = &bids[tid];
    if (bid->status != 0) return;
    
    for (int i = 0; i < ask_count; i++) {
        device Order* ask = &asks[i];
        
        if (ask->status != 0) continue;
        if (bid->price < ask->price) continue;
        
        uint32_t trade_qty = min(bid->quantity, ask->quantity);
        if (trade_qty == 0) continue;
        
        int trade_idx = atomic_fetch_add_explicit(trade_count, 1, memory_order_relaxed);
        if (trade_idx >= max_trades) {
            atomic_fetch_sub_explicit(trade_count, 1, memory_order_relaxed);
            break;
        }
        
        device Trade* trade = &trades[trade_idx];
        trade->trade_id = ((uint64_t)bid->order_id << 32) | ask->order_id;
        trade->buy_order_id = bid->order_id;
        trade->sell_order_id = ask->order_id;
        trade->price = ask->price;
        trade->quantity = trade_qty;
        trade->timestamp = max(bid->timestamp, ask->timestamp);
        
        atomic_fetch_sub_explicit((device atomic_uint*)&bid->quantity, 
                                  trade_qty, memory_order_relaxed);
        atomic_fetch_sub_explicit((device atomic_uint*)&ask->quantity, 
                                  trade_qty, memory_order_relaxed);
        
        if (bid->quantity == 0) {
            bid->status = 1;
            break;
        }
        if (ask->quantity == 0) {
            ask->status = 1;
        }
    }
}
)";
#endif // USE_MLX

// Cross-platform GPU matching engine
class GPUMatchingEngine {
private:
    static constexpr size_t MAX_ORDERS = 1000000;
    static constexpr size_t MAX_TRADES = 500000;
    static constexpr size_t MAX_PRICE_LEVELS = 10000;
    
#ifdef USE_CUDA
    // CUDA resources
    GPUOrder* d_bids;
    GPUOrder* d_asks;
    GPUTrade* d_trades;
    int* d_trade_count;
    uint32_t* d_price_levels;
    uint32_t* d_quantities;
    int* d_level_count;
    
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
#endif

#ifdef USE_MLX
    // Metal resources
    MTL::Device* device;
    MTL::CommandQueue* command_queue;
    MTL::ComputePipelineState* match_pipeline;
    MTL::Buffer* bid_buffer;
    MTL::Buffer* ask_buffer;
    MTL::Buffer* trade_buffer;
    MTL::Buffer* trade_count_buffer;
#endif
    
    // Host memory
    std::vector<GPUOrder> h_bids;
    std::vector<GPUOrder> h_asks;
    std::vector<GPUTrade> h_trades;
    
    // Statistics
    GPUStats stats;
    
public:
    GPUMatchingEngine() {
        initialize();
    }
    
    ~GPUMatchingEngine() {
        cleanup();
    }
    
    bool initialize() {
#ifdef USE_CUDA
        // Check for CUDA device
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            return false;
        }
        
        // Set device and print info
        cudaSetDevice(0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Using CUDA device: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  SM count: %d\n", prop.multiProcessorCount);
        
        // Allocate device memory
        cudaMalloc(&d_bids, MAX_ORDERS * sizeof(GPUOrder));
        cudaMalloc(&d_asks, MAX_ORDERS * sizeof(GPUOrder));
        cudaMalloc(&d_trades, MAX_TRADES * sizeof(GPUTrade));
        cudaMalloc(&d_trade_count, sizeof(int));
        cudaMalloc(&d_price_levels, MAX_PRICE_LEVELS * sizeof(uint32_t));
        cudaMalloc(&d_quantities, MAX_PRICE_LEVELS * sizeof(uint32_t));
        cudaMalloc(&d_level_count, sizeof(int));
        
        // Create stream for async operations
        cudaStreamCreate(&stream);
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        
        // Initialize counters
        cudaMemset(d_trade_count, 0, sizeof(int));
        cudaMemset(d_level_count, 0, sizeof(int));
#endif

#ifdef USE_MLX
        // Get default Metal device
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            return false;
        }
        
        printf("Using Metal device: %s\n", device->name()->utf8String());
        
        // Create command queue
        command_queue = device->newCommandQueue();
        
        // Compile shader
        NS::Error* error = nullptr;
        MTL::Library* library = device->newLibrary(
            NS::String::string(metal_match_shader, NS::UTF8StringEncoding),
            nullptr, &error);
        
        if (!library) {
            printf("Failed to compile Metal shader\n");
            return false;
        }
        
        MTL::Function* match_function = library->newFunction(
            NS::String::string("match_orders", NS::UTF8StringEncoding));
        
        match_pipeline = device->newComputePipelineState(match_function, &error);
        
        // Allocate Metal buffers
        bid_buffer = device->newBuffer(MAX_ORDERS * sizeof(GPUOrder),
                                       MTL::ResourceStorageModeShared);
        ask_buffer = device->newBuffer(MAX_ORDERS * sizeof(GPUOrder),
                                       MTL::ResourceStorageModeShared);
        trade_buffer = device->newBuffer(MAX_TRADES * sizeof(GPUTrade),
                                        MTL::ResourceStorageModeShared);
        trade_count_buffer = device->newBuffer(sizeof(int),
                                              MTL::ResourceStorageModeShared);
        
        // Initialize counters
        *(int*)trade_count_buffer->contents() = 0;
#endif
        
        // Reserve host memory
        h_bids.reserve(MAX_ORDERS);
        h_asks.reserve(MAX_ORDERS);
        h_trades.resize(MAX_TRADES);
        
        return true;
    }
    
    void cleanup() {
#ifdef USE_CUDA
        cudaFree(d_bids);
        cudaFree(d_asks);
        cudaFree(d_trades);
        cudaFree(d_trade_count);
        cudaFree(d_price_levels);
        cudaFree(d_quantities);
        cudaFree(d_level_count);
        
        cudaStreamDestroy(stream);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
#endif

#ifdef USE_MLX
        if (bid_buffer) bid_buffer->release();
        if (ask_buffer) ask_buffer->release();
        if (trade_buffer) trade_buffer->release();
        if (trade_count_buffer) trade_count_buffer->release();
        if (match_pipeline) match_pipeline->release();
        if (command_queue) command_queue->release();
        if (device) device->release();
#endif
    }
    
    // Match orders on GPU
    int match_orders(const std::vector<GPUOrder>& bids,
                     const std::vector<GPUOrder>& asks,
                     std::vector<GPUTrade>& trades) {
        
        int bid_count = bids.size();
        int ask_count = asks.size();
        int trade_count = 0;
        
        if (bid_count == 0 || ask_count == 0) {
            return 0;
        }
        
#ifdef USE_CUDA
        // Record start time
        cudaEventRecord(start_event, stream);
        
        // Copy orders to GPU
        cudaMemcpyAsync(d_bids, bids.data(), 
                       bid_count * sizeof(GPUOrder),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_asks, asks.data(),
                       ask_count * sizeof(GPUOrder),
                       cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(d_trade_count, 0, sizeof(int), stream);
        
        // Launch kernel
        int threads_per_block = 256;
        int blocks = (bid_count + threads_per_block - 1) / threads_per_block;
        
        match_orders_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_bids, bid_count,
            d_asks, ask_count,
            d_trades, d_trade_count,
            MAX_TRADES);
        
        // Copy results back
        cudaMemcpyAsync(&trade_count, d_trade_count, sizeof(int),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (trade_count > 0) {
            trades.resize(trade_count);
            cudaMemcpy(trades.data(), d_trades,
                      trade_count * sizeof(GPUTrade),
                      cudaMemcpyDeviceToHost);
        }
        
        // Record stop time and calculate elapsed
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        
        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        
        // Update statistics
        stats.orders_processed += bid_count + ask_count;
        stats.trades_executed += trade_count;
        stats.avg_latency_ns = (uint32_t)(elapsed_ms * 1000000 / bid_count);
#endif

#ifdef USE_MLX
        // Copy orders to Metal buffers
        memcpy(bid_buffer->contents(), bids.data(), bid_count * sizeof(GPUOrder));
        memcpy(ask_buffer->contents(), asks.data(), ask_count * sizeof(GPUOrder));
        *(int*)trade_count_buffer->contents() = 0;
        
        // Create command buffer
        MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
        
        // Set pipeline and buffers
        encoder->setComputePipelineState(match_pipeline);
        encoder->setBuffer(bid_buffer, 0, 0);
        encoder->setBuffer(ask_buffer, 0, 1);
        encoder->setBuffer(trade_buffer, 0, 2);
        encoder->setBuffer(trade_count_buffer, 0, 3);
        encoder->setBytes(&bid_count, sizeof(int), 4);
        encoder->setBytes(&ask_count, sizeof(int), 5);
        int max_trades = MAX_TRADES;
        encoder->setBytes(&max_trades, sizeof(int), 6);
        
        // Dispatch threads
        MTL::Size grid_size = MTL::Size(bid_count, 1, 1);
        MTL::Size thread_group_size = MTL::Size(
            std::min(bid_count, (int)match_pipeline->maxTotalThreadsPerThreadgroup()),
            1, 1);
        
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        
        // Execute and wait
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        
        // Get results
        trade_count = *(int*)trade_count_buffer->contents();
        if (trade_count > 0) {
            trades.resize(trade_count);
            memcpy(trades.data(), trade_buffer->contents(),
                  trade_count * sizeof(GPUTrade));
        }
        
        // Update statistics
        stats.orders_processed += bid_count + ask_count;
        stats.trades_executed += trade_count;
#endif
        
        return trade_count;
    }
    
    // Aggregate order book on GPU
    void aggregate_orderbook(const std::vector<GPUOrder>& orders,
                            std::vector<std::pair<uint32_t, uint32_t>>& levels) {
#ifdef USE_CUDA
        int order_count = orders.size();
        if (order_count == 0) return;
        
        // Copy orders to GPU
        cudaMemcpyAsync(d_bids, orders.data(),
                       order_count * sizeof(GPUOrder),
                       cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(d_level_count, 0, sizeof(int), stream);
        
        // Launch aggregation kernel
        int threads_per_block = 256;
        int blocks = (order_count + threads_per_block - 1) / threads_per_block;
        
        aggregate_orderbook_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_bids, order_count,
            d_price_levels, d_quantities,
            d_level_count, MAX_PRICE_LEVELS);
        
        // Get results
        int level_count;
        cudaMemcpyAsync(&level_count, d_level_count, sizeof(int),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (level_count > 0) {
            std::vector<uint32_t> prices(level_count);
            std::vector<uint32_t> quantities(level_count);
            
            cudaMemcpy(prices.data(), d_price_levels,
                      level_count * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
            cudaMemcpy(quantities.data(), d_quantities,
                      level_count * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
            
            levels.clear();
            for (int i = 0; i < level_count; i++) {
                levels.push_back({prices[i], quantities[i]});
            }
        }
#endif
        
        // MLX implementation would be similar
    }
    
    // Get performance statistics
    GPUStats get_stats() const {
        return stats;
    }
    
    // Get device info
    std::string get_device_info() const {
#ifdef USE_CUDA
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        return std::string(prop.name) + " (CUDA " + 
               std::to_string(prop.major) + "." + 
               std::to_string(prop.minor) + ")";
#endif
#ifdef USE_MLX
        return std::string(device->name()->utf8String()) + " (Metal)";
#endif
        return "No GPU";
    }
};

// C interface for integration with Go
extern "C" {
    void* create_gpu_engine() {
        GPUMatchingEngine* engine = new GPUMatchingEngine();
        if (!engine->initialize()) {
            delete engine;
            return nullptr;
        }
        return engine;
    }
    
    void destroy_gpu_engine(void* engine) {
        delete static_cast<GPUMatchingEngine*>(engine);
    }
    
    int gpu_match_orders(void* engine,
                        GPUOrder* bids, int bid_count,
                        GPUOrder* asks, int ask_count,
                        GPUTrade* trades, int max_trades) {
        GPUMatchingEngine* gpu = static_cast<GPUMatchingEngine*>(engine);
        
        std::vector<GPUOrder> bid_vec(bids, bids + bid_count);
        std::vector<GPUOrder> ask_vec(asks, asks + ask_count);
        std::vector<GPUTrade> trade_vec;
        
        int trade_count = gpu->match_orders(bid_vec, ask_vec, trade_vec);
        
        if (trade_count > 0 && trade_count <= max_trades) {
            memcpy(trades, trade_vec.data(), trade_count * sizeof(GPUTrade));
        }
        
        return trade_count;
    }
    
    void gpu_get_stats(void* engine, GPUStats* stats) {
        GPUMatchingEngine* gpu = static_cast<GPUMatchingEngine*>(engine);
        *stats = gpu->get_stats();
    }
    
    const char* gpu_get_device_info(void* engine) {
        GPUMatchingEngine* gpu = static_cast<GPUMatchingEngine*>(engine);
        static std::string info = gpu->get_device_info();
        return info.c_str();
    }
}