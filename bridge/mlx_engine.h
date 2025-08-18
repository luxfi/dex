#ifndef MLX_ENGINE_H
#define MLX_ENGINE_H

#include <cstdint>
#include <vector>
#include <string>

// Backend types
enum Backend {
    BACKEND_CPU,
    BACKEND_METAL,
    BACKEND_CUDA
};

// Order structure for C++ side
struct MLXOrder {
    uint64_t id;
    double price;
    double size;
    int side; // 0=buy, 1=sell
};

// Trade result
struct MLXTrade {
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    double price;
    double size;
};

class MLXEngine {
public:
    MLXEngine();
    ~MLXEngine();
    
    // Detect best available backend
    Backend detect_backend();
    
    // Get backend name
    std::string get_backend_name();
    
    // Check if GPU acceleration is available
    bool is_gpu_available();
    
    // Batch match orders on GPU
    std::vector<MLXTrade> batch_match(
        const std::vector<MLXOrder>& bids,
        const std::vector<MLXOrder>& asks
    );
    
    // Get device info
    std::string get_device_info();
    
    // Benchmark performance
    double benchmark(int num_orders);
    
private:
    Backend backend_;
    bool gpu_available_;
    std::string device_name_;
    
    // Backend-specific implementations
    std::vector<MLXTrade> match_cpu(
        const std::vector<MLXOrder>& bids,
        const std::vector<MLXOrder>& asks
    );
    
    std::vector<MLXTrade> match_metal(
        const std::vector<MLXOrder>& bids,
        const std::vector<MLXOrder>& asks
    );
    
    std::vector<MLXTrade> match_cuda(
        const std::vector<MLXOrder>& bids,
        const std::vector<MLXOrder>& asks
    );
};

// C interface for CGO
extern "C" {
    void* mlx_engine_create();
    void mlx_engine_destroy(void* engine);
    int mlx_engine_get_backend(void* engine);
    const char* mlx_engine_get_device_name(void* engine);
    int mlx_engine_is_gpu_available(void* engine);
    double mlx_engine_benchmark(void* engine, int num_orders);
    
    // Batch matching interface
    int mlx_engine_match_orders(
        void* engine,
        MLXOrder* bids, int num_bids,
        MLXOrder* asks, int num_asks,
        MLXTrade* trades, int max_trades
    );
}

#endif // MLX_ENGINE_H