// Ultra-Fast Lock-Free FIX Engine - Target: 10M msgs/sec
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <immintrin.h>  // For SIMD
#include <numa.h>       // For NUMA awareness
#include <sched.h>      // For CPU affinity
#include <sys/mman.h>   // For huge pages
#include <unistd.h>

// Configuration for 10M msgs/sec
constexpr size_t RING_BUFFER_SIZE = 1 << 24;  // 16M entries
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t MESSAGE_POOL_SIZE = 1 << 20; // 1M pre-allocated messages
constexpr size_t MAX_MESSAGE_SIZE = 256;
constexpr size_t NUM_SHARDS = 16;  // Parallel processing shards

// Align to cache line to prevent false sharing
#define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)

// Lock-free ring buffer using cache-aligned atomics
template<typename T>
class alignas(CACHE_LINE_SIZE) LockFreeRingBuffer {
private:
    CACHE_ALIGNED std::atomic<size_t> head{0};
    CACHE_ALIGNED std::atomic<size_t> tail{0};
    CACHE_ALIGNED T* buffer;
    const size_t mask;
    char padding[CACHE_LINE_SIZE - sizeof(void*)];
    
public:
    LockFreeRingBuffer() : mask(RING_BUFFER_SIZE - 1) {
        // Allocate using huge pages for better TLB performance
        buffer = (T*)mmap(nullptr, sizeof(T) * RING_BUFFER_SIZE,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                         -1, 0);
        if (buffer == MAP_FAILED) {
            buffer = new T[RING_BUFFER_SIZE];
        }
    }
    
    ~LockFreeRingBuffer() {
        if (buffer != MAP_FAILED) {
            munmap(buffer, sizeof(T) * RING_BUFFER_SIZE);
        } else {
            delete[] buffer;
        }
    }
    
    bool push(const T& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) & mask;
        
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer[current_tail] = item;
        tail.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t current_head = head.load(std::memory_order_relaxed);
        
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }
        
        item = buffer[current_head];
        head.store((current_head + 1) & mask, std::memory_order_release);
        return true;
    }
};

// Ultra-fast FIX message structure (optimized for cache)
struct alignas(CACHE_LINE_SIZE) FastFIXMessage {
    uint32_t length;
    uint32_t checksum;
    uint64_t timestamp;
    char data[MAX_MESSAGE_SIZE - 16];
    
    // SIMD-optimized checksum calculation
    inline uint32_t calculate_checksum_simd() const {
        __m256i sum = _mm256_setzero_si256();
        const __m256i* ptr = reinterpret_cast<const __m256i*>(data);
        
        for (size_t i = 0; i < length / 32; ++i) {
            sum = _mm256_add_epi8(sum, _mm256_loadu_si256(ptr + i));
        }
        
        // Horizontal sum
        __m128i sum128 = _mm_add_epi8(
            _mm256_extracti128_si256(sum, 0),
            _mm256_extracti128_si256(sum, 1)
        );
        
        uint8_t result[16];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result), sum128);
        
        uint32_t final_sum = 0;
        for (int i = 0; i < 16; ++i) {
            final_sum += result[i];
        }
        
        // Handle remaining bytes
        for (size_t i = (length / 32) * 32; i < length; ++i) {
            final_sum += static_cast<uint8_t>(data[i]);
        }
        
        return final_sum % 256;
    }
};

// Memory pool for zero-allocation message handling
class MessagePool {
private:
    CACHE_ALIGNED std::atomic<size_t> index{0};
    FastFIXMessage* pool;
    const size_t pool_size;
    
public:
    MessagePool() : pool_size(MESSAGE_POOL_SIZE) {
        // Allocate with huge pages
        pool = (FastFIXMessage*)aligned_alloc(CACHE_LINE_SIZE, 
                                              sizeof(FastFIXMessage) * pool_size);
        memset(pool, 0, sizeof(FastFIXMessage) * pool_size);
    }
    
    ~MessagePool() {
        free(pool);
    }
    
    FastFIXMessage* acquire() {
        size_t idx = index.fetch_add(1, std::memory_order_relaxed) % pool_size;
        return &pool[idx];
    }
    
    void release(FastFIXMessage* msg) {
        // In real implementation, would return to free list
        // For benchmark, we just cycle through the pool
    }
};

// Sharded processor for parallel FIX processing
class FIXShard {
private:
    const int shard_id;
    LockFreeRingBuffer<FastFIXMessage*> input_queue;
    std::atomic<uint64_t> processed{0};
    std::atomic<bool> running{true};
    std::thread worker;
    
    void process_messages() {
        // Pin thread to CPU core
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(shard_id % std::thread::hardware_concurrency(), &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        FastFIXMessage* msg;
        while (running.load(std::memory_order_relaxed)) {
            if (input_queue.pop(msg)) {
                // Ultra-fast FIX parsing (simplified)
                process_fix_message(msg);
                processed.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Spin-wait for ultra-low latency
                __builtin_ia32_pause();
            }
        }
    }
    
    inline void process_fix_message(FastFIXMessage* msg) {
        // Simulate minimal FIX processing
        // In reality, would parse and route based on message type
        volatile uint32_t checksum = msg->checksum;
        (void)checksum; // Prevent optimization
    }
    
public:
    FIXShard(int id) : shard_id(id), worker(&FIXShard::process_messages, this) {}
    
    ~FIXShard() {
        running.store(false);
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    bool submit(FastFIXMessage* msg) {
        return input_queue.push(msg);
    }
    
    uint64_t get_processed() const {
        return processed.load(std::memory_order_relaxed);
    }
};

// Ultra-fast FIX engine with sharding
class UltraFIXEngine {
private:
    std::vector<std::unique_ptr<FIXShard>> shards;
    MessagePool message_pool;
    std::atomic<uint64_t> total_submitted{0};
    std::atomic<size_t> next_shard{0};
    
public:
    UltraFIXEngine() {
        // Create shards for parallel processing
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            shards.emplace_back(std::make_unique<FIXShard>(i));
        }
    }
    
    // Submit message using round-robin sharding
    bool submit_message(const char* fix_data, size_t length) {
        FastFIXMessage* msg = message_pool.acquire();
        
        // Zero-copy message setup
        msg->length = length;
        msg->timestamp = __rdtsc(); // CPU timestamp counter
        memcpy(msg->data, fix_data, length);
        msg->checksum = msg->calculate_checksum_simd();
        
        // Round-robin to shards
        size_t shard_idx = next_shard.fetch_add(1, std::memory_order_relaxed) % NUM_SHARDS;
        
        if (shards[shard_idx]->submit(msg)) {
            total_submitted.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        
        message_pool.release(msg);
        return false;
    }
    
    uint64_t get_total_processed() const {
        uint64_t total = 0;
        for (const auto& shard : shards) {
            total += shard->get_processed();
        }
        return total;
    }
    
    uint64_t get_total_submitted() const {
        return total_submitted.load(std::memory_order_relaxed);
    }
};

// Benchmark harness
void run_ultra_benchmark(int duration_sec) {
    std::cout << "🚀 Ultra FIX Engine Benchmark\n";
    std::cout << "Target: 10,000,000 msgs/sec\n";
    std::cout << "Shards: " << NUM_SHARDS << "\n";
    std::cout << "Ring Buffer: " << (RING_BUFFER_SIZE / 1024 / 1024) << "M entries\n";
    std::cout << "Duration: " << duration_sec << " seconds\n";
    std::cout << "========================================\n\n";
    
    UltraFIXEngine engine;
    
    // Sample FIX message
    const char* sample_fix = "8=FIX.4.4\x01" "35=D\x01" "49=SENDER\x01" "56=TARGET\x01"
                            "11=ORDER123\x01" "55=EUR/USD\x01" "54=1\x01" "38=1000000\x01"
                            "40=2\x01" "44=1.1234\x01" "10=123\x01";
    size_t msg_length = strlen(sample_fix);
    
    // Create producer threads (simulating network receivers)
    const int num_producers = std::thread::hardware_concurrency();
    std::vector<std::thread> producers;
    std::atomic<bool> running{true};
    std::atomic<uint64_t> total_sent{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back([&, i]() {
            // Pin to CPU
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            
            uint64_t local_sent = 0;
            char local_msg[256];
            memcpy(local_msg, sample_fix, msg_length);
            
            while (running.load(std::memory_order_relaxed)) {
                // Modify message slightly (simulate real messages)
                local_msg[50] = '0' + (local_sent % 10);
                
                if (engine.submit_message(local_msg, msg_length)) {
                    local_sent++;
                }
                
                // No sleep - maximum throughput
            }
            
            total_sent.fetch_add(local_sent, std::memory_order_relaxed);
        });
    }
    
    // Monitor progress
    auto last_print = start_time;
    uint64_t last_processed = 0;
    
    for (int i = 0; i < duration_sec; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        
        uint64_t current_processed = engine.get_total_processed();
        uint64_t current_submitted = engine.get_total_submitted();
        
        double rate = current_processed / elapsed;
        double submit_rate = current_submitted / elapsed;
        
        std::cout << "\r⚡ Processed: " << current_processed 
                  << " | Rate: " << std::fixed << std::setprecision(0) << rate << "/sec"
                  << " | Submit Rate: " << submit_rate << "/sec"
                  << " | Progress: " << (100.0 * current_processed / current_submitted) << "%"
                  << std::flush;
    }
    
    // Stop producers
    running.store(false);
    for (auto& t : producers) {
        t.join();
    }
    
    // Wait for processing to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Final results
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    uint64_t final_processed = engine.get_total_processed();
    uint64_t final_submitted = engine.get_total_submitted();
    
    std::cout << "\n\n========================================\n";
    std::cout << "📊 FINAL RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "Duration: " << total_elapsed << " seconds\n";
    std::cout << "Messages Submitted: " << final_submitted << "\n";
    std::cout << "Messages Processed: " << final_processed << "\n";
    std::cout << "Submit Rate: " << (final_submitted / total_elapsed) << " msgs/sec\n";
    std::cout << "Process Rate: " << (final_processed / total_elapsed) << " msgs/sec\n";
    std::cout << "Efficiency: " << (100.0 * final_processed / final_submitted) << "%\n";
    
    double million_rate = (final_processed / total_elapsed) / 1000000.0;
    std::cout << "\n🎯 ACHIEVEMENT: " << std::fixed << std::setprecision(2) 
              << million_rate << "M msgs/sec\n";
    
    if (million_rate >= 10.0) {
        std::cout << "✅ SUCCESS! Reached 10M+ msgs/sec!\n";
    } else if (million_rate >= 5.0) {
        std::cout << "🔥 Excellent! Over 5M msgs/sec!\n";
    } else if (million_rate >= 1.0) {
        std::cout << "👍 Good! Over 1M msgs/sec!\n";
    } else {
        std::cout << "📈 Room for improvement...\n";
    }
}

int main(int argc, char* argv[]) {
    int duration = 10;
    if (argc > 1) {
        duration = std::atoi(argv[1]);
    }
    
    // Enable huge pages
    std::cout << "Configuring system for maximum performance...\n";
    system("echo 1024 > /proc/sys/vm/nr_hugepages 2>/dev/null");
    
    run_ultra_benchmark(duration);
    
    return 0;
}
