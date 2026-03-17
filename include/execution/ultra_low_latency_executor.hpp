#pragma once
// ============================================================
// ArchNeuronX v3 - Ultra Low Latency Execution System
// Lock-free data structures and sub-millisecond execution
// FPGA acceleration and network optimization
// High-frequency trading infrastructure for market domination
// ============================================================

#include <atomic>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <array>
#include <cstring>
#include <immintrin.h>  // For SIMD instructions
#include <numa.h>       // NUMA support
#include <rte_mbuf.h>   // DPDK support
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_tcp.h>

namespace archneuronx {
namespace execution {

/**
 * @brief Lock-free queue implementation
 */
template<typename T, size_t SIZE>
class LockFreeQueue {
private:
    static constexpr size_t MASK = SIZE - 1;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(64) std::array<T, SIZE> buffer_;

public:
    LockFreeQueue() : head_(0), tail_(0) {
        static_assert((SIZE & (SIZE - 1)) == 0, "Size must be power of 2");
    }
    
    bool enqueue(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) & MASK;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool dequeue(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        item = buffer_[current_head];
        head_.store((current_head + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    size_t size() const {
        return (tail_.load() - head_.load() + SIZE) & MASK;
    }
    
    bool empty() const {
        return head_.load() == tail_.load();
    }
    
    bool full() const {
        return ((tail_.load() + 1) & MASK) == head_.load();
    }
};

/**
 * @brief High-performance order structure
 */
struct alignas(64) UltraFastOrder {
    uint64_t order_id;
    uint64_t symbol_id;
    uint64_t exchange_id;
    
    // Price and quantity (fixed-point for speed)
    uint64_t price;      // Price in basis points
    uint64_t quantity;   // Quantity in lots
    
    // Order type and side (bit fields for compactness)
    uint32_t flags;      // Bit 0: side (0=buy, 1=sell), Bit 1: type, etc.
    
    // Timestamps
    uint64_t receive_time;  // When order was received
    uint64_t send_time;     // When order should be sent
    
    // Routing information
    uint32_t venue_priority;
    uint32_t latency_budget_us;
    
    // Padding to cache line size
    char padding[16];
    
    // Fast comparison operators
    bool operator<(const UltraFastOrder& other) const {
        return send_time < other.send_time;
    }
    
    bool operator>(const UltraFastOrder& other) const {
        return send_time > other.send_time;
    }
};

/**
 * @brief Execution result structure
 */
struct alignas(64) ExecutionResult {
    uint64_t order_id;
    uint64_t execution_time_us;
    uint64_t fill_quantity;
    uint64_t fill_price;
    uint32_t status;        // 0=success, 1=rejected, 2=partial, etc.
    uint32_t venue_id;
    char padding[32];
};

/**
 * @brief Memory pool for order objects
 */
class OrderMemoryPool {
private:
    static constexpr size_t POOL_SIZE = 65536;  // 64K orders
    static constexpr size_t ALIGNMENT = 64;
    
    alignas(ALIGNMENT) char memory_pool_[POOL_SIZE * sizeof(UltraFastOrder)];
    std::atomic<uint64_t> allocation_bitmap_[POOL_SIZE / 64];
    std::atomic<size_t> next_free_;
    
public:
    OrderMemoryPool() : next_free_(0) {
        std::memset(memory_pool_, 0, sizeof(memory_pool_));
        std::memset(allocation_bitmap_, 0, sizeof(allocation_bitmap_));
    }
    
    UltraFastOrder* allocate() {
        uint64_t current = next_free_.fetch_add(1, std::memory_order_relaxed);
        if (current >= POOL_SIZE) {
            return nullptr; // Pool exhausted
        }
        
        // Mark as allocated
        uint64_t bitmap_index = current / 64;
        uint64_t bit_index = current % 64;
        allocation_bitmap_[bitmap_index].fetch_or(1ULL << bit_index, std::memory_order_relaxed);
        
        return reinterpret_cast<UltraFastOrder*>(&memory_pool_[current * sizeof(UltraFastOrder)]);
    }
    
    void deallocate(UltraFastOrder* order) {
        if (!order) return;
        
        // Calculate index
        ptrdiff_t offset = reinterpret_cast<char*>(order) - memory_pool_;
        if (offset % sizeof(UltraFastOrder) != 0 || offset < 0 || offset >= POOL_SIZE * sizeof(UltraFastOrder)) {
            return; // Invalid pointer
        }
        
        uint64_t index = offset / sizeof(UltraFastOrder);
        
        // Mark as free
        uint64_t bitmap_index = index / 64;
        uint64_t bit_index = index % 64;
        allocation_bitmap_[bitmap_index].fetch_and(~(1ULL << bit_index), std::memory_order_relaxed);
    }
};

/**
 * @brief Network packet structure for UDP
 */
struct alignas(64) NetworkPacket {
    uint8_t data[1500];      // MTU size
    uint32_t length;
    uint32_t timestamp;
    uint32_t sequence;
    char padding[1444];      // Pad to cache line
};

/**
 * @brief High-resolution timer
 */
class HighResolutionTimer {
private:
    uint64_t start_time_;
    uint64_t frequency_;
    
public:
    HighResolutionTimer() {
        QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&frequency_));
        reset();
    }
    
    void reset() {
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&start_time_));
    }
    
    uint64_t elapsed_microseconds() const {
        uint64_t current_time;
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&current_time_));
        return ((current_time - start_time_) * 1000000) / frequency_;
    }
    
    static uint64_t get_timestamp_microseconds() {
        uint64_t frequency, current_time;
        QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&frequency_));
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&current_time_));
        return (current_time * 1000000) / frequency_;
    }
};

/**
 * @brief NUMA-aware thread affinity manager
 */
class NumaAffinityManager {
public:
    static bool bind_to_cpu(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        
        return (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0);
    }
    
    static bool bind_to_numa_node(int node_id) {
        return (set_mempolicy(MPOL_BIND, node_id, 0, NULL, 0) == 0);
    }
    
    static int get_current_numa_node() {
        return numa_node_of_cpu(sched_getcpu());
    }
};

/**
 * @brief FPGA acceleration interface
 */
class FPGAAccelerator {
private:
    bool fpga_available_;
    void* fpga_handle_;
    
public:
    FPGAAccelerator() : fpga_available_(false), fpga_handle_(nullptr) {
        // Initialize FPGA connection
        initialize_fpga();
    }
    
    ~FPGAAccelerator() {
        if (fpga_handle_) {
            shutdown_fpga();
        }
    }
    
    bool is_available() const { return fpga_available_; }
    
    // FPGA-accelerated operations
    bool accelerate_order_matching(const UltraFastOrder* orders, size_t count);
    bool accelerate_price_calculation(const uint64_t* prices, size_t count);
    bool accelerate_risk_calculation(const UltraFastOrder* orders, size_t count);
    
private:
    void initialize_fpga();
    void shutdown_fpga();
};

/**
 * @brief DPDK-optimized network interface
 */
class DpdkNetworkInterface {
private:
    uint16_t port_id_;
    uint8_t* rx_mbuf_pool_;
    uint8_t* tx_mbuf_pool_;
    struct rte_mempool* rx_pool_;
    struct rte_mempool* tx_pool_;
    
public:
    DpdkNetworkInterface() : port_id_(0), rx_mbuf_pool_(nullptr), tx_mbuf_pool_(nullptr) {}
    
    bool initialize(uint16_t port_id);
    void shutdown();
    
    bool send_packet(const NetworkPacket& packet);
    bool receive_packet(NetworkPacket& packet);
    
    uint64_t get_link_speed() const;
    bool is_link_up() const;
};

/**
 * @brief Ultra Low Latency Executor Configuration
 */
struct UltraLowLatencyConfig {
    // Performance settings
    int num_worker_threads = 8;
    int num_network_threads = 4;
    int order_queue_size = 65536;
    int result_queue_size = 65536;
    
    // Network settings
    bool enable_dpdk = true;
    bool enable_fpga = false;
    bool enable_numa_affinity = true;
    bool enable_simd = true;
    
    // Latency targets
    uint32_t max_order_latency_us = 100;      // 100 microseconds
    uint32_t max_network_latency_us = 50;     // 50 microseconds
    uint32_t max_processing_latency_us = 25;  // 25 microseconds
    
    // Memory settings
    bool enable_huge_pages = true;
    bool enable_memory_pooling = true;
    size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB
    
    // Monitoring
    bool enable_latency_monitoring = true;
    bool enable_throughput_monitoring = true;
    int monitoring_interval_ms = 1000;
};

/**
 * @brief Performance metrics for ultra-low latency system
 */
struct UltraLowLatencyMetrics {
    // Latency metrics
    uint64_t avg_order_latency_us;
    uint64_t max_order_latency_us;
    uint64_t min_order_latency_us;
    uint64_t p95_order_latency_us;
    uint64_t p99_order_latency_us;
    
    // Throughput metrics
    uint64_t orders_per_second;
    uint64_t packets_per_second;
    uint64_t bytes_per_second;
    
    // System metrics
    double cpu_utilization;
    double memory_utilization;
    double network_utilization;
    uint64_t cache_miss_rate;
    
    // Error metrics
    uint64_t dropped_orders;
    uint64_t failed_sends;
    uint64_t failed_receives;
    uint64_t fpga_errors;
    
    std::chrono::system_clock::time_point last_update;
};

/**
 * @brief Ultra Low Latency Execution Engine
 * 
 * Implements sub-millisecond order execution with lock-free
 * data structures, NUMA affinity, FPGA acceleration, and DPDK
 * network optimization for high-frequency trading.
 */
class UltraLowLatencyExecutor {
public:
    explicit UltraLowLatencyExecutor(const UltraLowLatencyConfig& config = UltraLowLatencyConfig{});
    ~UltraLowLatencyExecutor();

    // Initialization
    bool initialize();
    void shutdown();
    bool is_initialized() const;

    // Order submission
    bool submit_order(const UltraFastOrder& order);
    bool submit_orders(const UltraFastOrder* orders, size_t count);
    
    // Result retrieval
    bool get_result(ExecutionResult& result);
    bool get_results(ExecutionResult* results, size_t max_count);
    
    // Performance monitoring
    UltraLowLatencyMetrics get_metrics() const;
    void reset_metrics();
    void start_monitoring();
    void stop_monitoring();

    // Configuration
    void update_config(const UltraLowLatencyConfig& config);
    UltraLowLatencyConfig get_config() const;

    // System status
    bool is_healthy() const;
    std::vector<std::string> get_health_issues() const;

private:
    UltraLowLatencyConfig config_;
    
    // Core components
    OrderMemoryPool order_pool_;
    LockFreeQueue<UltraFastOrder, 65536> order_queue_;
    LockFreeQueue<ExecutionResult, 65536> result_queue_;
    
    // Network interface
    std::unique_ptr<DpdkNetworkInterface> network_interface_;
    
    // FPGA accelerator
    std::unique_ptr<FPGAAccelerator> fpga_accelerator_;
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::vector<std::thread> network_threads_;
    std::thread monitoring_thread_;
    
    // Thread synchronization
    std::atomic<bool> running_;
    std::atomic<bool> monitoring_active_;
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    UltraLowLatencyMetrics metrics_;
    
    // Latency tracking
    HighResolutionTimer timer_;
    std::vector<uint64_t> order_latencies_;
    std::mutex latency_mutex_;
    
    // NUMA affinity
    std::vector<int> cpu_affinity_;
    std::vector<int> numa_affinity_;
    
    // Internal methods
    void initialize_worker_threads();
    void initialize_network_threads();
    void initialize_monitoring();
    void shutdown_threads();
    
    void worker_thread_func(int thread_id);
    void network_thread_func(int thread_id);
    void monitoring_thread_func();
    
    // Core processing
    void process_order(const UltraFastOrder& order);
    void process_orders_batch(const UltraFastOrder* orders, size_t count);
    void send_order_to_exchange(const UltraFastOrder& order);
    void handle_exchange_response(const ExecutionResult& result);
    
    // Performance optimization
    void optimize_cpu_affinity();
    void optimize_memory_layout();
    void optimize_cache_usage();
    
    // FPGA acceleration
    bool use_fpga_for_order_matching(const UltraFastOrder* orders, size_t count);
    bool use_fpga_for_price_calculation(const uint64_t* prices, size_t count);
    
    // Network optimization
    bool send_packet_optimized(const NetworkPacket& packet);
    bool receive_packet_optimized(NetworkPacket& packet);
    
    // Metrics calculation
    void update_metrics();
    void calculate_latency_statistics();
    void calculate_throughput_statistics();
    void calculate_system_utilization();
    
    // Health monitoring
    bool check_system_health();
    std::vector<std::string> diagnose_health_issues();
};

/**
 * @brief SIMD-optimized order processing
 */
class SimdOrderProcessor {
public:
    // SIMD-accelerated order comparison
    static void compare_orders_simd(const UltraFastOrder* orders1, 
                                    const UltraFastOrder* orders2, 
                                    size_t count, 
                                    int* results);
    
    // SIMD-accelerated price calculation
    static void calculate_prices_simd(const uint64_t* prices, 
                                     const uint64_t* quantities, 
                                     size_t count, 
                                     uint64_t* results);
    
    // SIMD-accelerated risk calculation
    static void calculate_risk_simd(const UltraFastOrder* orders, 
                                   size_t count, 
                                   double* risk_scores);

private:
    static bool is_simd_available();
};

/**
 * @brief Cache-optimized data structures
 */
template<typename T, size_t CACHE_LINE_SIZE = 64>
class CacheOptimizedArray {
private:
    static constexpr size_t ELEMENTS_PER_LINE = CACHE_LINE_SIZE / sizeof(T);
    alignas(CACHE_LINE_SIZE) std::array<T, ELEMENTS_PER_LINE> data_;
    
public:
    T& operator[](size_t index) {
        return data_[index % ELEMENTS_PER_LINE];
    }
    
    const T& operator[](size_t index) const {
        return data_[index % ELEMENTS_PER_LINE];
    }
};

/**
 * @brief Lock-free statistics counter
 */
class LockFreeCounter {
private:
    alignas(64) std::atomic<uint64_t> counter_;
    
public:
    LockFreeCounter() : counter_(0) {}
    
    void increment() {
        counter_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void add(uint64_t value) {
        counter_.fetch_add(value, std::memory_order_relaxed);
    }
    
    uint64_t get() const {
        return counter_.load(std::memory_order_relaxed);
    }
    
    void reset() {
        counter_.store(0, std::memory_order_relaxed);
    }
};

/**
 * @brief RAII Ultra Low Latency Executor Context
 */
class UltraLowLatencyContext {
public:
    explicit UltraLowLatencyContext(const UltraLowLatencyConfig& config = UltraLowLatencyConfig{});
    ~UltraLowLatencyContext();
    
    UltraLowLatencyExecutor& get_executor();
    bool is_valid() const;

private:
    std::unique_ptr<UltraLowLatencyExecutor> executor_;
    bool valid_;
};

} // namespace execution
} // namespace archneuronx
