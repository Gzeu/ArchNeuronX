#pragma once
// ============================================================
// ArchNeuronX v3 - Colocation and Infrastructure Optimization
// Low-latency infrastructure for high-frequency trading
// Network optimization, NUMA tuning, and hardware acceleration
// Production deployment and scaling strategies
// ============================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <numa.h>
#include <sched.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <rte_eal.h>
#include <rte_mbuf.h>
#include <rte_ethdev.h>

namespace archneuronx {
namespace infrastructure {

/**
 * @brief Server node configuration
 */
struct ServerNode {
    std::string node_id;
    std::string hostname;
    std::string ip_address;
    std::vector<std::string> network_interfaces;
    
    // Hardware specifications
    int cpu_cores;
    int cpu_frequency_mhz;
    int memory_gb;
    std::vector<std::string> gpus;
    std::vector<std::string> fpgas;
    
    // Network specifications
    std::vector<std::string> exchange_connections;
    double network_bandwidth_gbps;
    double latency_to_exchange_us;
    
    // Location
    std::string data_center;
    std::string rack;
    std::string row;
    std::string cabinet;
    
    // Status
    bool is_active;
    bool is_healthy;
    double cpu_utilization;
    double memory_utilization;
    double network_utilization;
    std::chrono::system_clock::time_point last_health_check;
};

/**
 *brief Network optimization configuration
 */
struct NetworkConfig {
    // DPDK settings
    bool enable_dpdk = true;
    int num_rx_queues = 8;
    int num_tx_queues = 8;
    int rx_queue_size = 4096;
    int tx_queue_size = 4096;
    int mbuf_pool_size = 8192;
    int burst_size = 32;
    
    // Network tuning
    bool enable_rss = true;
    bool enable_tso = true;
    bool enable_tx_offload = true;
    bool enable_rx_csum_offload = true;
    
    // Performance settings
    int rx_descriptor_size = 1024;
    int tx_descriptor_size = 1024;
    int rx_free_thresh = 32;
    int tx_free_thresh = 32;
    
    // Hardware acceleration
    bool enable_rss_acceleration = true;
    bool enable_tx_acceleration = true;
    bool enable_vec_mbuf = true;
    
    // NUMA settings
    bool enable_numa = true;
    int preferred_numa_node = 0;
    bool enable_cpu_affinity = true;
    std::vector<int> cpu_affinity_mask;
    
    // CPU frequency scaling
    bool enable_turbo_boost = true;
    bool enable_pstate = true;
    int max_performance_pstate = 0;
    
    // Hyper-threading
    bool enable_hyperthreading = true;
    int num_hyper_threads_per_core = 2;
    
    // Power management
    bool enable_cpu_scaling = true;
    int min_performance_pstate = 0;
    int max_performance_pstate = 0;
    
    // Instruction optimization
    bool enable_simd = true;
    bool enable_vector_instructions = true;
    bool enable_prefetch_instructions = true;
    
    // Monitoring
    bool enable_cpu_monitoring = true;
    int cpu_update_interval_ms = 1000;
};

/**
 *brief Memory optimization configuration
 */
struct MemoryConfig {
    // Huge pages
    bool enable_huge_pages = true;
    size_t huge_page_size = 2 * 1024 * 1024; // 2MB
    
    // Memory allocation
    size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB
    bool enable_memory_pooling = true;
    bool enable_cache_alignment = true;
    int cache_line_size = 64;
    
    // NUMA optimization
    bool enable_numa_allocation = true;
    int preferred_numa_node = 0;
    
    // Lock-free structures
    bool enable_lock_free = true;
    size_t lock_free_pool_size = 65536;
    
    // Prefetching
    bool enable_prefetching = true;
    int prefetch_distance = 4;
    
    // Monitoring
    bool enable_memory_monitoring = true;
    int memory_update_interval_ms = 1000;
};

/**
 *brief CPU optimization configuration
 */
struct CPUConfig {
    // CPU affinity
    bool enable_cpu_affinity = true;
    bool enable_thread_priority = true;
    std::vector<int> worker_cpu_mask;
    std::vector<int> network_cpu_mask;
    std::vector<int> monitoring_cpu_mask;
    
    // CPU frequency scaling
    bool enable_turbo_boost = true;
    bool enable_pstate = true;
    int max_performance_pstate = 0;
    
    // Hyper-threading
    bool enable_hyperthreading = true;
    int num_hyper_threads_per_core = 2;
    
    // Power management
    bool enable_cpu_scaling = true;
    int min_performance_pstate = 0;
    int max_performance_pstate = 0;
    
    // Instruction optimization
    bool enable_simd = true;
    bool enable_vector_instructions = true;
    bool enable_prefetch_instructions = true;
    
    // Monitoring
    bool enable_cpu_monitoring = true;
    int cpu_update_interval_ms = 1000;
};

/**
 *brief Storage optimization configuration
 */
struct StorageConfig {
    // SSD optimization
    bool enable_nvme = true;
    bool enable_trim = true;
    bool enable_write_back_cache = true;
    int write_back_cache_size_gb = 4;
    
    // I/O optimization
    bool enable_io_uring = true;
    int io_queue_depth = 256;
    int scheduler_deadline = 0;
    
    // File system
    bool enable_direct_io = true;
    bool enable_async_io = true;
    size_t read_ahead_size = 1024 * 1024; // 1MB
    
    // Monitoring
    bool enable_storage_monitoring = true;
    int storage_update_interval_ms = 5000;
};

/**
 *brief Colocation configuration
 */
struct ColocationConfig {
    // Exchange locations
    std::unordered_map<std::string, std::string> exchange_locations;
    
    // Latency targets
    std::unordered_map<std::string, double> latency_targets_us;
    std::unordered_map<std::string, std::string> preferred_venues;
    
    // Network topology
    bool enable_mesh_network = true;
    std::vector<std::string> network_topology;
    double network_redundancy = 2.0;
    
    // Data center selection
    std::vector<std::string> preferred_data_centers;
    std::string primary_data_center;
    std::string backup_data_center;
    
    // Compliance
    bool enable_compliance_monitoring = true;
    std::vector<std::string> regulatory_requirements;
    
    // Cost optimization
    bool enable_cost_optimization = true;
    double max_monthly_budget = 10000.0;
    std::unordered_map<std::string, double> exchange_costs;
};

/**
 *brief Infrastructure optimization configuration
 */
struct InfrastructureConfig {
    // Overall settings
    std::string environment = "production";
    bool enable_auto_optimization = true;
    bool enable_monitoring = true;
    int optimization_interval_hours = 6;
    
    // Component configurations
    NetworkConfig network_config;
    MemoryConfig memory_config;
    CPUConfig cpu_config;
    StorageConfig storage_config;
    ColocationConfig colocation_config;
    
    // Performance targets
    double target_latency_us = 50.0;
    double target_throughput_mbps = 10000.0;
    double target_cpu_utilization = 0.7;
    double target_memory_utilization = 0.8;
    
    // Scaling settings
    int max_nodes = 10;
    bool enable_horizontal_scaling = true;
    bool enable_vertical_scaling = true;
    bool enable_auto_scaling = true;
    double scale_up_threshold = 0.8;
    double scale_down_threshold = 0.3;
    
    // Monitoring
    bool enable_performance_monitoring = true;
    bool enable_health_monitoring = true;
    bool enable_alerting = true;
    int monitoring_interval_ms = 1000;
    
    // Compliance
    bool enable_audit_logging = true;
    std::vector<std::string> compliance_requirements;
    bool enable_encryption = true;
    std::string encryption_level = "AES-256";
};

/**
 * @brief Performance metrics for infrastructure
 */
struct InfrastructureMetrics {
    // Latency metrics
    double avg_latency_us;
    double p50_latency_us;
    double p95_latency_us;
    double p99_latency_us;
    double max_latency_us;
    
    // Throughput metrics
    double packets_per_second;
    double bytes_per_second;
    double orders_per_second;
    double trades_per_second;
    
    // Resource utilization
    double cpu_utilization;
    double memory_utilization;
    double network_utilization;
    double storage_utilization;
    
    // Error rates
    double packet_loss_rate;
    double order_rejection_rate;
    double system_error_rate;
    
    // System health
    double system_health_score;
    std::vector<std::string> health_issues;
    
    std::chrono::system_clock::time_point last_update;
};

/**
 * @brief Infrastructure Optimizer
 * 
 * Optimizes trading infrastructure for maximum performance and reliability.
 * Implements colocation strategy, network optimization, and resource
 * management for high-frequency trading environments.
 */
class InfrastructureOptimizer {
public:
    explicit InfrastructureOptimizer(const InfrastructureConfig& config = InfrastructureConfig{});
    ~InfrastructureOptimizer();

    // Initialization
    bool initialize();
    void shutdown();
    bool is_initialized() const;

    // Node management
    bool add_node(const ServerNode& node);
    bool remove_node(const std::string& node_id);
    std::vector<ServerNode> get_active_nodes() const;
    ServerNode get_node(const std::string& node_id) const;
    
    // Network optimization
    bool optimize_network_settings();
    bool configure_dpdk(const NetworkConfig& config);
    bool optimize_network_topology();
    std::vector<std::string> get_network_topology() const;
    
    // Memory optimization
    bool optimize_memory_settings();
    bool configure_huge_pages();
    bool optimize_numa_affinity();
    void optimize_cache_settings();
    
    // CPU optimization
    bool optimize_cpu_settings();
    bool configure_cpu_affinity();
    bool configure_power_management();
    void optimize_cpu_frequency();
    
    // Storage optimization
    bool optimize_storage_settings();
    bool configure_nvme_settings();
    void optimize_io_scheduler();
    
    // Colocation optimization
    bool optimize_colocation_strategy();
    std::vector<std::string> get_colocation_recommendations() const;
    double calculate_latency_improvement(const std::string& from_location, const std::string& to_location);
    
    // Auto-optimization
    void start_auto_optimization();
    void stop_auto_optimization();
    bool run_optimization_cycle();
    
    // Performance monitoring
    InfrastructureMetrics get_metrics() const;
    void reset_metrics();
    std::vector<std::string> get_performance_insights() const;
    std::vector<std::string> get_optimization_recommendations() const;
    
    // Health monitoring
    bool is_healthy() const;
    std::vector<std::string> get_health_issues() const;
    void run_health_check();
    
    // Configuration management
    void update_config(const InfrastructureConfig& config);
    InfrastructureConfig get_config() const;

private:
    InfrastructureConfig config_;
    
    // Node management
    std::unordered_map<std::string, ServerNode> nodes_;
    mutable std::mutex nodes_mutex_;
    
    // Network state
    std::vector<std::string> network_topology_;
    std::unordered_map<std::string, double> network_latencies_;
    mutable std::mutex network_mutex_;
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    InfrastructureMetrics metrics_;
    
    // Threading
    std::atomic<bool> running_;
    std::vector<std::thread> optimization_threads_;
    std::thread monitoring_thread_;
    
    // Auto-optimization
    std::atomic<bool> auto_optimization_active_;
    std::thread auto_optimization_thread_;
    
    // Internal methods
    void initialize_background_threads();
    void shutdown_background_threads();
    void optimization_thread_func();
    void monitoring_thread_func();
    void auto_optimization_thread_func();
    
    // Optimization methods
    void optimize_dpdk_settings();
    void optimize_memory_layout();
    void optimize_cpu_topology();
    optimize_network_routing();
    optimize_storage_layout();
    optimize_colocation_placement();
    
    // Performance analysis
    void analyze_performance_bottlenecks();
    void analyze_resource_utilization();
    void analyze_latency_patterns();
    void identify_optimization_opportunities();
    
    // Health monitoring
    void check_system_health();
    std::vector<std::string> diagnose_health_issues();
    
    // Utility methods
    bool setup_dpdk_environment();
    bool setup_huge_pages();
    bool setup_numa_affinity();
    bool configure_cpu_frequency_scaling();
    void calculate_optimal_cpu_affinity();
    void calculate_optimal_network_topology();
    void calculate_optimal_colocation();
    
    // Monitoring utilities
    void update_metrics();
    void calculate_latency_statistics();
    void calculate_throughputput_statistics();
    void calculate_resource_utilization();
    void calculate_system_health_score();
    
    // Auto-optimization logic
    void analyze_performance_trends();
    void generate_optimization_recommendations();
    void apply_optimization(const std::string& recommendation);
};

/**
 * @brief RAII Infrastructure Optimizer Context
 */
class InfrastructureOptimizerContext {
public:
    explicit InfrastructureOptimizerContext(const InfrastructureConfig& config = InfrastructureConfig{});
    ~InfrastructureOptimizerContext();
    
    InfrastructureOptimizer& get_optimizer();
    bool is_valid() const;

private:
    std::unique_ptr<InfrastructureOptimizer> optimizer_;
    bool valid_;
};

} // namespace infrastructure
} // namespace archneuronx
