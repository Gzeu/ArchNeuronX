// ============================================================
// ArchNeuronX v2 - Metrics Collector Implementation
// Collects and aggregates performance metrics across the system
// ============================================================
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mutex>
#include <thread>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace archneuronx {
namespace utils {

class MetricsCollector {
public:
    struct MetricPoint {
        std::string name;
        double value;
        std::string unit;
        std::chrono::steady_clock::time_point timestamp;
        std::map<std::string, std::string> tags;
    };

    struct AggregatedMetric {
        std::string name;
        double min_value;
        double max_value;
        double avg_value;
        double sum_value;
        size_t count;
        std::string unit;
        std::chrono::steady_clock::time_point first_seen;
        std::chrono::steady_clock::time_point last_seen;
    };

    static MetricsCollector& instance() {
        static MetricsCollector instance;
        return instance;
    }

    void record_metric(const std::string& name, double value, 
                       const std::string& unit = "",
                       const std::map<std::string, std::string>& tags = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        MetricPoint point{
            name,
            value,
            unit,
            std::chrono::steady_clock::now(),
            tags
        };
        
        metrics_.push_back(point);
        
        // Update aggregation
        update_aggregation(name, value, unit);
        
        // Cleanup old metrics if buffer is full
        if (metrics_.size() > max_metrics_buffer_) {
            metrics_.erase(metrics_.begin(), metrics_.begin() + cleanup_batch_size_);
        }
    }

    void record_latency(const std::string& operation, double latency_us) {
        record_metric(operation + "_latency_us", latency_us, "μs");
    }

    void record_throughput(const std::string& operation, double operations_per_second) {
        record_metric(operation + "_throughput", operations_per_second, "ops/s");
    }

    void record_error_rate(const std::string& operation, double error_rate) {
        record_metric(operation + "_error_rate", error_rate, "%");
    }

    void record_memory_usage(const std::string& component, double memory_mb) {
        record_metric(component + "_memory_mb", memory_mb, "MB");
    }

    void record_gpu_utilization(const std::string& gpu_id, double utilization) {
        record_metric("gpu_utilization", utilization, "%", {{"gpu_id", gpu_id}});
    }

    void record_model_accuracy(const std::string& model_name, double accuracy) {
        record_metric("model_accuracy", accuracy, "%", {{"model", model_name}});
    }

    void record_trading_signal(const std::string& symbol, const std::string& action) {
        record_metric("trading_signals", 1.0, "count", {
            {"symbol", symbol}, 
            {"action", action}
        });
    }

    AggregatedMetric get_aggregated_metric(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = aggregated_metrics_.find(name);
        if (it != aggregated_metrics_.end()) {
            return it->second;
        }
        
        return {};
    }

    std::vector<AggregatedMetric> get_all_aggregated_metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::vector<AggregatedMetric> result;
        result.reserve(aggregated_metrics_.size());
        
        for (const auto& pair : aggregated_metrics_) {
            result.push_back(pair.second);
        }
        
        return result;
    }

    std::vector<MetricPoint> get_recent_metrics(const std::string& name, 
                                               std::chrono::seconds duration = std::chrono::seconds(300)) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto cutoff = std::chrono::steady_clock::now() - duration;
        std::vector<MetricPoint> result;
        
        for (const auto& metric : metrics_) {
            if (metric.name == name && metric.timestamp >= cutoff) {
                result.push_back(metric);
            }
        }
        
        return result;
    }

    void clear_metrics() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.clear();
        aggregated_metrics_.clear();
    }

    void export_to_csv(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for metrics export: " << filename << std::endl;
            return;
        }
        
        // Write header
        file << "timestamp,name,value,unit";
        
        // Write tag headers
        std::set<std::string> all_tags;
        for (const auto& metric : metrics_) {
            for (const auto& tag : metric.tags) {
                all_tags.insert(tag.first);
            }
        }
        
        for (const auto& tag : all_tags) {
            file << "," << tag;
        }
        file << std::endl;
        
        // Write data
        for (const auto& metric : metrics_) {
            auto time_t = std::chrono::duration_cast<std::chrono::milliseconds>(
                metric.timestamp.time_since_epoch()).count();
            
            file << time_t << "," << metric.name << "," << metric.value << "," << metric.unit;
            
            for (const auto& tag : all_tags) {
                auto it = metric.tags.find(tag);
                if (it != metric.tags.end()) {
                    file << "," << it->second;
                } else {
                    file << ",";
                }
            }
            file << std::endl;
        }
        
        file.close();
        std::cout << "Metrics exported to: " << filename << std::endl;
    }

    void start_periodic_export(const std::string& filename, 
                              std::chrono::seconds interval = std::chrono::seconds(60)) {
        stop_periodic_export();
        
        export_running_ = true;
        export_thread_ = std::thread(&MetricsCollector::export_loop, this, filename, interval);
    }

    void stop_periodic_export() {
        if (export_running_) {
            export_running_ = false;
            if (export_thread_.joinable()) {
                export_thread_.join();
            }
        }
    }

    void print_summary() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n=== Metrics Summary ===" << std::endl;
        std::cout << "Total metrics collected: " << metrics_.size() << std::endl;
        std::cout << "Aggregated metrics: " << aggregated_metrics_.size() << std::endl;
        
        for (const auto& pair : aggregated_metrics_) {
            const auto& agg = pair.second;
            std::cout << "\n" << agg.name << " (" << agg.unit << "):" << std::endl;
            std::cout << "  Count: " << agg.count << std::endl;
            std::cout << "  Min: " << agg.min_value << std::endl;
            std::cout << "  Max: " << agg.max_value << std::endl;
            std::cout << "  Avg: " << std::fixed << std::setprecision(2) << agg.avg_value << std::endl;
        }
        std::cout << "=====================\n" << std::endl;
    }

private:
    MetricsCollector() = default;
    ~MetricsCollector() {
        stop_periodic_export();
    }

    mutable std::mutex mutex_;
    std::vector<MetricPoint> metrics_;
    std::map<std::string, AggregatedMetric> aggregated_metrics_;
    
    size_t max_metrics_buffer_ = 100000;
    size_t cleanup_batch_size_ = 10000;
    
    bool export_running_ = false;
    std::thread export_thread_;

    void update_aggregation(const std::string& name, double value, const std::string& unit) {
        auto& agg = aggregated_metrics_[name];
        
        if (agg.count == 0) {
            agg.name = name;
            agg.unit = unit;
            agg.min_value = value;
            agg.max_value = value;
            agg.sum_value = value;
            agg.avg_value = value;
            agg.count = 1;
            agg.first_seen = std::chrono::steady_clock::now();
            agg.last_seen = agg.first_seen;
        } else {
            agg.min_value = std::min(agg.min_value, value);
            agg.max_value = std::max(agg.max_value, value);
            agg.sum_value += value;
            agg.avg_value = agg.sum_value / agg.count;
            agg.count++;
            agg.last_seen = std::chrono::steady_clock::now();
        }
    }

    void export_loop(const std::string& filename, std::chrono::seconds interval) {
        while (export_running_) {
            std::this_thread::sleep_for(interval);
            
            if (export_running_) {
                export_to_csv(filename);
            }
        }
    }
};

// Global convenience functions
void record_metric(const std::string& name, double value, 
                  const std::string& unit,
                  const std::map<std::string, std::string>& tags) {
    MetricsCollector::instance().record_metric(name, value, unit, tags);
}

void record_latency(const std::string& operation, double latency_us) {
    MetricsCollector::instance().record_latency(operation, latency_us);
}

void record_throughput(const std::string& operation, double operations_per_second) {
    MetricsCollector::instance().record_throughput(operation, operations_per_second);
}

void record_model_accuracy(const std::string& model_name, double accuracy) {
    MetricsCollector::instance().record_model_accuracy(model_name, accuracy);
}

void record_trading_signal(const std::string& symbol, const std::string& action) {
    MetricsCollector::instance().record_trading_signal(symbol, action);
}

void export_metrics_to_csv(const std::string& filename) {
    MetricsCollector::instance().export_to_csv(filename);
}

void print_metrics_summary() {
    MetricsCollector::instance().print_summary();
}

} // namespace utils
} // namespace archneuronx
