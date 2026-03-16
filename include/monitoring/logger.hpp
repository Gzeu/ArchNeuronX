#pragma once
// ============================================================
// ArchNeuronX v2 - Structured Logger
// Wraps spdlog with trading-specific context
// ============================================================
#include <string>
#include <string_view>
#include <memory>
#include <format>  // C++20

#ifdef USE_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#endif

namespace archneuronx {
namespace monitoring {

enum class LogLevel {
    TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL
};

class Logger {
public:
    static Logger& instance();

    void init(const std::string& log_dir,
              LogLevel level = LogLevel::INFO,
              size_t max_file_size_mb = 100,
              size_t max_files = 5);

    // Structured log with key-value pairs
    template<typename... Args>
    void info(std::string_view msg, Args&&... args) {
        log(LogLevel::INFO, msg, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(std::string_view msg, Args&&... args) {
        log(LogLevel::WARN, msg, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(std::string_view msg, Args&&... args) {
        log(LogLevel::ERROR, msg, std::forward<Args>(args)...);
    }

    // Trading-specific log helpers
    void log_signal(std::string_view symbol, std::string_view action,
                    float confidence, double price);
    void log_trade(std::string_view symbol, std::string_view side,
                   double quantity, double price, const std::string& order_id);
    void log_inference(std::string_view model, uint64_t latency_us,
                       bool gpu_used);
    void log_risk_event(std::string_view event_type,
                        std::string_view symbol, double value);

private:
    Logger() = default;

    template<typename... Args>
    void log(LogLevel level, std::string_view msg, Args&&... args);

#ifdef USE_SPDLOG
    std::shared_ptr<spdlog::logger> logger_;
#endif
};

// Convenience macros
#define LOG_INFO(...)   archneuronx::monitoring::Logger::instance().info(__VA_ARGS__)
#define LOG_WARN(...)   archneuronx::monitoring::Logger::instance().warn(__VA_ARGS__)
#define LOG_ERROR(...)  archneuronx::monitoring::Logger::instance().error(__VA_ARGS__)

} // namespace monitoring
} // namespace archneuronx
