// ============================================================
// ArchNeuronX v2 - Structured Logger Implementation
// Wraps spdlog with trading-specific context
// ============================================================
#include "monitoring/logger.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace archneuronx {
namespace monitoring {

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::init(const std::string& log_dir,
                  LogLevel level,
                  size_t max_file_size_mb,
                  size_t max_files) {
    
#ifdef USE_SPDLOG
    try {
        // Create log directory if it doesn't exist
        std::filesystem::create_directories(log_dir);
        
        // Create file sink with rotation
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_dir + "/archneuronx.log",
            max_file_size_mb * 1024 * 1024, // Convert MB to bytes
            max_files
        );
        
        // Create console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        
        // Create logger with both sinks
        std::vector<spdlog::sink_ptr> sinks = {file_sink, console_sink};
        logger_ = std::make_shared<spdlog::logger>("archneuronx", sinks.begin(), sinks.end());
        
        // Set log level
        switch (level) {
            case LogLevel::TRACE:
                logger_->set_level(spdlog::level::trace);
                break;
            case LogLevel::DEBUG:
                logger_->set_level(spdlog::level::debug);
                break;
            case LogLevel::INFO:
                logger_->set_level(spdlog::level::info);
                break;
            case LogLevel::WARN:
                logger_->set_level(spdlog::level::warn);
                break;
            case LogLevel::ERROR:
                logger_->set_level(spdlog::level::err);
                break;
            case LogLevel::CRITICAL:
                logger_->set_level(spdlog::level::critical);
                break;
        }
        
        // Set custom pattern with timestamp and thread info
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] [%l] %v");
        
        // Register logger
        spdlog::register_logger(logger_);
        spdlog::set_default_logger(logger_);
        
        logger_->info("Logger initialized. Log directory: {}", log_dir);
        
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        // Fallback to console logging
        logger_ = spdlog::stdout_color_mt("archneuronx_fallback");
    }
#else
    std::cout << "spdlog support disabled. Using console logging." << std::endl;
#endif
}

template<typename... Args>
void Logger::log(LogLevel level, std::string_view msg, Args&&... args) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_current_timestamp();
    auto level_str = log_level_to_string(level);
    
    // Format message with arguments
    std::string formatted_msg;
    if constexpr (sizeof...(args) > 0) {
        try {
            formatted_msg = std::vformat(msg, std::make_format_args(args...));
        } catch (const std::exception& e) {
            formatted_msg = std::string(msg) + " [FORMAT ERROR: " + e.what() + "]";
        }
    } else {
        formatted_msg = msg;
    }

#ifdef USE_SPDLOG
    if (logger_) {
        switch (level) {
            case LogLevel::TRACE:
                logger_->trace(formatted_msg);
                break;
            case LogLevel::DEBUG:
                logger_->debug(formatted_msg);
                break;
            case LogLevel::INFO:
                logger_->info(formatted_msg);
                break;
            case LogLevel::WARN:
                logger_->warn(formatted_msg);
                break;
            case LogLevel::ERROR:
                logger_->error(formatted_msg);
                break;
            case LogLevel::CRITICAL:
                logger_->critical(formatted_msg);
                break;
        }
    }
#else
    // Fallback console logging
    std::cout << "[" << timestamp << "] [" << level_str << "] " << formatted_msg << std::endl;
#endif
}

void Logger::log_signal(std::string_view symbol, std::string_view action,
                       float confidence, double price) {
    info("SIGNAL_GENERATED symbol={} action={} confidence={:.3f} price={:.6f}",
         symbol, action, confidence, price);
}

void Logger::log_trade(std::string_view symbol, std::string_view side,
                       double quantity, double price, const std::string& order_id) {
    info("TRADE_EXECUTED symbol={} side={} quantity={:.6f} price={:.6f} order_id={}",
         symbol, side, quantity, price, order_id);
}

void Logger::log_inference(std::string_view model, uint64_t latency_us,
                           bool gpu_used) {
    info("INFERENCE_COMPLETED model={} latency_us={} gpu_used={}",
         model, latency_us, gpu_used ? "true" : "false");
}

void Logger::log_risk_event(std::string_view event_type,
                            std::string_view symbol, double value) {
    warn("RISK_EVENT type={} symbol={} value={:.6f}",
         event_type, symbol, value);
}

std::string Logger::get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARN: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

// Static members
std::mutex Logger::log_mutex_;

} // namespace monitoring
} // namespace archneuronx
