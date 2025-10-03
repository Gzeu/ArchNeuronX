/**
 * @file logger.hpp
 * @brief Comprehensive logging framework for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <thread>

namespace ArchNeuronX {
namespace Core {

/**
 * @enum LogLevel
 * @brief Defines logging levels for message filtering
 */
enum class LogLevel {
    TRACE = 0,  ///< Detailed trace information
    DEBUG = 1,  ///< Debug information for development
    INFO = 2,   ///< General information messages
    WARN = 3,   ///< Warning messages
    ERROR = 4,  ///< Error messages
    FATAL = 5   ///< Fatal error messages
};

/**
 * @enum LogOutput
 * @brief Defines where logs should be output
 */
enum class LogOutput {
    CONSOLE_ONLY,    ///< Output only to console
    FILE_ONLY,       ///< Output only to file
    CONSOLE_AND_FILE ///< Output to both console and file
};

/**
 * @class Logger
 * @brief Thread-safe logging framework with multiple output options
 */
class Logger {
public:
    /**
     * @brief Get singleton instance of logger
     * @return Reference to logger instance
     */
    static Logger& getInstance();
    
    /**
     * @brief Initialize logger with configuration
     * @param level Minimum log level to output
     * @param output Where to output logs
     * @param filename File path for log output (if applicable)
     * @param max_file_size Maximum file size before rotation (bytes)
     */
    void initialize(LogLevel level, LogOutput output, 
                   const std::string& filename = "logs/archneuronx.log",
                   size_t max_file_size = 10 * 1024 * 1024); // 10MB default
    
    /**
     * @brief Log a message with specified level
     * @param level Log level for this message
     * @param message Log message content
     * @param file Source file name (automatically filled by macros)
     * @param function Function name (automatically filled by macros)
     * @param line Line number (automatically filled by macros)
     */
    void log(LogLevel level, const std::string& message,
             const char* file = "", const char* function = "", int line = 0);
    
    /**
     * @brief Set minimum log level
     * @param level New minimum log level
     */
    void setLogLevel(LogLevel level) { min_level_ = level; }
    
    /**
     * @brief Get current minimum log level
     * @return Current minimum log level
     */
    LogLevel getLogLevel() const { return min_level_; }
    
    /**
     * @brief Check if a log level is enabled
     * @param level Log level to check
     * @return True if level is enabled
     */
    bool isLevelEnabled(LogLevel level) const { return level >= min_level_; }
    
    /**
     * @brief Flush all pending log messages
     */
    void flush();
    
    /**
     * @brief Rotate log files (create new file, archive old one)
     */
    void rotateLogFile();
    
private:
    Logger() = default;
    ~Logger();
    
    // Prevent copying
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    /**
     * @brief Format timestamp for log messages
     * @return Formatted timestamp string
     */
    std::string formatTimestamp();
    
    /**
     * @brief Convert log level to string
     * @param level Log level to convert
     * @return String representation of log level
     */
    std::string levelToString(LogLevel level);
    
    /**
     * @brief Extract filename from full path
     * @param filepath Full file path
     * @return Just the filename
     */
    std::string extractFilename(const std::string& filepath);
    
    /**
     * @brief Check if log file needs rotation
     * @return True if rotation is needed
     */
    bool needsRotation();
    
    LogLevel min_level_ = LogLevel::INFO;
    LogOutput output_type_ = LogOutput::CONSOLE_ONLY;
    std::string log_filename_;
    std::unique_ptr<std::ofstream> log_file_;
    size_t max_file_size_ = 10 * 1024 * 1024; // 10MB
    
    mutable std::mutex log_mutex_;
    bool initialized_ = false;
};

// Convenience macros for logging
#define LOG_TRACE(msg) do { \
    std::ostringstream oss; oss << msg; \
    ArchNeuronX::Core::Logger::getInstance().log(ArchNeuronX::Core::LogLevel::TRACE, oss.str(), __FILE__, __FUNCTION__, __LINE__); \
} while(0)

#define LOG_DEBUG(msg) do { \
    std::ostringstream oss; oss << msg; \
    ArchNeuronX::Core::Logger::getInstance().log(ArchNeuronX::Core::LogLevel::DEBUG, oss.str(), __FILE__, __FUNCTION__, __LINE__); \
} while(0)

#define LOG_INFO(msg) do { \
    std::ostringstream oss; oss << msg; \
    ArchNeuronX::Core::Logger::getInstance().log(ArchNeuronX::Core::LogLevel::INFO, oss.str(), __FILE__, __FUNCTION__, __LINE__); \
} while(0)

#define LOG_WARN(msg) do { \
    std::ostringstream oss; oss << msg; \
    ArchNeuronX::Core::Logger::getInstance().log(ArchNeuronX::Core::LogLevel::WARN, oss.str(), __FILE__, __FUNCTION__, __LINE__); \
} while(0)

#define LOG_ERROR(msg) do { \
    std::ostringstream oss; oss << msg; \
    ArchNeuronX::Core::Logger::getInstance().log(ArchNeuronX::Core::LogLevel::ERROR, oss.str(), __FILE__, __FUNCTION__, __LINE__); \
} while(0)

#define LOG_FATAL(msg) do { \
    std::ostringstream oss; oss << msg; \
    ArchNeuronX::Core::Logger::getInstance().log(ArchNeuronX::Core::LogLevel::FATAL, oss.str(), __FILE__, __FUNCTION__, __LINE__); \
} while(0)

} // namespace Core
} // namespace ArchNeuronX