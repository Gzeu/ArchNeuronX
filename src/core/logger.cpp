/**
 * @file logger.cpp
 * @brief Implementation of comprehensive logging framework
 * @author George Pricop
 * @date 2025-10-02
 */

#include "core/logger.hpp"
#include <filesystem>
#include <algorithm>

namespace ArchNeuronX {
namespace Core {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::initialize(LogLevel level, LogOutput output, 
                       const std::string& filename, size_t max_file_size) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    min_level_ = level;
    output_type_ = output;
    log_filename_ = filename;
    max_file_size_ = max_file_size;
    
    // Create logs directory if it doesn't exist
    if (output_type_ == LogOutput::FILE_ONLY || output_type_ == LogOutput::CONSOLE_AND_FILE) {
        std::filesystem::path log_path(filename);
        std::filesystem::path log_dir = log_path.parent_path();
        
        if (!log_dir.empty() && !std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }
        
        log_file_ = std::make_unique<std::ofstream>(filename, std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
            output_type_ = LogOutput::CONSOLE_ONLY;
        }
    }
    
    initialized_ = true;
    
    // Log initialization message
    log(LogLevel::INFO, "Logger initialized successfully", __FILE__, __FUNCTION__, __LINE__);
}

void Logger::log(LogLevel level, const std::string& message,
                const char* file, const char* function, int line) {
    if (!isLevelEnabled(level)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    // Check if file rotation is needed
    if (needsRotation()) {
        rotateLogFile();
    }
    
    // Format log message
    std::ostringstream log_stream;
    log_stream << "[" << formatTimestamp() << "] "
              << "[" << levelToString(level) << "] "
              << "[" << std::this_thread::get_id() << "] ";
    
    if (level >= LogLevel::DEBUG && strlen(file) > 0) {
        log_stream << "[" << extractFilename(file) << ":" << line << "] ";
    }
    
    log_stream << message;
    
    std::string formatted_message = log_stream.str();
    
    // Output to console if required
    if (output_type_ == LogOutput::CONSOLE_ONLY || output_type_ == LogOutput::CONSOLE_AND_FILE) {
        if (level >= LogLevel::ERROR) {
            std::cerr << formatted_message << std::endl;
        } else {
            std::cout << formatted_message << std::endl;
        }
    }
    
    // Output to file if required
    if ((output_type_ == LogOutput::FILE_ONLY || output_type_ == LogOutput::CONSOLE_AND_FILE) 
        && log_file_ && log_file_->is_open()) {
        *log_file_ << formatted_message << std::endl;
        log_file_->flush(); // Ensure immediate write
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (output_type_ == LogOutput::CONSOLE_ONLY || output_type_ == LogOutput::CONSOLE_AND_FILE) {
        std::cout.flush();
        std::cerr.flush();
    }
    
    if ((output_type_ == LogOutput::FILE_ONLY || output_type_ == LogOutput::CONSOLE_AND_FILE) 
        && log_file_ && log_file_->is_open()) {
        log_file_->flush();
    }
}

void Logger::rotateLogFile() {
    if (!log_file_ || !log_file_->is_open()) {
        return;
    }
    
    log_file_->close();
    
    // Create backup filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream backup_name;
    std::filesystem::path log_path(log_filename_);
    backup_name << log_path.stem().string() << "_" 
                << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
                << log_path.extension().string();
    
    std::filesystem::path backup_path = log_path.parent_path() / backup_name.str();
    
    // Move current log to backup
    try {
        std::filesystem::rename(log_filename_, backup_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to rotate log file: " << e.what() << std::endl;
    }
    
    // Open new log file
    log_file_ = std::make_unique<std::ofstream>(log_filename_, std::ios::app);
    if (!log_file_->is_open()) {
        std::cerr << "Failed to open new log file after rotation: " << log_filename_ << std::endl;
    }
}

Logger::~Logger() {
    if (log_file_ && log_file_->is_open()) {
        log_file_->close();
    }
}

std::string Logger::formatTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNW";
    }
}

std::string Logger::extractFilename(const std::string& filepath) {
    std::filesystem::path path(filepath);
    return path.filename().string();
}

bool Logger::needsRotation() {
    if (!log_file_ || !log_file_->is_open()) {
        return false;
    }
    
    try {
        auto file_size = std::filesystem::file_size(log_filename_);
        return file_size >= max_file_size_;
    } catch (const std::exception&) {
        return false; // If we can't check size, don't rotate
    }
}

} // namespace Core
} // namespace ArchNeuronX