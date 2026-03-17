// ============================================================
// ArchNeuronX v2 - System Monitor Implementation
// Monitors CPU, memory, GPU, and system health
// ============================================================
#include "monitoring/metrics.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#elif __linux__
#include <sys/sysinfo.h>
#include <proc/readproc.h>
#include <unistd.h>
#endif

namespace archneuronx {
namespace monitoring {

class SystemMonitor {
public:
    struct SystemStats {
        double cpu_usage_percent = 0.0;
        double memory_usage_percent = 0.0;
        double memory_used_mb = 0.0;
        double memory_total_mb = 0.0;
        double gpu_usage_percent = 0.0;
        double gpu_memory_usage_percent = 0.0;
        double gpu_memory_used_mb = 0.0;
        double gpu_memory_total_mb = 0.0;
        double disk_usage_percent = 0.0;
        double network_receive_mb = 0.0;
        double network_transmit_mb = 0.0;
        int64_t uptime_seconds = 0;
        std::string timestamp;
    };

    explicit SystemMonitor(int update_interval_seconds = 5) 
        : update_interval_(update_interval_seconds), running_(false) {}

    void start() {
        if (running_) {
            return;
        }
        
        running_ = true;
        monitor_thread_ = std::thread(&SystemMonitor::monitor_loop, this);
        std::cout << "System monitor started (update interval: " << update_interval_ << "s)" << std::endl;
    }

    void stop() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
        std::cout << "System monitor stopped" << std::endl;
    }

    SystemStats get_current_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return current_stats_;
    }

    bool is_healthy() const {
        auto stats = get_current_stats();
        
        // Health checks
        if (stats.cpu_usage_percent > 95.0) return false;
        if (stats.memory_usage_percent > 95.0) return false;
        if (stats.disk_usage_percent > 95.0) return false;
        
        return true;
    }

private:
    int update_interval_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    mutable std::mutex stats_mutex_;
    SystemStats current_stats_;

#ifdef _WIN32
    HANDLE cpu_query_handle_ = nullptr;
    PDH_HCOUNTER cpu_counter_ = nullptr;
    ULARGE_INTEGER last_cpu_time_ = {};
    ULARGE_INTEGER last_sys_cpu_time_ = {};
    ULARGE_INTEGER last_usr_cpu_time_ = {};
    HANDLE process_handle_ = GetCurrentProcess();
#endif

    void monitor_loop() {
        initialize_monitoring();
        
        while (running_) {
            try {
                update_stats();
                log_stats();
                check_alerts();
                
                std::this_thread::sleep_for(std::chrono::seconds(update_interval_));
            } catch (const std::exception& e) {
                std::cerr << "System monitor error: " << e.what() << std::endl;
            }
        }
    }

    void initialize_monitoring() {
#ifdef _WIN32
        // Initialize PDH for CPU monitoring
        PdhOpenQuery(nullptr, 0, &cpu_query_handle_);
        PdhAddEnglishCounter(cpu_query_handle_, "\\Processor(_Total)\\% Processor Time", 0, &cpu_counter_);
        PdhCollectQueryData(cpu_query_handle_);
        
        // Get initial CPU times
        FILETIME ftime, fsys, fuser;
        GetSystemTimeAsFileTime(&ftime);
        memcpy(&last_cpu_time_, &ftime, sizeof(FILETIME));
        
        GetProcessTimes(process_handle_, &ftime, &ftime, &fsys, &fuser);
        memcpy(&last_sys_cpu_time_, &fsys, sizeof(FILETIME));
        memcpy(&last_usr_cpu_time_, &fuser, sizeof(FILETIME));
#endif
    }

    void update_stats() {
        SystemStats stats;
        stats.timestamp = get_current_timestamp();

        // CPU Usage
        stats.cpu_usage_percent = get_cpu_usage();

        // Memory Usage
        get_memory_usage(stats.memory_usage_percent, stats.memory_used_mb, stats.memory_total_mb);

        // GPU Usage (if available)
        get_gpu_usage(stats.gpu_usage_percent, stats.gpu_memory_usage_percent,
                     stats.gpu_memory_used_mb, stats.gpu_memory_total_mb);

        // Disk Usage
        stats.disk_usage_percent = get_disk_usage();

        // Network Usage
        get_network_usage(stats.network_receive_mb, stats.network_transmit_mb);

        // Uptime
        stats.uptime_seconds = get_uptime();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            current_stats_ = stats;
        }
    }

    double get_cpu_usage() {
#ifdef _WIN32
        PDH_FMT_COUNTERVALUE counter_val;
        if (PdhCollectQueryData(cpu_query_handle_) == ERROR_SUCCESS) {
            if (PdhGetFormattedCounterValue(cpu_counter_, PDH_FMT_DOUBLE, nullptr, &counter_val) == ERROR_SUCCESS) {
                return counter_val.doubleValue;
            }
        }
        return 0.0;
#elif __linux__
        std::ifstream file("/proc/stat");
        std::string line;
        if (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string cpu;
            long user, nice, system, idle, iowait, irq, softirq, steal;
            iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
            
            long total = user + nice + system + idle + iowait + irq + softirq + steal;
            long idle_time = idle + iowait;
            
            static long prev_total = 0, prev_idle = 0;
            long total_diff = total - prev_total;
            long idle_diff = idle_time - prev_idle;
            
            prev_total = total;
            prev_idle = idle_time;
            
            if (total_diff > 0) {
                return (1.0 - (double)idle_diff / total_diff) * 100.0;
            }
        }
        return 0.0;
#else
        return 0.0; // Placeholder for other platforms
#endif
    }

    void get_memory_usage(double& usage_percent, double& used_mb, double& total_mb) {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        
        total_mb = static_cast<double>(memInfo.ullTotalPhys) / (1024 * 1024);
        used_mb = static_cast<double>(memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024 * 1024);
        usage_percent = memInfo.dwMemoryLoad;
#elif __linux__
        std::ifstream file("/proc/meminfo");
        std::string line;
        long mem_total = 0, mem_available = 0;
        
        while (std::getline(file, line)) {
            if (line.find("MemTotal:") == 0) {
                std::istringstream iss(line);
                std::string label;
                iss >> label >> mem_total;
            } else if (line.find("MemAvailable:") == 0) {
                std::istringstream iss(line);
                std::string label;
                iss >> label >> mem_available;
            }
        }
        
        if (mem_total > 0) {
            total_mb = mem_total / 1024.0;
            used_mb = (mem_total - mem_available) / 1024.0;
            usage_percent = (used_mb / total_mb) * 100.0;
        }
#endif
    }

    void get_gpu_usage(double& gpu_usage, double& gpu_mem_usage, 
                      double& gpu_mem_used, double& gpu_mem_total) {
        // This would require NVIDIA ML library or similar
        // For now, return zeros
        gpu_usage = 0.0;
        gpu_mem_usage = 0.0;
        gpu_mem_used = 0.0;
        gpu_mem_total = 0.0;
    }

    double get_disk_usage() {
#ifdef _WIN32
        ULARGE_INTEGER free_bytes, total_bytes;
        if (GetDiskFreeSpaceExA("C:", nullptr, &total_bytes, &free_bytes)) {
            return ((double)(total_bytes.QuadPart - free_bytes.QuadPart) / total_bytes.QuadPart) * 100.0;
        }
#elif __linux__
        std::ifstream file("/proc/mounts");
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("/dev/") == 0) {
                std::istringstream iss(line);
                std::string device, mount_point, fs_type;
                iss >> device >> mount_point >> fs_type;
                
                if (mount_point == "/") {
                    struct statvfs stat;
                    if (statvfs(mount_point.c_str(), &stat) == 0) {
                        unsigned long total = stat.f_blocks * stat.f_frsize;
                        unsigned long free = stat.f_bfree * stat.f_frsize;
                        return ((double)(total - free) / total) * 100.0;
                    }
                }
            }
        }
#endif
        return 0.0;
    }

    void get_network_usage(double& receive_mb, double& transmit_mb) {
        // Network monitoring would require platform-specific implementation
        // For now, return zeros
        receive_mb = 0.0;
        transmit_mb = 0.0;
    }

    int64_t get_uptime() {
#ifdef _WIN32
        return GetTickCount64() / 1000;
#elif __linux__
        std::ifstream file("/proc/uptime");
        double uptime;
        file >> uptime;
        return static_cast<int64_t>(uptime);
#else
        return 0;
#endif
    }

    std::string get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }

    void log_stats() {
        auto stats = get_current_stats();
        std::cout << "[SYSTEM] CPU: " << stats.cpu_usage_percent << "% "
                  << "Memory: " << stats.memory_usage_percent << "% "
                  << "Disk: " << stats.disk_usage_percent << "% "
                  << "Uptime: " << stats.uptime_seconds << "s" << std::endl;
    }

    void check_alerts() {
        auto stats = get_current_stats();
        
        if (stats.cpu_usage_percent > 90.0) {
            std::cout << "[ALERT] High CPU usage: " << stats.cpu_usage_percent << "%" << std::endl;
        }
        
        if (stats.memory_usage_percent > 90.0) {
            std::cout << "[ALERT] High memory usage: " << stats.memory_usage_percent << "%" << std::endl;
        }
        
        if (stats.disk_usage_percent > 90.0) {
            std::cout << "[ALERT] High disk usage: " << stats.disk_usage_percent << "%" << std::endl;
        }
    }
};

// Global system monitor instance
static std::unique_ptr<SystemMonitor> g_system_monitor;

void start_system_monitor(int update_interval_seconds) {
    if (!g_system_monitor) {
        g_system_monitor = std::make_unique<SystemMonitor>(update_interval_seconds);
    }
    g_system_monitor->start();
}

void stop_system_monitor() {
    if (g_system_monitor) {
        g_system_monitor->stop();
    }
}

SystemMonitor::SystemStats get_system_stats() {
    if (g_system_monitor) {
        return g_system_monitor->get_current_stats();
    }
    return {};
}

bool is_system_healthy() {
    if (g_system_monitor) {
        return g_system_monitor->is_healthy();
    }
    return true;
}

} // namespace monitoring
} // namespace archneuronx
