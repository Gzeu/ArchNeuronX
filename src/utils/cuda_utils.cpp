// ============================================================
// ArchNeuronX v2 - CUDA Utilities Implementation
// GPU memory management, device selection, and optimization
// ============================================================
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <nvml.h>
#endif

namespace archneuronx {
namespace utils {

class CudaManager {
public:
    struct DeviceInfo {
        int device_id;
        std::string name;
        size_t total_memory_mb;
        size_t free_memory_mb;
        int compute_capability_major;
        int compute_capability_minor;
        int multiprocessor_count;
        int max_threads_per_block;
        size_t max_shared_memory_per_block;
        bool is_available;
    };

    static CudaManager& instance() {
        static CudaManager instance;
        return instance;
    }

    bool initialize() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (initialized_) {
            return true;
        }

#ifdef USE_CUDA
        try {
            // Initialize CUDA
            cudaError_t err = cudaSetDevice(0);
            if (err != cudaSuccess) {
                std::cerr << "CUDA initialization failed: " << cudaGetErrorString(err) << std::endl;
                return false;
            }

            // Initialize cuBLAS
            cublasStatus_t blas_status = cublasCreate(&cublas_handle_);
            if (blas_status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS initialization failed" << std::endl;
                return false;
            }

            // Initialize cuDNN
            cudnnStatus_t dnn_status = cudnnCreate(&cudnn_handle_);
            if (dnn_status != CUDNN_STATUS_SUCCESS) {
                std::cerr << "cuDNN initialization failed" << std::endl;
                return false;
            }

            // Initialize NVML for detailed device info
            nvmlReturn_t nvml_status = nvmlInit();
            if (nvml_status == NVML_SUCCESS) {
                nvml_available_ = true;
            }

            // Enumerate devices
            enumerate_devices();
            
            // Select best device
            select_optimal_device();

            initialized_ = true;
            std::cout << "CUDA Manager initialized successfully" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "CUDA Manager initialization error: " << e.what() << std::endl;
            return false;
        }
#else
        std::cout << "CUDA support disabled" << std::endl;
        return false;
#endif
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!initialized_) {
            return;
        }

#ifdef USE_CUDA
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }

        if (cudnn_handle_) {
            cudnnDestroy(cudnn_handle_);
            cudnn_handle_ = nullptr;
        }

        if (nvml_available_) {
            nvmlShutdown();
            nvml_available_ = false;
        }

        // Free all allocated memory
        for (auto& ptr : allocated_ptrs_) {
            cudaFree(ptr);
        }
        allocated_ptrs_.clear();

        initialized_ = false;
        std::cout << "CUDA Manager shutdown complete" << std::endl;
#endif
    }

    std::vector<DeviceInfo> get_available_devices() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return devices_;
    }

    DeviceInfo get_current_device_info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_device_id_ >= 0 && current_device_id_ < devices_.size()) {
            return devices_[current_device_id_];
        }
        return {};
    }

    bool set_device(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (device_id < 0 || device_id >= devices_.size()) {
            return false;
        }

#ifdef USE_CUDA
        cudaError_t err = cudaSetDevice(device_id);
        if (err == cudaSuccess) {
            current_device_id_ = device_id;
            return true;
        }
        return false;
#else
        return false;
#endif
    }

    void* allocate_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!initialized_) {
            return nullptr;
        }

#ifdef USE_CUDA
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err == cudaSuccess) {
            allocated_ptrs_.push_back(ptr);
            memory_usage_ += size;
            return ptr;
        }
        
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
#else
        return nullptr;
#endif
    }

    void free_memory(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!ptr) {
            return;
        }

#ifdef USE_CUDA
        auto it = std::find(allocated_ptrs_.begin(), allocated_ptrs_.end(), ptr);
        if (it != allocated_ptrs_.end()) {
            cudaFree(ptr);
            allocated_ptrs_.erase(it);
        }
#endif
    }

    size_t get_memory_usage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return memory_usage_;
    }

    bool optimize_memory() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!initialized_) {
            return false;
        }

#ifdef USE_CUDA
        // Force garbage collection
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return false;
        }

        // Reset device to clear memory fragmentation
        int current_device;
        cudaGetDevice(&current_device);
        cudaDeviceReset();
        cudaSetDevice(current_device);

        // Reinitialize handles
        cublasCreate(&cublas_handle_);
        cudnnCreate(&cudnn_handle_);

        return true;
#else
        return false;
#endif
    }

    bool is_memory_available(size_t required_size) const {
        auto device_info = get_current_device_info();
        return device_info.free_memory_mb >= (required_size / (1024 * 1024));
    }

private:
    CudaManager() = default;
    ~CudaManager() {
        shutdown();
    }

    bool initialized_ = false;
    bool nvml_available_ = false;
    int current_device_id_ = -1;
    std::vector<DeviceInfo> devices_;
    std::vector<void*> allocated_ptrs_;
    size_t memory_usage_ = 0;
    mutable std::mutex mutex_;

#ifdef USE_CUDA
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
#endif

    void enumerate_devices() {
#ifdef USE_CUDA
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        devices_.clear();
        devices_.reserve(device_count);

        for (int i = 0; i < device_count; ++i) {
            DeviceInfo info;
            info.device_id = i;

            // Get device properties
            cudaDeviceProp prop;
            err = cudaGetDeviceProperties(&prop, i);
            if (err == cudaSuccess) {
                info.name = prop.name;
                info.compute_capability_major = prop.major;
                info.compute_capability_minor = prop.minor;
                info.multiprocessor_count = prop.multiProcessorCount;
                info.max_threads_per_block = prop.maxThreadsPerBlock;
                info.max_shared_memory_per_block = prop.sharedMemPerBlock;
                info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
                info.is_available = true;

                // Get free memory
                size_t free_mem, total_mem;
                err = cudaSetDevice(i);
                if (err == cudaSuccess) {
                    err = cudaMemGetInfo(&free_mem, &total_mem);
                    if (err == cudaSuccess) {
                        info.free_memory_mb = free_mem / (1024 * 1024);
                    }
                }
            } else {
                info.is_available = false;
            }

            devices_.push_back(info);
        }
#endif
    }

    void select_optimal_device() {
        if (devices_.empty()) {
            return;
        }

        // Select device with most free memory and highest compute capability
        int best_device = 0;
        double best_score = 0.0;

        for (size_t i = 0; i < devices_.size(); ++i) {
            if (!devices_[i].is_available) {
                continue;
            }

            // Score based on compute capability and available memory
            double compute_score = devices_[i].compute_capability_major + 
                                 devices_[i].compute_capability_minor * 0.1;
            double memory_score = devices_[i].free_memory_mb / 1024.0; // GB
            double total_score = compute_score * 10 + memory_score;

            if (total_score > best_score) {
                best_score = total_score;
                best_device = static_cast<int>(i);
            }
        }

        set_device(best_device);
        std::cout << "Selected optimal CUDA device: " << devices_[best_device].name 
                  << " (ID: " << best_device << ")" << std::endl;
    }
};

// Global functions for CUDA utilities
bool initialize_cuda() {
    return CudaManager::instance().initialize();
}

void shutdown_cuda() {
    CudaManager::instance().shutdown();
}

std::vector<CudaManager::DeviceInfo> get_cuda_devices() {
    return CudaManager::instance().get_available_devices();
}

CudaManager::DeviceInfo get_current_cuda_device() {
    return CudaManager::instance().get_current_device_info();
}

void* cuda_allocate(size_t size) {
    return CudaManager::instance().allocate_memory(size);
}

void cuda_free(void* ptr) {
    CudaManager::instance().free_memory(ptr);
}

size_t get_cuda_memory_usage() {
    return CudaManager::instance().get_memory_usage();
}

bool optimize_cuda_memory() {
    return CudaManager::instance().optimize_memory();
}

} // namespace utils
} // namespace archneuronx
