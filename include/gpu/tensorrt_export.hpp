#pragma once
// ============================================================
// ArchNeuronX v3 - TensorRT Export System
// Production deployment optimization for regime-aware ensemble
// 3-5x inference speed improvement vs PyTorch
// ============================================================

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace archneuronx {
namespace gpu {

/**
 * @brief TensorRT export configuration
 */
struct TensorRTConfig {
    // Engine settings
    int max_batch_size = 32;
    int max_workspace_size = 1 << 30;     // 1GB
    bool enable_fp16 = true;               // FP16 optimization
    bool enable_int8 = false;              // INT8 calibration
    bool enable_dynamic_shapes = true;     // Dynamic batch sizes
    
    // Optimization settings
    bool enable_layer_norm_fusion = true;  // LayerNorm fusion
    bool enable_skip_layer_norm_fusion = true;
    bool enable_multi_head_attention_fusion = true;
    bool enable_qat_fusion = true;          // Quantization aware training
    
    // Precision settings
    nvinfer1::BuilderFlags precision_flags = nvinfer1::BuilderFlag::kFP16;
    
    // Calibration settings for INT8
    std::string calibration_cache_file = "tensorrt_calibration.cache";
    int calibration_batch_size = 16;
    int calibration_iterations = 100;
    
    // Export settings
    std::string engine_file_extension = ".trt";
    bool enable_timing_cache = true;
    std::string timing_cache_file = "tensorrt_timing.cache";
    
    // Profiling settings
    bool enable_profiling = false;
    int profiling_iterations = 100;
};

/**
 * @brief TensorRT engine wrapper
 */
class TensorRTEngine {
public:
    explicit TensorRTEngine(const std::string& engine_path);
    ~TensorRTEngine();
    
    bool load_engine();
    bool is_loaded() const;
    
    // Inference
    std::vector<torch::Tensor> infer(const std::vector<torch::Tensor>& inputs);
    std::vector<torch::Tensor> infer_async(const std::vector<torch::Tensor>& inputs, cudaStream_t stream = 0);
    
    // Engine information
    std::string get_engine_info() const;
    int get_max_batch_size() const;
    std::vector<std::vector<int>> get_input_shapes() const;
    std::vector<std::vector<int>> get_output_shapes() const;
    
    // Performance
    double benchmark_inference(const std::vector<torch::Tensor>& inputs, int iterations = 100);
    
private:
    std::string engine_path_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> device_buffers_;
    std::vector<size_t> buffer_sizes_;
    cudaStream_t stream_;
    bool loaded_;
    
    bool allocate_buffers(const std::vector<torch::Tensor>& inputs);
    void free_buffers();
};

/**
 * @brief TensorRT exporter for PyTorch models
 * 
 * Converts PyTorch models to optimized TensorRT engines for
 * maximum inference performance in production environments.
 */
class TensorRTExporter {
public:
    explicit TensorRTExporter(const TensorRTConfig& config = TensorRTConfig{});
    ~TensorRTExporter();
    
    // Export methods
    bool export_model_to_onnx(torch::jit::script::Module& model, 
                             const std::string& onnx_path,
                             const std::vector<std::vector<int>>& input_shapes);
    
    bool export_onnx_to_tensorrt(const std::string& onnx_path,
                                const std::string& engine_path,
                                const std::vector<std::vector<int>>& input_shapes);
    
    bool export_pytorch_to_tensorrt(torch::jit::script::Module& model,
                                    const std::string& engine_path,
                                    const std::vector<std::vector<int>>& input_shapes);
    
    // Batch export for ensemble
    bool export_ensemble_models(const std::vector<std::pair<std::string, torch::jit::script::Module>>& models,
                                 const std::string& output_directory,
                                 const std::vector<std::vector<int>>& input_shapes);
    
    // Calibration for INT8
    bool calibrate_int8_model(const std::string& onnx_path,
                              const std::string& engine_path,
                              const std::vector<torch::Tensor>& calibration_data);
    
    // Validation
    bool validate_exported_engine(const std::string& engine_path,
                                 torch::jit::script::Module& original_model,
                                 const std::vector<torch::Tensor>& test_inputs,
                                 double tolerance = 1e-3);
    
    // Performance comparison
    struct PerformanceComparison {
        double pytorch_time_ms;
        double tensorrt_time_ms;
        double speedup_factor;
        double accuracy_difference;
        bool passed_validation;
    };
    
    PerformanceComparison compare_performance(torch::jit::script::Module& original_model,
                                            const std::string& engine_path,
                                            const std::vector<torch::Tensor>& test_inputs,
                                            int iterations = 100);
    
    // Utility methods
    bool is_tensorrt_available() const;
    std::string get_tensorrt_version() const;
    std::vector<std::string> get_supported_precisions() const;
    
    // Engine management
    std::unique_ptr<TensorRTEngine> load_engine(const std::string& engine_path);
    bool optimize_engine(const std::string& engine_path, const std::string& optimized_path);

private:
    TensorRTConfig config_;
    std::unique_ptr<nvinfer1::IBuilder> builder_;
    std::unique_ptr<nvinfer1::INetworkDefinition> network_;
    std::unique_ptr<nvonnxparser::IParser> parser_;
    std::unique_ptr<nvinfer1::IHostMemory> plan_;
    
    // Internal methods
    bool initialize_tensorrt();
    void cleanup_tensorrt();
    
    // ONNX export helpers
    bool export_to_onnx(torch::jit::script::Module& model,
                        const std::string& onnx_path,
                        const std::vector<std::vector<int>>& input_shapes);
    
    // TensorRT optimization helpers
    bool build_tensorrt_engine(const std::string& onnx_path,
                               const std::string& engine_path,
                               const std::vector<std::vector<int>>& input_shapes);
    
    // Calibration helpers
    class Int8Calibrator;
    std::unique_ptr<Int8Calibrator> create_calibrator(const std::vector<torch::Tensor>& data);
    
    // Validation helpers
    bool compare_outputs(const torch::Tensor& pytorch_output,
                        const torch::Tensor& tensorrt_output,
                        double tolerance);
    
    // Utility helpers
    std::vector<nvinfer1::Dims> convert_shapes_to_dims(const std::vector<std::vector<int>>& shapes);
    torch::Tensor convert_tensorrt_output(void* device_ptr, const std::vector<int>& shape);
};

/**
 * @brief INT8 calibration implementation
 */
class TensorRTExporter::Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(const std::vector<torch::Tensor>& calibration_data,
                   const std::string& cache_file,
                   int batch_size);
    
    ~Int8Calibrator() override;
    
    // Calibration interface
    int getBatchSize() const override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* cache, size_t length) override;
    
private:
    std::vector<torch::Tensor> calibration_data_;
    std::string cache_file_;
    int batch_size_;
    int current_batch_;
    std::vector<char> calibration_cache_;
    void* device_input_;
    
    bool allocate_device_memory();
    void free_device_memory();
};

/**
 * @brief RAII TensorRT context manager
 */
class TensorRTContext {
public:
    explicit TensorRTContext(const TensorRTConfig& config = TensorRTConfig{});
    ~TensorRTContext();
    
    bool is_valid() const;
    TensorRTExporter& get_exporter();

private:
    std::unique_ptr<TensorRTExporter> exporter_;
    bool valid_;
};

/**
 * @brief Performance profiler for TensorRT engines
 */
class TensorRTProfiler {
public:
    explicit TensorRTProfiler(const std::string& engine_path);
    
    struct LayerProfile {
        std::string layer_name;
        double time_ms;
        double memory_mb;
        int execution_count;
    };
    
    std::vector<LayerProfile> profile_layers(int iterations = 100);
    std::string generate_report() const;
    void export_report(const std::string& filename) const;

private:
    std::string engine_path_;
    std::vector<LayerProfile> layer_profiles_;
    
    void profile_layer_execution(nvinfer1::IExecutionContext* context, 
                               const std::string& layer_name,
                               int iterations);
};

// Utility functions
namespace tensorrt_utils {
    bool check_tensorrt_availability();
    std::string get_device_info();
    bool validate_cuda_compatibility();
    std::vector<std::string> list_available_engines(const std::string& directory);
    bool remove_engine(const std::string& engine_path);
    size_t get_engine_size(const std::string& engine_path);
}

} // namespace gpu
} // namespace archneuronx
