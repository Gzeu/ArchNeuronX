/**
 * @file tensorrt_export.cpp
 * @brief TensorRT export implementation for production optimization
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "gpu/tensorrt_export.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>

namespace archneuronx {
namespace gpu {

// TensorRTEngine implementation

TensorRTEngine::TensorRTEngine(const std::string& engine_path)
    : engine_path_(engine_path), stream_(0), loaded_(false) {
    
    device_buffers_.clear();
    buffer_sizes_.clear();
}

TensorRTEngine::~TensorRTEngine() {
    free_buffers();
    
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool TensorRTEngine::load_engine() {
    try {
        // Read engine file
        std::ifstream file(engine_path_, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open engine file: " << engine_path_ << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        // Create runtime and engine
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(nullptr));
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_data.data(), size));
        
        if (!engine_) {
            std::cerr << "Failed to deserialize CUDA engine" << std::endl;
            return false;
        }
        
        // Create execution context
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Create CUDA stream
        auto error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream" << std::endl;
            return false;
        }
        
        loaded_ = true;
        std::cout << "TensorRT engine loaded successfully: " << engine_path_ << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading TensorRT engine: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRTEngine::is_loaded() const {
    return loaded_ && engine_ && context_;
}

std::vector<torch::Tensor> TensorRTEngine::infer(const std::vector<torch::Tensor>& inputs) {
    if (!is_loaded()) {
        throw std::runtime_error("Engine not loaded");
    }
    
    // Allocate buffers if needed
    if (device_buffers_.empty()) {
        if (!allocate_buffers(inputs)) {
            throw std::runtime_error("Failed to allocate buffers");
        }
    }
    
    // Copy inputs to device
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input_tensor = inputs[i].contiguous();
        auto error = cudaMemcpyAsync(device_buffers_[i], 
                                   input_tensor.data_ptr(),
                                   input_tensor.numel() * input_tensor.element_size(),
                                   cudaMemcpyHostToDevice, stream_);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to copy input to device");
        }
    }
    
    // Execute inference
    if (!context_->enqueueV2(device_buffers_.data(), stream_, nullptr)) {
        throw std::runtime_error("Failed to execute inference");
    }
    
    // Copy outputs from device
    std::vector<torch::Tensor> outputs;
    auto num_bindings = engine_->getNbBindings();
    
    for (int i = 0; i < num_bindings; ++i) {
        if (engine_->bindingIsInput(i)) continue;
        
        // Get output shape
        auto dims = engine_->getBindingDimensions(i);
        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; ++j) {
            shape.push_back(dims.d[j]);
        }
        
        // Create output tensor
        auto output_tensor = torch::empty(shape, torch::kFloat32);
        
        // Copy from device
        auto error = cudaMemcpyAsync(output_tensor.data_ptr(),
                                   device_buffers_[i],
                                   output_tensor.numel() * output_tensor.element_size(),
                                   cudaMemcpyDeviceToHost, stream_);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to copy output from device");
        }
        
        outputs.push_back(output_tensor);
    }
    
    // Synchronize stream
    cudaStreamSynchronize(stream_);
    
    return outputs;
}

std::vector<torch::Tensor> TensorRTEngine::infer_async(const std::vector<torch::Tensor>& inputs, cudaStream_t stream) {
    // Similar to infer() but uses provided stream
    cudaStream_t target_stream = stream ? stream : stream_;
    return infer(inputs); // Simplified for now
}

std::string TensorRTEngine::get_engine_info() const {
    if (!is_loaded()) {
        return "Engine not loaded";
    }
    
    std::stringstream info;
    info << "TensorRT Engine Info:\n";
    info << "  Max batch size: " << engine_->getMaxBatchSize() << "\n";
    info << "  Number of bindings: " << engine_->getNbBindings() << "\n";
    info << "  Device: " << engine_->getDevice() << "\n";
    
    return info.str();
}

int TensorRTEngine::get_max_batch_size() const {
    return is_loaded() ? engine_->getMaxBatchSize() : 0;
}

std::vector<std::vector<int>> TensorRTEngine::get_input_shapes() const {
    std::vector<std::vector<int>> shapes;
    
    if (!is_loaded()) {
        return shapes;
    }
    
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            auto dims = engine_->getBindingDimensions(i);
            std::vector<int> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            shapes.push_back(shape);
        }
    }
    
    return shapes;
}

std::vector<std::vector<int>> TensorRTEngine::get_output_shapes() const {
    std::vector<std::vector<int>> shapes;
    
    if (!is_loaded()) {
        return shapes;
    }
    
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (!engine_->bindingIsInput(i)) {
            auto dims = engine_->getBindingDimensions(i);
            std::vector<int> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            shapes.push_back(shape);
        }
    }
    
    return shapes;
}

double TensorRTEngine::benchmark_inference(const std::vector<torch::Tensor>& inputs, int iterations) {
    if (!is_loaded()) {
        return -1.0;
    }
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        infer(inputs);
    }
    
    // Benchmark
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        infer(inputs);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return static_cast<double>(duration.count()) / iterations / 1000.0; // Return average time in ms
}

bool TensorRTEngine::allocate_buffers(const std::vector<torch::Tensor>& inputs) {
    if (!is_loaded()) {
        return false;
    }
    
    device_buffers_.resize(engine_->getNbBindings());
    buffer_sizes_.resize(engine_->getNbBindings());
    
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        auto dims = engine_->getBindingDimensions(i);
        size_t size = 1;
        
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        
        // Multiply by element size (assuming float32)
        size *= sizeof(float);
        
        auto error = cudaMalloc(&device_buffers_[i], size);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for binding " << i << std::endl;
            return false;
        }
        
        buffer_sizes_[i] = size;
    }
    
    return true;
}

void TensorRTEngine::free_buffers() {
    for (auto buffer : device_buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    
    device_buffers_.clear();
    buffer_sizes_.clear();
}

// TensorRTExporter implementation

TensorRTExporter::TensorRTExporter(const TensorRTConfig& config)
    : config_(config) {
    
    if (!initialize_tensorrt()) {
        std::cerr << "Failed to initialize TensorRT" << std::endl;
    }
}

TensorRTExporter::~TensorRTExporter() {
    cleanup_tensorrt();
}

bool TensorRTExporter::export_model_to_onnx(torch::jit::script::Module& model,
                                           const std::string& onnx_path,
                                           const std::vector<std::vector<int>>& input_shapes) {
    try {
        // Set model to eval mode
        model.eval();
        
        // Create example inputs
        std::vector<torch::jit::IValue> example_inputs;
        for (const auto& shape : input_shapes) {
            auto input = torch::randn(shape, torch::kFloat32);
            example_inputs.push_back(input);
        }
        
        // Export to ONNX
        torch::jit::export_onnx(
            model,
            onnx_path,
            example_inputs,
            torch::jit::ExportONNXOptions(),
            {},
            {},
            true // Operator version 11
        );
        
        std::cout << "Model exported to ONNX: " << onnx_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error exporting to ONNX: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRTExporter::export_onnx_to_tensorrt(const std::string& onnx_path,
                                             const std::string& engine_path,
                                             const std::vector<std::vector<int>>& input_shapes) {
    return build_tensorrt_engine(onnx_path, engine_path, input_shapes);
}

bool TensorRTExporter::export_pytorch_to_tensorrt(torch::jit::script::Module& model,
                                                 const std::string& engine_path,
                                                 const std::vector<std::vector<int>>& input_shapes) {
    // First export to ONNX
    std::string temp_onnx_path = engine_path + ".temp.onnx";
    if (!export_model_to_onnx(model, temp_onnx_path, input_shapes)) {
        return false;
    }
    
    // Then convert to TensorRT
    bool success = export_onnx_to_tensorrt(temp_onnx_path, engine_path, input_shapes);
    
    // Clean up temporary ONNX file
    std::remove(temp_onnx_path.c_str());
    
    return success;
}

bool TensorRTExporter::export_ensemble_models(const std::vector<std::pair<std::string, torch::jit::script::Module>>& models,
                                              const std::string& output_directory,
                                              const std::vector<std::vector<int>>& input_shapes) {
    // Create output directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + output_directory;
    system(mkdir_cmd.c_str());
    
    bool all_success = true;
    
    for (const auto& [name, model] : models) {
        std::string engine_path = output_directory + "/" + name + ".trt";
        
        std::cout << "Exporting model: " << name << std::endl;
        bool success = export_pytorch_to_tensorrt(
            const_cast<torch::jit::script::Module&>(model), // Remove const for export
            engine_path,
            input_shapes
        );
        
        if (!success) {
            std::cerr << "Failed to export model: " << name << std::endl;
            all_success = false;
        }
    }
    
    return all_success;
}

bool TensorRTExporter::validate_exported_engine(const std::string& engine_path,
                                              torch::jit::script::Module& original_model,
                                              const std::vector<torch::Tensor>& test_inputs,
                                              double tolerance) {
    try {
        // Load TensorRT engine
        auto trt_engine = std::make_unique<TensorRTEngine>(engine_path);
        if (!trt_engine->load_engine()) {
            std::cerr << "Failed to load TensorRT engine for validation" << std::endl;
            return false;
        }
        
        // Run PyTorch inference
        original_model.eval();
        std::vector<torch::jit::IValue> pytorch_inputs;
        for (const auto& input : test_inputs) {
            pytorch_inputs.push_back(input);
        }
        auto pytorch_output = original_model.forward(pytorch_inputs).toTensor();
        
        // Run TensorRT inference
        auto trt_outputs = trt_engine->infer(test_inputs);
        if (trt_outputs.empty()) {
            std::cerr << "TensorRT inference returned no outputs" << std::endl;
            return false;
        }
        
        auto trt_output = trt_outputs[0];
        
        // Compare outputs
        bool validation_passed = compare_outputs(pytorch_output, trt_output, tolerance);
        
        std::cout << "Engine validation: " << (validation_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Max difference: " << torch::max(torch::abs(pytorch_output - trt_output)).item<double>() << std::endl;
        
        return validation_passed;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during validation: " << e.what() << std::endl;
        return false;
    }
}

TensorRTExporter::PerformanceComparison TensorRTExporter::compare_performance(torch::jit::script::Module& original_model,
                                                                             const std::string& engine_path,
                                                                             const std::vector<torch::Tensor>& test_inputs,
                                                                             int iterations) {
    PerformanceComparison comparison{};
    
    try {
        // Load TensorRT engine
        auto trt_engine = std::make_unique<TensorRTEngine>(engine_path);
        if (!trt_engine->load_engine()) {
            std::cerr << "Failed to load TensorRT engine for comparison" << std::endl;
            return comparison;
        }
        
        // Benchmark PyTorch
        original_model.eval();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            std::vector<torch::jit::IValue> inputs;
            for (const auto& input : test_inputs) {
                inputs.push_back(input);
            }
            original_model.forward(inputs);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto pytorch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        comparison.pytorch_time_ms = static_cast<double>(pytorch_duration.count()) / iterations / 1000.0;
        
        // Benchmark TensorRT
        comparison.tensorrt_time_ms = trt_engine->benchmark_inference(test_inputs, iterations);
        
        // Calculate speedup
        if (comparison.tensorrt_time_ms > 0) {
            comparison.speedup_factor = comparison.pytorch_time_ms / comparison.tensorrt_time_ms;
        }
        
        // Validate accuracy
        std::vector<torch::jit::IValue> pytorch_inputs;
        for (const auto& input : test_inputs) {
            pytorch_inputs.push_back(input);
        }
        auto pytorch_output = original_model.forward(pytorch_inputs).toTensor();
        auto trt_outputs = trt_engine->infer(test_inputs);
        
        if (!trt_outputs.empty()) {
            auto trt_output = trt_outputs[0];
            auto max_diff = torch::max(torch::abs(pytorch_output - trt_output)).item<double>();
            comparison.accuracy_difference = max_diff;
            comparison.passed_validation = max_diff < 1e-3;
        }
        
        // Print results
        std::cout << "Performance Comparison Results:" << std::endl;
        std::cout << "  PyTorch time: " << comparison.pytorch_time_ms << "ms" << std::endl;
        std::cout << "  TensorRT time: " << comparison.tensorrt_time_ms << "ms" << std::endl;
        std::cout << "  Speedup: " << comparison.speedup_factor << "x" << std::endl;
        std::cout << "  Accuracy diff: " << comparison.accuracy_difference << std::endl;
        std::cout << "  Validation: " << (comparison.passed_validation ? "PASSED" : "FAILED") << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during performance comparison: " << e.what() << std::endl;
    }
    
    return comparison;
}

bool TensorRTExporter::is_tensorrt_available() const {
    return check_tensorrt_availability();
}

std::string TensorRTExporter::get_tensorrt_version() const {
    // This would return actual TensorRT version
    return "8.0.0"; // Placeholder
}

std::vector<std::string> TensorRTExporter::get_supported_precisions() const {
    std::vector<std::string> precisions;
    precisions.push_back("FP32");
    
    if (config_.enable_fp16) {
        precisions.push_back("FP16");
    }
    
    if (config_.enable_int8) {
        precisions.push_back("INT8");
    }
    
    return precisions;
}

std::unique_ptr<TensorRTEngine> TensorRTExporter::load_engine(const std::string& engine_path) {
    auto engine = std::make_unique<TensorRTEngine>(engine_path);
    if (engine->load_engine()) {
        return engine;
    }
    
    return nullptr;
}

bool TensorRTExporter::optimize_engine(const std::string& engine_path, const std::string& optimized_path) {
    // This would implement additional optimizations
    // For now, just copy the file
    std::ifstream src(engine_path, std::ios::binary);
    std::ofstream dst(optimized_path, std::ios::binary);
    
    dst << src.rdbuf();
    
    return true;
}

// Private methods

bool TensorRTExporter::initialize_tensorrt() {
    try {
        builder_ = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(nullptr));
        if (!builder_) {
            std::cerr << "Failed to create TensorRT builder" << std::endl;
            return false;
        }
        
        network_ = std::unique_ptr<nvinfer1::INetworkDefinition>(builder_->createNetworkV2(1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        if (!network_) {
            std::cerr << "Failed to create TensorRT network" << std::endl;
            return false;
        }
        
        parser_ = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_, nullptr));
        if (!parser_) {
            std::cerr << "Failed to create ONNX parser" << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing TensorRT: " << e.what() << std::endl;
        return false;
    }
}

void TensorRTExporter::cleanup_tensorrt() {
    parser_.reset();
    network_.reset();
    builder_.reset();
    plan_.reset();
}

bool TensorRTExporter::build_tensorrt_engine(const std::string& onnx_path,
                                             const std::string& engine_path,
                                             const std::vector<std::vector<int>>& input_shapes) {
    try {
        // Parse ONNX model
        std::ifstream file(onnx_path, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open ONNX file: " << onnx_path << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> onnx_data(size);
        file.read(onnx_data.data(), size);
        file.close();
        
        if (!parser_->parse(onnx_data.data(), size)) {
            std::cerr << "Failed to parse ONNX model" << std::endl;
            return false;
        }
        
        // Configure builder
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
        config->setMaxWorkspaceSize(config_.max_workspace_size);
        
        // Set precision flags
        if (config_.enable_fp16) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        if (config_.enable_int8) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // Would need to set calibrator here
        }
        
        // Build engine
        std::cout << "Building TensorRT engine..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        plan_ = std::unique_ptr<nvinfer1::IHostMemory>(builder_->buildSerializedNetwork(*network_, *config));
        if (!plan_) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "Engine built in " << build_time.count() << " seconds" << std::endl;
        
        // Save engine to file
        std::ofstream engine_file(engine_path, std::ios::binary);
        engine_file.write(reinterpret_cast<const char*>(plan_->data()), plan_->size());
        engine_file.close();
        
        std::cout << "TensorRT engine saved: " << engine_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error building TensorRT engine: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRTExporter::compare_outputs(const torch::Tensor& pytorch_output,
                                      const torch::Tensor& tensorrt_output,
                                      double tolerance) {
    auto diff = torch::abs(pytorch_output - tensorrt_output);
    auto max_diff = torch::max(diff).item<double>();
    
    return max_diff < tolerance;
}

// TensorRTContext implementation

TensorRTContext::TensorRTContext(const TensorRTConfig& config)
    : valid_(false) {
    
    try {
        exporter_ = std::make_unique<TensorRTExporter>(config);
        valid_ = exporter_->is_tensorrt_available();
        
        if (valid_) {
            std::cout << "TensorRT context initialized" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing TensorRT context: " << e.what() << std::endl;
    }
}

TensorRTContext::~TensorRTContext() {
    // Cleanup handled by unique_ptr
}

bool TensorRTContext::is_valid() const {
    return valid_ && exporter_;
}

TensorRTExporter& TensorRTContext::get_exporter() {
    if (!is_valid()) {
        throw std::runtime_error("TensorRT context not valid");
    }
    
    return *exporter_;
}

// Utility functions

namespace tensorrt_utils {
    
bool check_tensorrt_availability() {
    try {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(nullptr));
        return builder != nullptr;
    } catch (...) {
        return false;
    }
}

std::string get_device_info() {
    int device = 0;
    cudaDeviceProp prop;
    auto error = cudaGetDeviceProperties(&prop, device);
    
    if (error != cudaSuccess) {
        return "Unknown device";
    }
    
    std::stringstream info;
    info << prop.name << " (Compute " << prop.major << "." << prop.minor << ")";
    return info.str();
}

bool validate_cuda_compatibility() {
    int device = 0;
    cudaDeviceProp prop;
    auto error = cudaGetDeviceProperties(&prop, device);
    
    if (error != cudaSuccess) {
        return false;
    }
    
    // Check for minimum compute capability
    return (prop.major >= 6) || (prop.major == 5 && prop.minor >= 3);
}

} // namespace tensorrt_utils

} // namespace gpu
} // namespace archneuronx
