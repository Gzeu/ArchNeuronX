/**
 * @file model_factory.hpp
 * @brief Model factory for creating and managing neural network models
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>

#include "core/base_model.hpp"
#include "mlp_model.hpp"
#include "cnn_model.hpp"
#include "hybrid_model.hpp"
#include "core/config.hpp"
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Models {

/**
 * @struct ModelCreationParams
 * @brief Parameters for model creation
 */
struct ModelCreationParams {
    std::string model_name;
    ModelType model_type;
    int64_t input_size;
    int64_t output_size;
    int64_t sequence_length = 60;  ///< For CNN and Hybrid models
    
    // Configuration variants
    std::optional<MLPConfig> mlp_config;
    std::optional<CNNConfig> cnn_config;
    std::optional<HybridConfig> hybrid_config;
    
    /**
     * @brief Validate creation parameters
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate() const;
};

/**
 * @struct ModelMetadata
 * @brief Metadata for registered models
 */
struct ModelMetadata {
    std::string name;
    ModelType type;
    std::string description;
    std::string version;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_trained;
    double best_accuracy = 0.0;
    int64_t parameter_count = 0;
    std::map<std::string, double> metrics;
    
    /**
     * @brief Convert to JSON string
     * @return JSON representation
     */
    std::string toJson() const;
};

/**
 * @brief Model creation function type
 */
using ModelCreator = std::function<std::unique_ptr<BaseModel>(const ModelCreationParams&)>;

/**
 * @class ModelFactory
 * @brief Factory for creating and managing neural network models
 */
class ModelFactory {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to model factory
     */
    static ModelFactory& getInstance();
    
    /**
     * @brief Register a model creator
     * @param type Model type
     * @param creator Creator function
     */
    void registerModelCreator(ModelType type, ModelCreator creator);
    
    /**
     * @brief Create model from parameters
     * @param params Model creation parameters
     * @return Unique pointer to created model
     */
    std::unique_ptr<BaseModel> createModel(const ModelCreationParams& params);
    
    /**
     * @brief Create model from configuration
     * @param config_path Path to model configuration file
     * @return Unique pointer to created model
     */
    std::unique_ptr<BaseModel> createModelFromConfig(const std::string& config_path);
    
    /**
     * @brief Create model from saved state
     * @param model_path Path to saved model file
     * @return Unique pointer to loaded model
     */
    std::unique_ptr<BaseModel> loadModel(const std::string& model_path);
    
    /**
     * @brief Save model with metadata
     * @param model Model to save
     * @param filepath Path to save model
     * @param metadata Model metadata
     */
    void saveModelWithMetadata(const BaseModel& model, 
                              const std::string& filepath,
                              const ModelMetadata& metadata);
    
    /**
     * @brief Get available model types
     * @return Vector of registered model types
     */
    std::vector<ModelType> getAvailableModelTypes() const;
    
    /**
     * @brief Get model metadata
     * @param model_path Path to model file
     * @return Model metadata
     */
    ModelMetadata getModelMetadata(const std::string& model_path);
    
    /**
     * @brief List all saved models in directory
     * @param models_dir Directory containing models
     * @return Vector of model metadata
     */
    std::vector<ModelMetadata> listSavedModels(const std::string& models_dir);
    
    /**
     * @brief Clone model architecture with new parameters
     * @param source_model Source model to clone
     * @param new_name Name for cloned model
     * @return Unique pointer to cloned model
     */
    std::unique_ptr<BaseModel> cloneModelArchitecture(
        const BaseModel& source_model,
        const std::string& new_name
    );
    
    /**
     * @brief Create model ensemble
     * @param ensemble_name Name for ensemble
     * @param model_configs Vector of model configurations
     * @return Unique pointer to ensemble model
     */
    std::unique_ptr<BaseModel> createEnsemble(
        const std::string& ensemble_name,
        const std::vector<ModelCreationParams>& model_configs
    );
    
    /**
     * @brief Get default configuration for model type
     * @param type Model type
     * @param input_size Input size
     * @param output_size Output size
     * @param sequence_length Sequence length (for CNN/Hybrid)
     * @return Default model creation parameters
     */
    ModelCreationParams getDefaultConfig(ModelType type, 
                                        int64_t input_size,
                                        int64_t output_size,
                                        int64_t sequence_length = 60);
    
    /**
     * @brief Compare model performance
     * @param model_paths Vector of model file paths
     * @param test_data Test dataset for evaluation
     * @return Map of model path to performance metrics
     */
    std::map<std::string, ModelMetrics> compareModels(
        const std::vector<std::string>& model_paths,
        const torch::data::DataLoader<>& test_data
    );
    
    /**
     * @brief Auto-select best model architecture
     * @param train_data Training dataset
     * @param val_data Validation dataset
     * @param input_size Input feature size
     * @param output_size Output size
     * @param max_models Maximum models to try
     * @return Best performing model
     */
    std::unique_ptr<BaseModel> autoSelectBestModel(
        const torch::data::DataLoader<>& train_data,
        const torch::data::DataLoader<>& val_data,
        int64_t input_size,
        int64_t output_size,
        int max_models = 5
    );
    
private:
    ModelFactory() {
        registerDefaultCreators();
    }
    
    // Prevent copying
    ModelFactory(const ModelFactory&) = delete;
    ModelFactory& operator=(const ModelFactory&) = delete;
    
    /**
     * @brief Register default model creators
     */
    void registerDefaultCreators();
    
    /**
     * @brief Parse configuration file
     * @param config_path Path to configuration file
     * @return Model creation parameters
     */
    ModelCreationParams parseConfigFile(const std::string& config_path);
    
    std::map<ModelType, ModelCreator> creators_;
    mutable std::mutex factory_mutex_;
};

/**
 * @class ModelManager
 * @brief Advanced model management with versioning and deployment
 */
class ModelManager {
public:
    /**
     * @brief Constructor
     * @param models_directory Directory for storing models
     */
    explicit ModelManager(const std::string& models_directory);
    
    /**
     * @brief Register a trained model
     * @param model Model to register
     * @param metadata Model metadata
     * @return Model version ID
     */
    std::string registerModel(std::unique_ptr<BaseModel> model, 
                             const ModelMetadata& metadata);
    
    /**
     * @brief Deploy model for production use
     * @param model_id Model ID to deploy
     * @return True if deployment successful
     */
    bool deployModel(const std::string& model_id);
    
    /**
     * @brief Get production model
     * @return Currently deployed model
     */
    std::shared_ptr<BaseModel> getProductionModel();
    
    /**
     * @brief Get model by ID
     * @param model_id Model ID
     * @return Model instance
     */
    std::shared_ptr<BaseModel> getModel(const std::string& model_id);
    
    /**
     * @brief List all registered models
     * @return Vector of model metadata
     */
    std::vector<ModelMetadata> listModels() const;
    
    /**
     * @brief Delete model and its files
     * @param model_id Model ID to delete
     * @return True if deletion successful
     */
    bool deleteModel(const std::string& model_id);
    
    /**
     * @brief Get model performance history
     * @param model_id Model ID
     * @return Vector of historical metrics
     */
    std::vector<ModelMetrics> getModelHistory(const std::string& model_id);
    
private:
    std::string models_dir_;
    std::map<std::string, std::shared_ptr<BaseModel>> loaded_models_;
    std::string production_model_id_;
    mutable std::mutex manager_mutex_;
    
    /**
     * @brief Generate unique model ID
     * @return Unique model ID string
     */
    std::string generateModelId();
};

} // namespace Models
} // namespace ArchNeuronX