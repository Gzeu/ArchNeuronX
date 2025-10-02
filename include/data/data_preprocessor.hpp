/**
 * @file data_preprocessor.hpp
 * @brief Data preprocessing pipeline for ArchNeuronX ML models
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <optional>
#include <functional>
#include <torch/torch.h>

#include "market_data.hpp"
#include "technical_indicators.hpp"
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Data {

/**
 * @enum NormalizationMethod
 * @brief Data normalization methods
 */
enum class NormalizationMethod {
    MIN_MAX,        ///< Min-Max normalization (0-1)
    Z_SCORE,        ///< Z-score standardization (mean=0, std=1)
    ROBUST,         ///< Robust scaling (median and IQR)
    QUANTILE,       ///< Quantile normalization
    LOG_TRANSFORM,  ///< Logarithmic transformation
    NONE            ///< No normalization
};

/**
 * @enum ImputationMethod
 * @brief Missing data imputation methods
 */
enum class ImputationMethod {
    FORWARD_FILL,   ///< Forward fill (carry last observation)
    BACKWARD_FILL,  ///< Backward fill
    LINEAR_INTERP,  ///< Linear interpolation
    MEAN_FILL,      ///< Fill with mean value
    MEDIAN_FILL,    ///< Fill with median value
    ZERO_FILL,      ///< Fill with zero
    DROP_ROWS       ///< Drop rows with missing values
};

/**
 * @struct NormalizationParams
 * @brief Parameters for data normalization
 */
struct NormalizationParams {
    NormalizationMethod method = NormalizationMethod::Z_SCORE;
    double min_value = 0.0;     ///< Min value for min-max scaling
    double max_value = 1.0;     ///< Max value for min-max scaling
    double mean = 0.0;          ///< Mean for z-score
    double std_dev = 1.0;       ///< Standard deviation for z-score
    double median = 0.0;        ///< Median for robust scaling
    double iqr = 1.0;           ///< Interquartile range for robust scaling
    bool fit_on_train = true;   ///< Fit parameters on training data only
    
    /**
     * @brief Convert to JSON string
     * @return JSON representation
     */
    std::string toJson() const;
    
    /**
     * @brief Create from JSON string
     * @param json_str JSON string
     * @return NormalizationParams object
     */
    static NormalizationParams fromJson(const std::string& json_str);
};

/**
 * @struct PreprocessingConfig
 * @brief Configuration for data preprocessing pipeline
 */
struct PreprocessingConfig {
    // Sequence parameters
    int sequence_length = 60;           ///< Input sequence length
    int prediction_horizon = 1;         ///< Prediction horizon (steps ahead)
    int step_size = 1;                 ///< Step size for sliding window
    
    // Feature engineering
    bool include_technical_indicators = true;
    bool include_price_differences = true;
    bool include_returns = true;
    bool include_volatility = true;
    bool include_volume_features = true;
    
    // Data cleaning
    ImputationMethod imputation_method = ImputationMethod::LINEAR_INTERP;
    double outlier_threshold = 3.0;     ///< Z-score threshold for outlier detection
    bool remove_outliers = false;
    bool handle_missing_data = true;
    
    // Normalization
    NormalizationMethod normalization_method = NormalizationMethod::Z_SCORE;
    bool normalize_features = true;
    bool normalize_targets = true;
    
    // Data splits
    double train_ratio = 0.7;
    double validation_ratio = 0.2;
    double test_ratio = 0.1;
    
    // Feature selection
    bool enable_feature_selection = false;
    int max_features = 100;
    double correlation_threshold = 0.95;
    
    /**
     * @brief Validate configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
};

/**
 * @struct ProcessedDataset
 * @brief Container for processed dataset
 */
struct ProcessedDataset {
    torch::Tensor features;              ///< Input features tensor
    torch::Tensor targets;               ///< Target values tensor
    torch::Tensor timestamps;            ///< Timestamp tensor
    std::vector<std::string> symbols;    ///< Symbol names
    std::vector<std::string> feature_names; ///< Feature names
    
    // Dataset splits
    torch::Tensor train_features;
    torch::Tensor train_targets;
    torch::Tensor val_features;
    torch::Tensor val_targets;
    torch::Tensor test_features;
    torch::Tensor test_targets;
    
    // Normalization parameters
    std::map<std::string, NormalizationParams> feature_params;
    std::map<std::string, NormalizationParams> target_params;
    
    /**
     * @brief Get dataset information
     * @return Formatted dataset info string
     */
    std::string getInfo() const;
    
    /**
     * @brief Save dataset to disk
     * @param filepath Path to save dataset
     */
    void save(const std::string& filepath) const;
    
    /**
     * @brief Load dataset from disk
     * @param filepath Path to load dataset from
     */
    void load(const std::string& filepath);
};

/**
 * @class DataPreprocessor
 * @brief Comprehensive data preprocessing pipeline for ML models
 */
class DataPreprocessor {
public:
    /**
     * @brief Constructor
     * @param config Preprocessing configuration
     */
    explicit DataPreprocessor(const PreprocessingConfig& config);
    
    /**
     * @brief Destructor
     */
    ~DataPreprocessor() = default;
    
    /**
     * @brief Process raw market data for ML training
     * @param market_data Vector of market data points
     * @param target_column Target column name ("close", "high", "low", etc.)
     * @return Processed dataset ready for ML training
     */
    ProcessedDataset processMarketData(
        const std::vector<MarketDataPoint>& market_data,
        const std::string& target_column = "close"
    );
    
    /**
     * @brief Process single data point for real-time inference
     * @param data_point Single market data point
     * @param sequence_buffer Previous sequence buffer
     * @return Normalized feature tensor for inference
     */
    torch::Tensor processRealtimeData(
        const MarketDataPoint& data_point,
        const std::vector<MarketDataPoint>& sequence_buffer
    );
    
    /**
     * @brief Create sequences from time series data
     * @param features Feature matrix
     * @param targets Target vector
     * @param timestamps Timestamp vector
     * @return Tuple of (sequence features, sequence targets, sequence timestamps)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> createSequences(
        const torch::Tensor& features,
        const torch::Tensor& targets,
        const torch::Tensor& timestamps
    );
    
    /**
     * @brief Extract features from market data
     * @param market_data Vector of market data points
     * @return Tuple of (feature matrix, feature names)
     */
    std::tuple<torch::Tensor, std::vector<std::string>> extractFeatures(
        const std::vector<MarketDataPoint>& market_data
    );
    
    /**
     * @brief Extract targets from market data
     * @param market_data Vector of market data points
     * @param target_column Target column name
     * @return Target tensor
     */
    torch::Tensor extractTargets(
        const std::vector<MarketDataPoint>& market_data,
        const std::string& target_column
    );
    
    /**
     * @brief Normalize features using fitted parameters
     * @param features Feature tensor
     * @param feature_names Feature names
     * @param fit_params Whether to fit normalization parameters
     * @return Normalized feature tensor
     */
    torch::Tensor normalizeFeatures(
        const torch::Tensor& features,
        const std::vector<std::string>& feature_names,
        bool fit_params = false
    );
    
    /**
     * @brief Denormalize features back to original scale
     * @param normalized_features Normalized feature tensor
     * @param feature_names Feature names
     * @return Denormalized feature tensor
     */
    torch::Tensor denormalizeFeatures(
        const torch::Tensor& normalized_features,
        const std::vector<std::string>& feature_names
    );
    
    /**
     * @brief Handle missing values in dataset
     * @param features Feature tensor
     * @param method Imputation method
     * @return Tensor with missing values handled
     */
    torch::Tensor handleMissingValues(
        const torch::Tensor& features,
        ImputationMethod method
    );
    
    /**
     * @brief Detect and remove outliers
     * @param features Feature tensor
     * @param threshold Z-score threshold for outlier detection
     * @return Tuple of (cleaned features, outlier mask)
     */
    std::tuple<torch::Tensor, torch::Tensor> removeOutliers(
        const torch::Tensor& features,
        double threshold = 3.0
    );
    
    /**
     * @brief Split dataset into train/validation/test sets
     * @param features Feature tensor
     * @param targets Target tensor
     * @param timestamps Timestamp tensor
     * @return Tuple of split datasets
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor> splitDataset(
        const torch::Tensor& features,
        const torch::Tensor& targets,
        const torch::Tensor& timestamps
    );
    
    /**
     * @brief Calculate feature correlation matrix
     * @param features Feature tensor
     * @return Correlation matrix
     */
    torch::Tensor calculateCorrelationMatrix(const torch::Tensor& features);
    
    /**
     * @brief Select features based on correlation and importance
     * @param features Feature tensor
     * @param feature_names Feature names
     * @param targets Target tensor (optional for supervised selection)
     * @return Tuple of (selected features, selected feature names)
     */
    std::tuple<torch::Tensor, std::vector<std::string>> selectFeatures(
        const torch::Tensor& features,
        const std::vector<std::string>& feature_names,
        const std::optional<torch::Tensor>& targets = std::nullopt
    );
    
    /**
     * @brief Get preprocessing statistics
     * @return Map of statistic name to value
     */
    std::map<std::string, double> getStatistics() const;
    
    // Getters
    const PreprocessingConfig& getConfig() const { return config_; }
    const std::map<std::string, NormalizationParams>& getFeatureParams() const {
        return feature_normalization_params_;
    }
    const std::map<std::string, NormalizationParams>& getTargetParams() const {
        return target_normalization_params_;
    }
    
private:
    /**
     * @brief Fit normalization parameters on data
     * @param data Data tensor
     * @param method Normalization method
     * @return Fitted normalization parameters
     */
    NormalizationParams fitNormalizationParams(
        const torch::Tensor& data,
        NormalizationMethod method
    );
    
    /**
     * @brief Apply normalization to data
     * @param data Data tensor
     * @param params Normalization parameters
     * @return Normalized data tensor
     */
    torch::Tensor applyNormalization(
        const torch::Tensor& data,
        const NormalizationParams& params
    );
    
    /**
     * @brief Create technical indicator features
     * @param market_data Vector of market data points
     * @return Tuple of (indicator features, feature names)
     */
    std::tuple<torch::Tensor, std::vector<std::string>> createTechnicalFeatures(
        const std::vector<MarketDataPoint>& market_data
    );
    
    /**
     * @brief Create price difference features
     * @param ohlcv_data Vector of OHLCV data
     * @return Tuple of (price diff features, feature names)
     */
    std::tuple<torch::Tensor, std::vector<std::string>> createPriceDiffFeatures(
        const std::vector<OHLCV>& ohlcv_data
    );
    
    /**
     * @brief Create return features
     * @param prices Vector of prices
     * @param periods Vector of return periods
     * @return Tuple of (return features, feature names)
     */
    std::tuple<torch::Tensor, std::vector<std::string>> createReturnFeatures(
        const std::vector<double>& prices,
        const std::vector<int>& periods = {1, 5, 10, 20}
    );
    
    PreprocessingConfig config_;
    
    // Normalization parameters
    std::map<std::string, NormalizationParams> feature_normalization_params_;
    std::map<std::string, NormalizationParams> target_normalization_params_;
    
    // Statistics
    mutable std::map<std::string, double> statistics_;
    
    // Feature selection
    std::vector<int> selected_feature_indices_;
    torch::Tensor correlation_matrix_;
};

/**
 * @brief Convert NormalizationMethod to string
 * @param method Normalization method
 * @return String representation
 */
std::string normalizationMethodToString(NormalizationMethod method);

/**
 * @brief Convert ImputationMethod to string
 * @param method Imputation method
 * @return String representation
 */
std::string imputationMethodToString(ImputationMethod method);

} // namespace Data
} // namespace ArchNeuronX