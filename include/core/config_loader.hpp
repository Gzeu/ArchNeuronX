/**
 * @file config_loader.hpp
 * @brief Configuration loader interface for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

#include "data/data_aggregator.hpp"
#include "data/data_manager.hpp"
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Core {

/**
 * @enum PositionSizingMethod
 * @brief Position sizing methods for trading
 */
enum class PositionSizingMethod {
    FIXED,              ///< Fixed position size
    KELLY_CRITERION,   ///< Kelly criterion sizing
    RISK_PARITY,       ///< Risk parity allocation
    VOLATILITY_TARGET,  ///< Volatility targeting
    OPTIMAL_F          ///< Optimal f calculation
};

/**
 * @struct DataProviderConfig
 * @brief Configuration for individual data providers
 */
struct DataProviderConfig {
    bool enabled = false;
    std::string api_key;
    std::string api_secret;
    std::string user_id;
    int account_id = 0;
    bool use_testnet = false;
    std::string base_url;
    std::string ws_url;
    
    struct RateLimits {
        int requests_per_second = 10;
        int requests_per_minute = 600;
    } rate_limits;
    
    std::vector<std::string> supported_assets;
    std::vector<std::string> features;
    double reliability_score = 0.0;
    double latency_score = 0.0;
    std::string data_quality;
};

/**
 * @struct DataProvidersConfig
 * @brief Configuration for all data providers
 */
struct DataProvidersConfig {
    DataProviderConfig binance;
    DataProviderConfig coinbase;
    DataProviderConfig kraken;
    DataProviderConfig alpha_vantage;
    DataProviderConfig yahoo_finance;
    DataProviderConfig metatrader;
};

/**
 * @struct MonitoringConfig
 * @brief Monitoring and alerting configuration
 */
struct MonitoringConfig {
    int health_check_interval_seconds = 30;
    
    struct AlertThresholds {
        int latency_ms = 1000;
        double error_rate_percent = 5.0;
        int data_gap_minutes = 15;
    } alert_thresholds;
    
    bool enable_slack_alerts = false;
    bool enable_email_alerts = false;
    bool enable_prometheus_metrics = false;
};

/**
 * @struct TradingConfig
 * @brief Trading and risk management configuration
 */
struct TradingConfig {
    bool paper_trading = false;
    double max_position_size = 10000.0;
    bool enable_risk_management = true;
    bool simulation_mode = false;
    bool enable_circuit_breakers = true;
    double max_daily_loss = 5000.0;
    double var_confidence = 0.95;
    PositionSizingMethod position_sizing_method = PositionSizingMethod::KELLY_CRITERION;
    bool enable_stop_loss = true;
    bool enable_take_profit = true;
    double stop_loss_atr_multiplier = 2.0;
    double take_profit_atr_multiplier = 3.0;
    int max_positions_per_symbol = 5;
};

/**
 * @struct LoggingConfig
 * @brief Logging configuration
 */
struct LoggingConfig {
    std::string level = "info";
    bool enable_file_logging = true;
    bool enable_console_logging = true;
    std::string log_directory = "logs/";
    int max_log_file_size_mb = 500;
    bool enable_log_rotation = true;
    bool enable_structured_logging = true;
    bool enable_performance_logging = false;
};

/**
 * @struct Configuration
 * @brief Complete system configuration
 */
struct Configuration {
    std::string profile_name = "default";
    std::string environment = "development";
    
    DataProvidersConfig data_providers;
    Data::AggregationConfig data_aggregation;
    Data::DataManagerConfig data_manager;
    MonitoringConfig monitoring;
    TradingConfig trading;
    LoggingConfig logging;
    
    /**
     * @brief Convert configuration to JSON string
     * @return JSON representation
     */
    std::string to_json() const;
    
    /**
     * @brief Validate configuration
     * @return True if configuration is valid
     */
    bool validate() const;
};

/**
 * @class ConfigLoader
 * @brief Configuration loader and manager for ArchNeuronX
 */
class ConfigLoader {
public:
    /**
     * @brief Constructor
     */
    ConfigLoader();
    
    /**
     * @brief Destructor
     */
    ~ConfigLoader();
    
    /**
     * @brief Load configuration profile by name
     * @param profile_name Name of the profile to load
     * @return Unique pointer to configuration
     */
    std::unique_ptr<Configuration> load_profile(const std::string& profile_name);
    
    /**
     * @brief Load configuration from file path
     * @param file_path Path to configuration file
     * @return Unique pointer to configuration
     */
    std::unique_ptr<Configuration> load_from_file(const std::string& file_path);
    
    /**
     * @brief Load configuration from JSON string
     * @param config_string JSON configuration string
     * @return Unique pointer to configuration
     */
    std::unique_ptr<Configuration> load_from_string(const std::string& config_string);
    
    /**
     * @brief Get list of available configuration profiles
     * @return Vector of profile names
     */
    std::vector<std::string> get_available_profiles() const;
    
    /**
     * @brief Save configuration to profile
     * @param profile_name Name of the profile
     * @param config Configuration to save
     * @return True if saved successfully
     */
    bool save_profile(const std::string& profile_name, const Configuration& config);
    
    /**
     * @brief Reload configuration from profile
     * @param profile_name Name of the profile to reload
     * @return True if reloaded successfully
     */
    bool reload_profile(const std::string& profile_name) {
        auto config = load_profile(profile_name);
        if (config) {
            current_config_ = std::move(config);
            return true;
        }
        return false;
    }
    
    /**
     * @brief Get current configuration
     * @return Reference to current configuration
     */
    const Configuration& get_current_config() const {
        return *current_config_;
    }
    
    /**
     * @brief Set configuration change callback
     * @param callback Function to call on configuration change
     */
    void set_change_callback(std::function<void(const Configuration&)> callback) {
        change_callback_ = callback;
    }

private:
    std::unique_ptr<Configuration> current_config_;
    std::function<void(const Configuration&)> change_callback_;
    
    // Private methods
    std::string get_config_path(const std::string& profile_name) const;
    std::string get_profiles_directory() const;
    std::unique_ptr<Configuration> load_default_config();
    
    void parse_configuration(const nlohmann::json& config_json, Configuration& config);
    void apply_environment_overrides(Configuration& config);
    bool validate_configuration(const Configuration& config);
    
    // Utility methods
    Data::AggregationMethod parse_aggregation_method(const std::string& method_str);
    PositionSizingMethod parse_position_sizing_method(const std::string& method_str);
    nlohmann::json serialize_configuration(const Configuration& config);
    std::string aggregation_method_to_string(Data::AggregationMethod method);
};

} // namespace Core
} // namespace ArchNeuronX
