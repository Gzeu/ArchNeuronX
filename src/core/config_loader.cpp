/**
 * @file config_loader.cpp
 * @brief Configuration loader implementation for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#include "core/config_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Core {

ConfigLoader::ConfigLoader() {
    // Default constructor
}

ConfigLoader::~ConfigLoader() {
    // Destructor
}

std::unique_ptr<Configuration> ConfigLoader::load_profile(const std::string& profile_name) {
    try {
        std::string config_path = get_config_path(profile_name);
        
        if (!std::filesystem::exists(config_path)) {
            LOG_ERROR("Configuration profile not found: {}", config_path);
            return load_default_config();
        }
        
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            LOG_ERROR("Failed to open configuration file: {}", config_path);
            return load_default_config();
        }
        
        json config_json;
        config_file >> config_json;
        
        auto config = std::make_unique<Configuration>();
        parse_configuration(config_json, *config);
        
        // Override with environment variables
        apply_environment_overrides(*config);
        
        // Validate configuration
        if (!validate_configuration(*config)) {
            LOG_ERROR("Invalid configuration in profile: {}", profile_name);
            return load_default_config();
        }
        
        LOG_INFO("Loaded configuration profile: {}", profile_name);
        return config;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load configuration profile '{}': {}", profile_name, e.what());
        return load_default_config();
    }
}

std::unique_ptr<Configuration> ConfigLoader::load_from_file(const std::string& file_path) {
    try {
        if (!std::filesystem::exists(file_path)) {
            LOG_ERROR("Configuration file not found: {}", file_path);
            return load_default_config();
        }
        
        std::ifstream config_file(file_path);
        if (!config_file.is_open()) {
            LOG_ERROR("Failed to open configuration file: {}", file_path);
            return load_default_config();
        }
        
        json config_json;
        config_file >> config_json;
        
        auto config = std::make_unique<Configuration>();
        parse_configuration(config_json, *config);
        
        // Override with environment variables
        apply_environment_overrides(*config);
        
        // Validate configuration
        if (!validate_configuration(*config)) {
            LOG_ERROR("Invalid configuration in file: {}", file_path);
            return load_default_config();
        }
        
        LOG_INFO("Loaded configuration from file: {}", file_path);
        return config;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load configuration from file '{}': {}", file_path, e.what());
        return load_default_config();
    }
}

std::unique_ptr<Configuration> ConfigLoader::load_from_string(const std::string& config_string) {
    try {
        json config_json = json::parse(config_string);
        
        auto config = std::make_unique<Configuration>();
        parse_configuration(config_json, *config);
        
        // Override with environment variables
        apply_environment_overrides(*config);
        
        // Validate configuration
        if (!validate_configuration(*config)) {
            LOG_ERROR("Invalid configuration in string");
            return load_default_config();
        }
        
        LOG_INFO("Loaded configuration from string");
        return config;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse configuration string: {}", e.what());
        return load_default_config();
    }
}

std::vector<std::string> ConfigLoader::get_available_profiles() const {
    std::vector<std::string> profiles;
    
    std::string profiles_dir = get_profiles_directory();
    
    if (std::filesystem::exists(profiles_dir) && 
        std::filesystem::is_directory(profiles_dir)) {
        
        for (const auto& entry : std::filesystem::directory_iterator(profiles_dir)) {
            if (entry.path().extension() == ".json") {
                profiles.push_back(entry.path().stem().string());
            }
        }
    }
    
    std::sort(profiles.begin(), profiles.end());
    
    LOG_INFO("Found {} configuration profiles", profiles.size());
    return profiles;
}

bool ConfigLoader::save_profile(const std::string& profile_name, const Configuration& config) {
    try {
        std::string config_path = get_config_path(profile_name);
        
        // Create directory if it doesn't exist
        std::filesystem::path dir = std::filesystem::path(config_path).parent_path();
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
        
        json config_json = serialize_configuration(config);
        
        std::ofstream config_file(config_path);
        if (!config_file.is_open()) {
            LOG_ERROR("Failed to create configuration file: {}", config_path);
            return false;
        }
        
        config_file << config_json.dump(4);
        config_file.close();
        
        LOG_INFO("Saved configuration profile: {}", profile_name);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to save configuration profile '{}': {}", profile_name, e.what());
        return false;
    }
}

std::string ConfigLoader::get_config_path(const std::string& profile_name) const {
    return get_profiles_directory() + "/" + profile_name + ".json";
}

std::string ConfigLoader::get_profiles_directory() const {
    return "config/profiles";
}

std::unique_ptr<Configuration> ConfigLoader::load_default_config() {
    auto config = std::make_unique<Configuration>();
    
    // Set sensible defaults
    config->data_manager.historical_days = 365;
    config->data_manager.cache_ttl_minutes = 5;
    config->data_manager.update_interval_ms = 1000;
    config->data_manager.enable_caching = true;
    
    config->data_aggregation.method = AggregationMethod::WEIGHTED_AVERAGE;
    config->data_aggregation.min_sources = 2;
    config->data_aggregation.enable_quality_scoring = true;
    
    config->monitoring.health_check_interval_seconds = 30;
    config->monitoring.alert_thresholds.latency_ms = 1000;
    
    config->trading.max_position_size = 10000;
    config->trading.enable_risk_management = true;
    
    config->logging.level = "info";
    config->logging.enable_file_logging = true;
    
    LOG_INFO("Loaded default configuration");
    return config;
}

void ConfigLoader::parse_configuration(const json& config_json, Configuration& config) {
    // Parse data providers
    if (config_json.contains("data_providers")) {
        const auto& providers = config_json["data_providers"];
        
        // Binance
        if (providers.contains("binance")) {
            const auto& binance = providers["binance"];
            config.data_providers.binance.enabled = binance.value("enabled", true);
            config.data_providers.binance.api_key = binance.value("api_key", "");
            config.data_providers.binance.api_secret = binance.value("api_secret", "");
            config.data_providers.binance.use_testnet = binance.value("use_testnet", false);
            
            if (binance.contains("rate_limits")) {
                const auto& limits = binance["rate_limits"];
                config.data_providers.binance.rate_limits.requests_per_second = 
                    limits.value("requests_per_second", 10);
                config.data_providers.binance.rate_limits.requests_per_minute = 
                    limits.value("requests_per_minute", 1200);
            }
        }
        
        // Alpha Vantage
        if (providers.contains("alpha_vantage")) {
            const auto& av = providers["alpha_vantage"];
            config.data_providers.alpha_vantage.enabled = av.value("enabled", true);
            config.data_providers.alpha_vantage.api_key = av.value("api_key", "");
            
            if (av.contains("rate_limits")) {
                const auto& limits = av["rate_limits"];
                config.data_providers.alpha_vantage.rate_limits.requests_per_minute = 
                    limits.value("requests_per_minute", 5);
            }
        }
        
        // Yahoo Finance
        if (providers.contains("yahoo_finance")) {
            const auto& yahoo = providers["yahoo_finance"];
            config.data_providers.yahoo_finance.enabled = yahoo.value("enabled", true);
            
            if (yahoo.contains("rate_limits")) {
                const auto& limits = yahoo["rate_limits"];
                config.data_providers.yahoo_finance.rate_limits.requests_per_minute = 
                    limits.value("requests_per_minute", 100);
            }
        }
        
        // MetaTrader
        if (providers.contains("metatrader")) {
            const auto& mt = providers["metatrader"];
            config.data_providers.metatrader.enabled = mt.value("enabled", false);
            config.data_providers.metatrader.api_key = mt.value("api_key", "");
            config.data_providers.metatrader.user_id = mt.value("user_id", "");
            config.data_providers.metatrader.account_id = mt.value("account_id", 0);
        }
        
        // Kraken
        if (providers.contains("kraken")) {
            const auto& kraken = providers["kraken"];
            config.data_providers.kraken.enabled = kraken.value("enabled", false);
            config.data_providers.kraken.api_key = kraken.value("api_key", "");
            config.data_providers.kraken.api_secret = kraken.value("api_secret", "");
        }
        
        // Coinbase
        if (providers.contains("coinbase")) {
            const auto& coinbase = providers["coinbase"];
            config.data_providers.coinbase.enabled = coinbase.value("enabled", false);
            config.data_providers.coinbase.api_key = coinbase.value("api_key", "");
            config.data_providers.coinbase.api_secret = coinbase.value("api_secret", "");
            config.data_providers.coinbase.passphrase = coinbase.value("passphrase", "");
        }
    }
    
    // Parse data aggregation
    if (config_json.contains("data_aggregation")) {
        const auto& agg = config_json["data_aggregation"];
        
        config.data_aggregation.method = parse_aggregation_method(
            agg.value("method", "weighted_average"));
        config.data_aggregation.min_sources = agg.value("min_sources", 2);
        config.data_aggregation.max_sources = agg.value("max_sources", 5);
        config.data_aggregation.enable_quality_scoring = agg.value("enable_quality_scoring", true);
        config.data_aggregation.enable_outlier_detection = agg.value("enable_outlier_detection", true);
        config.data_aggregation.outlier_threshold = agg.value("outlier_threshold", 3.0);
        config.data_aggregation.staleness_penalty_minutes = agg.value("staleness_penalty_minutes", 5.0);
        config.data_aggregation.enable_cross_validation = agg.value("enable_cross_validation", true);
        
        if (agg.contains("source_weights")) {
            const auto& weights = agg["source_weights"];
            for (const auto& [provider, weight] : weights.items()) {
                config.data_aggregation.source_weights[provider] = weight.get<double>();
            }
        }
    }
    
    // Parse data manager
    if (config_json.contains("data_manager")) {
        const auto& dm = config_json["data_manager"];
        
        config.data_manager.historical_days = dm.value("historical_days", 365);
        config.data_manager.cache_ttl_minutes = dm.value("cache_ttl_minutes", 5);
        config.data_manager.cache_size_mb = dm.value("cache_size_mb", 1024);
        config.data_manager.update_interval_ms = dm.value("update_interval_ms", 1000);
        config.data_manager.enable_caching = dm.value("enable_caching", true);
        config.data_manager.enable_persistence = dm.value("enable_persistence", true);
        config.data_manager.data_directory = dm.value("data_directory", "data/");
        config.data_manager.max_concurrent_requests = dm.value("max_concurrent_requests", 10);
        config.data_manager.enable_data_validation = dm.value("enable_data_validation", true);
        config.data_manager.enable_technical_indicators = dm.value("enable_technical_indicators", true);
        config.data_manager.backfill_missing_data = dm.value("backfill_missing_data", true);
    }
    
    // Parse monitoring
    if (config_json.contains("monitoring")) {
        const auto& mon = config_json["monitoring"];
        
        config.monitoring.health_check_interval_seconds = mon.value("health_check_interval_seconds", 30);
        
        if (mon.contains("alert_thresholds")) {
            const auto& thresholds = mon["alert_thresholds"];
            config.monitoring.alert_thresholds.latency_ms = 
                thresholds.value("latency_ms", 1000);
            config.monitoring.alert_thresholds.error_rate_percent = 
                thresholds.value("error_rate_percent", 5.0);
            config.monitoring.alert_thresholds.data_gap_minutes = 
                thresholds.value("data_gap_minutes", 15);
        }
    }
    
    // Parse trading
    if (config_json.contains("trading")) {
        const auto& trading = config_json["trading"];
        
        config.trading.paper_trading = trading.value("paper_trading", false);
        config.trading.max_position_size = trading.value("max_position_size", 10000);
        config.trading.enable_risk_management = trading.value("enable_risk_management", true);
        config.trading.simulation_mode = trading.value("simulation_mode", false);
        config.trading.max_daily_loss = trading.value("max_daily_loss", 5000);
        config.trading.var_confidence = trading.value("var_confidence", 0.95);
        config.trading.position_sizing_method = parse_position_sizing_method(
            trading.value("position_sizing_method", "kelly_criterion"));
    }
    
    // Parse logging
    if (config_json.contains("logging")) {
        const auto& logging = config_json["logging"];
        
        config.logging.level = logging.value("level", "info");
        config.logging.enable_file_logging = logging.value("enable_file_logging", true);
        config.logging.enable_console_logging = logging.value("enable_console_logging", true);
        config.logging.log_directory = logging.value("log_directory", "logs/");
        config.logging.max_log_file_size_mb = logging.value("max_log_file_size_mb", 500);
        config.logging.enable_log_rotation = logging.value("enable_log_rotation", true);
        config.logging.enable_structured_logging = logging.value("enable_structured_logging", true);
    }
}

AggregationMethod ConfigLoader::parse_aggregation_method(const std::string& method_str) {
    if (method_str == "first_available") return AggregationMethod::FIRST_AVAILABLE;
    if (method_str == "weighted_average") return AggregationMethod::WEIGHTED_AVERAGE;
    if (method_str == "median") return AggregationMethod::MEDIAN;
    if (method_str == "consensus") return AggregationMethod::CONSENSUS;
    if (method_str == "latest_timestamp") return AggregationMethod::LATEST_TIMESTAMP;
    if (method_str == "best_quality") return AggregationMethod::BEST_QUALITY;
    if (method_str == "custom") return AggregationMethod::CUSTOM;
    
    return AggregationMethod::WEIGHTED_AVERAGE; // Default
}

PositionSizingMethod ConfigLoader::parse_position_sizing_method(const std::string& method_str) {
    if (method_str == "fixed") return PositionSizingMethod::FIXED;
    if (method_str == "kelly_criterion") return PositionSizingMethod::KELLY_CRITERION;
    if (method_str == "risk_parity") return PositionSizingMethod::RISK_PARITY;
    if (method_str == "volatility_target") return PositionSizingMethod::VOLATILITY_TARGET;
    if (method_str == "optimal_f") return PositionSizingMethod::OPTIMAL_F;
    
    return PositionSizingMethod::KELLY_CRITERION; // Default
}

void ConfigLoader::apply_environment_overrides(Configuration& config) {
    // Override with environment variables
    const char* env_var;
    
    // Binance API key
    if ((env_var = std::getenv("BINANCE_API_KEY")) != nullptr) {
        config.data_providers.binance.api_key = env_var;
    }
    
    // Binance API secret
    if ((env_var = std::getenv("BINANCE_API_SECRET")) != nullptr) {
        config.data_providers.binance.api_secret = env_var;
    }
    
    // Alpha Vantage API key
    if ((env_var = std::getenv("ALPHA_VANTAGE_API_KEY")) != nullptr) {
        config.data_providers.alpha_vantage.api_key = env_var;
    }
    
    // MetaTrader API key
    if ((env_var = std::getenv("METATRADER_API_KEY")) != nullptr) {
        config.data_providers.metatrader.api_key = env_var;
    }
    
    // MetaTrader user ID
    if ((env_var = std::getenv("METATRADER_USER_ID")) != nullptr) {
        config.data_providers.metatrader.user_id = env_var;
    }
    
    // MetaTrader account ID
    if ((env_var = std::getenv("METATRADER_ACCOUNT_ID")) != nullptr) {
        config.data_providers.metatrader.account_id = std::stoi(env_var);
    }
    
    // Profile name
    if ((env_var = std::getenv("ARCHNEURONX_PROFILE")) != nullptr) {
        config.profile_name = env_var;
    }
    
    // Environment (dev/test/prod)
    if ((env_var = std::getenv("ARCHNEURONX_ENV")) != nullptr) {
        config.environment = env_var;
    }
    
    LOG_INFO("Applied environment variable overrides");
}

bool ConfigLoader::validate_configuration(const Configuration& config) {
    // Validate data providers
    if (config.data_providers.binance.enabled) {
        if (config.data_providers.binance.api_key.empty()) {
            LOG_ERROR("Binance enabled but API key not provided");
            return false;
        }
    }
    
    if (config.data_providers.alpha_vantage.enabled) {
        if (config.data_providers.alpha_vantage.api_key.empty()) {
            LOG_ERROR("Alpha Vantage enabled but API key not provided");
            return false;
        }
    }
    
    // Validate data aggregation
    if (config.data_aggregation.min_sources < 1) {
        LOG_ERROR("Data aggregation min_sources must be >= 1");
        return false;
    }
    
    if (config.data_aggregation.max_sources < config.data_aggregation.min_sources) {
        LOG_ERROR("Data aggregation max_sources must be >= min_sources");
        return false;
    }
    
    // Validate data manager
    if (config.data_manager.historical_days < 1) {
        LOG_ERROR("Historical days must be >= 1");
        return false;
    }
    
    if (config.data_manager.cache_size_mb < 1) {
        LOG_ERROR("Cache size must be >= 1 MB");
        return false;
    }
    
    // Validate trading
    if (config.trading.max_position_size <= 0) {
        LOG_ERROR("Max position size must be > 0");
        return false;
    }
    
    if (config.trading.var_confidence <= 0 || config.trading.var_confidence >= 1) {
        LOG_ERROR("VaR confidence must be between 0 and 1");
        return false;
    }
    
    // Validate monitoring
    if (config.monitoring.health_check_interval_seconds <= 0) {
        LOG_ERROR("Health check interval must be > 0");
        return false;
    }
    
    LOG_INFO("Configuration validation passed");
    return true;
}

json ConfigLoader::serialize_configuration(const Configuration& config) {
    json config_json;
    
    // Serialize data providers
    json providers;
    
    json binance;
    binance["enabled"] = config.data_providers.binance.enabled;
    binance["api_key"] = config.data_providers.binance.api_key;
    binance["use_testnet"] = config.data_providers.binance.use_testnet;
    
    json binance_limits;
    binance_limits["requests_per_second"] = config.data_providers.binance.rate_limits.requests_per_second;
    binance_limits["requests_per_minute"] = config.data_providers.binance.rate_limits.requests_per_minute;
    binance["rate_limits"] = binance_limits;
    providers["binance"] = binance;
    
    json alpha_vantage;
    alpha_vantage["enabled"] = config.data_providers.alpha_vantage.enabled;
    alpha_vantage["api_key"] = config.data_providers.alpha_vantage.api_key;
    
    json av_limits;
    av_limits["requests_per_minute"] = config.data_providers.alpha_vantage.rate_limits.requests_per_minute;
    alpha_vantage["rate_limits"] = av_limits;
    providers["alpha_vantage"] = alpha_vantage;
    
    json yahoo_finance;
    yahoo_finance["enabled"] = config.data_providers.yahoo_finance.enabled;
    
    json yahoo_limits;
    yahoo_limits["requests_per_minute"] = config.data_providers.yahoo_finance.rate_limits.requests_per_minute;
    yahoo_finance["rate_limits"] = yahoo_limits;
    providers["yahoo_finance"] = yahoo_finance;
    
    providers["coinbase"] = json::object();
    providers["kraken"] = json::object();
    providers["metatrader"] = json::object();
    
    config_json["data_providers"] = providers;
    
    // Serialize data aggregation
    json aggregation;
    aggregation["method"] = aggregation_method_to_string(config.data_aggregation.method);
    aggregation["min_sources"] = config.data_aggregation.min_sources;
    aggregation["max_sources"] = config.data_aggregation.max_sources;
    aggregation["enable_quality_scoring"] = config.data_aggregation.enable_quality_scoring;
    aggregation["enable_outlier_detection"] = config.data_aggregation.enable_outlier_detection;
    aggregation["outlier_threshold"] = config.data_aggregation.outlier_threshold;
    aggregation["staleness_penalty_minutes"] = config.data_aggregation.staleness_penalty_minutes;
    aggregation["enable_cross_validation"] = config.data_aggregation.enable_cross_validation;
    aggregation["source_weights"] = config.data_aggregation.source_weights;
    
    config_json["data_aggregation"] = aggregation;
    
    // Serialize other sections...
    // (Similar serialization for data_manager, monitoring, trading, logging)
    
    return config_json;
}

std::string ConfigLoader::aggregation_method_to_string(AggregationMethod method) {
    switch (method) {
        case AggregationMethod::FIRST_AVAILABLE: return "first_available";
        case AggregationMethod::WEIGHTED_AVERAGE: return "weighted_average";
        case AggregationMethod::MEDIAN: return "median";
        case AggregationMethod::CONSENSUS: return "consensus";
        case AggregationMethod::LATEST_TIMESTAMP: return "latest_timestamp";
        case AggregationMethod::BEST_QUALITY: return "best_quality";
        case AggregationMethod::CUSTOM: return "custom";
        default: return "weighted_average";
    }
}

} // namespace Core
} // namespace ArchNeuronX
