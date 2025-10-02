/**
 * @file config.hpp
 * @brief Configuration management system for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <variant>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace ArchNeuronX {
namespace Core {

/**
 * @brief Configuration value type that can hold various data types
 */
using ConfigValue = std::variant<bool, int, double, std::string, std::vector<std::string>>;

/**
 * @class ConfigurationException
 * @brief Exception thrown for configuration-related errors
 */
class ConfigurationException : public std::runtime_error {
public:
    explicit ConfigurationException(const std::string& message)
        : std::runtime_error("Configuration Error: " + message) {}
};

/**
 * @class Config
 * @brief Comprehensive configuration management system
 */
class Config {
public:
    /**
     * @brief Get singleton instance of configuration manager
     * @return Reference to config instance
     */
    static Config& getInstance();
    
    /**
     * @brief Load configuration from JSON file
     * @param filename Path to configuration file
     * @throws ConfigurationException if file cannot be loaded or parsed
     */
    void loadFromFile(const std::string& filename);
    
    /**
     * @brief Save current configuration to JSON file
     * @param filename Path to save configuration
     * @throws ConfigurationException if file cannot be written
     */
    void saveToFile(const std::string& filename) const;
    
    /**
     * @brief Set a configuration value
     * @param key Configuration key (supports dot notation like "model.learning_rate")
     * @param value Configuration value
     */
    void set(const std::string& key, const ConfigValue& value);
    
    /**
     * @brief Get configuration value with type checking
     * @tparam T Expected return type
     * @param key Configuration key
     * @param default_value Default value if key doesn't exist
     * @return Configuration value cast to requested type
     * @throws ConfigurationException if type conversion fails
     */
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const;
    
    /**
     * @brief Check if configuration key exists
     * @param key Configuration key to check
     * @return True if key exists
     */
    bool has(const std::string& key) const;
    
    /**
     * @brief Remove configuration key
     * @param key Configuration key to remove
     * @return True if key was removed, false if it didn't exist
     */
    bool remove(const std::string& key);
    
    /**
     * @brief Get all configuration keys
     * @return Vector of all configuration keys
     */
    std::vector<std::string> getKeys() const;
    
    /**
     * @brief Clear all configuration values
     */
    void clear();
    
    /**
     * @brief Load default configuration values
     */
    void loadDefaults();
    
    /**
     * @brief Validate current configuration
     * @throws ConfigurationException if configuration is invalid
     */
    void validate() const;
    
    /**
     * @brief Get configuration as formatted string
     * @return String representation of configuration
     */
    std::string toString() const;
    
private:
    Config() = default;
    ~Config() = default;
    
    // Prevent copying
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    
    /**
     * @brief Parse JSON string to configuration map
     * @param json_content JSON string content
     */
    void parseJson(const std::string& json_content);
    
    /**
     * @brief Convert configuration map to JSON string
     * @return JSON string representation
     */
    std::string toJson() const;
    
    /**
     * @brief Convert string to ConfigValue based on content
     * @param str Input string
     * @return ConfigValue with appropriate type
     */
    ConfigValue stringToValue(const std::string& str) const;
    
    /**
     * @brief Convert ConfigValue to string
     * @param value Input ConfigValue
     * @return String representation
     */
    std::string valueToString(const ConfigValue& value) const;
    
    /**
     * @brief Get nested configuration value using dot notation
     * @param key Dot-separated key path
     * @return Pointer to ConfigValue or nullptr if not found
     */
    ConfigValue* getNestedValue(const std::string& key);
    const ConfigValue* getNestedValue(const std::string& key) const;
    
    /**
     * @brief Set nested configuration value using dot notation
     * @param key Dot-separated key path
     * @param value Value to set
     */
    void setNestedValue(const std::string& key, const ConfigValue& value);
    
    std::map<std::string, ConfigValue> config_values_;
    mutable std::mutex config_mutex_;
};

// Template implementation
template<typename T>
T Config::get(const std::string& key, const T& default_value) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    const ConfigValue* value = getNestedValue(key);
    if (!value) {
        return default_value;
    }
    
    try {
        if constexpr (std::is_same_v<T, bool>) {
            return std::get<bool>(*value);
        } else if constexpr (std::is_same_v<T, int>) {
            return std::get<int>(*value);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::get<double>(*value);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return std::get<std::string>(*value);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            return std::get<std::vector<std::string>>(*value);
        } else {
            static_assert(sizeof(T) == 0, "Unsupported configuration value type");
        }
    } catch (const std::bad_variant_access&) {
        throw ConfigurationException("Type mismatch for key: " + key);
    }
}

/**
 * @brief Convenience macros for configuration access
 */
#define CONFIG ArchNeuronX::Core::Config::getInstance()
#define CONFIG_GET(key, default_val) CONFIG.get(key, default_val)
#define CONFIG_SET(key, val) CONFIG.set(key, val)
#define CONFIG_HAS(key) CONFIG.has(key)

} // namespace Core
} // namespace ArchNeuronX