/**
 * @file data_manager.cpp
 * @brief Data manager implementation - orchestrates multiple data providers
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/data_manager.hpp"
#include <algorithm>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

DataManager::DataManager(const DataManagerConfig& config) 
    : config_(config), running_(false) {
    
    // Initialize data providers based on configuration
    initialize_providers();
    
    // Start background threads
    running_ = true;
    data_collection_thread_ = std::thread(&DataManager::data_collection_loop, this);
    cache_cleanup_thread_ = std::thread(&DataManager::cache_cleanup_loop, this);
    
    LOG_INFO("Data manager initialized with {} providers", providers_.size());
}

DataManager::~DataManager() {
    shutdown();
}

void DataManager::initialize_providers() {
    // Initialize Binance provider if configured
    if (config_.enable_binance) {
        try {
            DataProviderConfig binance_config;
            binance_config.provider_type = DataProviderType::CRYPTO_EXCHANGE;
            binance_config.api_key = config_.binance_api_key;
            binance_config.api_secret = config_.binance_api_secret;
            binance_config.use_testnet = config_.use_testnet;
            
            auto binance_provider = std::make_shared<BinanceProvider>(binance_config);
            providers_["binance"] = binance_provider;
            
            LOG_INFO("Binance provider initialized");
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to initialize Binance provider: {}", e.what());
        }
    }
    
    // Initialize Alpha Vantage provider if configured
    if (config_.enable_alpha_vantage) {
        try {
            DataProviderConfig av_config;
            av_config.provider_type = DataProviderType::FOREX_PROVIDER;
            av_config.api_key = config_.alpha_vantage_api_key;
            
            auto av_provider = std::make_shared<AlphaVantageProvider>(av_config);
            providers_["alpha_vantage"] = av_provider;
            
            LOG_INFO("Alpha Vantage provider initialized");
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to initialize Alpha Vantage provider: {}", e.what());
        }
    }
    
    // Initialize Coinbase provider if configured
    if (config_.enable_coinbase) {
        try {
            DataProviderConfig coinbase_config;
            coinbase_config.provider_type = DataProviderType::CRYPTO_EXCHANGE;
            coinbase_config.api_key = config_.coinbase_api_key;
            coinbase_config.api_secret = config_.coinbase_api_secret;
            coinbase_config.passphrase = config_.coinbase_passphrase;
            
            auto coinbase_provider = std::make_shared<CoinbaseProvider>(coinbase_config);
            providers_["coinbase"] = coinbase_provider;
            
            LOG_INFO("Coinbase provider initialized");
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to initialize Coinbase provider: {}", e.what());
        }
    }
}

void DataManager::shutdown() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Disconnect all providers
    for (auto& [name, provider] : providers_) {
        provider->disconnect();
    }
    
    // Wait for threads to finish
    if (data_collection_thread_.joinable()) {
        data_collection_thread_.join();
    }
    
    if (cache_cleanup_thread_.joinable()) {
        cache_cleanup_thread_.join();
    }
    
    LOG_INFO("Data manager shutdown complete");
}

bool DataManager::connect_all() {
    bool all_connected = true;
    
    for (auto& [name, provider] : providers_) {
        if (!provider->connect()) {
            LOG_ERROR("Failed to connect provider: {}", name);
            all_connected = false;
        } else {
            LOG_INFO("Connected provider: {}", name);
        }
    }
    
    return all_connected;
}

void DataManager::disconnect_all() {
    for (auto& [name, provider] : providers_) {
        provider->disconnect();
        LOG_INFO("Disconnected provider: {}", name);
    }
}

std::future<std::vector<OHLCV>> DataManager::get_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end,
    const std::string& preferred_provider
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end, preferred_provider]() {
        std::vector<OHLCV> data;
        
        // Try preferred provider first
        if (!preferred_provider.empty()) {
            auto it = providers_.find(preferred_provider);
            if (it != providers_.end() && it->second->is_connected()) {
                try {
                    auto future = it->second->get_historical_data(symbol, timeframe, start, end);
                    data = future.get();
                    
                    if (!data.empty()) {
                        cache_historical_data(symbol, timeframe, data);
                        return data;
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Preferred provider {} failed: {}", preferred_provider, e.what());
                }
            }
        }
        
        // Try all available providers
        for (auto& [name, provider] : providers_) {
            if (!provider->is_connected()) {
                continue;
            }
            
            try {
                auto future = provider->get_historical_data(symbol, timeframe, start, end);
                auto provider_data = future.get();
                
                if (!provider_data.empty()) {
                    data = provider_data;
                    cache_historical_data(symbol, timeframe, data);
                    LOG_INFO("Retrieved {} records from provider: {}", data.size(), name);
                    break;
                }
            } catch (const std::exception& e) {
                LOG_WARN("Provider {} failed: {}", name, e.what());
            }
        }
        
        return data;
    });
}

std::future<double> DataManager::get_current_price(
    const std::string& symbol,
    const std::string& preferred_provider
) {
    return std::async(std::launch::async, [this, symbol, preferred_provider]() {
        // Check cache first
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto cache_key = symbol + "_price";
            auto it = price_cache_.find(cache_key);
            if (it != price_cache_.end()) {
                auto age = std::chrono::steady_clock::now() - it->second.timestamp;
                if (age < std::chrono::seconds(30)) { // 30 second cache
                    return it->second.price;
                }
            }
        }
        
        // Try preferred provider first
        if (!preferred_provider.empty()) {
            auto it = providers_.find(preferred_provider);
            if (it != providers_.end() && it->second->is_connected()) {
                try {
                    auto future = it->second->get_current_price(symbol);
                    double price = future.get();
                    
                    if (price > 0.0) {
                        cache_price(symbol, price);
                        return price;
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Preferred provider {} failed: {}", preferred_provider, e.what());
                }
            }
        }
        
        // Try all available providers
        for (auto& [name, provider] : providers_) {
            if (!provider->is_connected()) {
                continue;
            }
            
            try {
                auto future = provider->get_current_price(symbol);
                double price = future.get();
                
                if (price > 0.0) {
                    cache_price(symbol, price);
                    LOG_DEBUG("Got price {} from provider: {}", price, name);
                    return price;
                }
            } catch (const std::exception& e) {
                LOG_WARN("Provider {} failed: {}", name, e.what());
            }
        }
        
        return 0.0;
    });
}

std::future<OrderBook> DataManager::get_order_book(
    const std::string& symbol,
    int depth,
    const std::string& preferred_provider
) {
    return std::async(std::launch::async, [this, symbol, depth, preferred_provider]() {
        // Try preferred provider first
        if (!preferred_provider.empty()) {
            auto it = providers_.find(preferred_provider);
            if (it != providers_.end() && it->second->is_connected()) {
                try {
                    auto future = it->second->get_order_book(symbol, depth);
                    auto order_book = future.get();
                    
                    if (!order_book.bids.empty() || !order_book.asks.empty()) {
                        return order_book;
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Preferred provider {} failed: {}", preferred_provider, e.what());
                }
            }
        }
        
        // Try all available providers
        for (auto& [name, provider] : providers_) {
            if (!provider->is_connected()) {
                continue;
            }
            
            try {
                auto future = provider->get_order_book(symbol, depth);
                auto order_book = future.get();
                
                if (!order_book.bids.empty() || !order_book.asks.empty()) {
                    LOG_DEBUG("Got order book from provider: {}", name);
                    return order_book;
                }
            } catch (const std::exception& e) {
                LOG_WARN("Provider {} failed: {}", name, e.what());
            }
        }
        
        return OrderBook{};
    });
}

void DataManager::subscribe_to_real_time_data(
    const std::string& symbol,
    std::function<void(const TickData&)> callback,
    const std::string& preferred_provider
) {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    
    Subscription subscription;
    subscription.symbol = symbol;
    subscription.callback = callback;
    subscription.provider = preferred_provider;
    subscription.active = true;
    
    subscriptions_.push_back(subscription);
    
    LOG_INFO("Subscribed to real-time data for: {}", symbol);
}

void DataManager::unsubscribe_from_real_time_data(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    
    auto it = std::remove_if(subscriptions_.begin(), subscriptions_.end(),
        [&symbol](const Subscription& sub) {
            return sub.symbol == symbol;
        });
    
    subscriptions_.erase(it, subscriptions_.end());
    
    LOG_INFO("Unsubscribed from real-time data for: {}", symbol);
}

DataManagerStats DataManager::get_statistics() const {
    DataManagerStats stats;
    
    stats.total_providers = providers_.size();
    stats.connected_providers = 0;
    stats.active_subscriptions = 0;
    stats.cache_size = price_cache_.size();
    
    for (const auto& [name, provider] : providers_) {
        if (provider->is_connected()) {
            stats.connected_providers++;
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        stats.active_subscriptions = subscriptions_.size();
    }
    
    return stats;
}

std::vector<std::string> DataManager::get_available_providers() const {
    std::vector<std::string> providers;
    
    for (const auto& [name, provider] : providers_) {
        providers.push_back(name);
    }
    
    return providers;
}

bool DataManager::is_provider_connected(const std::string& provider_name) const {
    auto it = providers_.find(provider_name);
    if (it != providers_.end()) {
        return it->second->is_connected();
    }
    return false;
}

// Private methods

void DataManager::data_collection_loop() {
    while (running_) {
        try {
            // Update real-time subscriptions
            update_subscriptions();
            
            // Collect provider statistics
            collect_provider_stats();
            
            // Sleep for a short interval
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
        } catch (const std::exception& e) {
            LOG_ERROR("Error in data collection loop: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
}

void DataManager::cache_cleanup_loop() {
    while (running_) {
        try {
            cleanup_expired_cache();
            
            // Sleep for cache cleanup interval
            std::this_thread::sleep_for(std::chrono::minutes(5));
            
        } catch (const std::exception& e) {
            LOG_ERROR("Error in cache cleanup loop: {}", e.what());
        }
    }
}

void DataManager::update_subscriptions() {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    
    for (auto& subscription : subscriptions_) {
        if (!subscription.active) {
            continue;
        }
        
        // Find suitable provider for this subscription
        std::string provider_name = subscription.provider;
        if (provider_name.empty()) {
            // Auto-select provider based on symbol type
            provider_name = select_provider_for_symbol(subscription.symbol);
        }
        
        auto it = providers_.find(provider_name);
        if (it != providers_.end() && it->second->is_connected()) {
            // Start real-time data stream
            auto future = it->second->get_real_time_ticks(
                subscription.symbol, subscription.callback);
            
            // Store future for management
            subscription.data_future = std::move(future);
        }
    }
}

std::string DataManager::select_provider_for_symbol(const std::string& symbol) {
    // Simple heuristic: use Binance for crypto, Alpha Vantage for stocks/forex
    if (symbol.find("BTC") != std::string::npos || 
        symbol.find("ETH") != std::string::npos ||
        symbol.find("/") != std::string::npos) {
        
        // Prefer Binance for crypto
        if (is_provider_connected("binance")) return "binance";
        if (is_provider_connected("coinbase")) return "coinbase";
    } else {
        // Prefer Alpha Vantage for stocks/forex
        if (is_provider_connected("alpha_vantage")) return "alpha_vantage";
    }
    
    // Fallback to any available provider
    for (const auto& [name, provider] : providers_) {
        if (provider->is_connected()) {
            return name;
        }
    }
    
    return "";
}

void DataManager::collect_provider_stats() {
    for (const auto& [name, provider] : providers_) {
        auto status = provider->get_status();
        
        // Log provider status changes
        static std::map<std::string, ConnectionStatus> last_status;
        
        if (last_status[name] != status) {
            LOG_INFO("Provider {} status changed: {} -> {}", 
                     name, 
                     static_cast<int>(last_status[name]),
                     static_cast<int>(status));
            last_status[name] = status;
        }
    }
}

void DataManager::cache_price(const std::string& symbol, double price) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    CachedPrice cached_price;
    cached_price.price = price;
    cached_price.timestamp = std::chrono::steady_clock::now();
    
    price_cache_[symbol + "_price"] = cached_price;
}

void DataManager::cache_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::vector<OHLCV>& data
) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string cache_key = symbol + "_" + timeframe;
    
    CachedHistoricalData cached_data;
    cached_data.data = data;
    cached_data.timestamp = std::chrono::steady_clock::now();
    
    historical_cache_[cache_key] = cached_data;
}

void DataManager::cleanup_expired_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    // Cleanup price cache (older than 5 minutes)
    auto price_it = price_cache_.begin();
    while (price_it != price_cache_.end()) {
        if (now - price_it->second.timestamp > std::chrono::minutes(5)) {
            price_it = price_cache_.erase(price_it);
        } else {
            ++price_it;
        }
    }
    
    // Cleanup historical cache (older than 1 hour)
    auto hist_it = historical_cache_.begin();
    while (hist_it != historical_cache_.end()) {
        if (now - hist_it->second.timestamp > std::chrono::hours(1)) {
            hist_it = historical_cache_.erase(hist_it);
        } else {
            ++hist_it;
        }
    }
}

} // namespace Data
} // namespace ArchNeuronX
