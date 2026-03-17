// ============================================================
// ArchNeuronX v2 - Rate Limiting Implementation
// Token bucket algorithm with Redis support
// ============================================================
#include "api/server.hpp"
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <algorithm>

namespace archneuronx {
namespace api {

class TokenBucket {
public:
    TokenBucket(double refill_rate, int max_tokens) 
        : refill_rate_(refill_rate), max_tokens_(max_tokens), tokens_(max_tokens) {
        last_refill_ = std::chrono::steady_clock::now();
    }
    
    bool consume(int tokens = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        refill();
        
        if (tokens_ >= tokens) {
            tokens_ -= tokens;
            return true;
        }
        
        return false;
    }
    
    int available_tokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(tokens_);
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        tokens_ = max_tokens_;
        last_refill_ = std::chrono::steady_clock::now();
    }
    
private:
    void refill() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_refill_);
        
        double tokens_to_add = (duration.count() / 1000.0) * refill_rate_;
        tokens_ = std::min(max_tokens_, tokens_ + tokens_to_add);
        last_refill_ = now;
    }
    
    double refill_rate_;
    int max_tokens_;
    double tokens_;
    std::chrono::steady_clock::time_point last_refill_;
    mutable std::mutex mutex_;
};

class RateLimiter {
public:
    RateLimiter(int requests_per_minute = 1000, int requests_per_second = 50) {
        // Create buckets for different time windows
        minute_bucket_ = std::make_unique<TokenBucket>(requests_per_minute / 60.0, requests_per_minute);
        second_bucket_ = std::make_unique<TokenBucket>(requests_per_second, requests_per_second);
    }
    
    bool is_allowed(const std::string& client_id) {
        // Get or create buckets for this client
        auto& client_buckets = client_buckets_[client_id];
        if (!client_buckets.minute) {
            client_buckets.minute = std::make_unique<TokenBucket>(1000.0 / 60.0, 1000); // 1000 req/min
            client_buckets.second = std::make_unique<TokenBucket>(50.0, 50);       // 50 req/sec
        }
        
        // Check both time windows
        bool minute_allowed = client_buckets.minute->consume();
        bool second_allowed = client_buckets.second->consume();
        
        return minute_allowed && second_allowed;
    }
    
    void cleanup_old_clients() {
        auto now = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::minutes(5); // Remove clients inactive for 5 minutes
        
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = client_buckets_.begin(); it != client_buckets_.end();) {
            // Simple cleanup based on last access time (would need to track this)
            // For now, just remove if buckets are empty
            if (it->second.minute->available_tokens() == 1000 && 
                it->second.second->available_tokens() == 50) {
                it = client_buckets_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    size_t get_client_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return client_buckets_.size();
    }
    
    struct ClientBuckets {
        std::unique_ptr<TokenBucket> minute;
        std::unique_ptr<TokenBucket> second;
        std::chrono::steady_clock::time_point last_access;
    };
    
private:
    std::unique_ptr<TokenBucket> minute_bucket_;
    std::unique_ptr<TokenBucket> second_bucket_;
    std::unordered_map<std::string, ClientBuckets> client_buckets_;
    mutable std::mutex mutex_;
};

// Sliding window rate limiter for more precise control
class SlidingWindowRateLimiter {
public:
    SlidingWindowRateLimiter(int max_requests, std::chrono::seconds window_size)
        : max_requests_(max_requests), window_size_(window_size) {}
    
    bool is_allowed(const std::string& client_id) {
        auto now = std::chrono::steady_clock::now();
        auto cutoff = now - window_size_;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Remove old requests
        auto& requests = client_requests_[client_id];
        requests.erase(std::remove_if(requests.begin(), requests.end(),
                                   [cutoff](const auto& timestamp) {
                                       return timestamp < cutoff;
                                   }), requests.end());
        
        // Check if under limit
        if (requests.size() < static_cast<size_t>(max_requests_)) {
            requests.push_back(now);
            return true;
        }
        
        return false;
    }
    
private:
    int max_requests_;
    std::chrono::seconds window_size_;
    std::unordered_map<std::string, std::vector<std::chrono::steady_clock::time_point>> client_requests_;
    mutable std::mutex mutex_;
};

// Factory function to create appropriate rate limiter
std::unique_ptr<RateLimiter> create_rate_limiter(const APIConfig& config) {
    return std::make_unique<RateLimiter>(
        config.max_requests_per_minute,
        config.max_requests_per_second
    );
}

std::unique_ptr<SlidingWindowRateLimiter> create_sliding_rate_limiter(
    int max_requests, 
    std::chrono::seconds window_size
) {
    return std::make_unique<SlidingWindowRateLimiter>(max_requests, window_size);
}

} // namespace api
} // namespace archneuronx
