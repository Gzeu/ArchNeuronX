// ============================================================
// ArchNeuronX v2 - Authentication & Authorization
// API key validation and JWT token handling
// ============================================================
#include "api/server.hpp"
#include <nlohmann/json.hpp>
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <chrono>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;
namespace chrono = std::chrono;

namespace archneuronx {
namespace api {

class AuthManager {
public:
    explicit AuthManager(const APIConfig& config) : config_(config) {}
    
    // API Key validation
    bool validate_api_key(const std::string& api_key) const {
        if (!config_.require_api_key) {
            return true;
        }
        
        return std::find(config_.api_keys.begin(), config_.api_keys.end(), api_key) 
               != config_.api_keys.end();
    }
    
    // JWT token generation
    std::string generate_jwt(const std::string& user_id, const std::string& role = "user") {
        if (!config_.enable_jwt) {
            return "";
        }
        
        auto now = chrono::system_clock::now();
        auto exp = now + chrono::hours(24); // 24 hour expiration
        
        json header = {
            {"alg", "HS256"},
            {"typ", "JWT"}
        };
        
        json payload = {
            {"sub", user_id},
            {"role", role},
            {"iat", chrono::duration_cast<chrono::seconds>(now.time_since_epoch()).count()},
            {"exp", chrono::duration_cast<chrono::seconds>(exp.time_since_epoch()).count()}
        };
        
        std::string header_b64 = base64_url_encode(header.dump());
        std::string payload_b64 = base64_url_encode(payload.dump());
        std::string signature = hmac_sha256(config_.jwt_secret, header_b64 + "." + payload_b64);
        
        return header_b64 + "." + payload_b64 + "." + signature;
    }
    
    // JWT token validation
    bool validate_jwt(const std::string& token) {
        if (!config_.enable_jwt || token.empty()) {
            return false;
        }
        
        try {
            size_t first_dot = token.find('.');
            size_t second_dot = token.find('.', first_dot + 1);
            
            if (first_dot == std::string::npos || second_dot == std::string::npos) {
                return false;
            }
            
            std::string header_b64 = token.substr(0, first_dot);
            std::string payload_b64 = token.substr(first_dot + 1, second_dot - first_dot - 1);
            std::string signature = token.substr(second_dot + 1);
            
            // Verify signature
            std::string expected_signature = hmac_sha256(config_.jwt_secret, header_b64 + "." + payload_b64);
            if (signature != expected_signature) {
                return false;
            }
            
            // Verify expiration
            json payload = json::parse(base64_url_decode(payload_b64));
            auto now = chrono::system_clock::now();
            auto exp = payload["exp"].get<long>();
            auto exp_time = chrono::system_clock::from_time_t(exp);
            
            return now < exp_time;
            
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    // Extract user info from JWT
    std::string get_user_from_jwt(const std::string& token) {
        if (!validate_jwt(token)) {
            return "";
        }
        
        try {
            size_t first_dot = token.find('.');
            size_t second_dot = token.find('.', first_dot + 1);
            std::string payload_b64 = token.substr(first_dot + 1, second_dot - first_dot - 1);
            json payload = json::parse(base64_url_decode(payload_b64));
            return payload["sub"].get<std::string>();
        } catch (const std::exception&) {
            return "";
        }
    }
    
private:
    APIConfig config_;
    
    // Base64 URL-safe encoding
    std::string base64_url_encode(const std::string& input) {
        BIO *bio, *b64;
        BUF_MEM *bufferPtr;
        
        b64 = BIO_new(BIO_f_base64());
        bio = BIO_new(BIO_s_mem());
        bio = BIO_push(b64, bio);
        
        BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
        BIO_write(bio, input.c_str(), input.length());
        BIO_flush(bio);
        BIO_get_mem_ptr(bio, &bufferPtr);
        
        std::string result(bufferPtr->data, bufferPtr->length);
        BIO_free_all(bio);
        
        // URL-safe encoding
        std::replace(result.begin(), result.end(), '+', '-');
        std::replace(result.begin(), result.end(), '/', '_');
        result.erase(std::remove(result.begin(), result.end(), '='), result.end());
        
        return result;
    }
    
    // Base64 URL-safe decoding
    std::string base64_url_decode(const std::string& input) {
        std::string modified = input;
        std::replace(modified.begin(), modified.end(), '-', '+');
        std::replace(modified.begin(), modified.end(), '/', '/');
        
        // Add padding
        while (modified.length() % 4) {
            modified += '=';
        }
        
        BIO *bio, *b64;
        char buffer[input.length()];
        
        bio = BIO_new_mem_buf(modified.c_str(), modified.length());
        b64 = BIO_new(BIO_f_base64());
        bio = BIO_push(b64, bio);
        
        BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
        int length = BIO_read(bio, buffer, input.length());
        BIO_free_all(bio);
        
        return std::string(buffer, length);
    }
    
    // HMAC-SHA256
    std::string hmac_sha256(const std::string& key, const std::string& data) {
        unsigned char* digest;
        digest = HMAC(EVP_sha256(), key.c_str(), key.length(),
                      (unsigned char*)data.c_str(), data.length(), NULL, NULL);
        
        std::stringstream ss;
        for(int i = 0; i < 32; i++) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
        }
        
        return ss.str();
    }
};

// Rate limiter implementation
class RateLimiter {
public:
    explicit RateLimiter(int requests_per_minute, int requests_per_second)
        : rpm_(requests_per_minute), rps_(requests_per_second) {}
    
    bool is_allowed(const std::string& client_ip) {
        auto now = chrono::steady_clock::now();
        auto minute_key = std::to_string(chrono::duration_cast<chrono::minutes>(now.time_since_epoch()).count());
        auto second_key = std::to_string(chrono::duration_cast<chrono::seconds>(now.time_since_epoch()).count());
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check per-minute limit
        auto& minute_count = buckets_[client_ip + ":" + minute_key];
        if (minute_count >= rpm_) {
            return false;
        }
        
        // Check per-second limit
        auto& second_count = buckets_[client_ip + ":" + second_key];
        if (second_count >= rps_) {
            return false;
        }
        
        minute_count++;
        second_count++;
        return true;
    }
    
private:
    int rpm_, rps_;
    std::unordered_map<std::string, int> buckets_;
    std::mutex mutex_;
};

} // namespace api
} // namespace archneuronx
