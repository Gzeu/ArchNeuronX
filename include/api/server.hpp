#pragma once
// ============================================================
// ArchNeuronX v2 - REST API + WebSocket Server
// Uses cpp-httplib (header-only, no external service deps)
// Endpoints:
//   POST /api/v1/predict      - Generate trading signal
//   POST /api/v1/backtest     - Run backtest [NEW v2]
//   GET  /api/v1/portfolio    - Portfolio status [NEW v2]
//   GET  /api/v1/models       - List available models
//   POST /api/v1/train        - Trigger training
//   GET  /api/v1/status       - Health check
//   GET  /api/v1/metrics      - Live metrics
//   GET  /api/v1/signals/history - Signal history
//   WS   /ws/signals          - Real-time signal stream [NEW v2]
// ============================================================
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>

#ifdef USE_HTTPLIB
#include <httplib.h>
#endif

namespace archneuronx {
namespace api {

struct APIConfig {
    uint16_t port               = 8080;
    uint16_t metrics_port       = 9090;
    std::string host            = "0.0.0.0";
    bool enable_ssl             = false;
    std::string ssl_cert_path;
    std::string ssl_key_path;

    // Auth
    bool require_api_key        = true;
    std::vector<std::string> api_keys;
    bool enable_jwt             = false;
    std::string jwt_secret;

    // Rate limiting
    int max_requests_per_minute = 1000;
    int max_requests_per_second = 50;

    // CORS
    bool enable_cors            = true;
    std::string cors_origins    = "*";

    // Timeouts
    int request_timeout_sec     = 30;
    int keepalive_timeout_sec   = 60;
};

struct SignalResponse {
    std::string symbol;
    std::string action;        // "BUY", "SELL", "HOLD"
    float confidence;
    double price_target;
    double stop_loss;
    double take_profit;
    std::string timestamp;
    uint64_t latency_us;
    std::string explanation;
    // Risk
    double suggested_position_size;
    double var_95;
    std::string market_regime;
};

struct BacktestRequest {
    std::string symbol;
    std::string start_date;   // ISO 8601
    std::string end_date;
    std::string model_name;
    double initial_capital;
    double commission_pct;    // default 0.001 (0.1%)
    std::string timeframe;    // "1h", "4h", "1d"
};

struct BacktestResult {
    double total_return_pct;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown_pct;
    double win_rate;
    double avg_win_pct;
    double avg_loss_pct;
    double profit_factor;
    int total_trades;
    int winning_trades;
    int losing_trades;
    double final_equity;
    std::string report_url;   // HTML report path
};

class APIServer {
public:
    explicit APIServer(APIConfig config = {});
    ~APIServer();

    // Start blocking (call in dedicated thread)
    void start();
    void stop();

    // Inject dependencies
    void set_inference_engine(std::shared_ptr<void> engine);
    void set_risk_manager(std::shared_ptr<void> risk_mgr);

    // WebSocket: broadcast signal to all connected clients
    void broadcast_signal(const SignalResponse& signal);

    [[nodiscard]] bool is_running() const { return running_; }
    [[nodiscard]] uint64_t request_count() const { return request_count_; }

private:
    APIConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> request_count_{0};

    // WebSocket connections
    std::vector<std::function<void(const std::string&)>> ws_clients_;
    std::mutex ws_mutex_;

    // Rate limiter state (token bucket per IP)
    std::unordered_map<std::string, int> rate_limit_buckets_;
    std::mutex rate_mutex_;

    // Route handlers
    void handle_predict(const void* req, void* res);
    void handle_backtest(const void* req, void* res);
    void handle_portfolio(const void* req, void* res);
    void handle_models(const void* req, void* res);
    void handle_train(const void* req, void* res);
    void handle_status(const void* req, void* res);
    void handle_metrics(const void* req, void* res);
    void handle_signal_history(const void* req, void* res);

    // Auth middleware
    [[nodiscard]] bool authenticate(const void* req) const;
    [[nodiscard]] bool check_rate_limit(const std::string& client_ip);

    // Helpers
    std::string to_json(const SignalResponse& s) const;
    std::string to_json(const BacktestResult& b) const;
    std::string error_json(int code, const std::string& msg) const;
};

} // namespace api
} // namespace archneuronx
