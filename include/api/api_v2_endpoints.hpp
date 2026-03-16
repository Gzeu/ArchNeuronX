/**
 * @file api_v2_endpoints.hpp
 * @brief REST API v2 Endpoints - ArchNeuronX
 * @version 2.0.0
 *
 * New endpoints in v2:
 *   POST /api/v1/backtest        - Strategy backtesting
 *   GET  /api/v1/portfolio       - Portfolio state
 *   POST /api/v1/portfolio/rebalance - Portfolio rebalancing
 *   GET  /api/v1/risk            - Risk metrics
 *   WS   /ws/v1/market           - Real-time market data WebSocket
 *   WS   /ws/v1/signals          - Real-time signal WebSocket
 *
 * Authentication:
 *   Bearer token or API key via X-API-Key header
 *   Rate limiting: 100 req/min per key
 *
 * OpenAPI 3.1 spec available at /api/v1/docs
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <functional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace API {

// ============================================================
// Request/Response structures
// ============================================================

/** POST /api/v1/backtest */
struct BacktestRequest {
    std::string strategy_id;         // "mlp", "cnn", "transformer"
    std::string symbol;              // e.g. "BTCUSDT"
    std::string start_date;          // ISO 8601
    std::string end_date;
    double initial_capital = 10000.0;
    std::string model_path;          // Path to .pt model file
    std::map<std::string, double> params;  // Strategy params
    bool include_costs = true;       // Include commission/slippage
    double commission_pct = 0.001;   // 0.1% per trade
    double slippage_pct = 0.0005;    // 0.05% slippage
};

struct TradeRecord {
    std::string timestamp;
    std::string symbol;
    std::string side;     // "BUY" | "SELL"
    double price;
    double quantity;
    double pnl;
    double portfolio_value;
};

struct BacktestResult {
    bool success;
    std::string error;

    // Performance metrics
    double total_return_pct;
    double annualized_return_pct;
    double sharpe_ratio;
    double sortino_ratio;
    double calmar_ratio;
    double max_drawdown_pct;
    double win_rate;
    double profit_factor;
    int total_trades;
    int winning_trades;
    int losing_trades;
    double avg_win_pct;
    double avg_loss_pct;
    double best_trade_pct;
    double worst_trade_pct;
    double var_95;
    double cvar_95;

    // Equity curve
    std::vector<double> equity_curve;
    std::vector<std::string> timestamps;
    std::vector<TradeRecord> trades;
};

/** GET /api/v1/portfolio */
struct PortfolioState {
    double total_value;
    double cash;
    double invested;
    double unrealized_pnl;
    double realized_pnl_today;
    double total_return_pct;
    double daily_pnl_pct;

    struct PositionInfo {
        std::string symbol;
        double quantity;
        double avg_cost;
        double current_price;
        double market_value;
        double unrealized_pnl;
        double unrealized_pnl_pct;
        double stop_loss;
        double take_profit;
    };
    std::vector<PositionInfo> positions;
};

/** WebSocket message types */
struct WsMarketMessage {
    std::string type;        // "tick", "candle", "orderbook"
    std::string symbol;
    std::string timestamp;
    double open, high, low, close, volume;
};

struct WsSignalMessage {
    std::string type = "signal";
    std::string symbol;
    std::string timestamp;
    std::string action;      // "BUY" | "SELL" | "HOLD"
    double confidence;       // [0.0, 1.0]
    double price;
    std::string model_id;
    std::map<std::string, double> metadata;
};

// ============================================================
// Handler interfaces
// ============================================================

class BacktestHandler {
public:
    virtual ~BacktestHandler() = default;

    /**
     * @brief Run backtest
     * Endpoint: POST /api/v1/backtest
     * Returns: BacktestResult as JSON
     */
    virtual json handle_backtest(const BacktestRequest& req) = 0;

    /**
     * @brief List available backtest runs
     * Endpoint: GET /api/v1/backtest/history
     */
    virtual json list_backtests() = 0;

    /**
     * @brief Get specific backtest by ID
     * Endpoint: GET /api/v1/backtest/{id}
     */
    virtual json get_backtest(const std::string& id) = 0;
};

class PortfolioHandler {
public:
    virtual ~PortfolioHandler() = default;

    /** GET /api/v1/portfolio */
    virtual json get_portfolio() = 0;

    /** POST /api/v1/portfolio/rebalance */
    virtual json rebalance(const std::map<std::string, double>& target_weights) = 0;

    /** GET /api/v1/portfolio/history?days=30 */
    virtual json get_history(int days = 30) = 0;
};

class RiskEndpointHandler {
public:
    virtual ~RiskEndpointHandler() = default;

    /** GET /api/v1/risk */
    virtual json get_risk_metrics() = 0;

    /** GET /api/v1/risk/var */
    virtual json get_var(int confidence = 95, int horizon_days = 1) = 0;
};

// ============================================================
// Rate Limiter
// ============================================================
class RateLimiter {
public:
    explicit RateLimiter(int requests_per_minute = 100,
                         int burst_size = 20);

    /** @return true if request is allowed */
    bool allow(const std::string& api_key);

    /** Get remaining requests for key */
    int remaining(const std::string& api_key) const;

    void reset(const std::string& api_key);

private:
    struct TokenBucket {
        double tokens;
        std::chrono::steady_clock::time_point last_refill;
    };
    std::map<std::string, TokenBucket> buckets_;
    double rate_per_second_;
    double burst_;
};

// ============================================================
// Auth Middleware
// ============================================================
struct AuthContext {
    bool authenticated = false;
    std::string api_key;
    std::string user_id;
    std::vector<std::string> permissions;  // "read", "trade", "admin"
};

AuthContext authenticate_request(const std::string& auth_header,
                                  const std::string& api_key_header);

} // namespace API
} // namespace ArchNeuronX

// ============================================================
// JSON serialization helpers (nlohmann/json)
// ============================================================
INLINE_NAMESPACE_BEGIN

namespace nlohmann {
    template<>
    struct adl_serializer<ArchNeuronX::API::BacktestResult> {
        static void to_json(json& j, const ArchNeuronX::API::BacktestResult& r);
        static void from_json(const json& j, ArchNeuronX::API::BacktestResult& r);
    };
} // namespace nlohmann

INLINE_NAMESPACE_END
