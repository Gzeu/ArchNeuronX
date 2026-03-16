#pragma once
// ============================================================
// ArchNeuronX v2 - Prometheus Metrics
// Exposes trading + model performance metrics
// Endpoint: GET /metrics (port 9090)
// ============================================================
#include <string>
#include <atomic>
#include <chrono>

#ifdef USE_PROMETHEUS
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <prometheus/exposer.h>
#endif

namespace archneuronx {
namespace monitoring {

class Metrics {
public:
    static Metrics& instance();

    void init(uint16_t port = 9090);

    // ---- Inference metrics ----
    void record_inference_latency_us(double latency_us);
    void increment_inference_count(bool success = true);

    // ---- Trading signal metrics ----
    void record_signal(const std::string& symbol,
                       const std::string& action,   // BUY/SELL/HOLD
                       float confidence);
    void record_signal_accuracy(bool correct);

    // ---- Risk metrics ----
    void update_portfolio_value(double value);
    void update_current_drawdown(double drawdown_pct);
    void update_var(double var_95);
    void increment_risk_events();

    // ---- API metrics ----
    void record_api_request(const std::string& endpoint,
                             int status_code,
                             double latency_ms);

    // ---- Model metrics ----
    void update_model_accuracy(const std::string& model_name,
                                double accuracy);

private:
    Metrics() = default;
    bool initialized_ = false;

#ifdef USE_PROMETHEUS
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<prometheus::Exposer>  exposer_;

    // Inference
    prometheus::Histogram* inference_latency_hist_ = nullptr;
    prometheus::Counter*   inference_total_       = nullptr;
    prometheus::Counter*   inference_errors_      = nullptr;

    // Signals
    prometheus::Counter*   signals_total_     = nullptr;
    prometheus::Gauge*     signal_confidence_ = nullptr;
    prometheus::Counter*   signal_accuracy_   = nullptr;

    // Risk
    prometheus::Gauge* portfolio_value_  = nullptr;
    prometheus::Gauge* current_drawdown_ = nullptr;
    prometheus::Gauge* var_95_           = nullptr;
    prometheus::Counter* risk_events_    = nullptr;

    // API
    prometheus::Histogram* api_latency_  = nullptr;
    prometheus::Counter*   api_requests_ = nullptr;
#endif
};

} // namespace monitoring
} // namespace archneuronx
