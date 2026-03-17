// ============================================================
// ArchNeuronX v2 - Prometheus Metrics Implementation
// Exposes trading + model performance metrics on port 9090
// ============================================================
#include "monitoring/metrics.hpp"
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>

namespace archneuronx {
namespace monitoring {

Metrics& Metrics::instance() {
    static Metrics instance;
    return instance;
}

void Metrics::init(uint16_t port) {
    if (initialized_) {
        return;
    }

#ifdef USE_PROMETHEUS
    try {
        // Create Prometheus registry
        registry_ = std::make_shared<prometheus::Registry>();
        
        // Create exposer
        exposer_ = std::make_unique<prometheus::Exposer>("0.0.0.0:" + std::to_string(port));
        exposer_->RegisterCollectable(registry_);

        // Create and register metrics
        
        // Inference metrics
        auto& inference_family = prometheus::BuildHistogram()
            .Name("archneuronx_inference_latency_microseconds")
            .Help("Inference latency in microseconds")
            .Register(*registry_);
        inference_latency_hist_ = &inference_family.Add({});

        inference_total_ = &prometheus::BuildCounter()
            .Name("archneuronx_inference_total")
            .Help("Total number of inference requests")
            .Register(*registry_)
            .Add({});

        inference_errors_ = &prometheus::BuildCounter()
            .Name("archneuronx_inference_errors_total")
            .Help("Total number of inference errors")
            .Register(*registry_)
            .Add({});

        // Signal metrics
        signals_total_ = &prometheus::BuildCounter()
            .Name("archneuronx_signals_total")
            .Help("Total number of trading signals generated")
            .Register(*registry_)
            .Add({{"type", "total"}});

        signal_confidence_ = &prometheus::BuildGauge()
            .Name("archneuronx_signal_confidence")
            .Help("Current signal confidence level")
            .Register(*registry_)
            .Add({});

        signal_accuracy_ = &prometheus::BuildCounter()
            .Name("archneuronx_signal_accuracy_total")
            .Help("Total number of accurate signals")
            .Register(*registry_)
            .Add({});

        // Risk metrics
        portfolio_value_ = &prometheus::BuildGauge()
            .Name("archneuronx_portfolio_value")
            .Help("Current portfolio value")
            .Register(*registry_)
            .Add({});

        current_drawdown_ = &prometheus::BuildGauge()
            .Name("archneuronx_current_drawdown_percent")
            .Help("Current drawdown percentage")
            .Register(*registry_)
            .Add({});

        var_95_ = &prometheus::BuildGauge()
            .Name("archneuronx_var_95")
            .Help("95% Value at Risk")
            .Register(*registry_)
            .Add({});

        risk_events_ = &prometheus::BuildCounter()
            .Name("archneuronx_risk_events_total")
            .Help("Total number of risk events")
            .Register(*registry_)
            .Add({});

        // API metrics
        auto& api_latency_family = prometheus::BuildHistogram()
            .Name("archneuronx_api_request_duration_milliseconds")
            .Help("API request duration in milliseconds")
            .Register(*registry_);
        api_latency_ = &api_latency_family.Add({});

        api_requests_ = &prometheus::BuildCounter()
            .Name("archneuronx_api_requests_total")
            .Help("Total number of API requests")
            .Register(*registry_)
            .Add({});

        std::cout << "Prometheus metrics server started on port " << port << std::endl;
        std::cout << "Metrics available at: http://localhost:" << port << "/metrics" << std::endl;
        
        initialized_ = true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Prometheus metrics: " << e.what() << std::endl;
    }
#else
    std::cout << "Prometheus support disabled. Metrics will be logged locally." << std::endl;
    initialized_ = true;
#endif
}

void Metrics::record_inference_latency_us(double latency_us) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (inference_latency_hist_) {
        inference_latency_hist_->Observe(latency_us);
    }
#else
    std::cout << "[METRIC] Inference latency: " << latency_us << " μs" << std::endl;
#endif
}

void Metrics::increment_inference_count(bool success) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (success && inference_total_) {
        inference_total_->Increment();
    } else if (!success && inference_errors_) {
        inference_errors_->Increment();
    }
#else
    std::cout << "[METRIC] Inference " << (success ? "success" : "error") << std::endl;
#endif
}

void Metrics::record_signal(const std::string& symbol,
                           const std::string& action,
                           float confidence) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (signals_total_) {
        signals_total_->Increment({{"symbol", symbol}, {"action", action}});
    }
    if (signal_confidence_) {
        signal_confidence_->Set(confidence);
    }
#else
    std::cout << "[METRIC] Signal: " << symbol << " " << action 
              << " (confidence: " << confidence << ")" << std::endl;
#endif
}

void Metrics::record_signal_accuracy(bool correct) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (correct && signal_accuracy_) {
        signal_accuracy_->Increment();
    }
#else
    std::cout << "[METRIC] Signal accuracy: " << (correct ? "correct" : "incorrect") << std::endl;
#endif
}

void Metrics::update_portfolio_value(double value) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (portfolio_value_) {
        portfolio_value_->Set(value);
    }
#else
    std::cout << "[METRIC] Portfolio value: $" << value << std::endl;
#endif
}

void Metrics::update_current_drawdown(double drawdown_pct) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (current_drawdown_) {
        current_drawdown_->Set(drawdown_pct);
    }
#else
    std::cout << "[METRIC] Current drawdown: " << drawdown_pct << "%" << std::endl;
#endif
}

void Metrics::update_var(double var_95) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (var_95_) {
        var_95_->Set(var_95);
    }
#else
    std::cout << "[METRIC] VaR 95: " << var_95 << std::endl;
#endif
}

void Metrics::increment_risk_events() {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (risk_events_) {
        risk_events_->Increment();
    }
#else
    std::cout << "[METRIC] Risk event triggered" << std::endl;
#endif
}

void Metrics::record_api_request(const std::string& endpoint,
                                int status_code,
                                double latency_ms) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    if (api_latency_) {
        api_latency_->Observe(latency_ms);
    }
    if (api_requests_) {
        api_requests_->Increment({{"endpoint", endpoint}, {"status", std::to_string(status_code)}});
    }
#else
    std::cout << "[METRIC] API request: " << endpoint << " " << status_code 
              << " (" << latency_ms << "ms)" << std::endl;
#endif
}

void Metrics::update_model_accuracy(const std::string& model_name, double accuracy) {
    if (!initialized_) return;

#ifdef USE_PROMETHEUS
    // This would require creating a gauge for model accuracy
    // For now, we'll just log it
#else
    std::cout << "[METRIC] Model " << model_name << " accuracy: " << accuracy << std::endl;
#endif
}

} // namespace monitoring
} // namespace archneuronx
