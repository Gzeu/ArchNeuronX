# Risk Management Guide - ArchNeuronX v2.0

## Overview

ArchNeuronX v2.0 includes a comprehensive risk management system designed for live algorithmic trading. This guide covers position sizing, stop-loss automation, VaR calculation, and portfolio-level risk controls.

## Position Sizing Methods

### 1. Volatility-Adjusted (Default)

Sizes positions inversely proportional to recent volatility (ATR):

```
Position Size = (Portfolio Value × Risk Per Trade) / (ATR × ATR Multiplier × Price)
```

**Example** - BTCUSDT, Portfolio $10,000, Risk 1%, ATR = $800, Multiplier = 2:
```
Size = (10000 × 0.01) / (800 × 2 × 45000) = 0.00014 BTC
```

### 2. Kelly Criterion

Optimal fraction based on historical win rate and payoff ratio:

```
Kelly % = Win Rate - (Loss Rate / Reward:Risk Ratio)
```

ArchNeuronX uses **Quarter Kelly** (25% of full Kelly) for safety.

### 3. Fixed Fractional

Fixed percentage of portfolio per trade (e.g., 1-2%).

### 4. Risk Parity

Equal risk contribution from each position (inverse volatility weighting).

## Stop-Loss Automation

### ATR-Based Dynamic Stop (Recommended)

```json
{
  "use_atr_stop": true,
  "atr_multiplier": 2.0,
  "atr_period": 14
}
```

Stop is placed at: `Entry Price ± (ATR × Multiplier)`

### Trailing Stop

Automatically moves stop in direction of profit:

```json
{
  "trailing_stop_pct": 0.03
}
```

For a BUY at $45,000 with 3% trailing:
- Initial stop: $43,650
- If price reaches $48,000 → stop moves to $46,560

### Take-Profit (2:1 Risk-Reward Minimum)

```json
{
  "risk_reward_ratio": 2.0,
  "use_partial_profit": true
}
```

With partial profit enabled: close 50% at 1:1, let 50% run to target.

## Value at Risk (VaR)

### Historical Simulation (Default)

1. Collect last N days of returns
2. Sort returns ascending
3. VaR 95% = 5th percentile return × portfolio value
4. CVaR 95% = Mean of returns below VaR 95%

### Parametric (Gaussian)

```
VaR 95% = Portfolio Value × (μ - 1.645σ) × √horizon
```

### API Endpoint

```bash
curl http://localhost:8080/api/v1/risk/var?confidence=95&horizon=1 \
  -H "X-API-Key: your_key"

# Response
{
  "var_95": 450.00,
  "var_99": 720.00,
  "cvar_95": 580.00,
  "cvar_99": 890.00,
  "method": "historical",
  "portfolio_value": 10000.00,
  "horizon_days": 1
}
```

## Circuit Breaker

Automatically halts all trading when drawdown exceeds threshold:

```json
{
  "max_drawdown_limit": 0.15
}
```

When portfolio drops 15% from peak:
- All new trades blocked
- Open positions held (no forced liquidation)
- Alert sent via configured callback
- Manual reset required: `POST /api/v1/risk/reset-circuit-breaker`

## Portfolio Exposure Limits

| Limit | Default | Description |
|-------|---------|-------------|
| Max per position | 10% | Single asset cap |
| Max total exposure | 95% | Total invested |
| Max sector exposure | 30% | Per sector/exchange |
| Max correlation | 0.70 | Block correlated trades |
| Max risk per trade | 2% | Single trade risk |

## Market Regime Adaptation

Risk parameters auto-adjust based on detected regime:

| Regime | Position Scale | Description |
|--------|---------------|-------------|
| TRENDING_UP | 1.2× | Increase size in bull trends |
| TRENDING_DOWN | 1.0× | Normal sizing |
| RANGING | 0.8× | Reduce size in consolidation |
| HIGH_VOLATILITY | 0.5× | Halve size in volatile markets |
| LOW_VOLATILITY | 1.0× | Normal sizing |

## Configuration

```json
{
  "risk": {
    "max_portfolio_risk": 0.02,
    "max_total_exposure": 0.95,
    "max_drawdown_limit": 0.15,
    "max_position_size": 0.10,
    "sizing_method": "VOLATILITY_ADJUSTED",
    "kelly_fraction": 0.25,
    "risk_per_trade": 0.01,
    "use_atr_stop": true,
    "atr_multiplier": 2.0,
    "trailing_stop_pct": 0.03,
    "risk_reward_ratio": 2.0
  }
}
```

## Risk Metrics Dashboard

```bash
curl http://localhost:8080/api/v1/risk -H "X-API-Key: your_key"
```

Returns:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown, Current Drawdown
- Win Rate, Profit Factor
- VaR 95%/99%, CVaR
- Number of open positions
- Current market regime

## Prometheus Metrics

Available at `http://localhost:9090/metrics`:

```
archneuronx_portfolio_value_usd
archneuronx_drawdown_pct
archneuronx_sharpe_ratio
archneuronx_open_positions_count
archneuronx_var_95_usd
archneuronx_circuit_breaker_active
archneuronx_trades_total{side="buy|sell"}
archneuronx_inference_latency_ms{model="mlp|cnn|transformer"}
```
