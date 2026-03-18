# ArchNeuronX v4.0 Live Trading Guide

## Overview

ArchNeuronX v4.0 Live Trading is a comprehensive real-time trading system that combines quantum-enhanced AI, autonomous trading agents, and LLM integration for intelligent market execution. This guide covers setup, configuration, and operation of the live trading system.

## 🚀 Quick Start

### Prerequisites

**System Requirements:**
- **OS**: Linux, macOS, or Windows 10+
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet connection

**Software Dependencies:**
- **C++20** compatible compiler
- **CMake 3.20+**
- **Python 3.8+**
- **PyTorch 2.6+**
- **curl** (for API calls)
- **nlohmann/json** (for JSON parsing)

**Exchange Requirements:**
- **API Keys**: Trading API keys from supported exchanges
- **Permissions**: Trading permissions enabled
- **IP Whitelist**: Your server IP whitelisted

### Installation

#### 1. Build the System
```bash
# Clone repository
git clone https://github.com/Gzeu/ArchNeuronX.git
cd ArchNeuronX

# Build live trading system
./scripts/run_live_trading.sh build

# Or on Windows
.\scripts\run_live_trading.bat build
```

#### 2. Configure API Keys
```bash
# Create configuration file
cp config/live_trading.example.yaml config/live_trading.yaml

# Edit with your API keys
nano config/live_trading.yaml
```

#### 3. Start Trading
```bash
# Interactive mode (recommended for beginners)
./scripts/run_live_trading.sh interactive

# Automated mode (for production)
./scripts/run_live_trading.sh automated
```

## 📋 Trading Modes

### 🎯 Interactive Mode

**Purpose**: Manual control with real-time commands
**Best for**: Learning, testing, manual trading
**Features**:
- Real-time market data display
- Manual order placement
- Interactive portfolio management
- Live performance metrics

**Commands**:
```bash
status      # Show system status
market      # Show market data
orders      # Show open orders
positions   # Show positions
buy BTCUSDT 0.1 50000    # Place buy order
sell ETHUSDT 1.0 3000   # Place sell order
cancel order_id          # Cancel order
risk       # Show risk metrics
stop       # Stop trading
```

### 🤖 Automated Mode

**Purpose**: Fully automated trading
**Best for**: Production trading, 24/7 operation
**Features**:
- Autonomous signal generation
- Automatic order execution
- Real-time risk management
- Alert system
- Background operation

**Configuration**:
```yaml
trading:
  symbols: ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
  interval_ms: 1000
  risk_per_trade: 100.0
  max_position_size: 1000.0
  
risk_management:
  max_daily_loss: 1000.0
  max_drawdown: 0.1
  leverage_limit: 2.0
  stop_loss_pct: 0.02
  take_profit_pct: 0.05
```

### 📊 Backtesting Mode

**Purpose**: Test strategies on historical data
**Best for**: Strategy validation, performance analysis
**Features**:
- Historical data simulation
- Performance metrics calculation
- Risk analysis
- Strategy optimization

**Usage**:
```bash
# Run backtesting
./scripts/run_live_trading.sh backtest

# Custom backtesting
./build/archneuronx_live_trading --backtest \
  --data-path data/historical/ \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### 📝 Paper Trading Mode

**Purpose**: Test with simulated money
**Best for**: Strategy testing without real money
**Features**:
- Simulated portfolio
- Real market data
- No financial risk
- Full trading features

**Usage**:
```bash
# Start paper trading
./scripts/run_live_trading.sh paper

# Custom paper trading
./build/archneuronx_live_trading --paper \
  --balance 10000.0 \
  --symbols BTCUSDT,ETHUSDT
```

## 🔧 Configuration

### Exchange Configuration

**Binance Configuration**:
```yaml
exchange:
  name: "binance"
  api_key: "your_api_key_here"
  api_secret: "your_api_secret_here"
  base_url: "https://api.binance.com"
  testnet: false  # Set to true for testing
  
  # Trading pairs
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "BNBUSDT"
    - "ADAUSDT"
    - "SOLUSDT"
```

**Coinbase Configuration**:
```yaml
exchange:
  name: "coinbase"
  api_key: "your_api_key_here"
  api_secret: "your_api_secret_here"
  passphrase: "your_passphrase_here"
  base_url: "https://api.pro.coinbase.com"
  testnet: false
```

### Trading Configuration

**Basic Trading Settings**:
```yaml
trading:
  # Symbols to trade
  symbols: ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
  
  # Trading interval (milliseconds)
  interval_ms: 1000
  
  # Risk parameters
  risk_per_trade: 100.0      # Risk per trade in USD
  max_position_size: 1000.0   # Maximum position size
  max_daily_trades: 50        # Maximum trades per day
  
  # Order settings
  order_type: "market"        # market, limit, stop
  slippage_tolerance: 0.001   # 0.1% slippage tolerance
```

**Advanced Trading Settings**:
```yaml
trading:
  # Signal generation
  quantum_weight: 0.4         # Weight for quantum signals
  agent_weight: 0.3           # Weight for agent signals
  llm_weight: 0.3             # Weight for LLM signals
  
  # Position management
  position_sizing: "fixed"    # fixed, percentage, volatility
  rebalance_interval: 3600    # Rebalance every hour
  
  # Execution settings
  execution_delay: 100        # Execution delay in ms
  retry_attempts: 3          # Order retry attempts
  timeout_seconds: 30         # Order timeout
```

### Risk Management Configuration

**Basic Risk Settings**:
```yaml
risk_management:
  # Position limits
  max_position_size: 1000.0
  max_positions_per_symbol: 1
  max_total_positions: 5
  
  # Loss limits
  max_daily_loss: 1000.0
  max_drawdown: 0.1           # 10% max drawdown
  stop_loss_pct: 0.02         # 2% stop loss
  take_profit_pct: 0.05       # 5% take profit
  
  # Leverage
  leverage_limit: 2.0
  margin_call_pct: 0.8        # 80% margin call
  liquidation_pct: 0.9        # 90% liquidation
```

**Advanced Risk Settings**:
```yaml
risk_management:
  # VaR settings
  var_confidence: 0.95        # 95% VaR
  var_timeframe: 1            # 1 day VaR
  
  # Correlation limits
  max_correlation: 0.7        # Max correlation between positions
  sector_exposure: 0.3        # Max sector exposure
  
  # Volatility limits
  max_volatility: 0.05       # 5% max volatility
  volatility_adjustment: true  # Adjust position size based on volatility
  
  # Risk alerts
  risk_alerts:
    - "daily_loss_exceeded"
    - "drawdown_exceeded"
    - "position_limit_exceeded"
    - "margin_call"
```

### AI Configuration

**Quantum Neural Network Settings**:
```yaml
ai:
  quantum_neural_network:
    model_path: "models/quantum_nn.pt"
    input_features: 10
    hidden_layers: [128, 64, 32]
    output_classes: 3          # BUY, SELL, HOLD
    
    # Training parameters
    learning_rate: 0.001
    batch_size: 32
    epochs: 100
    
    # Quantum parameters
    quantum_heads: 16
    quantum_layers: 6
    quantum_states: 8
    coherence_threshold: 0.8
```

**Trading Agent Settings**:
```yaml
ai:
  quantum_trading_agents:
    num_agents: 5
    state_dim: 10
    action_dim: 3              # BUY, SELL, HOLD
    
    # Learning parameters
    learning_rate: 0.001
    discount_factor: 0.95
    exploration_rate: 0.1
    
    # Memory parameters
    memory_size: 10000
    batch_size: 64
    
    # Coordination parameters
    coordination_enabled: true
    entanglement_strength: 0.5
```

**LLM Integration Settings**:
```yaml
ai:
  llm_integration:
    provider: "huggingface"
    model: "mistralai/Mistral-7B-v0.1"
    
    # Model settings
    max_tokens: 512
    temperature: 0.7
    top_p: 0.9
    
    # Prompt settings
    trading_prompt: "Analyze market conditions and provide trading recommendation"
    analysis_prompt: "Provide detailed market analysis"
    risk_prompt: "Assess risk factors and provide risk management advice"
    
    # Cache settings
    cache_enabled: true
    cache_size: 1000
    cache_ttl: 3600            # 1 hour cache TTL
```

## 📊 Monitoring and Analytics

### Real-time Monitoring

**Dashboard Metrics**:
- **Portfolio Value**: Total portfolio value in USD
- **Cash Balance**: Available cash for trading
- **Total P&L**: Overall profit and loss
- **Daily P&L**: Today's profit and loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Maximum portfolio drawdown

**Performance Metrics**:
- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of losing trades
- **Average Win**: Average profit from winning trades
- **Average Loss**: Average loss from losing trades
- **Profit Factor**: Ratio of total wins to total losses

### Risk Monitoring

**Risk Metrics**:
- **VaR 95%**: Value at Risk at 95% confidence
- **Beta**: Portfolio beta relative to market
- **Alpha**: Portfolio alpha (excess return)
- **Sortino Ratio**: Downside risk-adjusted return
- **Volatility**: Portfolio volatility
- **Correlation**: Correlation with market

**Risk Alerts**:
- **Daily Loss Alert**: When daily loss exceeds limit
- **Drawdown Alert**: When drawdown exceeds limit
- **Margin Call Alert**: When margin usage is high
- **Position Limit Alert**: When position limits are exceeded

### Logging and Audit

**Log Levels**:
- **DEBUG**: Detailed debugging information
- **INFO**: General information messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical error messages

**Log Files**:
- **live_trading.log**: Main trading log
- **orders.log**: Order execution log
- **risk.log**: Risk management log
- **performance.log**: Performance metrics log

## 🚨 Alert System

### Alert Types

**Trade Alerts**:
- Order placed
- Order filled
- Order cancelled
- Position opened/closed

**Risk Alerts**:
- Daily loss exceeded
- Maximum drawdown exceeded
- Position limit exceeded
- Margin call warning
- High volatility detected

**Performance Alerts**:
- Daily P&L target reached
- Win rate threshold crossed
- Performance milestone achieved

**System Alerts**:
- Exchange connection lost
- API rate limit exceeded
- System error occurred
- Maintenance mode activated

### Alert Configuration

**Email Alerts**:
```yaml
alerts:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your_email@gmail.com"
    password: "your_app_password"
    recipients:
      - "trader@example.com"
      - "risk@example.com"
```

**SMS Alerts**:
```yaml
alerts:
  sms:
    enabled: true
    provider: "twilio"
    account_sid: "your_account_sid"
    auth_token: "your_auth_token"
    phone_numbers:
      - "+1234567890"
```

**Webhook Alerts**:
```yaml
alerts:
  webhook:
    enabled: true
    url: "https://your-webhook-url.com/alerts"
    headers:
      Authorization: "Bearer your_token"
      Content-Type: "application/json"
```

## 🔒 Security Best Practices

### API Security

**API Key Management**:
- Use dedicated API keys for trading
- Set appropriate permissions (read/write)
- Use IP whitelisting
- Regularly rotate API keys
- Never commit API keys to version control

**Rate Limiting**:
- Respect exchange rate limits
- Implement request throttling
- Use exponential backoff for retries
- Monitor API usage

### Data Security

**Data Encryption**:
- Encrypt sensitive configuration files
- Use TLS for all network communications
- Secure database connections
- Encrypt log files containing sensitive data

**Access Control**:
- Implement role-based access control
- Use strong passwords
- Enable two-factor authentication
- Regular security audits

## 📈 Performance Optimization

### System Optimization

**Hardware Optimization**:
- Use SSD for fast I/O
- Ensure sufficient RAM
- Use multi-core processors
- Optimize network connectivity

**Software Optimization**:
- Enable compiler optimizations
- Use efficient data structures
- Minimize memory allocations
- Optimize database queries

### Trading Optimization

**Latency Optimization**:
- Use co-located servers
- Optimize network routes
- Use efficient algorithms
- Minimize computational overhead

**Throughput Optimization**:
- Parallelize signal generation
- Batch order processing
- Use efficient data structures
- Optimize database operations

## 🛠️ Troubleshooting

### Common Issues

**Connection Issues**:
- Check internet connectivity
- Verify API keys and permissions
- Check exchange status
- Verify IP whitelisting

**Performance Issues**:
- Monitor system resources
- Check for memory leaks
- Optimize configuration
- Scale hardware resources

**Trading Issues**:
- Verify market data availability
- Check order execution
- Monitor risk limits
- Review trading logic

### Debug Mode

**Enable Debug Logging**:
```bash
# Set debug mode
export ARCHNEURONX_LOG_LEVEL=DEBUG

# Run with debug
./build/archneuronx_live_trading --debug
```

**Debug Commands**:
```bash
# Check system status
./scripts/run_live_trading.sh status

# View logs
./scripts/run_live_trading.sh logs

# Test configuration
./build/archneuronx_live_trading --test-config
```

## 📚 Advanced Features

### Multi-Exchange Trading

**Configuration**:
```yaml
exchanges:
  - name: "binance"
    api_key: "binance_key"
    api_secret: "binance_secret"
    symbols: ["BTCUSDT", "ETHUSDT"]
    
  - name: "coinbase"
    api_key: "coinbase_key"
    api_secret: "coinbase_secret"
    symbols: ["BTC-USD", "ETH-USD"]
```

### Arbitrage Trading

**Configuration**:
```yaml
arbitrage:
  enabled: true
  min_spread: 0.001          # 0.1% minimum spread
  max_position: 1000.0
  exchanges: ["binance", "coinbase"]
  symbols: ["BTCUSDT", "BTC-USD"]
```

### Social Trading

**Configuration**:
```yaml
social_trading:
  enabled: true
  follow_traders:
    - "trader1"
    - "trader2"
  copy_percentage: 0.1      # Copy 10% of positions
  max_copy_size: 500.0
```

## 📞 Support

### Documentation

- **API Reference**: [API Documentation](docs/api/live_trading_api.md)
- **Configuration Guide**: [Configuration Guide](docs/configuration.md)
- **Troubleshooting**: [Troubleshooting Guide](docs/troubleshooting.md)

### Community

- **GitHub Issues**: [Report Issues](https://github.com/Gzeu/ArchNeuronX/issues)
- **Discussions**: [Community Forum](https://github.com/Gzeu/ArchNeuronX/discussions)
- **Wiki**: [Knowledge Base](https://github.com/Gzeu/ArchNeuronX/wiki)

### Professional Support

- **Email**: support@archneuronx.com
- **Discord**: [Join Discord](https://discord.gg/archneuronx)
- **Telegram**: [Join Telegram](https://t.me/archneuronx)

---

**⚠️ DISCLAIMER**: This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Always do your own research and never risk more than you can afford to lose.

**🚀 START TRADING WITH ARCHNEURONX V4.0 TODAY!**
