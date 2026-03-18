#!/usr/bin/env python3
"""
ArchNeuronX v4.0 - Capacity Planning Model
Ultra-Low Latency Trading System Resource Forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchNeuronXCapacityModel:
    """
    Capacity planning model for ArchNeuronX v4.0 trading system.
    Handles resource forecasting, scaling recommendations, and cost optimization.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the capacity model with configuration.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.historical_data = None
        self.forecasts = {}
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "forecast_horizon_days": 90,
            "growth_rate_daily": 0.002,  # 0.2% daily growth
            "seasonal_factor": 1.2,  # 20% seasonal variation
            "buffer_percentage": 0.3,  # 30% buffer capacity
            "max_gpu_utilization": 0.85,
            "max_cpu_utilization": 0.80,
            "max_memory_utilization": 0.85,
            "cost_per_gpu_hour": 4.50,
            "cost_per_cpu_hour": 0.15,
            "cost_per_gb_hour": 0.02
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except FileNotFoundError:
                logger.warning(f"Config file {config_file} not found, using defaults")
        
        return default_config
    
    def load_historical_data(self, data_file: str) -> None:
        """
        Load historical performance data.
        
        Args:
            data_file: Path to historical data CSV file
        """
        try:
            self.historical_data = pd.read_csv(data_file, parse_dates=['timestamp'])
            self.historical_data.set_index('timestamp', inplace=True)
            logger.info(f"Loaded {len(self.historical_data)} historical data points")
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic historical data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-03-18', freq='5T')
        
        # Generate synthetic metrics with realistic patterns
        base_requests = 100000
        base_latency = 15  # microseconds
        base_throughput = 500000  # ops/sec
        
        data = []
        for i, date in enumerate(dates):
            # Add daily and weekly patterns
            daily_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * i / 288)  # 24-hour pattern
            weekly_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * i / 2016)  # 7-day pattern
            
            # Add growth trend
            growth_factor = 1.0 + self.config['growth_rate_daily'] * i
            
            # Add random noise
            noise = np.random.normal(1.0, 0.05)
            
            combined_factor = daily_pattern * weekly_pattern * growth_factor * noise
            
            data.append({
                'timestamp': date,
                'requests_per_second': base_requests * combined_factor,
                'avg_latency_us': base_latency * (2.0 - combined_factor),  # Inverse relationship
                'throughput_ops_per_sec': base_throughput * combined_factor,
                'gpu_utilization': 0.7 * combined_factor,
                'cpu_utilization': 0.6 * combined_factor,
                'memory_utilization': 0.65 * combined_factor,
                'active_connections': int(1000 * combined_factor),
                'error_rate': 0.001 * (2.0 - combined_factor)  # Inverse relationship
            })
        
        self.historical_data = pd.DataFrame(data)
        self.historical_data.set_index('timestamp', inplace=True)
        logger.info("Generated synthetic historical data")
    
    def forecast_demand(self, days: int = None) -> pd.DataFrame:
        """
        Forecast demand for specified period.
        
        Args:
            days: Number of days to forecast (default from config)
            
        Returns:
            DataFrame with forecasted metrics
        """
        if days is None:
            days = self.config['forecast_horizon_days']
        
        if self.historical_data is None:
            raise ValueError("No historical data available for forecasting")
        
        # Generate forecast dates
        last_date = self.historical_data.index.max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(minutes=5),
            periods=days * 288,  # 5-minute intervals
            freq='5T'
        )
        
        # Extract patterns from historical data
        daily_pattern = self._extract_daily_pattern()
        weekly_pattern = self._extract_weekly_pattern()
        growth_trend = self._calculate_growth_trend()
        
        # Generate forecasts
        forecasts = []
        for i, date in enumerate(forecast_dates):
            # Calculate combined factors
            daily_factor = daily_pattern.get(date.hour, 1.0)
            weekly_factor = weekly_pattern.get(date.weekday(), 1.0)
            growth_factor = 1.0 + self.config['growth_rate_daily'] * (i / 288)
            
            # Add seasonal variation
            seasonal_factor = 1.0 + self.config['seasonal_factor'] * np.sin(
                2 * np.pi * date.timetuple().tm_yday / 365
            )
            
            combined_factor = daily_factor * weekly_factor * growth_factor * seasonal_factor
            
            # Get base values from recent historical data
            recent_avg = self.historical_data.tail('1D').mean()
            
            forecasts.append({
                'timestamp': date,
                'forecast_requests_per_second': recent_avg['requests_per_second'] * combined_factor,
                'forecast_latency_us': recent_avg['avg_latency_us'] / combined_factor,
                'forecast_throughput_ops_per_sec': recent_avg['throughput_ops_per_sec'] * combined_factor,
                'forecast_gpu_utilization': min(0.95, recent_avg['gpu_utilization'] * combined_factor),
                'forecast_cpu_utilization': min(0.95, recent_avg['cpu_utilization'] * combined_factor),
                'forecast_memory_utilization': min(0.95, recent_avg['memory_utilization'] * combined_factor),
                'forecast_active_connections': int(recent_avg['active_connections'] * combined_factor),
                'forecast_error_rate': recent_avg['error_rate'] / combined_factor
            })
        
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.set_index('timestamp', inplace=True)
        
        self.forecasts['demand'] = forecast_df
        return forecast_df
    
    def _extract_daily_pattern(self) -> Dict[int, float]:
        """Extract daily usage pattern from historical data."""
        hourly_avg = self.historical_data.groupby(self.historical_data.index.hour).mean()
        
        # Normalize to get relative pattern
        pattern = {}
        for hour in range(24):
            if hour in hourly_avg.index:
                pattern[hour] = hourly_avg.loc[hour, 'requests_per_second'] / hourly_avg['requests_per_second'].mean()
            else:
                pattern[hour] = 1.0
        
        return pattern
    
    def _extract_weekly_pattern(self) -> Dict[int, float]:
        """Extract weekly usage pattern from historical data."""
        daily_avg = self.historical_data.groupby(self.historical_data.index.weekday).mean()
        
        # Normalize to get relative pattern
        pattern = {}
        for day in range(7):
            if day in daily_avg.index:
                pattern[day] = daily_avg.loc[day, 'requests_per_second'] / daily_avg['requests_per_second'].mean()
            else:
                pattern[day] = 1.0
        
        return pattern
    
    def _calculate_growth_trend(self) -> float:
        """Calculate daily growth trend from historical data."""
        if len(self.historical_data) < 7:
            return self.config['growth_rate_daily']
        
        # Calculate week-over-week growth
        weekly_avg = self.historical_data.resample('W').mean()
        if len(weekly_avg) < 2:
            return self.config['growth_rate_daily']
        
        growth_rate = (weekly_avg.iloc[-1]['requests_per_second'] / 
                      weekly_avg.iloc[0]['requests_per_second']) ** (1/len(weekly_avg)) - 1
        
        return max(0, growth_rate / 7)  # Convert to daily rate
    
    def calculate_resource_requirements(self) -> Dict[str, Dict]:
        """
        Calculate required resources based on forecasted demand.
        
        Returns:
            Dictionary with resource requirements and recommendations
        """
        if 'demand' not in self.forecasts:
            self.forecast_demand()
        
        forecast = self.forecasts['demand']
        
        # Calculate peak requirements
        peak_requests = forecast['forecast_requests_per_second'].max()
        peak_throughput = forecast['forecast_throughput_ops_per_sec'].max()
        peak_connections = forecast['forecast_active_connections'].max()
        
        # Calculate resource needs with buffer
        buffer_factor = 1.0 + self.config['buffer_percentage']
        
        # GPU requirements
        gpu_per_node = 50000  # ops/sec per GPU node
        required_gpu_nodes = max(2, np.ceil(peak_throughput * buffer_factor / gpu_per_node))
        
        # CPU requirements
        cpu_per_node = 25000  # ops/sec per CPU node
        required_cpu_nodes = max(4, np.ceil(peak_throughput * buffer_factor / cpu_per_node))
        
        # Memory requirements
        memory_per_connection = 10  # MB per connection
        required_memory_gb = max(16, np.ceil(peak_connections * memory_per_connection * buffer_factor / 1024))
        
        # Network requirements
        bandwidth_per_request = 1024  # bytes
        required_bandwidth_gbps = max(1, np.ceil(peak_requests * bandwidth_per_request * buffer_factor / (1024**3)))
        
        # Calculate costs
        daily_gpu_cost = required_gpu_nodes * self.config['cost_per_gpu_hour'] * 24
        daily_cpu_cost = required_cpu_nodes * self.config['cost_per_cpu_hour'] * 24
        daily_memory_cost = required_memory_gb * self.config['cost_per_gb_hour'] * 24
        total_daily_cost = daily_gpu_cost + daily_cpu_cost + daily_memory_cost
        
        requirements = {
            'peak_demand': {
                'requests_per_second': peak_requests,
                'throughput_ops_per_sec': peak_throughput,
                'active_connections': peak_connections
            },
            'resource_requirements': {
                'gpu_nodes': int(required_gpu_nodes),
                'cpu_nodes': int(required_cpu_nodes),
                'memory_gb': int(required_memory_gb),
                'bandwidth_gbps': int(required_bandwidth_gbps)
            },
            'cost_analysis': {
                'daily_gpu_cost': daily_gpu_cost,
                'daily_cpu_cost': daily_cpu_cost,
                'daily_memory_cost': daily_memory_cost,
                'total_daily_cost': total_daily_cost,
                'monthly_cost': total_daily_cost * 30,
                'annual_cost': total_daily_cost * 365
            },
            'scalability_recommendations': self._generate_scalability_recommendations(forecast),
            'risk_assessment': self._assess_capacity_risks(forecast)
        }
        
        return requirements
    
    def _generate_scalability_recommendations(self, forecast: pd.DataFrame) -> List[Dict]:
        """Generate scalability recommendations based on forecast."""
        recommendations = []
        
        # Analyze utilization trends
        peak_gpu_util = forecast['forecast_gpu_utilization'].max()
        peak_cpu_util = forecast['forecast_cpu_utilization'].max()
        peak_memory_util = forecast['forecast_memory_utilization'].max()
        
        # GPU recommendations
        if peak_gpu_util > self.config['max_gpu_utilization']:
            recommendations.append({
                'type': 'gpu_scaling',
                'priority': 'high',
                'description': f'GPU utilization will peak at {peak_gpu_util:.1%}, exceeding {self.config["max_gpu_utilization"]:.0%} threshold',
                'action': 'Scale GPU nodes or optimize GPU memory usage',
                'timeline': 'immediate'
            })
        
        # CPU recommendations
        if peak_cpu_util > self.config['max_cpu_utilization']:
            recommendations.append({
                'type': 'cpu_scaling',
                'priority': 'high',
                'description': f'CPU utilization will peak at {peak_cpu_util:.1%}, exceeding {self.config["max_cpu_utilization"]:.0%} threshold',
                'action': 'Scale CPU nodes or optimize processing',
                'timeline': 'immediate'
            })
        
        # Memory recommendations
        if peak_memory_util > self.config['max_memory_utilization']:
            recommendations.append({
                'type': 'memory_scaling',
                'priority': 'medium',
                'description': f'Memory utilization will peak at {peak_memory_util:.1%}, exceeding {self.config["max_memory_utilization"]:.0%} threshold',
                'action': 'Scale memory or optimize memory usage',
                'timeline': 'within 1 week'
            })
        
        # Growth recommendations
        growth_rate = self._calculate_growth_trend()
        if growth_rate > 0.005:  # 0.5% daily growth
            recommendations.append({
                'type': 'growth_planning',
                'priority': 'medium',
                'description': f'High growth rate detected ({growth_rate:.2%} daily)',
                'action': 'Plan for aggressive scaling in next quarter',
                'timeline': 'within 1 month'
            })
        
        # Seasonal recommendations
        if self.config['seasonal_factor'] > 0.1:
            recommendations.append({
                'type': 'seasonal_planning',
                'priority': 'low',
                'description': 'Significant seasonal variation detected',
                'action': 'Prepare for seasonal scaling patterns',
                'timeline': 'next season'
            })
        
        return recommendations
    
    def _assess_capacity_risks(self, forecast: pd.DataFrame) -> List[Dict]:
        """Assess capacity risks based on forecast."""
        risks = []
        
        # Calculate risk factors
        peak_requests = forecast['forecast_requests_per_second'].max()
        avg_requests = forecast['forecast_requests_per_second'].mean()
        volatility = forecast['forecast_requests_per_second'].std() / avg_requests
        
        # High volatility risk
        if volatility > 0.3:
            risks.append({
                'type': 'volatility',
                'severity': 'high',
                'description': f'High demand volatility detected ({volatility:.1%})',
                'impact': 'Resource exhaustion risk',
                'mitigation': 'Increase buffer capacity and implement auto-scaling'
            })
        
        # Growth risk
        growth_rate = self._calculate_growth_trend()
        if growth_rate > 0.01:  # 1% daily growth
            risks.append({
                'type': 'growth',
                'severity': 'medium',
                'description': f'High growth rate ({growth_rate:.1%} daily)',
                'impact': 'Capacity shortage risk',
                'mitigation': 'Proactive scaling and capacity planning'
            })
        
        # Single point of failure risk
        if forecast['forecast_gpu_utilization'].max() > 0.9:
            risks.append({
                'type': 'single_point_failure',
                'severity': 'high',
                'description': 'GPU utilization approaching capacity limits',
                'impact': 'System failure risk',
                'mitigation': 'Add redundancy and load balancing'
            })
        
        # Cost risk
        cost_growth = self._calculate_cost_growth(forecast)
        if cost_growth > 0.5:  # 50% cost increase
            risks.append({
                'type': 'cost',
                'severity': 'medium',
                'description': f'High cost growth projected ({cost_growth:.1%})',
                'impact': 'Budget overrun risk',
                'mitigation': 'Optimize resource usage and implement cost controls'
            })
        
        return risks
    
    def _calculate_cost_growth(self, forecast: pd.DataFrame) -> float:
        """Calculate projected cost growth rate."""
        current_cost = self._calculate_current_daily_cost()
        projected_cost = self._calculate_projected_daily_cost(forecast)
        return (projected_cost - current_cost) / current_cost
    
    def _calculate_current_daily_cost(self) -> float:
        """Calculate current daily cost based on current resources."""
        # This would typically query current infrastructure
        # For now, use baseline values
        return 1000.0  # Placeholder
    
    def _calculate_projected_daily_cost(self, forecast: pd.DataFrame) -> float:
        """Calculate projected daily cost based on forecast."""
        # Simplified cost calculation
        avg_requests = forecast['forecast_requests_per_second'].mean()
        base_cost = 1000.0
        growth_factor = avg_requests / 100000  # Normalized to baseline
        return base_cost * growth_factor
    
    def generate_capacity_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive capacity planning report.
        
        Args:
            output_file: Path to save report (optional)
            
        Returns:
            Report content as string
        """
        requirements = self.calculate_resource_requirements()
        
        report = f"""
# ArchNeuronX v4.0 - Capacity Planning Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Peak Demand Forecast
- **Requests/sec**: {requirements['peak_demand']['requests_per_second']:,.0f}
- **Throughput**: {requirements['peak_demand']['throughput_ops_per_sec']:,.0f} ops/sec
- **Active Connections**: {requirements['peak_demand']['active_connections']:,}
- **Forecast Horizon**: {self.config['forecast_horizon_days']} days

### Resource Requirements
- **GPU Nodes**: {requirements['resource_requirements']['gpu_nodes']}
- **CPU Nodes**: {requirements['resource_requirements']['cpu_nodes']}
- **Memory**: {requirements['resource_requirements']['memory_gb']} GB
- **Bandwidth**: {requirements['resource_requirements']['bandwidth_gbps']} Gbps

### Cost Analysis
- **Daily Cost**: ${requirements['cost_analysis']['total_daily_cost']:,.2f}
- **Monthly Cost**: ${requirements['cost_analysis']['monthly_cost']:,.2f}
- **Annual Cost**: ${requirements['cost_analysis']['annual_cost']:,.2f}
- **Cost per Request**: ${requirements['cost_analysis']['total_daily_cost'] / (requirements['peak_demand']['requests_per_second'] * 86400):.6f}

## Detailed Analysis

### Resource Utilization
- **GPU Utilization**: {self.config['max_gpu_utilization']:.0%} target
- **CPU Utilization**: {self.config['max_cpu_utilization']:.0%} target
- **Memory Utilization**: {self.config['max_memory_utilization']:.0%} target
- **Buffer Capacity**: {self.config['buffer_percentage']:.0%}

### Scalability Recommendations
"""
        
        for i, rec in enumerate(requirements['scalability_recommendations'], 1):
            report += f"""
{i}. **{rec['type'].title()}** ({rec['priority'].upper()})
   - **Description**: {rec['description']}
   - **Action**: {rec['action']}
   - **Timeline**: {rec['timeline']}
"""
        
        report += """
### Risk Assessment
"""
        
        for i, risk in enumerate(requirements['risk_assessment'], 1):
            report += f"""
{i}. **{risk['type'].title()}** ({risk['severity'].upper()})
   - **Description**: {risk['description']}
   - **Impact**: {risk['impact']}
   - **Mitigation**: {risk['mitigation']}
"""
        
        report += f"""

## Recommendations

### Immediate Actions (Next 7 Days)
1. **Resource Provisioning**: Scale infrastructure to meet peak demand
2. **Monitoring Enhancement**: Implement capacity monitoring alerts
3. **Auto-Scaling**: Configure automated scaling based on demand

### Short-term Actions (Next 30 Days)
1. **Cost Optimization**: Implement resource optimization strategies
2. **Redundancy**: Add failover capacity for critical components
3. **Performance Tuning**: Optimize resource utilization

### Long-term Actions (Next 90 Days)
1. **Capacity Planning**: Establish regular capacity planning process
2. **Predictive Scaling**: Implement ML-based demand forecasting
3. **Cost Management**: Implement cost controls and optimization

## Appendix

### Configuration
- **Growth Rate**: {self.config['growth_rate_daily']:.2%} daily
- **Seasonal Factor**: {self.config['seasonal_factor']:.1f}
- **Buffer Percentage**: {self.config['buffer_percentage']:.0%}
- **Forecast Horizon**: {self.config['forecast_horizon_days']} days

### Assumptions
- Linear growth pattern with seasonal variations
- Current resource utilization patterns continue
- No major market disruptions
- Technology stack remains stable

---
*Report generated by ArchNeuronX v4.0 Capacity Planning Model*
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def visualize_forecasts(self, output_dir: str = None) -> None:
        """
        Create visualizations of capacity forecasts.
        
        Args:
            output_dir: Directory to save plots
        """
        if 'demand' not in self.forecasts:
            self.forecast_demand()
        
        forecast = self.forecasts['demand']
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ArchNeuronX v4.0 - Capacity Forecast', fontsize=16)
        
        # Plot 1: Requests and Throughput
        axes[0, 0].plot(forecast.index, forecast['forecast_requests_per_second'], label='Requests/sec')
        axes[0, 0].plot(forecast.index, forecast['forecast_throughput_ops_per_sec'], label='Throughput ops/sec')
        axes[0, 0].set_title('Demand Forecast')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Resource Utilization
        axes[0, 1].plot(forecast.index, forecast['forecast_gpu_utilization'], label='GPU')
        axes[0, 1].plot(forecast.index, forecast['forecast_cpu_utilization'], label='CPU')
        axes[0, 1].plot(forecast.index, forecast['forecast_memory_utilization'], label='Memory')
        axes[0, 1].set_title('Resource Utilization Forecast')
        axes[0, 1].set_ylabel('Utilization (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Latency and Error Rate
        axes[1, 0].plot(forecast.index, forecast['forecast_latency_us'], label='Latency (μs)')
        axes[1, 0].set_title('Performance Forecast')
        axes[1, 0].set_ylabel('Latency (μs)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Active Connections
        axes[1, 1].plot(forecast.index, forecast['forecast_active_connections'], label='Connections')
        axes[1, 1].set_title('Connection Forecast')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            plot_file = os.path.join(output_dir, 'capacity_forecast.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {plot_file}")
        
        plt.show()
    
    def save_forecasts(self, output_file: str) -> None:
        """Save forecasts to JSON file."""
        if 'demand' not in self.forecasts:
            self.forecast_demand()
        
        forecasts_data = {
            'config': self.config,
            'forecasts': {
                'demand': self.forecasts['demand'].to_dict(),
                'resource_requirements': self.calculate_resource_requirements()
            },
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(forecasts_data, f, indent=2, default=str)
        
        logger.info(f"Forecasts saved to {output_file}")


def main():
    """Main function for testing the capacity model."""
    # Initialize model
    model = ArchNeuronXCapacityModel()
    
    # Load or generate data
    try:
        model.load_historical_data('historical_data.csv')
    except:
        logger.info("Using synthetic data for demonstration")
    
    # Generate forecasts
    forecast = model.forecast_demand()
    print(f"Generated {len(forecast)} forecast points")
    
    # Calculate requirements
    requirements = model.calculate_resource_requirements()
    print(f"Peak demand: {requirements['peak_demand']['requests_per_second']:,.0f} requests/sec")
    print(f"Required GPU nodes: {requirements['resource_requirements']['gpu_nodes']}")
    print(f"Daily cost: ${requirements['cost_analysis']['total_daily_cost']:,.2f}")
    
    # Generate report
    report = model.generate_capacity_report('capacity_report.md')
    print("Capacity report generated")
    
    # Create visualizations
    model.visualize_forecasts()
    
    # Save forecasts
    model.save_forecasts('forecasts.json')
    print("Forecasts saved to JSON")


if __name__ == "__main__":
    main()
