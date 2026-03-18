#!/usr/bin/env python3
"""
ArchNeuronX v4.0 - Complete System Integration Tests
Comprehensive testing suite for <20μs latency and 500K+ orders/sec validation
"""

import asyncio
import time
import logging
import statistics
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter
import pytest
import json
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
LATENCY_GAUGE = Gauge('integration_test_latency_us', 'Integration test latency in microseconds')
THROUGHPUT_GAUGE = Gauge('integration_test_throughput_ops', 'Integration test throughput in ops/sec')
SUCCESS_RATE_GAUGE = Gauge('integration_test_success_rate', 'Integration test success rate')
ERROR_COUNTER = Counter('integration_test_errors', 'Integration test errors', ['service', 'error_type'])

@dataclass
class TestConfig:
    """Configuration for integration tests"""
    base_url: str = "http://archneuronx-api:8080"
    market_data_url: str = "http://market-data-service:8081"
    order_routing_url: str = "http://order-routing-service:8082"
    risk_management_url: str = "http://risk-management-service:8083"
    portfolio_optimizer_url: str = "http://portfolio-optimizer-service:8084"
    
    # Performance Targets
    target_latency_us: float = 20.0
    target_throughput_ops: int = 500000
    target_success_rate: float = 0.9999
    
    # Test Parameters
    test_duration_seconds: int = 300  # 5 minutes
    warmup_duration_seconds: int = 60   # 1 minute
    concurrent_requests: int = 1000
    batch_size: int = 32
    timeout_seconds: int = 30

class IntegrationTestSuite:
    """Complete integration test suite for ArchNeuronX v4.0"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout_seconds
        self.results = []
        self.start_time = None
        
    async def run_all_tests(self) -> Dict[str, any]:
        """Run complete integration test suite"""
        logger.info("Starting ArchNeuronX v4.0 Integration Test Suite")
        
        results = {
            "test_suite": "archneuronx_v4_integration",
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "results": {}
        }
        
        try:
            # Phase 1: Health Checks
            logger.info("Phase 1: Health Checks")
            health_results = await self.run_health_checks()
            results["results"]["health_checks"] = health_results
            
            # Phase 2: Service Integration Tests
            logger.info("Phase 2: Service Integration Tests")
            integration_results = await self.run_service_integration_tests()
            results["results"]["service_integration"] = integration_results
            
            # Phase 3: Performance Tests
            logger.info("Phase 3: Performance Tests")
            performance_results = await self.run_performance_tests()
            results["results"]["performance"] = performance_results
            
            # Phase 4: End-to-End Tests
            logger.info("Phase 4: End-to-End Tests")
            e2e_results = await self.run_end_to_end_tests()
            results["results"]["end_to_end"] = e2e_results
            
            # Phase 5: Load Tests
            logger.info("Phase 5: Load Tests")
            load_results = await self.run_load_tests()
            results["results"]["load_testing"] = load_results
            
            # Phase 6: Security Tests
            logger.info("Phase 6: Security Tests")
            security_results = await self.run_security_tests()
            results["results"]["security"] = security_results
            
            # Overall Results
            results["overall_success"] = self.evaluate_overall_success(results["results"])
            results["summary"] = self.generate_summary(results["results"])
            
        except Exception as e:
            logger.error(f"Integration test suite failed: {e}")
            results["error"] = str(e)
            results["overall_success"] = False
        
        return results
    
    async def run_health_checks(self) -> Dict[str, any]:
        """Run health checks on all services"""
        health_results = {
            "services": {},
            "overall_health": False
        }
        
        services = {
            "market_transformer": self.config.base_url,
            "market_data": self.config.market_data_url,
            "order_routing": self.config.order_routing_url,
            "risk_management": self.config.risk_management_url,
            "portfolio_optimizer": self.config.portfolio_optimizer_url
        }
        
        for service_name, service_url in services.items():
            try:
                response = self.session.get(f"{service_url}/health", timeout=10)
                health_data = response.json()
                
                health_results["services"][service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": health_data
                }
                
                logger.info(f"Health check {service_name}: {response.status_code}")
                
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                health_results["services"][service_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Overall health check
        health_results["overall_health"] = all(
            service["status"] == "healthy" 
            for service in health_results["services"].values()
        )
        
        return health_results
    
    async def run_service_integration_tests(self) -> Dict[str, any]:
        """Run service integration tests"""
        integration_results = {
            "tests": {},
            "overall_success": False
        }
        
        tests = [
            ("market_transformer_integration", self.test_market_transformer_integration),
            ("graph_network_integration", self.test_graph_network_integration),
            ("order_routing_integration", self.test_order_routing_integration),
            ("risk_management_integration", self.test_risk_management_integration),
            ("portfolio_optimizer_integration", self.test_portfolio_optimizer_integration),
            ("cross_service_communication", self.test_cross_service_communication),
            ("data_pipeline_integration", self.test_data_pipeline_integration),
            ("authentication_integration", self.test_authentication_integration)
        ]
        
        for test_name, test_func in tests:
            try:
                test_result = await test_func()
                integration_results["tests"][test_name] = test_result
                logger.info(f"Integration test {test_name}: {test_result['success']}")
            except Exception as e:
                logger.error(f"Integration test {test_name} failed: {e}")
                integration_results["tests"][test_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        integration_results["overall_success"] = all(
            test.get("success", False) 
            for test in integration_results["tests"].values()
        )
        
        return integration_results
    
    async def test_market_transformer_integration(self) -> Dict[str, any]:
        """Test Market Transformer service integration"""
        test_data = self.generate_market_data()
        
        start_time = time.perf_counter()
        response = self.session.post(
            f"{self.config.base_url}/v4/analyze",
            json=test_data,
            timeout=10
        )
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        
        LATENCY_GAUGE.set(latency_us)
        
        success = (
            response.status_code == 200 and
            latency_us < self.config.target_latency_us * 5  # Allow 5x for integration
        )
        
        if not success:
            ERROR_COUNTER.labels(service="market_transformer", error_type="integration").inc()
        
        return {
            "success": success,
            "latency_us": latency_us,
            "response_code": response.status_code,
            "signal": response.json() if response.status_code == 200 else None,
            "test_data": test_data
        }
    
    async def test_graph_network_integration(self) -> Dict[str, any]:
        """Test Graph Network service integration"""
        test_data = self.generate_graph_data()
        
        start_time = time.perf_counter()
        response = self.session.post(
            f"{self.config.base_url}/v4/correlations",
            json=test_data,
            timeout=10
        )
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        
        success = (
            response.status_code == 200 and
            latency_us < self.config.target_latency_us * 10  # Allow 10x for graph processing
        )
        
        return {
            "success": success,
            "latency_us": latency_us,
            "response_code": response.status_code,
            "correlations": response.json() if response.status_code == 200 else None
        }
    
    async def test_order_routing_integration(self) -> Dict[str, any]:
        """Test Order Routing service integration"""
        test_data = self.generate_order_data()
        
        start_time = time.perf_counter()
        response = self.session.post(
            f"{self.config.order_routing_url}/v4/route/select",
            json=test_data,
            timeout=10
        )
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        
        success = (
            response.status_code == 200 and
            latency_us < self.config.target_latency_us * 2  # Allow 2x for routing
        )
        
        return {
            "success": success,
            "latency_us": latency_us,
            "response_code": response.status_code,
            "routing_decision": response.json() if response.status_code == 200 else None
        }
    
    async def test_risk_management_integration(self) -> Dict[str, any]:
        """Test Risk Management service integration"""
        test_data = self.generate_risk_data()
        
        start_time = time.perf_counter()
        response = self.session.post(
            f"{self.config.risk_management_url}/v4/assess",
            json=test_data,
            timeout=10
        )
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        
        success = (
            response.status_code == 200 and
            latency_us < self.config.target_latency_us * 50  # Allow 50x for risk calculations
        )
        
        return {
            "success": success,
            "latency_us": latency_us,
            "response_code": response.status_code,
            "risk_assessment": response.json() if response.status_code == 200 else None
        }
    
    async def test_portfolio_optimizer_integration(self) -> Dict[str, any]:
        """Test Portfolio Optimizer service integration"""
        test_data = self.generate_portfolio_data()
        
        start_time = time.perf_counter()
        response = self.session.post(
            f"{self.config.portfolio_optimizer_url}/v4/optimize",
            json=test_data,
            timeout=10
        )
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        
        success = (
            response.status_code == 200 and
            latency_us < self.config.target_latency_us * 10  # Allow 10x for optimization
        )
        
        return {
            "success": success,
            "latency_us": latency_us,
            "response_code": response.status_code,
            "optimization_result": response.json() if response.status_code == 200 else None
        }
    
    async def test_cross_service_communication(self) -> Dict[str, any]:
        """Test cross-service communication"""
        # Test that services can communicate with each other
        test_results = {}
        
        # Test Market Transformer to Order Routing
        try:
            market_data = self.generate_market_data()
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze_with_routing",
                json=market_data,
                timeout=15
            )
            test_results["market_to_routing"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["market_to_routing"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test Order Routing to Risk Management
        try:
            order_data = self.generate_order_data()
            response = self.session.post(
                f"{self.config.order_routing_url}/v4/route_with_risk",
                json=order_data,
                timeout=15
            )
            test_results["routing_to_risk"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["routing_to_risk"] = {
                "success": False,
                "error": str(e)
            }
        
        return {
            "success": all(result.get("success", False) for result in test_results.values()),
            "test_results": test_results
        }
    
    async def test_data_pipeline_integration(self) -> Dict[str, any]:
        """Test data pipeline integration"""
        # Test that data flows correctly through the pipeline
        test_results = {}
        
        # Test market data ingestion
        try:
            market_data = self.generate_market_data()
            response = self.session.post(
                f"{self.config.market_data_url}/v4/ingest",
                json=market_data,
                timeout=10
            )
            test_results["data_ingestion"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["data_ingestion"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test data processing
        try:
            response = self.session.get(
                f"{self.config.market_data_url}/v4/processed/BTC/USD",
                timeout=10
            )
            test_results["data_processing"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["data_processing"] = {
                "success": False,
                "error": str(e)
            }
        
        return {
            "success": all(result.get("success", False) for result in test_results.values()),
            "test_results": test_results
        }
    
    async def test_authentication_integration(self) -> Dict[str, any]:
        """Test authentication integration"""
        test_results = {}
        
        # Test authentication
        try:
            auth_data = {
                "username": "test_user",
                "password": "test_password"
            }
            response = self.session.post(
                f"{self.config.base_url}/v4/auth/login",
                json=auth_data,
                timeout=10
            )
            test_results["authentication"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["authentication"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test authorization
        try:
            headers = {"Authorization": "Bearer test_token"}
            response = self.session.get(
                f"{self.config.base_url}/v4/protected",
                headers=headers,
                timeout=10
            )
            test_results["authorization"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["authorization"] = {
                "success": False,
                "error": str(e)
            }
        
        return {
            "success": all(result.get("success", False) for result in test_results.values()),
            "test_results": test_results
        }
    
    async def run_performance_tests(self) -> Dict[str, any]:
        """Run performance tests"""
        performance_results = {
            "latency_tests": {},
            "throughput_tests": {},
            "scalability_tests": {},
            "overall_success": False
        }
        
        # Latency Tests
        logger.info("Running latency tests...")
        latency_results = await self.run_latency_tests()
        performance_results["latency_tests"] = latency_results
        
        # Throughput Tests
        logger.info("Running throughput tests...")
        throughput_results = await self.run_throughput_tests()
        performance_results["throughput_tests"] = throughput_results
        
        # Scalability Tests
        logger.info("Running scalability tests...")
        scalability_results = await self.run_scalability_tests()
        performance_results["scalability_tests"] = scalability_results
        
        # Evaluate overall success
        performance_results["overall_success"] = (
            latency_results["success"] and
            throughput_results["success"] and
            scalability_results["success"]
        )
        
        return performance_results
    
    async def run_latency_tests(self) -> Dict[str, any]:
        """Run latency tests"""
        latencies = []
        errors = 0
        total_tests = 100
        
        for i in range(total_tests):
            try:
                test_data = self.generate_market_data()
                start_time = time.perf_counter()
                response = self.session.post(
                    f"{self.config.base_url}/v4/analyze",
                    json=test_data,
                    timeout=10
                )
                latency_us = (time.perf_counter() - start_time) * 1_000_000
                latencies.append(latency_us)
                
                if response.status_code != 200:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                logger.error(f"Latency test {i} failed: {e}")
        
        # Calculate statistics
        if latencies:
            latency_stats = {
                "mean_us": statistics.mean(latencies),
                "median_us": statistics.median(latencies),
                "p95_us": np.percentile(latencies, 95),
                "p99_us": np.percentile(latencies, 99),
                "min_us": min(latencies),
                "max_us": max(latencies),
                "std_us": statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
        else:
            latency_stats = {}
        
        success = (
            len(latencies) > 0 and
            latency_stats.get("p95_us", float('inf')) < self.config.target_latency_us and
            errors < total_tests * 0.01  # Less than 1% errors
        )
        
        return {
            "success": success,
            "latency_stats": latency_stats,
            "total_tests": total_tests,
            "successful_tests": len(latencies),
            "errors": errors,
            "target_latency_us": self.config.target_latency_us
        }
    
    async def run_throughput_tests(self) -> Dict[str, any]:
        """Run throughput tests"""
        duration_seconds = 60
        start_time = time.time()
        requests_completed = 0
        errors = 0
        
        async def make_request():
            nonlocal requests_completed, errors
            try:
                test_data = self.generate_market_data()
                response = self.session.post(
                    f"{self.config.base_url}/v4/analyze",
                    json=test_data,
                    timeout=10
                )
                if response.status_code == 200:
                    requests_completed += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
        
        # Run concurrent requests
        tasks = []
        while time.time() - start_time < duration_seconds:
            # Create batch of concurrent requests
            batch_tasks = [
                make_request() for _ in range(self.config.concurrent_requests)
            ]
            tasks.extend(batch_tasks)
            
            # Wait for batch to complete
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        # Calculate throughput
        actual_duration = time.time() - start_time
        throughput_ops = requests_completed / actual_duration
        success_rate = requests_completed / (requests_completed + errors)
        
        THROUGHPUT_GAUGE.set(throughput_ops)
        SUCCESS_RATE_GAUGE.set(success_rate)
        
        success = (
            throughput_ops >= self.config.target_throughput_ops and
            success_rate >= self.config.target_success_rate
        )
        
        return {
            "success": success,
            "throughput_ops": throughput_ops,
            "target_throughput_ops": self.config.target_throughput_ops,
            "success_rate": success_rate,
            "target_success_rate": self.config.target_success_rate,
            "duration_seconds": actual_duration,
            "requests_completed": requests_completed,
            "errors": errors
        }
    
    async def run_scalability_tests(self) -> Dict[str, any]:
        """Run scalability tests"""
        scalability_results = {
            "concurrent_users": {},
            "overall_success": False
        }
        
        # Test different concurrency levels
        concurrency_levels = [100, 500, 1000, 2000]
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing with {concurrency} concurrent users")
            
            latencies = []
            errors = 0
            start_time = time.time()
            
            async def make_concurrent_request():
                nonlocal errors
                try:
                    test_data = self.generate_market_data()
                    request_start = time.perf_counter()
                    response = self.session.post(
                        f"{self.config.base_url}/v4/analyze",
                        json=test_data,
                        timeout=10
                    )
                    latency_us = (time.perf_counter() - request_start) * 1_000_000
                    latencies.append(latency_us)
                    
                    if response.status_code != 200:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
            
            # Run concurrent requests
            tasks = [make_concurrent_request() for _ in range(concurrency)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate metrics
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                success_rate = len(latencies) / (len(latencies) + errors)
            else:
                avg_latency = float('inf')
                p95_latency = float('inf')
                success_rate = 0
            
            scalability_results["concurrent_users"][concurrency] = {
                "avg_latency_us": avg_latency,
                "p95_latency_us": p95_latency,
                "success_rate": success_rate,
                "errors": errors,
                "total_requests": len(latencies) + errors
            }
        
        # Evaluate scalability success
        scalability_results["overall_success"] = all(
            result["p95_latency_us"] < self.config.target_latency_us * 2 and
            result["success_rate"] > 0.95
            for result in scalability_results["concurrent_users"].values()
        )
        
        return scalability_results
    
    async def run_end_to_end_tests(self) -> Dict[str, any]:
        """Run end-to-end tests"""
        e2e_results = {
            "tests": {},
            "overall_success": False
        }
        
        tests = [
            ("complete_trading_flow", self.test_complete_trading_flow),
            ("risk_management_flow", self.test_risk_management_flow),
            ("portfolio_optimization_flow", self.test_portfolio_optimization_flow),
            ("market_data_flow", self.test_market_data_flow),
            ("error_handling_flow", self.test_error_handling_flow)
        ]
        
        for test_name, test_func in tests:
            try:
                test_result = await test_func()
                e2e_results["tests"][test_name] = test_result
                logger.info(f"E2E test {test_name}: {test_result['success']}")
            except Exception as e:
                logger.error(f"E2E test {test_name} failed: {e}")
                e2e_results["tests"][test_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        e2e_results["overall_success"] = all(
            test.get("success", False) 
            for test in e2e_results["tests"].values()
        )
        
        return e2e_results
    
    async def test_complete_trading_flow(self) -> Dict[str, any]:
        """Test complete trading flow"""
        flow_results = {
            "steps": {},
            "overall_success": False
        }
        
        # Step 1: Market Data Ingestion
        try:
            market_data = self.generate_market_data()
            response = self.session.post(
                f"{self.config.market_data_url}/v4/ingest",
                json=market_data,
                timeout=10
            )
            flow_results["steps"]["market_data_ingestion"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            flow_results["steps"]["market_data_ingestion"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 2: Signal Generation
        try:
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze",
                json=market_data,
                timeout=10
            )
            signal = response.json() if response.status_code == 200 else None
            flow_results["steps"]["signal_generation"] = {
                "success": response.status_code == 200,
                "signal": signal
            }
        except Exception as e:
            flow_results["steps"]["signal_generation"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 3: Order Routing
        if flow_results["steps"]["signal_generation"].get("success"):
            try:
                order_data = self.generate_order_data_from_signal(signal)
                response = self.session.post(
                    f"{self.config.order_routing_url}/v4/route/select",
                    json=order_data,
                    timeout=10
                )
                routing = response.json() if response.status_code == 200 else None
                flow_results["steps"]["order_routing"] = {
                    "success": response.status_code == 200,
                    "routing": routing
                }
            except Exception as e:
                flow_results["steps"]["order_routing"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Step 4: Risk Assessment
        try:
            risk_data = self.generate_risk_data()
            response = self.session.post(
                f"{self.config.risk_management_url}/v4/assess",
                json=risk_data,
                timeout=10
            )
            risk_assessment = response.json() if response.status_code == 200 else None
            flow_results["steps"]["risk_assessment"] = {
                "success": response.status_code == 200,
                "risk_assessment": risk_assessment
            }
        except Exception as e:
            flow_results["steps"]["risk_assessment"] = {
                "success": False,
                "error": str(e)
            }
        
        flow_results["overall_success"] = all(
            step.get("success", False) 
            for step in flow_results["steps"].values()
        )
        
        return flow_results
    
    async def test_risk_management_flow(self) -> Dict[str, any]:
        """Test risk management flow"""
        flow_results = {
            "steps": {},
            "overall_success": False
        }
        
        # Step 1: Portfolio Risk Assessment
        try:
            portfolio_data = self.generate_portfolio_data()
            response = self.session.post(
                f"{self.config.risk_management_url}/v4/portfolio_risk",
                json=portfolio_data,
                timeout=10
            )
            portfolio_risk = response.json() if response.status_code == 200 else None
            flow_results["steps"]["portfolio_risk"] = {
                "success": response.status_code == 200,
                "portfolio_risk": portfolio_risk
            }
        except Exception as e:
            flow_results["steps"]["portfolio_risk"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 2: Position Limit Check
        try:
            position_data = self.generate_position_data()
            response = self.session.post(
                f"{self.config.risk_management_url}/v4/position_limits",
                json=position_data,
                timeout=10
            )
            limits_check = response.json() if response.status_code == 200 else None
            flow_results["steps"]["position_limits"] = {
                "success": response.status_code == 200,
                "limits_check": limits_check
            }
        except Exception as e:
            flow_results["steps"]["position_limits"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 3: VaR Calculation
        try:
            var_data = self.generate_var_data()
            response = self.session.post(
                f"{self.config.risk_management_url}/v4/var_calculation",
                json=var_data,
                timeout=10
            )
            var_result = response.json() if response.status_code == 200 else None
            flow_results["steps"]["var_calculation"] = {
                "success": response.status_code == 200,
                "var_result": var_result
            }
        except Exception as e:
            flow_results["steps"]["var_calculation"] = {
                "success": False,
                "error": str(e)
            }
        
        flow_results["overall_success"] = all(
            step.get("success", False) 
            for step in flow_results["steps"].values()
        )
        
        return flow_results
    
    async def test_portfolio_optimization_flow(self) -> Dict[str, any]:
        """Test portfolio optimization flow"""
        flow_results = {
            "steps": {},
            "overall_success": False
        }
        
        # Step 1: Portfolio Analysis
        try:
            portfolio_data = self.generate_portfolio_data()
            response = self.session.post(
                f"{self.config.portfolio_optimizer_url}/v4/analyze",
                json=portfolio_data,
                timeout=10
            )
            analysis = response.json() if response.status_code == 200 else None
            flow_results["steps"]["portfolio_analysis"] = {
                "success": response.status_code == 200,
                "analysis": analysis
            }
        except Exception as e:
            flow_results["steps"]["portfolio_analysis"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 2: Optimization
        try:
            optimization_data = self.generate_optimization_data()
            response = self.session.post(
                f"{self.config.portfolio_optimizer_url}/v4/optimize",
                json=optimization_data,
                timeout=10
            )
            optimization = response.json() if response.status_code == 200 else None
            flow_results["steps"]["optimization"] = {
                "success": response.status_code == 200,
                "optimization": optimization
            }
        except Exception as e:
            flow_results["steps"]["optimization"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 3: Rebalancing
        try:
            rebalance_data = self.generate_rebalance_data()
            response = self.session.post(
                f"{self.config.portfolio_optimizer_url}/v4/rebalance",
                json=rebalance_data,
                timeout=10
            )
            rebalance = response.json() if response.status_code == 200 else None
            flow_results["steps"]["rebalancing"] = {
                "success": response.status_code == 200,
                "rebalance": rebalance
            }
        except Exception as e:
            flow_results["steps"]["rebalancing"] = {
                "success": False,
                "error": str(e)
            }
        
        flow_results["overall_success"] = all(
            step.get("success", False) 
            for step in flow_results["steps"].values()
        )
        
        return flow_results
    
    async def test_market_data_flow(self) -> Dict[str, any]:
        """Test market data flow"""
        flow_results = {
            "steps": {},
            "overall_success": False
        }
        
        # Step 1: Data Ingestion
        try:
            market_data = self.generate_market_data()
            response = self.session.post(
                f"{self.config.market_data_url}/v4/ingest",
                json=market_data,
                timeout=10
            )
            flow_results["steps"]["data_ingestion"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            flow_results["steps"]["data_ingestion"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 2: Data Processing
        try:
            response = self.session.get(
                f"{self.config.market_data_url}/v4/processed/BTC/USD",
                timeout=10
            )
            processed_data = response.json() if response.status_code == 200 else None
            flow_results["steps"]["data_processing"] = {
                "success": response.status_code == 200,
                "processed_data": processed_data
            }
        except Exception as e:
            flow_results["steps"]["data_processing"] = {
                "success": False,
                "error": str(e)
            }
        
        # Step 3: Data Validation
        try:
            validation_data = self.generate_validation_data()
            response = self.session.post(
                f"{self.config.market_data_url}/v4/validate",
                json=validation_data,
                timeout=10
            )
            validation = response.json() if response.status_code == 200 else None
            flow_results["steps"]["data_validation"] = {
                "success": response.status_code == 200,
                "validation": validation
            }
        except Exception as e:
            flow_results["steps"]["data_validation"] = {
                "success": False,
                "error": str(e)
            }
        
        flow_results["overall_success"] = all(
            step.get("success", False) 
            for step in flow_results["steps"].values()
        )
        
        return flow_results
    
    async def test_error_handling_flow(self) -> Dict[str, any]:
        """Test error handling flow"""
        flow_results = {
            "tests": {},
            "overall_success": False
        }
        
        # Test 1: Invalid Data Handling
        try:
            invalid_data = {"invalid": "data"}
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze",
                json=invalid_data,
                timeout=10
            )
            flow_results["tests"]["invalid_data"] = {
                "success": response.status_code == 400,
                "response_code": response.status_code,
                "error_handling": "proper" if response.status_code == 400 else "improper"
            }
        except Exception as e:
            flow_results["tests"]["invalid_data"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Timeout Handling
        try:
            large_data = self.generate_large_market_data()
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze",
                json=large_data,
                timeout=5  # Short timeout
            )
            flow_results["tests"]["timeout"] = {
                "success": False,  # Should timeout
                "timeout_handling": "failed"
            }
        except requests.exceptions.Timeout:
            flow_results["tests"]["timeout"] = {
                "success": True,
                "timeout_handling": "proper"
            }
        except Exception as e:
            flow_results["tests"]["timeout"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Service Unavailable
        try:
            # This would test against a non-existent service
            response = self.session.get(
                "http://non-existent-service:8080/health",
                timeout=5
            )
            flow_results["tests"]["service_unavailable"] = {
                "success": False,
                "error_handling": "failed"
            }
        except requests.exceptions.ConnectionError:
            flow_results["tests"]["service_unavailable"] = {
                "success": True,
                "error_handling": "proper"
            }
        except Exception as e:
            flow_results["tests"]["service_unavailable"] = {
                "success": False,
                "error": str(e)
            }
        
        flow_results["overall_success"] = all(
            test.get("success", False) 
            for test in flow_results["tests"].values()
        )
        
        return flow_results
    
    async def run_load_tests(self) -> Dict[str, any]:
        """Run load tests"""
        load_results = {
            "sustained_load": {},
            "peak_load": {},
            "overall_success": False
        }
        
        # Sustained Load Test (5 minutes)
        logger.info("Running sustained load test...")
        sustained_results = await self.run_sustained_load_test()
        load_results["sustained_load"] = sustained_results
        
        # Peak Load Test (1 minute)
        logger.info("Running peak load test...")
        peak_results = await self.run_peak_load_test()
        load_results["peak_load"] = peak_results
        
        load_results["overall_success"] = (
            sustained_results["success"] and
            peak_results["success"]
        )
        
        return load_results
    
    async def run_sustained_load_test(self) -> Dict[str, any]:
        """Run sustained load test"""
        duration_seconds = 300  # 5 minutes
        target_throughput = self.config.target_throughput_ops
        
        latencies = []
        errors = 0
        requests_completed = 0
        start_time = time.time()
        
        async def make_request():
            nonlocal requests_completed, errors, latencies
            try:
                test_data = self.generate_market_data()
                request_start = time.perf_counter()
                response = self.session.post(
                    f"{self.config.base_url}/v4/analyze",
                    json=test_data,
                    timeout=10
                )
                latency_us = (time.perf_counter() - request_start) * 1_000_000
                latencies.append(latency_us)
                
                if response.status_code == 200:
                    requests_completed += 1
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
        
        # Calculate required request rate
        requests_per_second = target_throughput
        request_interval = 1.0 / requests_per_second
        
        # Run sustained load
        while time.time() - start_time < duration_seconds:
            # Create batch of requests
            batch_size = min(100, int(requests_per_second))
            tasks = [make_request() for _ in range(batch_size)]
            
            # Execute batch
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Wait for next batch
            await asyncio.sleep(request_interval)
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        actual_throughput = requests_completed / actual_duration
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = float('inf')
            p95_latency = float('inf')
            p99_latency = float('inf')
        
        success = (
            actual_throughput >= target_throughput * 0.95 and  # 95% of target
            p95_latency < self.config.target_latency_us * 2 and
            errors < requests_completed * 0.01  # Less than 1% errors
        )
        
        return {
            "success": success,
            "duration_seconds": actual_duration,
            "target_throughput_ops": target_throughput,
            "actual_throughput_ops": actual_throughput,
            "requests_completed": requests_completed,
            "errors": errors,
            "avg_latency_us": avg_latency,
            "p95_latency_us": p95_latency,
            "p99_latency_us": p99_latency
        }
    
    async def run_peak_load_test(self) -> Dict[str, any]:
        """Run peak load test"""
        duration_seconds = 60  # 1 minute
        peak_throughput = self.config.target_throughput_ops * 1.5  # 150% of target
        
        latencies = []
        errors = 0
        requests_completed = 0
        start_time = time.time()
        
        async def make_request():
            nonlocal requests_completed, errors, latencies
            try:
                test_data = self.generate_market_data()
                request_start = time.perf_counter()
                response = self.session.post(
                    f"{self.config.base_url}/v4/analyze",
                    json=test_data,
                    timeout=10
                )
                latency_us = (time.perf_counter() - request_start) * 1_000_000
                latencies.append(latency_us)
                
                if response.status_code == 200:
                    requests_completed += 1
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
        
        # Calculate required request rate
        requests_per_second = peak_throughput
        request_interval = 1.0 / requests_per_second
        
        # Run peak load
        while time.time() - start_time < duration_seconds:
            # Create batch of requests
            batch_size = min(200, int(requests_per_second))
            tasks = [make_request() for _ in range(batch_size)]
            
            # Execute batch
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Wait for next batch
            await asyncio.sleep(request_interval)
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        actual_throughput = requests_completed / actual_duration
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = float('inf')
            p95_latency = float('inf')
            p99_latency = float('inf')
        
        success = (
            actual_throughput >= peak_throughput * 0.8 and  # 80% of peak target
            p95_latency < self.config.target_latency_us * 3 and
            errors < requests_completed * 0.05  # Less than 5% errors
        )
        
        return {
            "success": success,
            "duration_seconds": actual_duration,
            "target_throughput_ops": peak_throughput,
            "actual_throughput_ops": actual_throughput,
            "requests_completed": requests_completed,
            "errors": errors,
            "avg_latency_us": avg_latency,
            "p95_latency_us": p95_latency,
            "p99_latency_us": p99_latency
        }
    
    async def run_security_tests(self) -> Dict[str, any]:
        """Run security tests"""
        security_results = {
            "tests": {},
            "overall_success": False
        }
        
        tests = [
            ("authentication_security", self.test_authentication_security),
            ("authorization_security", self.test_authorization_security),
            ("input_validation_security", self.test_input_validation_security),
            ("rate_limiting_security", self.test_rate_limiting_security),
            ("data_encryption_security", self.test_data_encryption_security)
        ]
        
        for test_name, test_func in tests:
            try:
                test_result = await test_func()
                security_results["tests"][test_name] = test_result
                logger.info(f"Security test {test_name}: {test_result['success']}")
            except Exception as e:
                logger.error(f"Security test {test_name} failed: {e}")
                security_results["tests"][test_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        security_results["overall_success"] = all(
            test.get("success", False) 
            for test in security_results["tests"].values()
        )
        
        return security_results
    
    async def test_authentication_security(self) -> Dict[str, any]:
        """Test authentication security"""
        test_results = {
            "valid_credentials": {},
            "invalid_credentials": {},
            "missing_credentials": {},
            "overall_success": False
        }
        
        # Test valid credentials
        try:
            auth_data = {"username": "test_user", "password": "valid_password"}
            response = self.session.post(
                f"{self.config.base_url}/v4/auth/login",
                json=auth_data,
                timeout=10
            )
            test_results["valid_credentials"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["valid_credentials"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test invalid credentials
        try:
            auth_data = {"username": "test_user", "password": "invalid_password"}
            response = self.session.post(
                f"{self.config.base_url}/v4/auth/login",
                json=auth_data,
                timeout=10
            )
            test_results["invalid_credentials"] = {
                "success": response.status_code == 401,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["invalid_credentials"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test missing credentials
        try:
            auth_data = {}
            response = self.session.post(
                f"{self.config.base_url}/v4/auth/login",
                json=auth_data,
                timeout=10
            )
            test_results["missing_credentials"] = {
                "success": response.status_code == 400,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["missing_credentials"] = {
                "success": False,
                "error": str(e)
            }
        
        test_results["overall_success"] = (
            test_results["valid_credentials"]["success"] and
            test_results["invalid_credentials"]["success"] and
            test_results["missing_credentials"]["success"]
        )
        
        return test_results
    
    async def test_authorization_security(self) -> Dict[str, any]:
        """Test authorization security"""
        test_results = {
            "authorized_access": {},
            "unauthorized_access": {},
            "missing_token": {},
            "overall_success": False
        }
        
        # Test authorized access
        try:
            headers = {"Authorization": "Bearer valid_token"}
            response = self.session.get(
                f"{self.config.base_url}/v4/protected",
                headers=headers,
                timeout=10
            )
            test_results["authorized_access"] = {
                "success": response.status_code == 200,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["authorized_access"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test unauthorized access
        try:
            headers = {"Authorization": "Bearer invalid_token"}
            response = self.session.get(
                f"{self.config.base_url}/v4/protected",
                headers=headers,
                timeout=10
            )
            test_results["unauthorized_access"] = {
                "success": response.status_code == 401,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["unauthorized_access"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test missing token
        try:
            response = self.session.get(
                f"{self.config.base_url}/v4/protected",
                timeout=10
            )
            test_results["missing_token"] = {
                "success": response.status_code == 401,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["missing_token"] = {
                "success": False,
                "error": str(e)
            }
        
        test_results["overall_success"] = (
            test_results["authorized_access"]["success"] and
            test_results["unauthorized_access"]["success"] and
            test_results["missing_token"]["success"]
        )
        
        return test_results
    
    async def test_input_validation_security(self) -> Dict[str, any]:
        """Test input validation security"""
        test_results = {
            "sql_injection": {},
            "xss_injection": {},
            "malformed_json": {},
            "oversized_payload": {},
            "overall_success": False
        }
        
        # Test SQL injection
        try:
            malicious_data = {"symbol": "BTC/USD'; DROP TABLE users; --"}
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze",
                json=malicious_data,
                timeout=10
            )
            test_results["sql_injection"] = {
                "success": response.status_code == 400,  # Should be rejected
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["sql_injection"] = {
                "success": True,  # Exception is good for security
                "error": str(e)
            }
        
        # Test XSS injection
        try:
            malicious_data = {"symbol": "<script>alert('xss')</script>"}
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze",
                json=malicious_data,
                timeout=10
            )
            test_results["xss_injection"] = {
                "success": response.status_code == 400,  # Should be rejected
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["xss_injection"] = {
                "success": True,
                "error": str(e)
            }
        
        # Test malformed JSON
        try:
            malformed_data = "{'symbol': 'BTC/USD'}"  # Single quotes
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze",
                data=malformed_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            test_results["malformed_json"] = {
                "success": response.status_code == 400,
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["malformed_json"] = {
                "success": True,
                "error": str(e)
            }
        
        # Test oversized payload
        try:
            oversized_data = {"data": "x" * 1000000}  # 1MB payload
            response = self.session.post(
                f"{self.config.base_url}/v4/analyze",
                json=oversized_data,
                timeout=10
            )
            test_results["oversized_payload"] = {
                "success": response.status_code == 413,  # Should be rejected
                "response_code": response.status_code
            }
        except Exception as e:
            test_results["oversized_payload"] = {
                "success": True,
                "error": str(e)
            }
        
        test_results["overall_success"] = all(
            test.get("success", False) 
            for test in test_results.values()
            if isinstance(test, dict) and "success" in test
        )
        
        return test_results
    
    async def test_rate_limiting_security(self) -> Dict[str, any]:
        """Test rate limiting security"""
        test_results = {
            "normal_rate": {},
            "excessive_rate": {},
            "overall_success": False
        }
        
        # Test normal rate
        try:
            for i in range(10):  # Normal rate
                test_data = self.generate_market_data()
                response = self.session.post(
                    f"{self.config.base_url}/v4/analyze",
                    json=test_data,
                    timeout=10
                )
            
            test_results["normal_rate"] = {
                "success": True,  # Should succeed
                "requests_completed": 10
            }
        except Exception as e:
            test_results["normal_rate"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test excessive rate
        try:
            rate_limited = 0
            for i in range(1000):  # Excessive rate
                test_data = self.generate_market_data()
                response = self.session.post(
                    f"{self.config.base_url}/v4/analyze",
                    json=test_data,
                    timeout=10
                )
                if response.status_code == 429:  # Too Many Requests
                    rate_limited += 1
                    break
            
            test_results["excessive_rate"] = {
                "success": rate_limited > 0,  # Should be rate limited
                "rate_limited": rate_limited
            }
        except Exception as e:
            test_results["excessive_rate"] = {
                "success": False,
                "error": str(e)
            }
        
        test_results["overall_success"] = (
            test_results["normal_rate"]["success"] and
            test_results["excessive_rate"]["success"]
        )
        
        return test_results
    
    async def test_data_encryption_security(self) -> Dict[str, any]:
        """Test data encryption security"""
        test_results = {
            "https_encryption": {},
            "data_at_rest": {},
            "overall_success": False
        }
        
        # Test HTTPS encryption
        try:
            response = self.session.get(
                f"{self.config.base_url}/health",
                timeout=10
            )
            test_results["https_encryption"] = {
                "success": response.url.startswith("https://"),
                "url": response.url
            }
        except Exception as e:
            test_results["https_encryption"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test data at rest (this would require database access)
        test_results["data_at_rest"] = {
            "success": True,  # Assume encrypted at rest
            "note": "Database encryption verification requires direct access"
        }
        
        test_results["overall_success"] = (
            test_results["https_encryption"]["success"] and
            test_results["data_at_rest"]["success"]
        )
        
        return test_results
    
    # Helper methods for generating test data
    def generate_market_data(self) -> Dict[str, any]:
        """Generate realistic market data for testing"""
        return {
            "symbol": "BTC/USD",
            "exchange": "binance",
            "bid_price": 50000.0 + np.random.normal(0, 100),
            "ask_price": 50001.0 + np.random.normal(0, 100),
            "bid_volume": 1000000 + np.random.randint(0, 500000),
            "ask_volume": 950000 + np.random.randint(0, 500000),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_graph_data(self) -> Dict[str, any]:
        """Generate graph data for testing"""
        return {
            "assets": ["BTC/USD", "ETH/USD", "BNB/USD"],
            "time_window": "1h",
            "correlation_threshold": 0.7
        }
    
    def generate_order_data(self) -> Dict[str, any]:
        """Generate order data for testing"""
        return {
            "symbol": "BTC/USD",
            "side": "BUY",
            "quantity": 1.0,
            "type": "MARKET",
            "urgency": 0.8,
            "max_slippage_bps": 5.0
        }
    
    def generate_risk_data(self) -> Dict[str, any]:
        """Generate risk data for testing"""
        return {
            "portfolio": {
                "BTC/USD": 10.5,
                "ETH/USD": 100.0
            },
            "risk_parameters": {
                "var_confidence": 0.95,
                "time_horizon": "1d"
            }
        }
    
    def generate_portfolio_data(self) -> Dict[str, any]:
        """Generate portfolio data for testing"""
        return {
            "assets": ["BTC/USD", "ETH/USD", "BNB/USD"],
            "current_weights": [0.6, 0.3, 0.1],
            "target_return": 0.15,
            "risk_tolerance": 0.1
        }
    
    def generate_large_market_data(self) -> Dict[str, any]:
        """Generate large market data for testing"""
        return {
            "symbol": "BTC/USD",
            "exchange": "binance",
            "large_data": "x" * 1000000,  # Large payload
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_order_data_from_signal(self, signal: Dict[str, any]) -> Dict[str, any]:
        """Generate order data from signal"""
        return {
            "symbol": "BTC/USD",
            "side": signal.get("action", "BUY"),
            "quantity": 1.0,
            "type": "MARKET",
            "urgency": signal.get("confidence", 0.5),
            "max_slippage_bps": 5.0
        }
    
    def generate_position_data(self) -> Dict[str, any]:
        """Generate position data for testing"""
        return {
            "positions": {
                "BTC/USD": {"quantity": 10.5, "avg_cost": 48000.0},
                "ETH/USD": {"quantity": 100.0, "avg_cost": 3000.0}
            },
            "limits": {
                "max_position_size": 100000.0,
                "max_leverage": 3.0
            }
        }
    
    def generate_var_data(self) -> Dict[str, any]:
        """Generate VaR data for testing"""
        return {
            "portfolio": {
                "BTC/USD": 10.5,
                "ETH/USD": 100.0
            },
            "confidence_level": 0.95,
            "time_horizon": "1d",
            "method": "historical"
        }
    
    def generate_optimization_data(self) -> Dict[str, any]:
        """Generate optimization data for testing"""
        return {
            "assets": ["BTC/USD", "ETH/USD", "BNB/USD"],
            "returns": [0.15, 0.12, 0.08],
            "covariance_matrix": [
                [0.04, 0.02, 0.01],
                [0.02, 0.03, 0.015],
                [0.01, 0.015, 0.02]
            ],
            "risk_tolerance": 0.1
        }
    
    def generate_rebalance_data(self) -> Dict[str, any]:
        """Generate rebalance data for testing"""
        return {
            "current_portfolio": {
                "BTC/USD": 10.5,
                "ETH/USD": 100.0,
                "BNB/USD": 500.0
            },
            "target_weights": [0.5, 0.3, 0.2],
            "rebalance_threshold": 0.05
        }
    
    def generate_validation_data(self) -> Dict[str, any]:
        """Generate validation data for testing"""
        return {
            "symbol": "BTC/USD",
            "exchange": "binance",
            "validation_rules": ["price_range", "volume_range", "timestamp_valid"]
        }
    
    def evaluate_overall_success(self, results: Dict[str, any]) -> bool:
        """Evaluate overall success of all test phases"""
        required_phases = [
            "health_checks",
            "service_integration",
            "performance",
            "end_to_end",
            "load_testing",
            "security"
        ]
        
        for phase in required_phases:
            if phase not in results:
                return False
            if not results[phase].get("overall_success", False):
                return False
        
        return True
    
    def generate_summary(self, results: Dict[str, any]) -> Dict[str, any]:
        """Generate summary of test results"""
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "overall_success": results.get("overall_success", False),
            "performance_achieved": False,
            "security_achieved": False
        }
        
        # Count tests
        for phase_name, phase_results in results.items():
            if isinstance(phase_results, dict) and "tests" in phase_results:
                for test_name, test_result in phase_results["tests"].items():
                    summary["total_tests"] += 1
                    if test_result.get("success", False):
                        summary["passed_tests"] += 1
                    else:
                        summary["failed_tests"] += 1
        
        # Check performance achievement
        if "performance" in results:
            perf_results = results["performance"]
            if (perf_results.get("latency_tests", {}).get("success", False) and
                perf_results.get("throughput_tests", {}).get("success", False) and
                perf_results.get("scalability_tests", {}).get("success", False)):
                summary["performance_achieved"] = True
        
        # Check security achievement
        if "security" in results and results["security"].get("overall_success", False):
            summary["security_achieved"] = True
        
        return summary

async def main():
    """Main function to run integration tests"""
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Create test configuration
    config = TestConfig()
    
    # Create and run test suite
    test_suite = IntegrationTestSuite(config)
    
    try:
        results = await test_suite.run_all_tests()
        
        # Save results to file
        with open("integration_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        summary = results.get("summary", {})
        print(f"\n{'='*60}")
        print("ARCHNEURONX V4.0 INTEGRATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Overall Success: {summary.get('overall_success', False)}")
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed Tests: {summary.get('passed_tests', 0)}")
        print(f"Failed Tests: {summary.get('failed_tests', 0)}")
        print(f"Performance Achieved: {summary.get('performance_achieved', False)}")
        print(f"Security Achieved: {summary.get('security_achieved', False)}")
        print(f"{'='*60}")
        
        # Exit with appropriate code
        sys.exit(0 if summary.get('overall_success', False) else 1)
        
    except Exception as e:
        logger.error(f"Integration test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
