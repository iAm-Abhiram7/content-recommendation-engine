"""
Performance Evaluation and Load Testing
Comprehensive performance monitoring and testing infrastructure
"""

import asyncio
import time
import logging
import json
import psutil
import aiohttp
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import requests
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    active_connections: int
    timestamp: datetime
    status_code: int = 200
    endpoint: str = ""
    user_agent: str = ""

@dataclass
class LoadTestConfig:
    """Load testing configuration"""
    target_url: str
    concurrent_users: int = 100
    duration_seconds: int = 300
    ramp_up_seconds: int = 60
    request_rate_per_second: Optional[int] = None
    endpoints: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    payloads: Dict[str, Any] = field(default_factory=dict)

class SystemMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics_history = deque(maxlen=1000)
        self.is_monitoring = False
        self._monitor_task = None
    
    async def start_monitoring(self):
        """Start continuous system monitoring"""
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.sampling_interval)
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_mb': memory_mb,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_metrics_summary(self, window_seconds: int = 60) -> Dict[str, Any]:
        """Get summary of metrics over time window"""
        if not self.metrics_history:
            return {}
        
        # Filter metrics within time window
        current_time = time.time()
        window_metrics = [
            m for m in self.metrics_history
            if current_time - m.get('timestamp', 0) <= window_seconds
        ]
        
        if not window_metrics:
            return {}
        
        summary = {}
        for key in ['cpu_percent', 'memory_percent', 'disk_percent']:
            values = [m.get(key, 0) for m in window_metrics if key in m]
            if values:
                summary[key] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary

class LatencyTracker:
    """Track response time latencies"""
    
    def __init__(self, max_samples: int = 10000):
        self.latencies = deque(maxlen=max_samples)
        self.lock = threading.Lock()
    
    def record_latency(self, latency_ms: float, endpoint: str = ""):
        """Record a latency measurement"""
        with self.lock:
            self.latencies.append({
                'latency_ms': latency_ms,
                'endpoint': endpoint,
                'timestamp': time.time()
            })
    
    def get_latency_stats(self, window_seconds: Optional[int] = None) -> Dict[str, float]:
        """Get latency statistics"""
        with self.lock:
            if not self.latencies:
                return {}
            
            # Filter by time window if specified
            current_time = time.time()
            if window_seconds:
                latencies = [
                    l['latency_ms'] for l in self.latencies
                    if current_time - l['timestamp'] <= window_seconds
                ]
            else:
                latencies = [l['latency_ms'] for l in self.latencies]
            
            if not latencies:
                return {}
            
            return {
                'min': min(latencies),
                'max': max(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'p99_9': np.percentile(latencies, 99.9),
                'count': len(latencies)
            }

class ThroughputMonitor:
    """Monitor request throughput"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.requests = deque()
        self.lock = threading.Lock()
    
    def record_request(self, endpoint: str = "", status_code: int = 200):
        """Record a completed request"""
        with self.lock:
            current_time = time.time()
            self.requests.append({
                'timestamp': current_time,
                'endpoint': endpoint,
                'status_code': status_code
            })
            
            # Remove old requests outside window
            cutoff_time = current_time - self.window_size
            while self.requests and self.requests[0]['timestamp'] < cutoff_time:
                self.requests.popleft()
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics"""
        with self.lock:
            current_time = time.time()
            
            # Count requests in different time windows
            counts_1s = sum(1 for r in self.requests if current_time - r['timestamp'] <= 1)
            counts_5s = sum(1 for r in self.requests if current_time - r['timestamp'] <= 5)
            counts_60s = len(self.requests)
            
            # Count by status code
            success_count = sum(1 for r in self.requests if 200 <= r['status_code'] < 300)
            error_count = len(self.requests) - success_count
            
            return {
                'rps_1s': counts_1s,
                'rps_5s': counts_5s / 5,
                'rps_60s': counts_60s / min(self.window_size, 60),
                'total_requests': len(self.requests),
                'success_rate': success_count / len(self.requests) if self.requests else 0,
                'error_rate': error_count / len(self.requests) if self.requests else 0
            }

class LoadTester:
    """Comprehensive load testing framework"""
    
    def __init__(self):
        self.latency_tracker = LatencyTracker()
        self.throughput_monitor = ThroughputMonitor()
        self.system_monitor = SystemMonitor()
        self.is_running = False
        self.test_results = []
    
    async def run_load_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Run comprehensive load test"""
        logger.info(f"Starting load test: {config.concurrent_users} users for {config.duration_seconds}s")
        
        # Start monitoring
        await self.system_monitor.start_monitoring()
        
        # Initialize test state
        self.is_running = True
        start_time = time.time()
        end_time = start_time + config.duration_seconds
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(config.concurrent_users)
        
        try:
            # Run the test
            await self._execute_load_test(config, semaphore, start_time, end_time)
            
            # Collect final results
            test_duration = time.time() - start_time
            results = await self._compile_results(test_duration, config)
            
            return results
            
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            return {"error": str(e)}
        
        finally:
            self.is_running = False
            await self.system_monitor.stop_monitoring()
    
    async def _execute_load_test(self, config: LoadTestConfig, semaphore: asyncio.Semaphore,
                                start_time: float, end_time: float):
        """Execute the actual load test"""
        tasks = []
        
        # Calculate ramp-up strategy
        ramp_up_delay = config.ramp_up_seconds / config.concurrent_users if config.ramp_up_seconds > 0 else 0
        
        # Start user sessions
        for user_id in range(config.concurrent_users):
            # Stagger user starts during ramp-up period
            delay = user_id * ramp_up_delay
            task = asyncio.create_task(
                self._user_session(config, semaphore, user_id, start_time + delay, end_time)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _user_session(self, config: LoadTestConfig, semaphore: asyncio.Semaphore,
                           user_id: int, start_time: float, end_time: float):
        """Simulate individual user session"""
        async with semaphore:
            # Wait for start time
            await asyncio.sleep(max(0, start_time - time.time()))
            
            session_connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=session_connector, timeout=timeout) as session:
                request_count = 0
                
                while time.time() < end_time and self.is_running:
                    try:
                        # Select endpoint
                        endpoint = self._select_endpoint(config.endpoints)
                        url = f"{config.target_url.rstrip('/')}/{endpoint.lstrip('/')}"
                        
                        # Make request
                        request_start = time.time()
                        
                        async with session.get(url, headers=config.headers) as response:
                            await response.read()  # Consume response body
                            
                            request_end = time.time()
                            latency_ms = (request_end - request_start) * 1000
                            
                            # Record metrics
                            self.latency_tracker.record_latency(latency_ms, endpoint)
                            self.throughput_monitor.record_request(endpoint, response.status)
                            
                            request_count += 1
                            
                            # Rate limiting
                            if config.request_rate_per_second:
                                await asyncio.sleep(1.0 / config.request_rate_per_second)
                            else:
                                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                    
                    except Exception as e:
                        logger.warning(f"Request failed for user {user_id}: {e}")
                        self.throughput_monitor.record_request(endpoint, 500)
                        await asyncio.sleep(0.1)  # Brief pause on error
                
                logger.debug(f"User {user_id} completed {request_count} requests")
    
    def _select_endpoint(self, endpoints: List[str]) -> str:
        """Select endpoint for request (round-robin or weighted)"""
        if not endpoints:
            return ""
        
        # Simple round-robin selection
        return endpoints[int(time.time() * 1000) % len(endpoints)]
    
    async def _compile_results(self, test_duration: float, config: LoadTestConfig) -> Dict[str, Any]:
        """Compile comprehensive test results"""
        latency_stats = self.latency_tracker.get_latency_stats()
        throughput_stats = self.throughput_monitor.get_throughput_stats()
        system_stats = self.system_monitor.get_metrics_summary()
        
        # Performance assessment
        performance_grade = self._assess_performance(latency_stats, throughput_stats)
        
        results = {
            'test_config': {
                'concurrent_users': config.concurrent_users,
                'duration_seconds': config.duration_seconds,
                'target_url': config.target_url
            },
            'test_results': {
                'actual_duration': test_duration,
                'total_requests': throughput_stats.get('total_requests', 0),
                'requests_per_second': throughput_stats.get('rps_60s', 0),
                'success_rate': throughput_stats.get('success_rate', 0),
                'error_rate': throughput_stats.get('error_rate', 0)
            },
            'latency_metrics': latency_stats,
            'throughput_metrics': throughput_stats,
            'system_metrics': system_stats,
            'performance_assessment': performance_grade,
            'recommendations': self._generate_recommendations(latency_stats, throughput_stats, system_stats),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _assess_performance(self, latency_stats: Dict[str, float], 
                          throughput_stats: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall performance and assign grades"""
        assessment = {
            'overall_grade': 'Unknown',
            'latency_grade': 'Unknown',
            'throughput_grade': 'Unknown',
            'reliability_grade': 'Unknown',
            'scores': {}
        }
        
        if not latency_stats or not throughput_stats:
            return assessment
        
        # Latency assessment (target: <100ms p95)
        p95_latency = latency_stats.get('p95', float('inf'))
        if p95_latency < 50:
            latency_grade = 'A'
            latency_score = 100
        elif p95_latency < 100:
            latency_grade = 'B'
            latency_score = 85
        elif p95_latency < 200:
            latency_grade = 'C'
            latency_score = 70
        elif p95_latency < 500:
            latency_grade = 'D'
            latency_score = 50
        else:
            latency_grade = 'F'
            latency_score = 25
        
        # Throughput assessment
        rps = throughput_stats.get('rps_60s', 0)
        if rps > 1000:
            throughput_grade = 'A'
            throughput_score = 100
        elif rps > 500:
            throughput_grade = 'B'
            throughput_score = 85
        elif rps > 100:
            throughput_grade = 'C'
            throughput_score = 70
        elif rps > 50:
            throughput_grade = 'D'
            throughput_score = 50
        else:
            throughput_grade = 'F'
            throughput_score = 25
        
        # Reliability assessment
        success_rate = throughput_stats.get('success_rate', 0)
        if success_rate > 0.99:
            reliability_grade = 'A'
            reliability_score = 100
        elif success_rate > 0.95:
            reliability_grade = 'B'
            reliability_score = 85
        elif success_rate > 0.90:
            reliability_grade = 'C'
            reliability_score = 70
        elif success_rate > 0.80:
            reliability_grade = 'D'
            reliability_score = 50
        else:
            reliability_grade = 'F'
            reliability_score = 25
        
        # Overall assessment (weighted average)
        overall_score = (latency_score * 0.4 + throughput_score * 0.3 + reliability_score * 0.3)
        
        if overall_score >= 90:
            overall_grade = 'A'
        elif overall_score >= 80:
            overall_grade = 'B'
        elif overall_score >= 70:
            overall_grade = 'C'
        elif overall_score >= 60:
            overall_grade = 'D'
        else:
            overall_grade = 'F'
        
        assessment.update({
            'overall_grade': overall_grade,
            'latency_grade': latency_grade,
            'throughput_grade': throughput_grade,
            'reliability_grade': reliability_grade,
            'scores': {
                'overall': overall_score,
                'latency': latency_score,
                'throughput': throughput_score,
                'reliability': reliability_score
            }
        })
        
        return assessment
    
    def _generate_recommendations(self, latency_stats: Dict[str, float],
                                throughput_stats: Dict[str, float],
                                system_stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not latency_stats or not throughput_stats:
            return ["Insufficient data for recommendations"]
        
        # Latency recommendations
        p95_latency = latency_stats.get('p95', 0)
        if p95_latency > 100:
            recommendations.append(f"High P95 latency ({p95_latency:.1f}ms). Consider caching, database optimization, or CDN.")
        
        if p95_latency > 500:
            recommendations.append("Critical latency issues detected. Immediate optimization required.")
        
        # Throughput recommendations
        rps = throughput_stats.get('rps_60s', 0)
        if rps < 100:
            recommendations.append(f"Low throughput ({rps:.1f} RPS). Consider horizontal scaling or connection pooling.")
        
        # Error rate recommendations
        error_rate = throughput_stats.get('error_rate', 0)
        if error_rate > 0.01:
            recommendations.append(f"High error rate ({error_rate:.1%}). Check application logs and error handling.")
        
        if error_rate > 0.05:
            recommendations.append("Critical error rate detected. Immediate investigation required.")
        
        # System resource recommendations
        if 'cpu_percent' in system_stats:
            cpu_avg = system_stats['cpu_percent'].get('avg', 0)
            if cpu_avg > 80:
                recommendations.append(f"High CPU usage ({cpu_avg:.1f}%). Consider CPU optimization or scaling.")
        
        if 'memory_percent' in system_stats:
            memory_avg = system_stats['memory_percent'].get('avg', 0)
            if memory_avg > 80:
                recommendations.append(f"High memory usage ({memory_avg:.1f}%). Check for memory leaks or increase memory allocation.")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable limits. Monitor trends for potential issues.")
        
        return recommendations

# Convenience function for quick load testing
async def quick_load_test(target_url: str, concurrent_users: int = 50, 
                         duration_seconds: int = 60) -> Dict[str, Any]:
    """Quick load test with default settings"""
    config = LoadTestConfig(
        target_url=target_url,
        concurrent_users=concurrent_users,
        duration_seconds=duration_seconds,
        endpoints=["/health", "/api/v1/recommendations", "/api/v1/info"]
    )
    
    load_tester = LoadTester()
    return await load_tester.run_load_test(config)
