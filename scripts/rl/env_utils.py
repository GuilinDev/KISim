#!/usr/bin/env python3
"""
Utility functions for Kubernetes RL Environment
Contains metric collection and helper functions.
"""

import logging
import requests
import time
from typing import Dict, Tuple
from prometheus_api_client import PrometheusConnect
from kubernetes import client

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Utility class for collecting metrics from various sources"""
    
    def __init__(self, prometheus_url: str, locust_url: str, namespace: str):
        self.prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        self.locust_url = locust_url
        self.namespace = namespace
        self.k8s_apps = client.AppsV1Api()
        
    def get_cpu_utilization(self) -> float:
        """Get current CPU utilization from Prometheus"""
        try:
            query = 'avg(100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))'
            result = self.prometheus.custom_query(query)
            if result:
                return float(result[0]['value'][1])
        except Exception as e:
            logger.error(f"Error getting CPU utilization: {e}")
        return 50.0  # Default value
        
    def get_memory_utilization(self) -> float:
        """Get current memory utilization from Prometheus"""
        try:
            query = '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
            result = self.prometheus.custom_query(query)
            if result:
                return float(result[0]['value'][1])
        except Exception as e:
            logger.error(f"Error getting memory utilization: {e}")
        return 50.0  # Default value
        
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization from DCGM metrics"""
        try:
            query = 'DCGM_FI_DEV_GPU_UTIL'
            result = self.prometheus.custom_query(query)
            if result:
                return float(result[0]['value'][1])
        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")
        return 30.0  # Default value based on baseline (43% during inference)
        
    def get_current_latency_p95(self) -> float:
        """Get current P95 latency from Locust metrics"""
        try:
            # Try to get from Locust API
            response = requests.get(f"{self.locust_url}/stats/requests", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                if stats.get('stats'):
                    for stat in stats['stats']:
                        if stat.get('name') == 'Aggregated':
                            return stat.get('current_response_time_percentile_95', 1000)
        except Exception as e:
            logger.error(f"Error getting latency: {e}")
        return 1000.0  # Default value
        
    def get_current_throughput(self) -> float:
        """Get current throughput from Locust metrics"""
        try:
            response = requests.get(f"{self.locust_url}/stats/requests", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                if stats.get('stats'):
                    for stat in stats['stats']:
                        if stat.get('name') == 'Aggregated':
                            return stat.get('current_rps', 5.0)
        except Exception as e:
            logger.error(f"Error getting throughput: {e}")
        return 5.0  # Default value
        
    def get_current_replicas(self) -> Tuple[int, int]:
        """Get current replica counts for GPU and CPU deployments"""
        try:
            gpu_deployment = self.k8s_apps.read_namespaced_deployment(
                name="mobilenetv4-triton-deployment", 
                namespace=self.namespace
            )
            cpu_deployment = self.k8s_apps.read_namespaced_deployment(
                name="mobilenetv4-triton-cpu-deployment", 
                namespace=self.namespace
            )
            return gpu_deployment.spec.replicas, cpu_deployment.spec.replicas
        except Exception as e:
            logger.error(f"Error getting replica counts: {e}")
            return 3, 3  # Default values
            
    def get_pod_resource_usage(self) -> Dict:
        """Get detailed resource usage for all pods"""
        try:
            # Get CPU usage per pod
            cpu_query = 'sum(rate(container_cpu_usage_seconds_total{namespace="workloads"}[5m])) by (pod)'
            cpu_result = self.prometheus.custom_query(cpu_query)
            
            # Get memory usage per pod
            memory_query = 'sum(container_memory_working_set_bytes{namespace="workloads"}) by (pod)'
            memory_result = self.prometheus.custom_query(memory_query)
            
            pod_usage = {}
            
            # Process CPU results
            for item in cpu_result:
                pod_name = item['metric']['pod']
                pod_usage[pod_name] = {'cpu': float(item['value'][1])}
                
            # Process memory results
            for item in memory_result:
                pod_name = item['metric']['pod']
                if pod_name in pod_usage:
                    pod_usage[pod_name]['memory'] = float(item['value'][1]) / (1024**3)  # Convert to GB
                else:
                    pod_usage[pod_name] = {'memory': float(item['value'][1]) / (1024**3)}
                    
            return pod_usage
            
        except Exception as e:
            logger.error(f"Error getting pod resource usage: {e}")
            return {}

class LoadPatternGenerator:
    """Generate different load patterns for RL training"""
    
    @staticmethod
    def generate_ramp_pattern(duration: int = 300, min_users: int = 10, max_users: int = 100) -> Dict:
        """Generate ramp load pattern configuration"""
        return {
            "pattern_type": "ramp",
            "duration": duration,
            "min_users": min_users,
            "max_users": max_users,
            "description": f"Linear increase from {min_users} to {max_users} users over {duration}s"
        }
        
    @staticmethod
    def generate_spike_pattern(duration: int = 100, base_users: int = 10, spike_users: int = 100) -> Dict:
        """Generate spike load pattern configuration"""
        return {
            "pattern_type": "spike",
            "duration": duration,
            "base_users": base_users,
            "spike_users": spike_users,
            "description": f"Spike from {base_users} to {spike_users} users with rapid transitions"
        }
        
    @staticmethod
    def generate_periodic_pattern(duration: int = 300, min_users: int = 10, max_users: int = 100, cycles: int = 3) -> Dict:
        """Generate periodic load pattern configuration"""
        return {
            "pattern_type": "periodic",
            "duration": duration,
            "min_users": min_users,
            "max_users": max_users,
            "cycles": cycles,
            "description": f"Sinusoidal pattern with {cycles} cycles between {min_users}-{max_users} users"
        }
        
    @staticmethod
    def generate_random_pattern(duration: int = 50, min_users: int = 10, max_users: int = 100) -> Dict:
        """Generate random load pattern configuration"""
        return {
            "pattern_type": "random",
            "duration": duration,
            "min_users": min_users,
            "max_users": max_users,
            "description": f"Random variations between {min_users}-{max_users} users over {duration}s"
        }

class RewardCalculator:
    """Advanced reward calculation utilities"""
    
    def __init__(self, baseline_metrics: Dict):
        self.baseline_metrics = baseline_metrics
        
    def calculate_latency_score(self, current_latency: float, pattern: str) -> float:
        """Calculate latency improvement score"""
        baseline_gpu = self.baseline_metrics["gpu_p95_latency"].get(pattern, 3000)
        baseline_cpu = self.baseline_metrics["cpu_p95_latency"].get(pattern, 4000)
        
        # Use better baseline for comparison
        if pattern in ["random", "spike"]:
            baseline = baseline_gpu  # GPU performs better
        else:
            baseline = min(baseline_gpu, baseline_cpu)  # Use best baseline
            
        if baseline > 0:
            improvement = max(0, (baseline - current_latency) / baseline)
            return min(1.0, improvement)  # Cap at 1.0
        return 0.0
        
    def calculate_throughput_score(self, current_throughput: float, pattern: str) -> float:
        """Calculate throughput improvement score"""
        baseline_gpu = self.baseline_metrics["gpu_throughput"].get(pattern, 10)
        baseline_cpu = self.baseline_metrics["cpu_throughput"].get(pattern, 10)
        
        # Use better baseline for comparison
        if pattern in ["random", "spike"]:
            baseline = baseline_gpu
        else:
            baseline = max(baseline_gpu, baseline_cpu)  # Use best baseline
            
        if baseline > 0:
            improvement = max(0, (current_throughput - baseline) / baseline)
            return min(1.0, improvement)  # Cap at 1.0
        return 0.0
        
    def calculate_efficiency_score(self, gpu_replicas: int, cpu_replicas: int, pattern: str) -> float:
        """Calculate resource efficiency score based on pattern"""
        total_replicas = gpu_replicas + cpu_replicas
        if total_replicas == 0:
            return 0.0
            
        gpu_ratio = gpu_replicas / total_replicas
        
        # Optimal ratios based on baseline findings
        if pattern == "random":
            # GPU shows 1.96x speedup - prefer GPU heavily
            optimal_gpu_ratio = 0.8
        elif pattern == "spike":
            # GPU shows 1.27x speedup - prefer GPU moderately
            optimal_gpu_ratio = 0.7
        elif pattern == "ramp":
            # GPU shows 1.16x speedup - slight GPU preference
            optimal_gpu_ratio = 0.6
        else:  # periodic
            # Similar performance - balanced approach
            optimal_gpu_ratio = 0.5
            
        # Calculate score based on distance from optimal ratio
        distance = abs(gpu_ratio - optimal_gpu_ratio)
        score = max(0, 1.0 - distance * 2)  # Penalty for deviation
        
        return score

def create_environment_config(
    namespace: str = "workloads",
    prometheus_url: str = "http://localhost:9090",
    locust_url: str = "http://localhost:8089",
    episode_duration: int = 300
) -> Dict:
    """Create environment configuration with sensible defaults"""
    
    return {
        "namespace": namespace,
        "prometheus_url": prometheus_url,
        "locust_url": locust_url,
        "max_replicas": 10,
        "min_replicas": 1,
        "observation_window": 60,
        "action_interval": 30,
        "episode_duration": episode_duration,
        "latency_weight": 0.4,
        "throughput_weight": 0.3,
        "resource_efficiency_weight": 0.2,
        "stability_weight": 0.1
    }
