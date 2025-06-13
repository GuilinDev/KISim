#!/usr/bin/env python3
"""
Basic tests for KISim components
"""

import unittest
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

class TestKISimBasics(unittest.TestCase):
    """Basic functionality tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = Path(__file__).parent.parent / "examples"
        self.sample_results_file = self.test_data_dir / "sample_results.json"
    
    def test_sample_results_format(self):
        """Test that sample results file has correct format"""
        self.assertTrue(self.sample_results_file.exists(), 
                       "Sample results file should exist")
        
        with open(self.sample_results_file, 'r') as f:
            data = json.load(f)
        
        # Check required top-level keys
        required_keys = [
            'experiment_info', 'workload_config', 'load_pattern',
            'performance_metrics', 'scaling_events', 'time_series', 'summary'
        ]
        
        for key in required_keys:
            self.assertIn(key, data, f"Required key '{key}' missing from results")
    
    def test_performance_metrics_structure(self):
        """Test performance metrics have correct structure"""
        with open(self.sample_results_file, 'r') as f:
            data = json.load(f)
        
        metrics = data['performance_metrics']
        
        # Check response time metrics
        response_time = metrics['response_time']
        required_rt_keys = ['avg_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'max_ms', 'min_ms']
        for key in required_rt_keys:
            self.assertIn(key, response_time, f"Response time key '{key}' missing")
            self.assertIsInstance(response_time[key], (int, float), 
                                f"Response time '{key}' should be numeric")
        
        # Check throughput metrics
        throughput = metrics['throughput']
        required_tp_keys = ['requests_per_second', 'total_requests', 'success_rate_percent']
        for key in required_tp_keys:
            self.assertIn(key, throughput, f"Throughput key '{key}' missing")
            self.assertIsInstance(throughput[key], (int, float), 
                                f"Throughput '{key}' should be numeric")
    
    def test_time_series_consistency(self):
        """Test that time series data is consistent"""
        with open(self.sample_results_file, 'r') as f:
            data = json.load(f)
        
        time_series = data['time_series']
        
        # All time series should have same length
        lengths = [
            len(time_series['timestamps']),
            len(time_series['user_counts']),
            len(time_series['response_times_p95']),
            len(time_series['requests_per_second']),
            len(time_series['cpu_utilization']),
            len(time_series['gpu_utilization'])
        ]
        
        self.assertEqual(len(set(lengths)), 1, 
                        "All time series should have same length")
    
    def test_scaling_events_format(self):
        """Test scaling events have correct format"""
        with open(self.sample_results_file, 'r') as f:
            data = json.load(f)
        
        scaling_events = data['scaling_events']
        self.assertIsInstance(scaling_events, list, "Scaling events should be a list")
        
        if scaling_events:  # If there are scaling events
            event = scaling_events[0]
            required_keys = ['timestamp', 'action', 'from_replicas', 'to_replicas', 'trigger']
            for key in required_keys:
                self.assertIn(key, event, f"Scaling event key '{key}' missing")
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        config_dir = Path(__file__).parent.parent / "configs"
        
        expected_configs = [
            "rl_production_config.json",
            "rl_test_config.json"
        ]
        
        for config_file in expected_configs:
            config_path = config_dir / config_file
            self.assertTrue(config_path.exists(), 
                           f"Configuration file {config_file} should exist")
            
            # Test that it's valid JSON
            with open(config_path, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError:
                    self.fail(f"Configuration file {config_file} is not valid JSON")

class TestKISimScripts(unittest.TestCase):
    """Test script functionality"""
    
    def test_scripts_directory_structure(self):
        """Test that scripts directory has expected structure"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        # Check for key script files
        expected_scripts = [
            "download_hf_onnx_model.py",
            "dynamic_load_controller.py",
            "generate_comprehensive_comparison.py"
        ]
        
        for script in expected_scripts:
            script_path = scripts_dir / script
            self.assertTrue(script_path.exists(), 
                           f"Script {script} should exist")
            self.assertTrue(script_path.is_file(), 
                           f"Script {script} should be a file")
    
    def test_rl_module_structure(self):
        """Test RL module has correct structure"""
        rl_dir = Path(__file__).parent.parent / "scripts" / "rl"
        
        if rl_dir.exists():
            expected_files = [
                "__init__.py",
                "kubernetes_env.py",
                "ppo_agent.py",
                "train_rl_agent.py"
            ]
            
            for file_name in expected_files:
                file_path = rl_dir / file_name
                self.assertTrue(file_path.exists(), 
                               f"RL module file {file_name} should exist")

class TestKISimKubernetes(unittest.TestCase):
    """Test Kubernetes manifests"""
    
    def test_kubernetes_manifests_exist(self):
        """Test that Kubernetes manifest files exist"""
        k8s_dir = Path(__file__).parent.parent / "kubernetes"
        
        if k8s_dir.exists():
            # Look for YAML files
            yaml_files = list(k8s_dir.glob("*.yaml"))
            self.assertGreater(len(yaml_files), 0, 
                              "Should have at least one Kubernetes manifest")
            
            # Check for key manifests
            expected_manifests = [
                "mobilenetv4-triton-deployment.yaml",
                "locust-deployment.yaml"
            ]
            
            existing_files = [f.name for f in yaml_files]
            for manifest in expected_manifests:
                if manifest in existing_files:
                    # Basic YAML syntax check could be added here
                    pass

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestKISimBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestKISimScripts))
    suite.addTests(loader.loadTestsFromTestCase(TestKISimKubernetes))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
