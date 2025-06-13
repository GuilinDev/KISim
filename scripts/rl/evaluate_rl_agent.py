#!/usr/bin/env python3
"""
Evaluation script for trained RL agent
Compares RL performance against baseline GPU and CPU results.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from kubernetes_env import KubernetesRLEnvironment, EnvironmentConfig
from ppo_agent import PPOAgent
from env_utils import create_environment_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLEvaluator:
    """
    RL Agent Evaluator
    
    Compares trained RL agent performance against baseline results:
    - GPU baseline performance
    - CPU baseline performance
    - Load pattern specific analysis
    """
    
    def __init__(self, model_path: str, config: Dict):
        self.model_path = model_path
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        self.results_dir = f"/home/guilin/allProjects/ecrl/experiments/rl/evaluation_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize environment
        env_config = EnvironmentConfig(**config['environment'])
        self.env = KubernetesRLEnvironment(env_config)
        
        # Load trained agent
        device = "cuda" if torch.cuda.is_available() and config.get('use_gpu', True) else "cpu"
        self.agent = PPOAgent(config=None, device=device)  # Config will be loaded from model
        self.agent.load_model(model_path)
        self.agent.set_eval_mode()
        
        # Load baseline metrics for comparison
        self.baseline_metrics = self._load_baseline_metrics()
        
        logger.info(f"Initialized RL evaluator with model: {model_path}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        
    def _load_baseline_metrics(self) -> Dict:
        """Load baseline performance metrics from experimental results"""
        
        # Default baseline metrics from our experiments
        baseline = {
            "gpu_p95_latency": {
                "random": 2600,
                "spike": 370,
                "ramp": 5800,
                "periodic": 2300
            },
            "cpu_p95_latency": {
                "random": 5100,
                "spike": 470,
                "ramp": 6700,
                "periodic": 2300
            },
            "gpu_throughput": {
                "random": 10.40,
                "spike": 6.40,
                "ramp": 10.30,
                "periodic": 11.50
            },
            "cpu_throughput": {
                "random": 10.80,
                "spike": 5.70,
                "ramp": 10.20,
                "periodic": 12.10
            }
        }
        
        # Try to load actual baseline data if available
        baseline_file = "/home/guilin/allProjects/ecrl/experiments/results/comparison/comprehensive_comparison_report.txt"
        try:
            if os.path.exists(baseline_file):
                logger.info("Loaded baseline metrics from experimental results")
        except Exception as e:
            logger.warning(f"Could not load baseline file: {e}")
            
        return baseline
        
    def evaluate_comprehensive(self, num_runs: int = 5) -> Dict:
        """
        Comprehensive evaluation across all load patterns
        
        Args:
            num_runs: Number of evaluation runs per pattern
            
        Returns:
            evaluation_results: Comprehensive evaluation results
        """
        
        logger.info(f"Starting comprehensive evaluation with {num_runs} runs per pattern")
        
        load_patterns = ['ramp', 'spike', 'periodic', 'random']
        results = {}
        
        for pattern in load_patterns:
            logger.info(f"Evaluating pattern: {pattern}")
            
            pattern_results = []
            
            for run in range(num_runs):
                # Force specific load pattern
                self.env.current_load_pattern = self.env.LoadPattern(pattern)
                
                # Run evaluation episode
                episode_result = self._run_evaluation_episode(pattern, run)
                pattern_results.append(episode_result)
                
                logger.info(f"  Run {run + 1}/{num_runs}: "
                           f"Avg Latency={episode_result['avg_latency']:.1f}ms, "
                           f"Avg Throughput={episode_result['avg_throughput']:.1f} req/s, "
                           f"Total Reward={episode_result['total_reward']:.3f}")
                           
            results[pattern] = pattern_results
            
        # Calculate aggregate statistics
        aggregate_results = self._calculate_aggregate_stats(results)
        
        # Compare with baselines
        comparison_results = self._compare_with_baselines(aggregate_results)
        
        # Save results
        evaluation_data = {
            'timestamp': self.timestamp,
            'model_path': self.model_path,
            'num_runs': num_runs,
            'pattern_results': results,
            'aggregate_results': aggregate_results,
            'baseline_comparison': comparison_results
        }
        
        results_file = os.path.join(self.results_dir, "comprehensive_evaluation.json")
        with open(results_file, 'w') as f:
            json.dump(evaluation_data, f, indent=2, default=str)
            
        # Generate visualization
        self._generate_evaluation_plots(evaluation_data)
        
        logger.info("Comprehensive evaluation completed")
        return evaluation_data
        
    def _run_evaluation_episode(self, pattern: str, run: int) -> Dict:
        """Run a single evaluation episode"""
        
        # Reset environment
        observation = self.env.reset()
        
        episode_data = {
            'pattern': pattern,
            'run': run,
            'steps': [],
            'total_reward': 0,
            'episode_length': 0
        }
        
        done = False
        step = 0
        
        while not done:
            # Get action from trained agent
            action, _, _ = self.agent.get_action(observation, deterministic=True)
            
            # Take step
            next_observation, reward, done, info = self.env.step(action)
            
            # Record step data
            step_data = {
                'step': step,
                'action': action.tolist(),
                'reward': reward,
                'performance_metrics': info.get('performance_metrics', {}),
                'action_info': info.get('action_info', {})
            }
            
            episode_data['steps'].append(step_data)
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1
            
            observation = next_observation
            step += 1
            
        # Get episode summary
        episode_summary = self.env.get_episode_summary()
        episode_data.update(episode_summary)
        
        return episode_data
        
    def _calculate_aggregate_stats(self, results: Dict) -> Dict:
        """Calculate aggregate statistics across all runs"""
        
        aggregate = {}
        
        for pattern, pattern_results in results.items():
            # Extract metrics
            latencies = [r['avg_latency'] for r in pattern_results]
            throughputs = [r['avg_throughput'] for r in pattern_results]
            rewards = [r['total_reward'] for r in pattern_results]
            
            aggregate[pattern] = {
                'avg_latency_mean': np.mean(latencies),
                'avg_latency_std': np.std(latencies),
                'avg_latency_min': np.min(latencies),
                'avg_latency_max': np.max(latencies),
                'avg_throughput_mean': np.mean(throughputs),
                'avg_throughput_std': np.std(throughputs),
                'avg_throughput_min': np.min(throughputs),
                'avg_throughput_max': np.max(throughputs),
                'total_reward_mean': np.mean(rewards),
                'total_reward_std': np.std(rewards),
                'num_runs': len(pattern_results)
            }
            
        return aggregate
        
    def _compare_with_baselines(self, aggregate_results: Dict) -> Dict:
        """Compare RL results with baseline performance"""
        
        comparison = {}
        
        for pattern in aggregate_results.keys():
            rl_latency = aggregate_results[pattern]['avg_latency_mean']
            rl_throughput = aggregate_results[pattern]['avg_throughput_mean']
            
            # Get baseline metrics
            gpu_latency = self.baseline_metrics['gpu_p95_latency'].get(pattern, 0)
            cpu_latency = self.baseline_metrics['cpu_p95_latency'].get(pattern, 0)
            gpu_throughput = self.baseline_metrics['gpu_throughput'].get(pattern, 0)
            cpu_throughput = self.baseline_metrics['cpu_throughput'].get(pattern, 0)
            
            # Calculate improvements
            comparison[pattern] = {
                'rl_latency': rl_latency,
                'gpu_latency': gpu_latency,
                'cpu_latency': cpu_latency,
                'rl_throughput': rl_throughput,
                'gpu_throughput': gpu_throughput,
                'cpu_throughput': cpu_throughput,
                
                # Speedup vs GPU
                'latency_speedup_vs_gpu': gpu_latency / rl_latency if rl_latency > 0 else 0,
                'throughput_improvement_vs_gpu': rl_throughput / gpu_throughput if gpu_throughput > 0 else 0,
                
                # Speedup vs CPU
                'latency_speedup_vs_cpu': cpu_latency / rl_latency if rl_latency > 0 else 0,
                'throughput_improvement_vs_cpu': rl_throughput / cpu_throughput if cpu_throughput > 0 else 0,
                
                # Best baseline comparison
                'best_baseline_latency': min(gpu_latency, cpu_latency) if gpu_latency > 0 and cpu_latency > 0 else max(gpu_latency, cpu_latency),
                'best_baseline_throughput': max(gpu_throughput, cpu_throughput),
                'latency_improvement_vs_best': min(gpu_latency, cpu_latency) / rl_latency if rl_latency > 0 and min(gpu_latency, cpu_latency) > 0 else 0,
                'throughput_improvement_vs_best': rl_throughput / max(gpu_throughput, cpu_throughput) if max(gpu_throughput, cpu_throughput) > 0 else 0
            }
            
        return comparison
        
    def _generate_evaluation_plots(self, evaluation_data: Dict):
        """Generate evaluation visualization plots"""
        
        patterns = list(evaluation_data['aggregate_results'].keys())
        comparison = evaluation_data['baseline_comparison']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency comparison
        rl_latencies = [evaluation_data['aggregate_results'][p]['avg_latency_mean'] for p in patterns]
        gpu_latencies = [comparison[p]['gpu_latency'] for p in patterns]
        cpu_latencies = [comparison[p]['cpu_latency'] for p in patterns]
        
        x = np.arange(len(patterns))
        width = 0.25
        
        ax1.bar(x - width, rl_latencies, width, label='RL Agent', color='red', alpha=0.7)
        ax1.bar(x, gpu_latencies, width, label='GPU Baseline', color='green', alpha=0.7)
        ax1.bar(x + width, cpu_latencies, width, label='CPU Baseline', color='blue', alpha=0.7)
        
        ax1.set_ylabel('P95 Latency (ms)')
        ax1.set_title('Latency Comparison: RL vs Baselines')
        ax1.set_xticks(x)
        ax1.set_xticklabels(patterns)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throughput comparison
        rl_throughputs = [evaluation_data['aggregate_results'][p]['avg_throughput_mean'] for p in patterns]
        gpu_throughputs = [comparison[p]['gpu_throughput'] for p in patterns]
        cpu_throughputs = [comparison[p]['cpu_throughput'] for p in patterns]
        
        ax2.bar(x - width, rl_throughputs, width, label='RL Agent', color='red', alpha=0.7)
        ax2.bar(x, gpu_throughputs, width, label='GPU Baseline', color='green', alpha=0.7)
        ax2.bar(x + width, cpu_throughputs, width, label='CPU Baseline', color='blue', alpha=0.7)
        
        ax2.set_ylabel('Throughput (req/s)')
        ax2.set_title('Throughput Comparison: RL vs Baselines')
        ax2.set_xticks(x)
        ax2.set_xticklabels(patterns)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Speedup factors
        latency_speedups_gpu = [comparison[p]['latency_speedup_vs_gpu'] for p in patterns]
        latency_speedups_cpu = [comparison[p]['latency_speedup_vs_cpu'] for p in patterns]
        
        ax3.bar(x - width/2, latency_speedups_gpu, width, label='vs GPU', color='green', alpha=0.7)
        ax3.bar(x + width/2, latency_speedups_cpu, width, label='vs CPU', color='blue', alpha=0.7)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No improvement')
        
        ax3.set_ylabel('Latency Speedup Factor')
        ax3.set_title('RL Latency Improvements')
        ax3.set_xticks(x)
        ax3.set_xticklabels(patterns)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Overall performance summary
        overall_improvements = []
        for pattern in patterns:
            best_latency_improvement = comparison[pattern]['latency_improvement_vs_best']
            best_throughput_improvement = comparison[pattern]['throughput_improvement_vs_best']
            # Combined score (harmonic mean of improvements)
            if best_latency_improvement > 0 and best_throughput_improvement > 0:
                combined_score = 2 / (1/best_latency_improvement + 1/best_throughput_improvement)
            else:
                combined_score = 0
            overall_improvements.append(combined_score)
            
        ax4.bar(patterns, overall_improvements, color='purple', alpha=0.7)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline performance')
        ax4.set_ylabel('Combined Performance Score')
        ax4.set_title('Overall RL Performance vs Best Baseline')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, "evaluation_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--runs', type=int, default=5, help='Number of evaluation runs per pattern')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'environment': {
            'namespace': 'workloads',
            'prometheus_url': 'http://localhost:9090',
            'locust_url': 'http://localhost:8089',
            'episode_duration': 300,
            'action_interval': 30
        },
        'use_gpu': torch.cuda.is_available()
    }
    
    # Load custom configuration if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
            
    # Initialize and run evaluator
    evaluator = RLEvaluator(args.model, config)
    results = evaluator.evaluate_comprehensive(num_runs=args.runs)
    
    # Print summary
    print("\n" + "="*80)
    print("RL AGENT EVALUATION SUMMARY")
    print("="*80)
    
    for pattern, comparison in results['baseline_comparison'].items():
        print(f"\n{pattern.upper()} PATTERN:")
        print(f"  RL Latency: {comparison['rl_latency']:.1f}ms")
        print(f"  Best Baseline: {comparison['best_baseline_latency']:.1f}ms")
        print(f"  Improvement: {comparison['latency_improvement_vs_best']:.2f}x")
        print(f"  RL Throughput: {comparison['rl_throughput']:.1f} req/s")
        print(f"  Best Baseline: {comparison['best_baseline_throughput']:.1f} req/s")
        print(f"  Improvement: {comparison['throughput_improvement_vs_best']:.2f}x")

if __name__ == "__main__":
    main()
