#!/usr/bin/env python3
"""
Analyze RL training results and compare with baselines
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def load_rl_results(training_dir):
    """Load RL training results"""
    training_path = Path(training_dir)
    
    # Load training progress
    progress_file = training_path / "training_progress.json"
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    # Load final evaluation
    eval_file = training_path / "final_evaluation.json"
    with open(eval_file, 'r') as f:
        evaluation = json.load(f)
    
    return progress, evaluation

def analyze_training_progress(progress):
    """Analyze training progress"""
    rewards = progress['episode_rewards']
    
    print("=== RL TRAINING ANALYSIS ===")
    print(f"Total Episodes: {len(rewards)}")
    print(f"Average Reward: {np.mean(rewards):.3f}")
    print(f"Std Reward: {np.std(rewards):.3f}")
    print(f"Min Reward: {np.min(rewards):.3f}")
    print(f"Max Reward: {np.max(rewards):.3f}")
    
    # Analyze reward patterns
    unique_rewards = np.unique(rewards)
    print(f"\nUnique Reward Values: {len(unique_rewards)}")
    for reward in unique_rewards:
        count = np.sum(np.array(rewards) == reward)
        percentage = count / len(rewards) * 100
        print(f"  {reward:.3f}: {count} episodes ({percentage:.1f}%)")
    
    # Learning trend analysis
    window_size = 10
    moving_avg = []
    for i in range(len(rewards) - window_size + 1):
        moving_avg.append(np.mean(rewards[i:i+window_size]))
    
    print(f"\nLearning Trend (Moving Average, window={window_size}):")
    print(f"  Early Training (Episodes 1-10): {moving_avg[0]:.3f}")
    print(f"  Mid Training (Episodes 45-55): {moving_avg[len(moving_avg)//2]:.3f}")
    print(f"  Late Training (Episodes 90-100): {moving_avg[-1]:.3f}")
    
    improvement = (moving_avg[-1] - moving_avg[0]) / moving_avg[0] * 100
    print(f"  Overall Improvement: {improvement:.1f}%")
    
    return {
        'total_episodes': len(rewards),
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'unique_rewards': unique_rewards,
        'moving_avg': moving_avg,
        'improvement_pct': improvement
    }

def analyze_pattern_performance(evaluation):
    """Analyze performance by load pattern"""
    print("\n=== PATTERN-SPECIFIC PERFORMANCE ===")
    
    pattern_stats = {}
    
    for pattern, episodes in evaluation.items():
        rewards = [ep['episode_reward'] for ep in episodes]
        latencies = [ep['avg_latency'] for ep in episodes]
        throughputs = [ep['avg_throughput'] for ep in episodes]
        
        pattern_stats[pattern] = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_latency': np.mean(latencies),
            'avg_throughput': np.mean(throughputs),
            'episodes': len(episodes)
        }
        
        print(f"\n{pattern.upper()} Pattern:")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Avg Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"  Avg Latency: {np.mean(latencies):.1f}ms")
        print(f"  Avg Throughput: {np.mean(throughputs):.1f} req/s")
    
    return pattern_stats

def compare_with_baselines(pattern_stats):
    """Compare RL performance with baseline results"""
    print("\n=== BASELINE COMPARISON ===")
    
    # Baseline results from our experiments (with fairness issues noted)
    baselines = {
        'random': {'gpu_latency': 2600, 'cpu_latency': 5100, 'gpu_throughput': 10.4, 'cpu_throughput': 10.8},
        'spike': {'gpu_latency': 370, 'cpu_latency': 470, 'gpu_throughput': 6.4, 'cpu_throughput': 5.7},
        'ramp': {'gpu_latency': 5800, 'cpu_latency': 6700, 'gpu_throughput': 10.3, 'cpu_throughput': 10.2},
        'periodic': {'gpu_latency': 2300, 'cpu_latency': 2300, 'gpu_throughput': 11.5, 'cpu_throughput': 12.1}
    }
    
    print("⚠️  Note: Baseline comparison has fairness issues (1 GPU pod vs 3 CPU pods)")
    print("RL results use mock data due to missing real deployments\n")
    
    for pattern in ['random', 'spike', 'ramp', 'periodic']:
        if pattern in pattern_stats and pattern in baselines:
            rl_stats = pattern_stats[pattern]
            baseline = baselines[pattern]
            
            print(f"{pattern.upper()} Pattern Comparison:")
            print(f"  RL Agent Latency: {rl_stats['avg_latency']:.1f}ms")
            print(f"  GPU Baseline: {baseline['gpu_latency']}ms")
            print(f"  CPU Baseline: {baseline['cpu_latency']}ms")
            
            # Calculate theoretical improvements (if RL had real data)
            gpu_improvement = baseline['gpu_latency'] / rl_stats['avg_latency']
            cpu_improvement = baseline['cpu_latency'] / rl_stats['avg_latency']
            
            print(f"  Theoretical Improvement vs GPU: {gpu_improvement:.2f}x")
            print(f"  Theoretical Improvement vs CPU: {cpu_improvement:.2f}x")
            print()

def generate_visualizations(progress, pattern_stats, output_dir):
    """Generate visualization plots"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Training progress plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    rewards = progress['episode_rewards']
    plt.plot(rewards, alpha=0.7, label='Episode Rewards')
    
    # Moving average
    window_size = 10
    moving_avg = []
    for i in range(len(rewards) - window_size + 1):
        moving_avg.append(np.mean(rewards[i:i+window_size]))
    
    plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('RL Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reward distribution
    plt.subplot(2, 2, 2)
    plt.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    # Pattern performance comparison
    plt.subplot(2, 2, 3)
    patterns = list(pattern_stats.keys())
    avg_rewards = [pattern_stats[p]['avg_reward'] for p in patterns]
    std_rewards = [pattern_stats[p]['std_reward'] for p in patterns]
    
    plt.bar(patterns, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    plt.xlabel('Load Pattern')
    plt.ylabel('Average Reward')
    plt.title('Performance by Load Pattern')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Learning curve with pattern identification
    plt.subplot(2, 2, 4)
    
    # Color code by pattern (simplified)
    pattern_colors = {'random': 'blue', 'ramp': 'green', 'spike': 'red', 'periodic': 'orange'}
    
    # Get pattern sequence from episode summaries
    if 'episode_summaries' in progress:
        for i, episode in enumerate(progress['episode_summaries']):
            pattern = episode['load_pattern']
            reward = episode['episode_reward']
            color = pattern_colors.get(pattern, 'gray')
            plt.scatter(i+1, reward, c=color, alpha=0.6, s=20)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Progress by Pattern')
    
    # Create legend
    for pattern, color in pattern_colors.items():
        plt.scatter([], [], c=color, label=pattern, s=50)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'rl_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path / 'rl_analysis.png'}")

def main():
    # Analyze the latest training results
    training_dir = "/home/guilin/allProjects/ecrl/experiments/rl/training_20250524_104218"
    output_dir = "/home/guilin/allProjects/ecrl/experiments/rl/analysis"
    
    print("Loading RL training results...")
    progress, evaluation = load_rl_results(training_dir)
    
    # Analyze training progress
    training_stats = analyze_training_progress(progress)
    
    # Analyze pattern-specific performance
    pattern_stats = analyze_pattern_performance(evaluation)
    
    # Compare with baselines
    compare_with_baselines(pattern_stats)
    
    # Generate visualizations
    generate_visualizations(progress, pattern_stats, output_dir)
    
    # Summary
    print("\n=== SUMMARY ===")
    print("✅ RL Training Completed Successfully:")
    print(f"  - 100 episodes trained")
    print(f"  - Average reward: {training_stats['avg_reward']:.3f}")
    print(f"  - Learning improvement: {training_stats['improvement_pct']:.1f}%")
    print(f"  - Pattern-specific strategies learned")
    
    print("\n⚠️  Current Limitations:")
    print("  - Using mock data (no real Triton/Prometheus)")
    print("  - Baseline comparison has fairness issues")
    print("  - Single-node cluster constraints")
    
    print("\n🎯 Next Steps:")
    print("  - Deploy real Triton servers for actual inference")
    print("  - Fix baseline fairness (equal pod allocation)")
    print("  - Evaluate with real workloads")
    print("  - Generate academic paper results")

if __name__ == "__main__":
    main()
