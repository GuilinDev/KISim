#!/usr/bin/env python3
"""
Generate publication-ready plots for KISim results
Supports both baseline and RL experiment visualization
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PlotGenerator:
    """Generate various types of plots for KISim experiments"""
    
    def __init__(self, results_dir: str = "./results", output_dir: str = "./figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "baseline").mkdir(exist_ok=True)
        (self.output_dir / "rl").mkdir(exist_ok=True)
        (self.output_dir / "comparison").mkdir(exist_ok=True)
    
    def load_baseline_results(self) -> Dict:
        """Load baseline experiment results"""
        baseline_dir = self.results_dir / "baseline" / "dynamic"
        results = {}

        if not baseline_dir.exists():
            return results

        patterns = ['ramp', 'spike', 'periodic', 'random']
        for pattern in patterns:
            # Find most recent results
            pattern_files = list(baseline_dir.glob(f"{pattern}_*_stats.json"))
            if pattern_files:
                latest_file = max(pattern_files, key=lambda x: x.stat().st_mtime)
                users_file = str(latest_file).replace("_stats.json", "_users.json")

                try:
                    with open(latest_file, 'r') as f:
                        stats_data = json.load(f)

                    # Check if users file exists
                    if os.path.exists(users_file):
                        with open(users_file, 'r') as f:
                            users_data = json.load(f)
                    else:
                        # Extract user counts from stats data if users file doesn't exist
                        users_data = {
                            'user_counts': [entry.get('user_count', 0) for entry in stats_data if 'user_count' in entry],
                            'timestamps': [entry.get('timestamp', '') for entry in stats_data if 'timestamp' in entry]
                        }

                    results[pattern] = {
                        'stats': stats_data,
                        'users': users_data
                    }
                except Exception as e:
                    print(f"Warning: Could not load {pattern} results: {e}")

        return results
    
    def load_rl_results(self) -> Dict:
        """Load RL training results"""
        rl_dir = self.results_dir / "rl"
        results = {}
        
        if not rl_dir.exists():
            return results
            
        # Find training directories
        training_dirs = list(rl_dir.glob("training_*"))
        for training_dir in training_dirs:
            try:
                # Load training metrics
                metrics_file = training_dir / "training_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        results[training_dir.name] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load RL results from {training_dir}: {e}")
        
        return results
    
    def plot_baseline_workloads(self, baseline_results: Dict):
        """Generate workload pattern plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        patterns = ['ramp', 'spike', 'periodic', 'random']
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, pattern in enumerate(patterns):
            if pattern not in baseline_results:
                continue
                
            data = baseline_results[pattern]
            users = data['users']['user_counts']
            
            # Extract response times from real data format
            response_times = []
            for entry in data['stats']:
                if isinstance(entry, dict):
                    # Check for current_response_time_percentiles (real data format)
                    if 'current_response_time_percentiles' in entry:
                        p95 = entry['current_response_time_percentiles'].get('response_time_percentile_0.95', 0)
                        if p95 is not None:
                            response_times.append(p95)
                    # Check for stats array with Aggregated data (real data format)
                    elif 'stats' in entry:
                        for stat in entry['stats']:
                            if stat.get('name') == 'Aggregated':
                                p95 = stat.get('response_time_percentile_0.95', 0)
                                if p95 is not None:
                                    response_times.append(p95)
                                break
            
            # Filter out None values and ensure same length
            response_times = [rt for rt in response_times if rt is not None and rt > 0]

            # If we have fewer response times than users, pad or truncate
            if len(response_times) < len(users):
                # Pad with last known value or default
                last_rt = response_times[-1] if response_times else 100
                response_times.extend([last_rt] * (len(users) - len(response_times)))

            min_len = min(len(users), len(response_times))
            users = users[:min_len]
            response_times = response_times[:min_len]
            
            ax = axes[i]
            
            # Plot user count
            ax1 = ax
            color = colors[i]
            ax1.set_xlabel('Time Points', fontsize=12)
            ax1.set_ylabel('User Count', color=color, fontsize=12)
            line1 = ax1.plot(range(len(users)), users, color=color, marker='o', 
                           linewidth=2, markersize=6, label='User Count')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Set Y-axis limit to provide space for legend
            max_users = max(users) if users else 100
            ax1.set_ylim(0, max(120, max_users * 1.3))
            
            # Plot response time on right y-axis
            if response_times:
                ax2 = ax1.twinx()
                color2 = 'red'
                ax2.set_ylabel('P95 Response Time (ms)', color=color2, fontsize=12)
                line2 = ax2.plot(range(len(response_times)), response_times, 
                               color=color2, marker='x', linewidth=2, markersize=8,
                               label='P95 Response Time (ms)')
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Set Y-axis limit for response time
                max_response = max(response_times) if response_times else 6000
                ax2.set_ylim(0, max(7000, max_response * 1.3))
                
                # Combined legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, 
                          loc='upper left', fontsize=10, framealpha=0.9)
            
            ax.set_title(f'{pattern.title()} Load Pattern', fontsize=14, pad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "baseline" / "workload_patterns.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Baseline workload plots saved to: {self.output_dir}/baseline/workload_patterns.png")
    
    def plot_rl_training(self, rl_results: Dict):
        """Generate RL training progress plots"""
        if not rl_results:
            print("No RL results found for plotting")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(rl_results)))
        
        for i, (run_name, data) in enumerate(rl_results.items()):
            color = colors[i]
            label = run_name.replace('training_', '')
            
            if 'episode_rewards' in data:
                episodes = range(len(data['episode_rewards']))
                
                # Episode rewards
                ax1.plot(episodes, data['episode_rewards'], 
                        color=color, alpha=0.7, label=label)
                
                # Moving average
                window = min(10, len(data['episode_rewards']) // 4)
                if window > 1:
                    moving_avg = np.convolve(data['episode_rewards'], 
                                           np.ones(window)/window, mode='valid')
                    ax1.plot(range(window-1, len(data['episode_rewards'])), moving_avg,
                            color=color, linewidth=2)
            
            if 'episode_lengths' in data:
                ax2.plot(range(len(data['episode_lengths'])), data['episode_lengths'],
                        color=color, alpha=0.7, label=label)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('RL Training Progress: Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('RL Training Progress: Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Additional plots can be added here for policy loss, value loss, etc.
        ax3.text(0.5, 0.5, 'Policy Loss\n(Coming Soon)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax4.text(0.5, 0.5, 'Value Loss\n(Coming Soon)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "rl" / "training_progress.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"RL training plots saved to: {self.output_dir}/rl/training_progress.png")
    
    def plot_comparison(self, baseline_results: Dict, rl_results: Dict):
        """Generate comparison plots between baseline and RL"""
        # This would compare RL performance against baseline
        # Implementation depends on having comparable metrics
        print("Comparison plots: Implementation pending RL evaluation results")
    
    def generate_all_plots(self):
        """Generate all available plots"""
        print("Loading baseline results...")
        baseline_results = self.load_baseline_results()
        
        print("Loading RL results...")
        rl_results = self.load_rl_results()
        
        if baseline_results:
            print("Generating baseline plots...")
            self.plot_baseline_workloads(baseline_results)
        else:
            print("No baseline results found")
        
        if rl_results:
            print("Generating RL plots...")
            self.plot_rl_training(rl_results)
        else:
            print("No RL results found")
        
        print("Generating comparison plots...")
        self.plot_comparison(baseline_results, rl_results)
        
        print(f"\nAll plots saved to: {self.output_dir}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate KISim plots')
    parser.add_argument('--results-dir', default='./results', 
                       help='Results directory (default: ./results)')
    parser.add_argument('--output-dir', default='./figures', 
                       help='Output directory (default: ./figures)')
    parser.add_argument('--type', choices=['baseline', 'rl', 'comparison', 'all'], 
                       default='all', help='Type of plots to generate')
    
    args = parser.parse_args()
    
    generator = PlotGenerator(args.results_dir, args.output_dir)
    
    if args.type == 'all':
        generator.generate_all_plots()
    elif args.type == 'baseline':
        baseline_results = generator.load_baseline_results()
        if baseline_results:
            generator.plot_baseline_workloads(baseline_results)
    elif args.type == 'rl':
        rl_results = generator.load_rl_results()
        if rl_results:
            generator.plot_rl_training(rl_results)
    elif args.type == 'comparison':
        baseline_results = generator.load_baseline_results()
        rl_results = generator.load_rl_results()
        generator.plot_comparison(baseline_results, rl_results)

if __name__ == "__main__":
    main()
