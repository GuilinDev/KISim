#!/usr/bin/env python3
"""
Generate baseline workload pattern plots matching the academic paper style.
Creates individual plots for each load pattern (ramp, spike, periodic, random)
with the same style as writing/figs/baseline/optimized/
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import glob

class BaselinePlotGenerator:
    def __init__(self, output_dir: str = "./figures/baseline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style for academic papers
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 16
        })
    
    def load_workload_data(self, users_file, stats_file):
        """Load workload data from separate users and stats JSON files"""
        if not os.path.exists(users_file) or not os.path.exists(stats_file):
            print(f"Warning: {users_file} or {stats_file} not found")
            return None, None, None

        # Load user count data
        with open(users_file, 'r') as f:
            users_data = json.load(f)

        timestamps = users_data.get('timestamps', [])
        user_counts = users_data.get('user_counts', [])

        # Load stats data
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)

        # Extract P95 response times from current_response_time_percentiles
        response_times = []
        for entry in stats_data:
            if isinstance(entry, dict):
                # Try to get P95 from current_response_time_percentiles first
                if 'current_response_time_percentiles' in entry:
                    percentiles = entry['current_response_time_percentiles']
                    if 'response_time_percentile_0.95' in percentiles:
                        response_times.append(percentiles['response_time_percentile_0.95'])
                        continue

                # Fallback: try to get from Aggregated stats
                if 'stats' in entry:
                    aggregated_stat = None
                    for stat in entry['stats']:
                        if stat.get('name') == 'Aggregated':
                            aggregated_stat = stat
                            break

                    if aggregated_stat and 'response_time_percentile_0.95' in aggregated_stat:
                        response_times.append(aggregated_stat['response_time_percentile_0.95'])
                        continue

                # If no P95 found, append None
                response_times.append(None)
            else:
                response_times.append(None)

        # Ensure user_counts and response_times have the same length
        min_length = min(len(user_counts), len(response_times))
        user_counts = user_counts[:min_length]
        response_times = response_times[:min_length]
        timestamps = timestamps[:min_length] if timestamps else []

        return timestamps, user_counts, response_times

    def create_optimized_plot(self, user_counts, response_times, pattern_name, output_file):
        """Create an optimized plot matching the academic paper style"""
        
        if not user_counts:
            print(f"No data for {pattern_name}")
            return
        
        # Create time labels - show only every nth point to reduce clutter
        n_points = len(user_counts)
        if n_points <= 10:
            step = 1
        elif n_points <= 20:
            step = 2
        else:
            step = max(1, n_points // 10)  # Show about 10 labels max
        
        # Create generic time labels
        time_labels = [f't{i+1}' for i in range(n_points)]
        
        # Create figure with optimal size for clean presentation
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot user count on left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Time Points', fontsize=16)
        ax1.set_ylabel('User Count', color=color, fontsize=16)

        # Create x-axis values starting from 1 (t1, t2, t3, ...)
        x_values = list(range(1, n_points + 1))

        line1 = ax1.plot(x_values, user_counts, color=color, marker='o',
                 linewidth=3, markersize=8, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1,
                 label='User Count')
        ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
        ax1.grid(True, alpha=0.3)

        # Set Y-axis limit - keep data range at 120, but provide space for legend
        max_users = max(user_counts) if user_counts else 100
        # Keep the upper limit at 120 for consistency with original style
        ax1.set_ylim(0, max(120, max_users * 1.2))
        
        # Set x-axis ticks to show time points (t1, t2, t3, ...)
        tick_positions = list(range(1, n_points + 1, step))
        if tick_positions[-1] != n_points:  # Ensure last point is shown
            tick_positions.append(n_points)

        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([f't{i}' for i in tick_positions], rotation=45, fontsize=14)
        
        # Plot response time on right y-axis if available
        if response_times and any(rt is not None for rt in response_times):
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('P95 Response Time (ms)', color=color, fontsize=16)

            # Filter out None values for plotting, but keep x-axis alignment
            valid_indices = [i for i, rt in enumerate(response_times) if rt is not None]
            valid_x_values = [x_values[i] for i in valid_indices]
            valid_response_times = [response_times[i] for i in valid_indices]

            line2 = ax2.plot(valid_x_values, valid_response_times, color=color, marker='x',
                    linewidth=3, markersize=8, label='P95 Response Time (ms)')
            ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
            
            # Set Y-axis range for response time
            valid_response_times = [rt for rt in response_times if rt is not None]
            if valid_response_times:
                max_rt = max(valid_response_times)
                ax2.set_ylim(0, max_rt * 1.1)
        
        # Set title
        plt.title(f'{pattern_name.title()} Load Pattern', fontsize=18, pad=20)

        # Add legend in upper left corner with proper spacing to avoid overlap
        lines1, labels1 = ax1.get_legend_handles_labels()
        if 'ax2' in locals():
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Use bbox_to_anchor to position legend with more precise control
            ax1.legend(lines1 + lines2, labels1 + labels2,
                      loc='upper left',
                      bbox_to_anchor=(0.02, 0.98),  # Slight offset from corner
                      fontsize=16,
                      framealpha=0.95,
                      fancybox=True,
                      shadow=True,
                      edgecolor='gray')
        else:
            ax1.legend(loc='upper left',
                      bbox_to_anchor=(0.02, 0.98),
                      fontsize=16,
                      framealpha=0.95,
                      fancybox=True,
                      shadow=True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Generated {pattern_name} plot: {output_file}")

    def find_latest_data_files(self, pattern, baseline_dir):
        """Find the latest users and stats files for a given pattern"""
        users_pattern = f"{pattern}_*_users.json"
        stats_pattern = f"{pattern}_*_stats.json"
        
        users_files = glob.glob(str(baseline_dir / "dynamic" / users_pattern))
        stats_files = glob.glob(str(baseline_dir / "dynamic" / stats_pattern))
        
        if not users_files or not stats_files:
            return None, None
        
        # Get the latest files (by timestamp in filename)
        latest_users_file = max(users_files)
        latest_stats_file = max(stats_files)
        
        return latest_users_file, latest_stats_file

    def generate_all_baseline_plots(self):
        """Generate all baseline workload pattern plots"""
        print("Generating baseline workload pattern plots...")
        
        # Define patterns
        patterns = ['ramp', 'spike', 'periodic', 'random']
        
        # Try to find the experiments directory
        experiments_paths = [
            Path("../experiments"),
            Path("../../experiments"),
            Path("/home/guilin/allProjects/ecrl/experiments")
        ]
        
        baseline_dir = None
        for path in experiments_paths:
            baseline_path = path / "results" / "baseline"
            if baseline_path.exists():
                baseline_dir = baseline_path
                break
        
        if not baseline_dir:
            print("Warning: Could not find baseline results directory!")
            self.generate_sample_plots()
            return
        
        print(f"Using baseline data from: {baseline_dir}")
        
        # Generate plots for each pattern
        for pattern in patterns:
            latest_users_file, latest_stats_file = self.find_latest_data_files(pattern, baseline_dir)
            
            if not latest_users_file or not latest_stats_file:
                print(f"No data files found for {pattern}")
                continue
            
            print(f"Using data from: {latest_users_file} and {latest_stats_file}")
            
            # Load data
            timestamps, user_counts, response_times = self.load_workload_data(latest_users_file, latest_stats_file)

            if user_counts:
                # Generate optimized plot
                output_file = self.output_dir / f"{pattern}_optimized.png"
                self.create_optimized_plot(user_counts, response_times, pattern, output_file)
            else:
                print(f"No valid data found for {pattern}")
        
        print(f"\n✅ All baseline plots generated in: {self.output_dir}")
        print(f"📊 Generated {len(list(self.output_dir.glob('*.png')))} plot files")

        # Note: We generate plots in KISim/figures/baseline/ to match the style of writing/figs/baseline/optimized/
        # but we don't overwrite the original optimized plots

    def generate_sample_plots(self):
        """Generate sample plots if no real data is available"""
        print("Generating sample baseline plots...")
        
        patterns = ['ramp', 'spike', 'periodic', 'random']
        
        for pattern in patterns:
            # Generate synthetic data for demonstration
            n_points = 20
            
            if pattern == 'ramp':
                user_counts = [i * 5 for i in range(n_points)]
                response_times = [1000 + i * 100 for i in range(n_points)]
            elif pattern == 'spike':
                user_counts = [50 if i == n_points//2 else 10 for i in range(n_points)]
                response_times = [5000 if i == n_points//2 else 1000 for i in range(n_points)]
            elif pattern == 'periodic':
                user_counts = [50 + 40 * np.sin(2 * np.pi * i / 6) for i in range(n_points)]
                response_times = [2000 + 1000 * np.sin(2 * np.pi * i / 6) for i in range(n_points)]
            else:  # random
                np.random.seed(42)
                user_counts = [20 + np.random.randint(-10, 30) for i in range(n_points)]
                response_times = [1500 + np.random.randint(-500, 1000) for i in range(n_points)]
            
            output_file = self.output_dir / f"{pattern}_optimized.png"
            self.create_optimized_plot(user_counts, response_times, pattern, output_file)



def main():
    generator = BaselinePlotGenerator()
    generator.generate_all_baseline_plots()

if __name__ == "__main__":
    main()
