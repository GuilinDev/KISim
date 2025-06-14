#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

def verify_random_plot():
    """Verify that the random plot starts from the correct point"""
    
    # Load the actual data
    users_file = "/home/guilin/allProjects/ecrl/experiments/results/baseline/dynamic/random_20250523_232836_users.json"
    stats_file = "/home/guilin/allProjects/ecrl/experiments/results/baseline/dynamic/random_20250523_232836_stats.json"
    
    # Load users data
    with open(users_file, 'r') as f:
        users_data = json.load(f)
    user_counts = users_data.get('user_counts', [])
    
    # Load stats data
    with open(stats_file, 'r') as f:
        stats_data = json.load(f)
    
    # Extract P95 response times
    response_times = []
    for entry in stats_data:
        if isinstance(entry, dict):
            if 'current_response_time_percentiles' in entry:
                percentiles = entry['current_response_time_percentiles']
                if 'response_time_percentile_0.95' in percentiles:
                    response_times.append(percentiles['response_time_percentile_0.95'])
                    continue
            response_times.append(None)
        else:
            response_times.append(None)
    
    # Ensure same length
    min_length = min(len(user_counts), len(response_times))
    user_counts = user_counts[:min_length]
    response_times = response_times[:min_length]
    
    print(f"First 5 user counts: {user_counts[:5]}")
    print(f"First 5 response times: {response_times[:5]}")
    
    # Create x-axis values starting from 1
    x_values = list(range(1, len(user_counts) + 1))
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot user count (blue line)
    color = 'tab:blue'
    ax1.set_xlabel('Time Points', fontsize=16)
    ax1.set_ylabel('User Count', color=color, fontsize=16)
    line1 = ax1.plot(x_values, user_counts, color=color, marker='o',
             linewidth=3, markersize=8, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1,
             label='User Count')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 120)
    
    # Plot response time (red line) - filter None values
    if response_times and any(rt is not None for rt in response_times):
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('P95 Response Time (ms)', color=color, fontsize=16)
        
        valid_indices = [i for i, rt in enumerate(response_times) if rt is not None]
        valid_x_values = [x_values[i] for i in valid_indices]
        valid_response_times = [response_times[i] for i in valid_indices]
        
        print(f"First 5 valid x values: {valid_x_values[:5]}")
        print(f"First 5 valid response times: {valid_response_times[:5]}")
        
        line2 = ax2.plot(valid_x_values, valid_response_times, color=color, marker='x',
                linewidth=3, markersize=8, label='P95 Response Time (ms)')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
        
        valid_response_times_clean = [rt for rt in response_times if rt is not None]
        if valid_response_times_clean:
            max_rt = max(valid_response_times_clean)
            ax2.set_ylim(0, max_rt * 1.1)
    
    # Set X-axis
    step = 1 if len(x_values) <= 10 else 2
    tick_positions = list(range(1, len(x_values) + 1, step))
    if tick_positions[-1] != len(x_values):
        tick_positions.append(len(x_values))
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f't{i}' for i in tick_positions], rotation=45, fontsize=14)
    
    # Add title with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    plt.title(f'Random Load Pattern - Verified {timestamp}', fontsize=18, pad=20)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if 'ax2' in locals():
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper left',
                  bbox_to_anchor=(0.02, 0.98),
                  fontsize=16,
                  framealpha=0.95,
                  fancybox=True,
                  shadow=True,
                  edgecolor='gray')
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp_file = datetime.now().strftime("%H%M%S")
    output_file = f'figures/baseline/random_verified_{timestamp_file}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Verified plot saved: {output_file}")
    print(f"Expected: Red line should start at t1 with value {valid_response_times[0] if valid_response_times else 'N/A'}")
    print(f"Expected: Blue line should start at t1 with value {user_counts[0]}")

if __name__ == "__main__":
    verify_random_plot()
