#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Test data based on debug output
x_values = [1, 2, 3, 4, 5]
user_counts = [26, 94, 78, 21, 21]  # Random pattern user counts
response_times = [5800.0, 1200.0, 5100.0, 4600.0, 630.0]  # Random pattern response times

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

# Plot response time (red line)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('P95 Response Time (ms)', color=color, fontsize=16)
line2 = ax2.plot(x_values, response_times, color=color, marker='x',
        linewidth=3, markersize=8, label='P95 Response Time (ms)')
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
ax2.set_ylim(0, max(response_times) * 1.1)

# Set X-axis
ax1.set_xticks(x_values)
ax1.set_xticklabels([f't{i}' for i in x_values], fontsize=14)

# Add title
plt.title('Test Random Load Pattern', fontsize=18, pad=20)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
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
plt.savefig('figures/baseline/test_random.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Test plot generated: figures/baseline/test_random.png")
print(f"Blue line should start at: ({x_values[0]}, {user_counts[0]})")
print(f"Red line should start at: ({x_values[0]}, {response_times[0]})")
