#!/usr/bin/env python3
"""
Generate academic-style RL training plots for KISim
Creates individual plots matching the academic paper style shown in the image.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

class RLPlotGenerator:
    """Generate academic-style RL training plots"""
    
    def __init__(self, output_dir: str = "./figures/rl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style for academic papers
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
    
    def generate_synthetic_rl_data(self):
        """Generate synthetic RL training data matching the academic paper figures"""
        episodes = 100

        # Generate episode rewards matching the paper's pattern
        # The paper shows rewards ranging from ~0.5 to ~2.7 with high variability
        episode_rewards = []
        np.random.seed(42)  # For reproducible results

        for ep in range(episodes):
            # Create pattern similar to the paper: high variability with some learning
            if ep < 20:
                # Early episodes: more random, lower average
                reward = np.random.choice([0.5, 1.4, 1.8, 2.1, 2.7], p=[0.3, 0.2, 0.2, 0.15, 0.15])
            elif ep < 50:
                # Mid episodes: slight improvement
                reward = np.random.choice([0.5, 1.4, 1.8, 2.1, 2.7], p=[0.2, 0.15, 0.25, 0.2, 0.2])
            else:
                # Later episodes: better performance but still variable
                reward = np.random.choice([0.5, 1.4, 1.8, 2.1, 2.7], p=[0.15, 0.1, 0.25, 0.25, 0.25])

            # Add small noise
            reward += np.random.normal(0, 0.05)
            episode_rewards.append(max(0.5, min(2.7, reward)))

        # Generate training losses matching the paper's pattern
        # Policy loss stays low (~0), Value loss starts high (~2800) and decreases
        policy_losses = []
        value_losses = []

        # Only 7 updates as shown in the paper
        for update in range(7):
            # Policy loss stays near 0 with small variations
            policy_loss = np.random.normal(0.02, 0.01)
            policy_losses.append(max(0, policy_loss))

            # Value loss decreases exponentially from ~2800 to ~100
            if update == 0:
                value_loss = 2800
            elif update == 1:
                value_loss = 200 + np.random.normal(0, 20)
            else:
                value_loss = 100 + np.random.normal(0, 10)

            value_losses.append(max(0, value_loss))

        # Generate pattern-specific performance matching the paper
        # Values from the bar chart: random=1.761, ramp=2.710, spike=0.530, periodic=2.160
        pattern_performance = {
            'random': 1.761,
            'ramp': 2.710,
            'spike': 0.530,
            'periodic': 2.160
        }

        # Generate learning by pattern data (scatter plot data)
        learning_by_pattern = {}
        pattern_rewards = {
            'random': 1.7,  # Blue dots around 1.7
            'ramp': 2.7,    # Green dots around 2.7
            'spike': 0.5,   # Red dots around 0.5
            'periodic': 2.1 # Orange dots around 2.1
        }

        for pattern, base_reward in pattern_rewards.items():
            pattern_episodes = []
            pattern_values = []
            # Generate episodes where this pattern appears (every 4th episode starting from pattern index)
            pattern_index = list(pattern_rewards.keys()).index(pattern)
            for ep in range(pattern_index, 100, 4):
                pattern_episodes.append(ep)
                # Add some noise around the base reward
                reward = base_reward + np.random.normal(0, 0.05)
                pattern_values.append(max(0.4, min(2.8, reward)))

            learning_by_pattern[pattern] = {
                'episodes': pattern_episodes,
                'rewards': pattern_values
            }

        # Generate alternative pattern performance (different values for second chart)
        alt_pattern_performance = {
            'ramp': 1.704,
            'spike': 1.944,
            'periodic': 2.081,
            'random': 1.618
        }

        # Generate reward distribution data matching the original paper exactly
        # Original shows 4 separate bars at specific reward values
        reward_distribution = {
            'reward_values': [0.5, 2.0, 2.5, 2.8],  # X positions of bars
            'frequencies': [28, 22, 32, 19],         # Heights matching the original
            'bar_width': 0.3                        # Width of each bar
        }

        return {
            'episode_rewards': episode_rewards,
            'policy_losses': policy_losses,
            'value_losses': value_losses,
            'pattern_performance': pattern_performance,
            'learning_by_pattern': learning_by_pattern,
            'alt_pattern_performance': alt_pattern_performance,
            'reward_distribution': reward_distribution
        }
    
    def plot_training_progress(self, episode_rewards: List[float]):
        """Generate RL Training Progress plot matching the academic paper"""
        fig, ax = plt.subplots(figsize=(10, 6))

        episodes = range(len(episode_rewards))

        # Plot episode rewards with light blue color and transparency
        ax.plot(episodes, episode_rewards, alpha=0.7, linewidth=1.5, color='steelblue', label='Episode Rewards')

        # Moving average with red color and thicker line
        window = 10
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episode_rewards)), moving_avg,
                   color='red', linewidth=3, label='Moving Avg (10)')

        # Styling to match the paper
        ax.set_xlabel('Episode', fontsize=16)
        ax.set_ylabel('Reward', fontsize=16)
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0.5, 2.8)

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'rl_training_progress.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"RL training progress plot saved to: {self.output_dir}/rl_training_progress.png")
    
    def plot_performance_by_pattern(self, pattern_performance: Dict[str, float]):
        """Generate Pattern Performance Training plot matching the academic paper"""
        fig, ax = plt.subplots(figsize=(10, 6))

        patterns = list(pattern_performance.keys())
        avg_rewards = list(pattern_performance.values())

        # Create bars with steelblue color
        bars = ax.bar(patterns, avg_rewards, alpha=0.8, color='steelblue', width=0.6)

        # Add value labels on bars with 3 decimal places to match the paper
        for bar, avg in zip(bars, avg_rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{avg:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Styling to match the paper
        ax.set_ylabel('Average Reward', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 3.5)

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_performance_training.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Pattern performance training plot saved to: {self.output_dir}/pattern_performance_training.png")
    
    def plot_training_losses(self, policy_losses: List[float], value_losses: List[float]):
        """Generate Training Losses plot matching the academic paper"""
        fig, ax = plt.subplots(figsize=(10, 6))

        updates = range(1, len(policy_losses) + 1)

        # Plot with colors and line width matching the paper
        ax.plot(updates, policy_losses, label='Policy Loss', color='orange', linewidth=3)
        ax.plot(updates, value_losses, label='Value Loss', color='purple', linewidth=3)

        # Styling to match the paper
        ax.set_xlabel('Update', fontsize=16)
        ax.set_ylabel('Loss', fontsize=16)
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 7)
        ax.set_ylim(0, max(value_losses) * 1.1)

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_losses.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Training losses plot saved to: {self.output_dir}/training_losses.png")

    def plot_learning_by_pattern(self, learning_by_pattern: Dict):
        """Generate RL Learning by Pattern scatter plot matching the academic paper"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Colors and markers for each pattern
        colors = {'random': 'blue', 'ramp': 'green', 'spike': 'red', 'periodic': 'orange'}

        # Plot scatter points for each pattern
        for pattern, data in learning_by_pattern.items():
            episodes = data['episodes']
            rewards = data['rewards']
            ax.scatter(episodes, rewards, color=colors[pattern], label=pattern,
                      s=60, alpha=0.8, edgecolors='white', linewidth=0.5)

        # Styling to match the paper
        ax.set_xlabel('Episode', fontsize=16)
        ax.set_ylabel('Reward', fontsize=16)
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0.5, 2.8)

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'rl_learning_by_pattern.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"RL learning by pattern plot saved to: {self.output_dir}/rl_learning_by_pattern.png")

    def plot_alt_pattern_performance(self, alt_pattern_performance: Dict[str, float]):
        """Generate alternative Pattern Performance plot matching the academic paper"""
        fig, ax = plt.subplots(figsize=(10, 6))

        patterns = list(alt_pattern_performance.keys())
        avg_rewards = list(alt_pattern_performance.values())

        # Create bars with steelblue color
        bars = ax.bar(patterns, avg_rewards, alpha=0.8, color='steelblue', width=0.6)

        # Add value labels on bars with 3 decimal places to match the paper
        for bar, avg in zip(bars, avg_rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{avg:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Styling to match the paper
        ax.set_ylabel('Average Reward', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 2.5)

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'rl_pattern_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"RL pattern performance plot saved to: {self.output_dir}/rl_pattern_performance.png")

    def plot_reward_distribution_histogram(self, reward_distribution: Dict):
        """Generate Reward Distribution histogram matching the academic paper exactly"""
        fig, ax = plt.subplots(figsize=(10, 6))

        reward_values = reward_distribution['reward_values']
        frequencies = reward_distribution['frequencies']
        bar_width = reward_distribution['bar_width']

        # Create individual bars at specific reward values (matching original paper)
        bars = ax.bar(reward_values, frequencies, width=bar_width, alpha=0.8, color='steelblue',
                     edgecolor='black', linewidth=0.5)

        # Styling to match the paper exactly
        ax.set_xlabel('Reward', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.2, 3.0)  # Wider range to show all bars clearly
        ax.set_ylim(0, 35)

        # Set specific x-axis ticks to match the original
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'rl_reward_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"RL reward distribution plot saved to: {self.output_dir}/rl_reward_distribution.png")
    
    def plot_learning_progress_by_pattern(self, pattern_performance: Dict[str, List[float]]):
        """Generate (a) Learning Progress by Pattern plot (bottom left) - with markers like paper"""
        fig, ax = plt.subplots(figsize=(8, 6))

        patterns = list(pattern_performance.keys())
        colors = ['green', 'blue', 'red', 'orange']
        markers = ['o', 's', '^', 'D']

        for i, pattern in enumerate(patterns):
            rewards = pattern_performance[pattern]
            episodes = range(1, len(rewards) + 1)
            # Add markers and make lines more visible like in the paper
            ax.plot(episodes, rewards, color=colors[i], marker=markers[i],
                   label=pattern, linewidth=2, markersize=6, alpha=0.8,
                   markerfacecolor=colors[i], markeredgecolor='white', markeredgewidth=1)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_progress_by_pattern.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Learning progress by pattern plot saved to: {self.output_dir}/learning_progress_by_pattern.png")
    
    def plot_reward_distribution(self, reward_data: Dict):
        """Generate (c) Reward Distribution plot (bottom right)"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bins = reward_data['bins']
        counts = reward_data['counts']
        
        # Create histogram
        ax.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, color='steelblue', 
               align='edge', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reward_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Reward distribution plot saved to: {self.output_dir}/reward_distribution.png")

    def plot_combined_paper_style(self, data: Dict):
        """Generate combined plots matching the exact paper layout (2x3)"""
        fig = plt.figure(figsize=(18, 12))

        # Fig. 3: RL Training Progress (top row)
        # (a) RL Training Progress
        ax1 = plt.subplot(2, 3, 1)
        episodes = range(1, len(data['episode_rewards']) + 1)
        ax1.plot(episodes, data['episode_rewards'], alpha=0.6, linewidth=1, color='steelblue', label='Episode Rewards')

        window = 10
        if len(data['episode_rewards']) >= window:
            moving_avg = np.convolve(data['episode_rewards'], np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(data['episode_rewards']) + 1), moving_avg,
                   color='red', linewidth=2.5, label='Moving Avg (10)')

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('(a) RL Training Progress')

        # (b) Performance by Load Pattern
        ax2 = plt.subplot(2, 3, 2)
        patterns = list(data['pattern_performance'].keys())
        avg_rewards = [np.mean(data['pattern_performance'][p]) for p in patterns]

        bars = ax2.bar(patterns, avg_rewards, alpha=0.7, color='steelblue')
        for bar, avg in zip(bars, avg_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{avg:.2f}', ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel('Average Reward')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('(b) Performance by Load Pattern')

        # (c) Training Losses
        ax3 = plt.subplot(2, 3, 3)
        episodes = range(1, len(data['policy_losses']) + 1)
        ax3.plot(episodes, data['policy_losses'], label='Policy Loss', color='orange', linewidth=2)
        ax3.plot(episodes, data['value_losses'], label='Value Loss', color='purple', linewidth=2)

        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('(c) Training Losses')

        # Fig. 4: Pattern-specific behavior (bottom row)
        # (a) Learning Progress by Pattern
        ax4 = plt.subplot(2, 3, 4)
        patterns = list(data['pattern_performance'].keys())
        colors = ['green', 'blue', 'red', 'orange']
        markers = ['o', 's', '^', 'D']

        for i, pattern in enumerate(patterns):
            rewards = data['pattern_performance'][pattern]
            episodes = range(1, len(rewards) + 1)
            ax4.plot(episodes, rewards, color=colors[i], marker=markers[i],
                   label=pattern, linewidth=2, markersize=6, alpha=0.8,
                   markerfacecolor=colors[i], markeredgecolor='white', markeredgewidth=1)

        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('(a) Learning Progress by Pattern')

        # (b) Performance by Load Pattern (duplicate for layout)
        ax5 = plt.subplot(2, 3, 5)
        bars = ax5.bar(patterns, avg_rewards, alpha=0.7, color='steelblue')
        for bar, avg in zip(bars, avg_rewards):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{avg:.2f}', ha='center', va='bottom', fontsize=10)

        ax5.set_ylabel('Average Reward')
        ax5.grid(True, alpha=0.3)
        ax5.set_title('(b) Performance by Load Pattern')

        # (c) Reward Distribution
        ax6 = plt.subplot(2, 3, 6)
        bins = data['reward_distribution']['bins']
        counts = data['reward_distribution']['counts']

        ax6.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, color='steelblue',
               align='edge', edgecolor='black', linewidth=0.5)

        ax6.set_xlabel('Reward')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        ax6.set_title('(c) Reward Distribution')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'paper_style_combined.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Combined paper-style plot saved to: {self.output_dir}/paper_style_combined.png")

    def generate_all_rl_plots(self):
        """Generate all RL plots matching the academic paper style"""
        print("Generating RL training data matching academic paper figures...")
        data = self.generate_synthetic_rl_data()

        print("Generating RL training plots...")

        # Generate all six plots matching the paper
        self.plot_training_progress(data['episode_rewards'])
        self.plot_performance_by_pattern(data['pattern_performance'])
        self.plot_training_losses(data['policy_losses'], data['value_losses'])
        self.plot_learning_by_pattern(data['learning_by_pattern'])
        self.plot_alt_pattern_performance(data['alt_pattern_performance'])
        self.plot_reward_distribution_histogram(data['reward_distribution'])

        print(f"\n✅ All RL plots saved to: {self.output_dir}")
        print("📊 Generated plots matching academic paper:")
        print("  - rl_training_progress.png (Episode rewards with moving average)")
        print("  - pattern_performance_training.png (Performance by load pattern)")
        print("  - training_losses.png (Policy and value losses)")
        print("  - rl_learning_by_pattern.png (Learning by pattern scatter plot)")
        print("  - rl_pattern_performance.png (Alternative pattern performance)")
        print("  - rl_reward_distribution.png (Reward distribution histogram)")

        # Create individual directory for academic paper structure
        individual_dir = self.output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)

        # Copy files to individual directory with academic naming
        import shutil
        shutil.copy2(self.output_dir / 'rl_training_progress.png', individual_dir / 'rl_training_progress.png')
        shutil.copy2(self.output_dir / 'pattern_performance_training.png', individual_dir / 'pattern_performance_training.png')
        shutil.copy2(self.output_dir / 'training_losses.png', individual_dir / 'training_losses.png')
        shutil.copy2(self.output_dir / 'rl_learning_by_pattern.png', individual_dir / 'rl_learning_by_pattern.png')
        shutil.copy2(self.output_dir / 'rl_pattern_performance.png', individual_dir / 'rl_pattern_performance.png')
        shutil.copy2(self.output_dir / 'rl_reward_distribution.png', individual_dir / 'rl_reward_distribution.png')

        print(f"📁 Individual plots also saved to: {individual_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate RL training plots')
    parser.add_argument('--output-dir', default='./figures/rl', help='Output directory for plots')
    
    args = parser.parse_args()
    
    generator = RLPlotGenerator(args.output_dir)
    generator.generate_all_rl_plots()

if __name__ == "__main__":
    main()
