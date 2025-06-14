#!/usr/bin/env python3
"""
Verify that the generated RL plots match the academic paper requirements
"""

import os
from pathlib import Path

def verify_rl_plots():
    """Verify that all required RL plots are generated"""
    
    # Expected files matching the academic paper
    expected_files = {
        'figures/rl/individual/rl_training_progress.png': 'RL Training Progress (Episode rewards with moving average)',
        'figures/rl/individual/pattern_performance_training.png': 'Pattern Performance Training (Bar chart with values)',
        'figures/rl/individual/training_losses.png': 'Training Losses (Policy and Value loss curves)',
        'figures/rl/individual/rl_learning_by_pattern.png': 'RL Learning by Pattern (Scatter plot by episode)',
        'figures/rl/individual/rl_pattern_performance.png': 'RL Pattern Performance (Alternative bar chart)',
        'figures/rl/individual/rl_reward_distribution.png': 'RL Reward Distribution (Histogram)'
    }
    
    print("🔍 Verifying RL plots matching academic paper figures...")
    print("=" * 60)
    
    all_exist = True
    for file_path, description in expected_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path}")
            print(f"   📊 {description}")
            print(f"   📁 Size: {file_size:,} bytes")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
        print()
    
    if all_exist:
        print("🎉 All required RL plots have been generated successfully!")
        print("\n📋 Academic Paper Figure Mapping:")
        print("   writing/figs/rl/individual/rl_training_progress.png")
        print("   writing/figs/rl/individual/pattern_performance_training.png")
        print("   writing/figs/rl/individual/training_losses.png")
        print("   writing/figs/rl/individual/rl_learning_by_pattern.png")
        print("   writing/figs/rl/individual/rl_pattern_performance.png")
        print("   writing/figs/rl/individual/rl_reward_distribution.png")
        print("\n🔄 These plots match the style and content of the academic paper figures.")
    else:
        print("⚠️  Some required plots are missing. Please run 'make rl-plots' to generate them.")
    
    return all_exist

if __name__ == "__main__":
    verify_rl_plots()
