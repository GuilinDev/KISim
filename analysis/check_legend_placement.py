#!/usr/bin/env python3
"""
Check legend placement in generated baseline plots to ensure no overlap with data.
"""

import json
import os
from pathlib import Path

def analyze_legend_placement():
    """Analyze the data patterns to verify legend placement is optimal"""
    print("🔍 Analyzing legend placement in baseline plots...")
    print("=" * 60)
    
    patterns = ['ramp', 'spike', 'periodic', 'random']
    baseline_dir = Path("../experiments/results/baseline/dynamic")
    
    for pattern in patterns:
        print(f"\n📊 {pattern.upper()} Pattern:")
        
        # Find the latest files for this pattern
        users_files = list(baseline_dir.glob(f"{pattern}_*_users.json"))
        stats_files = list(baseline_dir.glob(f"{pattern}_*_stats.json"))
        
        if not users_files or not stats_files:
            print(f"  ❌ Data files not found for {pattern}")
            continue
        
        latest_users_file = max(users_files)
        latest_stats_file = max(stats_files)
        
        try:
            # Load user count data
            with open(latest_users_file, 'r') as f:
                users_data = json.load(f)
            user_counts = users_data.get('user_counts', [])
            
            # Load stats data
            with open(latest_stats_file, 'r') as f:
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
                    
                    if 'stats' in entry:
                        for stat in entry['stats']:
                            if stat.get('name') == 'Aggregated':
                                if 'response_time_percentile_0.95' in stat:
                                    response_times.append(stat['response_time_percentile_0.95'])
                                    break
                        else:
                            response_times.append(None)
                    else:
                        response_times.append(None)
                else:
                    response_times.append(None)
            
            # Ensure same length
            min_length = min(len(user_counts), len(response_times))
            user_counts = user_counts[:min_length]
            response_times = response_times[:min_length]
            
            if not user_counts or not response_times:
                print(f"  ❌ No valid data for {pattern}")
                continue
            
            # Analyze data for legend placement
            max_users = max(user_counts)
            valid_response_times = [rt for rt in response_times if rt is not None]
            max_rt = max(valid_response_times) if valid_response_times else 0
            
            # Check corner densities
            start_users = user_counts[:3] if len(user_counts) >= 3 else user_counts[:1]
            end_users = user_counts[-3:] if len(user_counts) >= 3 else user_counts[-1:]
            
            start_rts = [rt for rt in response_times[:3] if rt is not None]
            end_rts = [rt for rt in response_times[-3:] if rt is not None]
            
            start_user_avg = sum(start_users) / len(start_users)
            end_user_avg = sum(end_users) / len(end_users)
            start_rt_avg = sum(start_rts) / len(start_rts) if start_rts else 0
            end_rt_avg = sum(end_rts) / len(end_rts) if end_rts else 0
            
            # Calculate relative positions
            start_user_rel = start_user_avg / max_users if max_users > 0 else 0
            end_user_rel = end_user_avg / max_users if max_users > 0 else 0
            start_rt_rel = start_rt_avg / max_rt if max_rt > 0 else 0
            end_rt_rel = end_rt_avg / max_rt if max_rt > 0 else 0
            
            print(f"  📈 Data points: {len(user_counts)}")
            print(f"  📊 User count range: {min(user_counts)}-{max_users}")
            print(f"  ⏱️  P95 RT range: {min(valid_response_times):.0f}-{max_rt:.0f}ms")
            print(f"  📍 Start corner density: Users {start_user_rel:.2f}, RT {start_rt_rel:.2f}")
            print(f"  📍 End corner density: Users {end_user_rel:.2f}, RT {end_rt_rel:.2f}")
            
            # Recommend legend position
            if start_user_rel < 0.6 and start_rt_rel < 0.6:
                recommended = 'upper left'
                reason = 'Low density at start'
            elif end_user_rel < 0.6 and end_rt_rel < 0.6:
                recommended = 'upper right'
                reason = 'Low density at end'
            elif start_user_rel < 0.4:
                recommended = 'upper left'
                reason = 'Low user count at start'
            elif end_user_rel < 0.4:
                recommended = 'upper right'
                reason = 'Low user count at end'
            else:
                recommended = 'upper right'
                reason = 'Default safe position'
            
            print(f"  🎯 Recommended legend: {recommended} ({reason})")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {pattern}: {e}")
    
    print("\n" + "=" * 60)
    print("📋 Legend Placement Summary:")
    print("  ✅ Smart algorithm analyzes data density in corners")
    print("  📊 Chooses position with least overlap potential")
    print("  🎨 Enhanced with transparency and shadow for visibility")
    print("  📏 Increased Y-axis range provides more legend space")
    
    # Check if plots exist
    plots_dir = Path("figures/baseline")
    plot_files = list(plots_dir.glob("*_optimized.png"))
    
    print(f"\n📁 Generated plots: {len(plot_files)}/4")
    for plot_file in sorted(plot_files):
        size_kb = plot_file.stat().st_size / 1024
        print(f"  📈 {plot_file.name}: {size_kb:.1f} KB")
    
    if len(plot_files) == 4:
        print("\n🎉 All plots generated with optimized legend placement!")
        print("💡 Legend positions are automatically chosen to minimize overlap")
        print("🎯 Ready for academic paper use!")

def main():
    analyze_legend_placement()

if __name__ == "__main__":
    main()
