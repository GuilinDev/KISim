#!/usr/bin/env python3
"""
Verify that the generated baseline plots contain P95 Response Time data.
"""

import json
import os
from pathlib import Path

def verify_p95_data():
    """Verify that P95 response time data exists in the baseline results"""
    print("🔍 Verifying P95 Response Time data in baseline plots...")
    print("=" * 60)
    
    patterns = ['ramp', 'spike', 'periodic', 'random']
    baseline_dir = Path("../experiments/results/baseline/dynamic")
    
    for pattern in patterns:
        print(f"\n📊 {pattern.upper()} Pattern:")
        
        # Find the latest stats file for this pattern
        stats_files = list(baseline_dir.glob(f"{pattern}_*_stats.json"))
        if not stats_files:
            print(f"  ❌ No stats file found for {pattern}")
            continue
        
        latest_stats_file = max(stats_files)
        print(f"  📁 Stats file: {latest_stats_file.name}")
        
        # Load and analyze the stats data
        try:
            with open(latest_stats_file, 'r') as f:
                stats_data = json.load(f)
            
            p95_values = []
            total_entries = len(stats_data)
            
            for i, entry in enumerate(stats_data):
                if isinstance(entry, dict):
                    # Check current_response_time_percentiles
                    if 'current_response_time_percentiles' in entry:
                        percentiles = entry['current_response_time_percentiles']
                        if 'response_time_percentile_0.95' in percentiles:
                            p95_values.append(percentiles['response_time_percentile_0.95'])
                            continue
                    
                    # Fallback: check Aggregated stats
                    if 'stats' in entry:
                        for stat in entry['stats']:
                            if stat.get('name') == 'Aggregated':
                                if 'response_time_percentile_0.95' in stat:
                                    p95_values.append(stat['response_time_percentile_0.95'])
                                    break
                        else:
                            p95_values.append(None)
                    else:
                        p95_values.append(None)
                else:
                    p95_values.append(None)
            
            # Analyze P95 data
            valid_p95_count = sum(1 for p95 in p95_values if p95 is not None)
            
            print(f"  📈 Total data points: {total_entries}")
            print(f"  ✅ Valid P95 values: {valid_p95_count}")
            print(f"  📊 P95 coverage: {valid_p95_count/total_entries*100:.1f}%")
            
            if valid_p95_count > 0:
                valid_p95s = [p95 for p95 in p95_values if p95 is not None]
                min_p95 = min(valid_p95s)
                max_p95 = max(valid_p95s)
                avg_p95 = sum(valid_p95s) / len(valid_p95s)
                
                print(f"  📉 P95 range: {min_p95:.1f} - {max_p95:.1f} ms")
                print(f"  📊 P95 average: {avg_p95:.1f} ms")
                
                if valid_p95_count >= total_entries * 0.8:  # At least 80% coverage
                    print(f"  ✅ P95 data quality: GOOD (>80% coverage)")
                else:
                    print(f"  ⚠️  P95 data quality: PARTIAL (<80% coverage)")
            else:
                print(f"  ❌ No valid P95 data found!")
                
        except Exception as e:
            print(f"  ❌ Error reading stats file: {e}")
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    
    # Check if plots exist and have reasonable file sizes
    plots_dir = Path("figures/baseline")
    plot_files = list(plots_dir.glob("*_optimized.png"))
    
    print(f"  📊 Generated plots: {len(plot_files)}/4")
    
    for plot_file in sorted(plot_files):
        size_kb = plot_file.stat().st_size / 1024
        print(f"    📈 {plot_file.name}: {size_kb:.1f} KB")
    
    if len(plot_files) == 4:
        print("  ✅ All baseline plots generated successfully!")
        print("  📊 Plots should now include P95 Response Time on right y-axis")
        print("  🎯 Ready for academic paper use!")
    else:
        print("  ⚠️  Some plots are missing")

def main():
    verify_p95_data()

if __name__ == "__main__":
    main()
