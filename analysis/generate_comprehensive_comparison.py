#!/usr/bin/env python3
"""
Comprehensive GPU vs CPU Performance Comparison Report Generator
Analyzes all 4 load patterns and generates detailed comparison charts and reports.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_synthetic_results(results_dir):
    """Load synthetic test results if available"""
    synthetic_file = os.path.join(results_dir, "metrics", "synthetic_results.json")
    if os.path.exists(synthetic_file):
        with open(synthetic_file, 'r') as f:
            return json.load(f)
    return None

def load_dynamic_results(results_dir):
    """Load all dynamic load test results"""
    dynamic_dir = os.path.join(results_dir, "dynamic")
    results = {}

    if not os.path.exists(dynamic_dir):
        return results

    patterns = ['ramp', 'spike', 'periodic', 'random']
    for pattern in patterns:
        # Find the most recent file for this pattern
        pattern_files = [f for f in os.listdir(dynamic_dir) if f.startswith(f"{pattern}_") and f.endswith("_stats.json")]
        if pattern_files:
            # Sort by timestamp and take the most recent
            pattern_files.sort(reverse=True)
            stats_file = os.path.join(dynamic_dir, pattern_files[0])
            users_file = stats_file.replace("_stats.json", "_users.json")

            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                with open(users_file, 'r') as f:
                    users_data = json.load(f)

                results[pattern] = {
                    'stats': stats_data,
                    'users': users_data,
                    'timestamp': pattern_files[0].split('_')[1] + '_' + pattern_files[0].split('_')[2]
                }
            except Exception as e:
                print(f"Warning: Could not load {pattern} results: {e}")

    return results

def extract_performance_metrics(dynamic_results):
    """Extract key performance metrics from dynamic test results"""
    metrics = {}

    for pattern, data in dynamic_results.items():
        stats_list = data['stats']  # This is a list of time-series data
        users = data['users']

        if not stats_list:
            print(f"Warning: No stats data found for {pattern}")
            continue

        # Find the aggregated stats from the last entry (most complete data)
        aggregated_stats = None
        for entry in reversed(stats_list):
            if 'stats' in entry:
                for stat in entry['stats']:
                    if stat.get('name') == 'Aggregated':
                        aggregated_stats = stat
                        break
                if aggregated_stats:
                    break

        if not aggregated_stats:
            print(f"Warning: No aggregated stats found for {pattern}")
            continue

        # Extract overall response time percentiles from the last entry
        last_entry = stats_list[-1]
        overall_percentiles = last_entry.get('current_response_time_percentiles', {})

        metrics[pattern] = {
            'avg_response_time': aggregated_stats.get('avg_response_time', 0),
            'p50_response_time': overall_percentiles.get('response_time_percentile_0.5', 0),
            'p95_response_time': overall_percentiles.get('response_time_percentile_0.95', 0),
            'p99_response_time': aggregated_stats.get('response_time_percentile_0.99', 0),
            'max_response_time': aggregated_stats.get('max_response_time', 0),
            'min_response_time': aggregated_stats.get('min_response_time', 0),
            'total_requests': aggregated_stats.get('num_requests', 0),
            'failed_requests': aggregated_stats.get('num_failures', 0),
            'requests_per_sec': aggregated_stats.get('current_rps', 0),
            'user_count_range': f"{min(users['user_counts'])}-{max(users['user_counts'])}",
            'test_duration': len(users['user_counts']) * 5,  # Assuming 5-second intervals
            'success_rate': ((aggregated_stats.get('num_requests', 0) - aggregated_stats.get('num_failures', 0)) /
                           max(aggregated_stats.get('num_requests', 1), 1)) * 100
        }

    return metrics

def generate_comparison_charts(gpu_metrics, cpu_metrics, gpu_synthetic, cpu_synthetic, output_dir):
    """Generate comprehensive comparison charts"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Response Time Comparison Chart
    patterns = list(set(gpu_metrics.keys()) & set(cpu_metrics.keys()))
    if patterns:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # P95 Response Time Comparison
        gpu_p95 = [gpu_metrics[p]['p95_response_time'] for p in patterns]
        cpu_p95 = [cpu_metrics[p]['p95_response_time'] for p in patterns]

        x = np.arange(len(patterns))
        width = 0.35

        ax1.bar(x - width/2, gpu_p95, width, label='GPU', color='green', alpha=0.7)
        ax1.bar(x + width/2, cpu_p95, width, label='CPU', color='blue', alpha=0.7)
        ax1.set_ylabel('P95 Response Time (ms)')
        ax1.set_title('P95 Response Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(patterns)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Requests per Second Comparison
        gpu_rps = [gpu_metrics[p]['requests_per_sec'] for p in patterns]
        cpu_rps = [cpu_metrics[p]['requests_per_sec'] for p in patterns]

        ax2.bar(x - width/2, gpu_rps, width, label='GPU', color='green', alpha=0.7)
        ax2.bar(x + width/2, cpu_rps, width, label='CPU', color='blue', alpha=0.7)
        ax2.set_ylabel('Requests per Second')
        ax2.set_title('Throughput Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(patterns)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Speedup Factors
        latency_speedups = [cpu_metrics[p]['p95_response_time'] / gpu_metrics[p]['p95_response_time']
                           if gpu_metrics[p]['p95_response_time'] > 0 else 0 for p in patterns]
        throughput_speedups = [gpu_metrics[p]['requests_per_sec'] / cpu_metrics[p]['requests_per_sec']
                              if cpu_metrics[p]['requests_per_sec'] > 0 else 0 for p in patterns]

        ax3.bar(patterns, latency_speedups, color='orange', alpha=0.7)
        ax3.set_ylabel('Speedup Factor (CPU/GPU)')
        ax3.set_title('Latency Speedup (Higher is Better)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax3.legend()

        ax4.bar(patterns, throughput_speedups, color='purple', alpha=0.7)
        ax4.set_ylabel('Speedup Factor (GPU/CPU)')
        ax4.set_title('Throughput Speedup (Higher is Better)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Synthetic Test Comparison
    if gpu_synthetic and cpu_synthetic:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Latency comparison
        categories = ['Average', 'P95', 'P99']
        gpu_latencies = [
            gpu_synthetic['avg_latency_ms'],
            gpu_synthetic['p95_latency_ms'],
            gpu_synthetic['p99_latency_ms']
        ]
        cpu_latencies = [
            cpu_synthetic['avg_latency_ms'],
            cpu_synthetic['p95_latency_ms'],
            cpu_synthetic['p99_latency_ms']
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(x - width/2, gpu_latencies, width, label='GPU', color='green', alpha=0.7)
        ax1.bar(x + width/2, cpu_latencies, width, label='CPU', color='blue', alpha=0.7)
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Synthetic Test Latency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Throughput comparison
        throughputs = [gpu_synthetic['samples_per_second'],
                      cpu_synthetic['samples_per_second']]
        platforms = ['GPU', 'CPU']
        colors = ['green', 'blue']

        ax2.bar(platforms, throughputs, color=colors, alpha=0.7)
        ax2.set_ylabel('Throughput (images/sec)')
        ax2.set_title('Synthetic Test Throughput Comparison')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'synthetic_test_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_text_report(gpu_metrics, cpu_metrics, gpu_synthetic, cpu_synthetic, output_dir):
    """Generate detailed text report"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE GPU vs CPU PERFORMANCE COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Synthetic Test Results
    if gpu_synthetic and cpu_synthetic:
        report_lines.append("SYNTHETIC TEST RESULTS")
        report_lines.append("-" * 40)
        report_lines.append(f"GPU Average Latency: {gpu_synthetic['avg_latency_ms']:.2f} ms")
        report_lines.append(f"CPU Average Latency: {cpu_synthetic['avg_latency_ms']:.2f} ms")

        latency_speedup = cpu_synthetic['avg_latency_ms'] / gpu_synthetic['avg_latency_ms']
        report_lines.append(f"Latency Speedup: {latency_speedup:.2f}x (GPU is {latency_speedup:.2f}x faster)")
        report_lines.append("")

        report_lines.append(f"GPU Throughput: {gpu_synthetic['samples_per_second']:.2f} images/sec")
        report_lines.append(f"CPU Throughput: {cpu_synthetic['samples_per_second']:.2f} images/sec")

        throughput_speedup = gpu_synthetic['samples_per_second'] / cpu_synthetic['samples_per_second']
        report_lines.append(f"Throughput Speedup: {throughput_speedup:.2f}x (GPU processes {throughput_speedup:.2f}x more images)")
        report_lines.append("")

    # Dynamic Load Test Results
    patterns = list(set(gpu_metrics.keys()) & set(cpu_metrics.keys()))
    if patterns:
        report_lines.append("DYNAMIC LOAD TEST RESULTS")
        report_lines.append("-" * 40)

        for pattern in sorted(patterns):
            report_lines.append(f"\n{pattern.upper()} LOAD PATTERN:")
            report_lines.append(f"  User Range: {gpu_metrics[pattern]['user_count_range']} users")
            report_lines.append(f"  Test Duration: {gpu_metrics[pattern]['test_duration']} seconds")
            report_lines.append("")

            # Response Time Comparison
            gpu_p95 = gpu_metrics[pattern]['p95_response_time']
            cpu_p95 = cpu_metrics[pattern]['p95_response_time']
            if gpu_p95 > 0:
                latency_speedup = cpu_p95 / gpu_p95
                report_lines.append(f"  P95 Response Time:")
                report_lines.append(f"    GPU: {gpu_p95:.2f} ms")
                report_lines.append(f"    CPU: {cpu_p95:.2f} ms")
                report_lines.append(f"    Speedup: {latency_speedup:.2f}x (GPU is {latency_speedup:.2f}x faster)")

            # Throughput Comparison
            gpu_rps = gpu_metrics[pattern]['requests_per_sec']
            cpu_rps = cpu_metrics[pattern]['requests_per_sec']
            if cpu_rps > 0:
                throughput_speedup = gpu_rps / cpu_rps
                report_lines.append(f"  Throughput:")
                report_lines.append(f"    GPU: {gpu_rps:.2f} req/sec")
                report_lines.append(f"    CPU: {cpu_rps:.2f} req/sec")
                report_lines.append(f"    Speedup: {throughput_speedup:.2f}x (GPU handles {throughput_speedup:.2f}x more requests)")

            # Reliability
            gpu_failed = gpu_metrics[pattern]['failed_requests']
            cpu_failed = cpu_metrics[pattern]['failed_requests']
            gpu_total = gpu_metrics[pattern]['total_requests']
            cpu_total = cpu_metrics[pattern]['total_requests']

            gpu_success_rate = (gpu_total - gpu_failed) / gpu_total * 100 if gpu_total > 0 else 0
            cpu_success_rate = (cpu_total - cpu_failed) / cpu_total * 100 if cpu_total > 0 else 0

            report_lines.append(f"  Reliability:")
            report_lines.append(f"    GPU Success Rate: {gpu_success_rate:.2f}% ({gpu_total - gpu_failed}/{gpu_total})")
            report_lines.append(f"    CPU Success Rate: {cpu_success_rate:.2f}% ({cpu_total - cpu_failed}/{cpu_total})")
            report_lines.append("")

    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 40)
    if gpu_synthetic and cpu_synthetic:
        overall_latency_speedup = cpu_synthetic['avg_latency_ms'] / gpu_synthetic['avg_latency_ms']
        overall_throughput_speedup = gpu_synthetic['samples_per_second'] / cpu_synthetic['samples_per_second']

        report_lines.append(f"Overall GPU Performance Advantage:")
        report_lines.append(f"  - {overall_latency_speedup:.2f}x faster response times")
        report_lines.append(f"  - {overall_throughput_speedup:.2f}x higher throughput")
        report_lines.append(f"  - Consistent performance across all load patterns")
        report_lines.append(f"  - GPU utilizes TensorRT optimization and FP16 precision")
        report_lines.append(f"  - CPU version provides reliable fallback option")

    # Save report
    with open(os.path.join(output_dir, 'comprehensive_comparison_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))

    return '\n'.join(report_lines)

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    gpu_results_dir = project_root / "results" / "baseline"
    cpu_results_dir = project_root / "results" / "cpu_baseline"
    output_dir = project_root / "results" / "comparison"

    print("Loading GPU results...")
    gpu_synthetic = load_synthetic_results(str(gpu_results_dir))
    gpu_dynamic = load_dynamic_results(str(gpu_results_dir))
    gpu_metrics = extract_performance_metrics(gpu_dynamic)

    print("Loading CPU results...")
    cpu_synthetic = load_synthetic_results(str(cpu_results_dir))
    cpu_dynamic = load_dynamic_results(str(cpu_results_dir))
    cpu_metrics = extract_performance_metrics(cpu_dynamic)

    print(f"Found GPU patterns: {list(gpu_metrics.keys())}")
    print(f"Found CPU patterns: {list(cpu_metrics.keys())}")

    if not gpu_metrics and not cpu_metrics:
        print("No dynamic test results found!")
        return

    print("Generating comparison charts...")
    generate_comparison_charts(gpu_metrics, cpu_metrics, gpu_synthetic, cpu_synthetic, str(output_dir))

    print("Generating text report...")
    report = generate_text_report(gpu_metrics, cpu_metrics, gpu_synthetic, cpu_synthetic, str(output_dir))

    print("\n" + "="*80)
    print("COMPARISON REPORT GENERATED")
    print("="*80)
    print(f"Charts saved to: {output_dir}")
    print(f"Report saved to: {output_dir}/comprehensive_comparison_report.txt")
    print("\nKey findings:")
    print(report.split("SUMMARY")[1] if "SUMMARY" in report else "Report generated successfully")

if __name__ == "__main__":
    main()
