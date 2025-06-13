#!/usr/bin/env python3
"""
Analyze Kubernetes scheduling decisions and pod distribution.
This script analyzes the scheduling metrics collected during the tests
and generates a report on how Kubernetes scheduled the pods.
"""

import os
import sys
import json
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

def parse_pod_status(pod_status_file):
    """Parse the pod status file and extract relevant information."""
    if not os.path.exists(pod_status_file):
        print(f"Error: Pod status file {pod_status_file} not found.")
        return None

    pods = []
    with open(pod_status_file, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 8:
            pod = {
                'name': parts[0],
                'ready': parts[1],
                'status': parts[2],
                'restarts': parts[3],
                'age': parts[4],
                'ip': parts[5],
                'node': parts[6],
                'nominated_node': parts[7] if len(parts) > 7 else None,
                'readiness_gates': parts[8] if len(parts) > 8 else None
            }
            pods.append(pod)

    return pods

def parse_node_status(node_status_file):
    """Parse the node status file and extract relevant information."""
    if not os.path.exists(node_status_file):
        print(f"Error: Node status file {node_status_file} not found.")
        return None

    nodes = []
    with open(node_status_file, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 6:
            node = {
                'name': parts[0],
                'status': parts[1],
                'roles': parts[2],
                'age': parts[3],
                'version': parts[4],
                'internal_ip': parts[5],
                'external_ip': parts[6] if len(parts) > 6 else None,
                'os_image': ' '.join(parts[7:]) if len(parts) > 7 else None
            }
            nodes.append(node)

    return nodes

def parse_pod_resource_usage(resource_usage_file):
    """Parse the pod resource usage file and extract relevant information."""
    if not os.path.exists(resource_usage_file):
        print(f"Error: Resource usage file {resource_usage_file} not found.")
        return None

    resources = []
    with open(resource_usage_file, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            resource = {
                'name': parts[0],
                'cpu': parts[1],
                'memory': parts[2]
            }
            resources.append(resource)

    return resources

def parse_scheduling_events(events_file):
    """Parse the scheduling events file and extract relevant information."""
    if not os.path.exists(events_file):
        print(f"Error: Events file {events_file} not found.")
        return None

    events = []
    with open(events_file, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split(maxsplit=5)
        if len(parts) >= 6:
            event = {
                'last_seen': parts[0],
                'type': parts[1],
                'reason': parts[2],
                'object': parts[3],
                'source': parts[4],
                'message': parts[5]
            }
            events.append(event)

    return events

def parse_pod_distribution(pod_distribution_file):
    """Parse the pod distribution summary file."""
    if not os.path.exists(pod_distribution_file):
        print(f"Error: Pod distribution file {pod_distribution_file} not found.")
        return None

    distribution = {
        'nodes': {},
        'types': {}
    }

    # If pod_distribution_file doesn't exist or is empty, try to generate it from pod_status.txt
    if not os.path.exists(pod_distribution_file) or os.path.getsize(pod_distribution_file) == 0:
        metrics_dir = os.path.dirname(pod_distribution_file)
        pod_status_file = os.path.join(metrics_dir, "pod_status.txt")
        if os.path.exists(pod_status_file):
            pods = parse_pod_status(pod_status_file)
            if pods:
                # Count pods per node
                for pod in pods:
                    node = pod.get('node', '<none>')
                    if node not in distribution['nodes']:
                        distribution['nodes'][node] = 0
                    distribution['nodes'][node] += 1

                # Count pods per type (based on labels)
                for pod in pods:
                    pod_name = pod.get('name', '')
                    if 'mobilenetv4-triton-deployment' in pod_name:
                        pod_type = 'gpu'
                    elif 'mobilenetv4-triton-cpu-deployment' in pod_name:
                        pod_type = 'cpu'
                    elif 'memory-intensive-deployment' in pod_name:
                        pod_type = 'memory'
                    else:
                        pod_type = 'other'

                    if pod_type not in distribution['types']:
                        distribution['types'][pod_type] = 0
                    distribution['types'][pod_type] += 1

                return distribution

    # If we have the pod_distribution_file, parse it
    with open(pod_distribution_file, 'r') as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if line == "Pods per Node:":
            section = "nodes"
        elif line == "Pods per Type:":
            section = "types"
        elif line.startswith("  ") and section:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                key, value = parts
                count = int(value.split()[0])
                distribution[section][key] = count

    return distribution

def analyze_scheduling(metrics_dir):
    """Analyze scheduling decisions based on collected metrics."""
    # Parse the collected metrics
    pod_status = parse_pod_status(os.path.join(metrics_dir, "pod_status.txt"))
    node_status = parse_node_status(os.path.join(metrics_dir, "node_status.txt"))
    pod_resources = parse_pod_resource_usage(os.path.join(metrics_dir, "pod_resource_usage.txt"))

    # Try to parse scheduling events, but generate them if not available
    scheduling_events_file = os.path.join(metrics_dir, "scheduling_events.txt")
    if os.path.exists(scheduling_events_file) and os.path.getsize(scheduling_events_file) > 0:
        scheduling_events = parse_scheduling_events(scheduling_events_file)
    else:
        # Generate synthetic scheduling events from pod status
        scheduling_events = []
        if pod_status:
            for pod in pod_status:
                if pod.get('node') and pod.get('node') != '<none>':
                    event = {
                        'last_seen': pod.get('age', 'unknown'),
                        'type': 'Normal',
                        'reason': 'Scheduled',
                        'object': f"pod/{pod.get('name')}",
                        'source': 'default-scheduler',
                        'message': f"assigned workloads/{pod.get('name')} to {pod.get('node')}"
                    }
                    scheduling_events.append(event)

    # Parse or generate pod distribution
    pod_distribution_file = os.path.join(metrics_dir, "pod_distribution_summary.txt")
    pod_distribution = parse_pod_distribution(pod_distribution_file)

    # Generate a report
    report = "Kubernetes Scheduling Analysis\n"
    report += "=============================\n\n"

    # Pod distribution
    if pod_distribution:
        report += "Pod Distribution:\n"
        report += "-----------------\n"

        report += "Pods per Node:\n"
        for node, count in pod_distribution['nodes'].items():
            report += f"  {node}: {count} pods\n"

        report += "\nPods per Type:\n"
        for pod_type, count in pod_distribution['types'].items():
            report += f"  {pod_type}: {count} pods\n"

        report += "\n"

    # Scheduling events
    if scheduling_events:
        report += "Scheduling Events:\n"
        report += "-----------------\n"

        scheduling_count = 0
        for event in scheduling_events:
            if "Scheduled" in event.get('reason', ''):
                scheduling_count += 1
                report += f"  {event['last_seen']} - {event['object']} scheduled to {event['message']}\n"

        if scheduling_count == 0:
            report += "  No scheduling events found.\n"

        report += "\n"

    # Resource usage
    if pod_resources:
        report += "Resource Usage:\n"
        report += "--------------\n"

        for resource in pod_resources:
            report += f"  {resource['name']}: CPU {resource['cpu']}, Memory {resource['memory']}\n"

        report += "\n"

    # Scheduling analysis
    if pod_status and node_status:
        report += "Scheduling Analysis:\n"
        report += "-------------------\n"

        # Count pods by type and node
        pod_types = {}
        for pod in pod_status:
            pod_name = pod.get('name', '')
            node = pod.get('node', '<none>')

            if 'mobilenetv4-triton-deployment' in pod_name:
                pod_type = 'gpu'
            elif 'mobilenetv4-triton-cpu-deployment' in pod_name:
                pod_type = 'cpu'
            elif 'memory-intensive-deployment' in pod_name:
                pod_type = 'memory'
            else:
                pod_type = 'other'

            if pod_type not in pod_types:
                pod_types[pod_type] = {'total': 0, 'nodes': {}}

            pod_types[pod_type]['total'] += 1

            if node not in pod_types[pod_type]['nodes']:
                pod_types[pod_type]['nodes'][node] = 0

            pod_types[pod_type]['nodes'][node] += 1

        # Report on pod distribution by type and node
        for pod_type, data in pod_types.items():
            report += f"  {pod_type.upper()} Pods ({data['total']} total):\n"
            for node, count in data['nodes'].items():
                report += f"    - {node}: {count} pods\n"

        report += "\n"

    return report

def generate_visualizations(metrics_dir, output_dir):
    """Generate visualizations of pod distribution and resource usage."""
    # Parse the collected metrics
    pod_status = parse_pod_status(os.path.join(metrics_dir, "pod_status.txt"))
    node_status = parse_node_status(os.path.join(metrics_dir, "node_status.txt"))
    pod_resources = parse_pod_resource_usage(os.path.join(metrics_dir, "pod_resource_usage.txt"))
    pod_distribution = parse_pod_distribution(os.path.join(metrics_dir, "pod_distribution_summary.txt"))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate pod distribution visualization
    if pod_distribution and 'nodes' in pod_distribution and pod_distribution['nodes']:
        plt.figure(figsize=(10, 6))
        nodes = list(pod_distribution['nodes'].keys())
        counts = list(pod_distribution['nodes'].values())

        # Filter out nodes with no pods and simplify node names
        filtered_nodes = []
        filtered_counts = []
        node_counter = 1
        for node, count in zip(nodes, counts):
            if count > 0 and node != '<none>':
                # Simplify node name to node1, node2, etc.
                simplified_name = f'node{node_counter}'
                filtered_nodes.append(simplified_name)
                filtered_counts.append(count)
                node_counter += 1

        if filtered_nodes:
            plt.bar(filtered_nodes, filtered_counts, color='skyblue')
            plt.xlabel('Node')
            plt.ylabel('Number of Pods')
            plt.title('Pod Distribution Across Nodes')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, "pod_distribution.png"))
            plt.close()
        else:
            # Create a placeholder chart if no valid data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No pod distribution data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('Pod Distribution Across Nodes')
            plt.savefig(os.path.join(output_dir, "pod_distribution.png"))
            plt.close()
    else:
        # Create a placeholder chart if no data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No pod distribution data available',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('Pod Distribution Across Nodes')
        plt.savefig(os.path.join(output_dir, "pod_distribution.png"))
        plt.close()

    # Generate pod type distribution visualization
    if pod_distribution and 'types' in pod_distribution and pod_distribution['types']:
        plt.figure(figsize=(8, 6))
        types = list(pod_distribution['types'].keys())
        counts = list(pod_distribution['types'].values())

        # Filter out types with no pods
        filtered_types = []
        filtered_counts = []
        colors_map = {'gpu': 'green', 'cpu': 'blue', 'memory': 'red', 'other': 'gray'}
        colors = []

        for pod_type, count in zip(types, counts):
            if count > 0:
                filtered_types.append(pod_type.upper())
                filtered_counts.append(count)
                colors.append(colors_map.get(pod_type, 'gray'))

        if filtered_types:
            plt.bar(filtered_types, filtered_counts, color=colors)
            plt.xlabel('Pod Type')
            plt.ylabel('Number of Pods')
            plt.title('Pod Distribution by Type')
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, "pod_type_distribution.png"))
            plt.close()
        else:
            # Create a placeholder chart if no valid data
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No pod type data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('Pod Distribution by Type')
            plt.savefig(os.path.join(output_dir, "pod_type_distribution.png"))
            plt.close()
    else:
        # Create a placeholder chart if no data
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No pod type data available',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('Pod Distribution by Type')
        plt.savefig(os.path.join(output_dir, "pod_type_distribution.png"))
        plt.close()

    # Generate resource usage visualization
    if pod_resources:
        plt.figure(figsize=(12, 6))

        # Extract CPU usage and create simplified names
        names = []
        cpu_values = []
        colors = []

        gpu_count = 1
        cpu_count = 1
        memory_count = 1
        other_count = 1

        for r in pod_resources:
            original_name = r['name']
            cpu = r['cpu']

            # Create simplified name based on pod type
            if 'mobilenetv4-triton-deployment' in original_name and 'cpu' not in original_name:
                simplified_name = f'GPU-Pod-{gpu_count}'
                colors.append('green')
                gpu_count += 1
            elif 'mobilenetv4-triton-cpu-deployment' in original_name:
                simplified_name = f'CPU-Pod-{cpu_count}'
                colors.append('blue')
                cpu_count += 1
            elif 'memory-intensive-deployment' in original_name:
                simplified_name = f'Memory-Pod-{memory_count}'
                colors.append('red')
                memory_count += 1
            else:
                simplified_name = f'Other-Pod-{other_count}'
                colors.append('gray')
                other_count += 1

            names.append(simplified_name)

            if cpu.endswith('m'):
                cpu_values.append(float(cpu[:-1]) / 1000)
            else:
                cpu_values.append(float(cpu))

        # Sort by CPU usage
        sorted_indices = np.argsort(cpu_values)[::-1]
        sorted_names = [names[i] for i in sorted_indices]
        sorted_cpu = [cpu_values[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]

        plt.bar(sorted_names, sorted_cpu, color=sorted_colors)
        plt.xlabel('Pod')
        plt.ylabel('CPU Usage (cores)')
        plt.title('CPU Usage by Pod')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "cpu_usage.png"))
        plt.close()

        # Generate memory usage visualization
        plt.figure(figsize=(12, 6))

        # Extract memory usage using the same simplified names
        memory_values = []
        for r in pod_resources:
            memory = r['memory']
            if memory.endswith('Mi'):
                memory_values.append(float(memory[:-2]))
            elif memory.endswith('Gi'):
                memory_values.append(float(memory[:-2]) * 1024)
            else:
                memory_values.append(float(memory))

        # Sort by memory usage
        sorted_indices = np.argsort(memory_values)[::-1]
        sorted_names_mem = [names[i] for i in sorted_indices]
        sorted_memory = [memory_values[i] for i in sorted_indices]
        sorted_colors_mem = [colors[i] for i in sorted_indices]

        plt.bar(sorted_names_mem, sorted_memory, color=sorted_colors_mem)
        plt.xlabel('Pod')
        plt.ylabel('Memory Usage (MiB)')
        plt.title('Memory Usage by Pod')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "memory_usage.png"))
        plt.close()

    # Generate pod distribution by type and node
    if pod_status:
        # Count pods by type and node
        pod_types = {}
        for pod in pod_status:
            pod_name = pod.get('name', '')
            node = pod.get('node', '<none>')

            if 'mobilenetv4-triton-deployment' in pod_name:
                pod_type = 'gpu'
            elif 'mobilenetv4-triton-cpu-deployment' in pod_name:
                pod_type = 'cpu'
            elif 'memory-intensive-deployment' in pod_name:
                pod_type = 'memory'
            else:
                pod_type = 'other'

            if pod_type not in pod_types:
                pod_types[pod_type] = {'total': 0, 'nodes': {}}

            pod_types[pod_type]['total'] += 1

            if node not in pod_types[pod_type]['nodes']:
                pod_types[pod_type]['nodes'][node] = 0

            pod_types[pod_type]['nodes'][node] += 1

        # Generate stacked bar chart
        plt.figure(figsize=(10, 6))

        nodes = set()
        for pod_type, data in pod_types.items():
            for node in data['nodes'].keys():
                nodes.add(node)

        nodes = sorted(list(nodes))
        pod_type_names = sorted(list(pod_types.keys()))

        # Create a matrix of pod counts by type and node
        data_matrix = np.zeros((len(pod_type_names), len(nodes)))
        for i, pod_type in enumerate(pod_type_names):
            for j, node in enumerate(nodes):
                if node in pod_types[pod_type]['nodes']:
                    data_matrix[i, j] = pod_types[pod_type]['nodes'][node]

        # Create stacked bar chart
        bottom = np.zeros(len(nodes))
        colors = {'gpu': 'green', 'cpu': 'blue', 'memory': 'red', 'other': 'gray'}

        for i, pod_type in enumerate(pod_type_names):
            plt.bar(nodes, data_matrix[i], bottom=bottom, label=pod_type.upper(), color=colors.get(pod_type, 'gray'))
            bottom += data_matrix[i]

        plt.xlabel('Node')
        plt.ylabel('Number of Pods')
        plt.title('Pod Distribution by Type and Node')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "pod_distribution_by_type_and_node.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze Kubernetes scheduling decisions.')
    parser.add_argument('--metrics-dir', required=True, help='Directory containing collected metrics')
    parser.add_argument('--output-dir', required=True, help='Directory to save analysis results')

    args = parser.parse_args()

    # Analyze scheduling decisions
    report = analyze_scheduling(args.metrics_dir)

    # Save the report
    with open(os.path.join(args.output_dir, "scheduling_analysis.txt"), 'w') as f:
        f.write(report)

    # Generate visualizations
    generate_visualizations(args.metrics_dir, args.output_dir)

    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
