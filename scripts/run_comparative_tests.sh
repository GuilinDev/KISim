#!/bin/bash
# Run comparative tests between GPU and CPU versions with the same load patterns
# and store results in a structured way for easy analysis

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Determine the absolute path to the project's root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/experiments/scripts"
RESULTS_DIR="$PROJECT_ROOT/experiments/results"
GPU_RESULTS_DIR="$RESULTS_DIR/baseline"
CPU_RESULTS_DIR="$RESULTS_DIR/cpu_baseline"

# Create results directories
mkdir -p "$GPU_RESULTS_DIR/dynamic"
mkdir -p "$CPU_RESULTS_DIR/dynamic"

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run comparative tests between GPU and CPU versions with the same load patterns"
    echo ""
    echo "Options:"
    echo "  -p, --pattern PATTERN   Load pattern to use (spike, ramp, periodic, random)"
    echo "  -d, --duration MINUTES  Test duration in minutes (default: 10)"
    echo "  -m, --min-users COUNT   Minimum number of users (default: 10)"
    echo "  -M, --max-users COUNT   Maximum number of users (default: 100)"
    echo "  -g, --gpu-only          Run tests only on GPU version"
    echo "  -c, --cpu-only          Run tests only on CPU version"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --pattern ramp --duration 10"
    exit 1
}

# Default values
PATTERN="ramp"
DURATION=10
MIN_USERS=10
MAX_USERS=100
RUN_GPU=true
RUN_CPU=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -m|--min-users)
            MIN_USERS="$2"
            shift 2
            ;;
        -M|--max-users)
            MAX_USERS="$2"
            shift 2
            ;;
        -g|--gpu-only)
            RUN_GPU=true
            RUN_CPU=false
            shift
            ;;
        -c|--cpu-only)
            RUN_GPU=false
            RUN_CPU=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate pattern
if [[ "$PATTERN" != "spike" && "$PATTERN" != "ramp" && "$PATTERN" != "periodic" && "$PATTERN" != "random" ]]; then
    echo "Error: Invalid pattern '$PATTERN'"
    usage
fi

# Check if Python dependencies are installed
check_dependencies() {
    echo "Checking Python dependencies..."
    python3 -c "import numpy, matplotlib, requests" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing required Python packages..."
        pip install numpy matplotlib requests
    fi
}

# Run a specific load pattern on a specific target
run_pattern() {
    local pattern=$1
    local duration=$2
    local target_host=$3
    local output_dir=$4
    local target_name=$5

    echo "Running $pattern load pattern for $duration minutes on $target_name..."

    # Make sure output directory exists
    mkdir -p "$output_dir/dynamic"

    # Kill any existing port forwards
    pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true
    sleep 2

    # Start port forwarding to Locust web interface
    echo "Starting port forwarding to Locust web interface..."
    $KUBECTL port-forward -n workloads svc/locust-master 8089:8089 &
    PORT_FORWARD_PID=$!

    # Wait for port forwarding to be established
    echo "Waiting for port forwarding to be established..."
    sleep 5

    # Check if port forwarding is working
    if ! curl -s http://localhost:8089 > /dev/null; then
        echo "ERROR: Port forwarding not working. Trying again..."
        kill $PORT_FORWARD_PID 2>/dev/null || true
        pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true
        sleep 2
        $KUBECTL port-forward -n workloads svc/locust-master 8089:8089 &
        PORT_FORWARD_PID=$!
        sleep 5

        if ! curl -s http://localhost:8089 > /dev/null; then
            echo "ERROR: Port forwarding still not working. Skipping test."
            kill $PORT_FORWARD_PID 2>/dev/null || true
            pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true
            return 1
        fi
    fi

    # Run the dynamic load controller
    echo "Starting dynamic load controller..."
    python3 "$SCRIPTS_DIR/dynamic_load_controller.py" \
        --pattern "$pattern" \
        --duration "$duration" \
        --min-users "$MIN_USERS" \
        --max-users "$MAX_USERS" \
        --host "http://localhost:8089" \
        --target-host "$target_host" \
        --output-dir "$output_dir/dynamic"

    # Check if results were saved
    if [ ! -d "$output_dir/dynamic" ] || [ -z "$(ls -A "$output_dir/dynamic")" ]; then
        echo "WARNING: No results were saved to $output_dir/dynamic"
    else
        echo "Results saved to $output_dir/dynamic"
        ls -la "$output_dir/dynamic"
    fi

    # Stop port forwarding
    echo "Stopping port forwarding..."
    kill $PORT_FORWARD_PID 2>/dev/null || true
    pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true

    # Wait a moment before starting the next test
    echo "Waiting for resources to settle..."
    sleep 10
}

# Collect additional metrics and logs
collect_metrics() {
    local output_dir=$1
    local deployment=$2
    local target_name=$3

    echo "Collecting metrics and logs for $target_name..."

    # Create metrics directory and ensure it exists
    mkdir -p "$output_dir/metrics"

    # Check if directory was created successfully
    if [ ! -d "$output_dir/metrics" ]; then
        echo "ERROR: Failed to create metrics directory: $output_dir/metrics"
        echo "Current directory: $(pwd)"
        echo "Trying to create directory with absolute path..."
        mkdir -p "$(realpath "$output_dir/metrics")"
    fi

    # Get pod status
    $KUBECTL get pods -n workloads -o wide > "$output_dir/metrics/pod_status.txt"

    # Get node status
    $KUBECTL get nodes -o wide > "$output_dir/metrics/node_status.txt"

    # Get detailed node information
    $KUBECTL describe nodes > "$output_dir/metrics/node_details.txt"

    # Get scheduling events
    $KUBECTL get events -n workloads --sort-by='.lastTimestamp' > "$output_dir/metrics/scheduling_events.txt"

    # Get logs from Triton server pods (collect logs from each pod separately)
    mkdir -p "$output_dir/metrics/pod_logs"

    # Get list of pods for this deployment
    local pods=$($KUBECTL get pods -n workloads -l app=$deployment -o jsonpath='{.items[*].metadata.name}')

    # Collect logs from each pod
    for pod in $pods; do
        echo "Collecting logs from pod $pod..."
        $KUBECTL logs -n workloads $pod > "$output_dir/metrics/pod_logs/${pod}.log" 2>/dev/null || echo "No logs found for pod $pod or error collecting."
    done

    # Get resource usage for all pods
    $KUBECTL top pods -n workloads > "$output_dir/metrics/pod_resource_usage.txt" 2>/dev/null || echo "Could not get resource usage (metrics-server may not be enabled)"

    # Get resource usage for nodes
    $KUBECTL top nodes > "$output_dir/metrics/node_resource_usage.txt" 2>/dev/null || echo "Could not get node resource usage (metrics-server may not be enabled)"

    # If it's the GPU deployment, get GPU metrics
    if [[ "$deployment" == "mobilenetv4-triton" ]]; then
        # Get GPU metrics if nvidia-smi is available
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi > "$output_dir/metrics/gpu_metrics.txt"
        fi
    fi

    # Run synthetic test for consistent comparison
    echo "Running synthetic test for $target_name..."
    local service_name
    if [[ "$deployment" == "mobilenetv4-triton" ]]; then
        service_name="mobilenetv4-triton-svc"
    else
        service_name="mobilenetv4-triton-cpu-svc"
    fi

    TRITON_IP=$($KUBECTL get svc -n workloads $service_name -o jsonpath='{.spec.clusterIP}')
    python3 "$SCRIPTS_DIR/evaluate_synthetic.py" \
        --url "http://$TRITON_IP:8000" \
        --model-name mobilenetv4 \
        --num-samples 100 \
        --output-file "$output_dir/metrics/synthetic_results.json"

    # Collect pod distribution information
    echo "Collecting pod distribution information..."

    # Get pod to node mapping
    $KUBECTL get pods -n workloads -o wide > "$output_dir/metrics/pod_to_node_mapping.txt"

    # Create a summary of pod distribution
    echo "Pod Distribution Summary:" > "$output_dir/metrics/pod_distribution_summary.txt"
    echo "=========================" >> "$output_dir/metrics/pod_distribution_summary.txt"
    echo "" >> "$output_dir/metrics/pod_distribution_summary.txt"

    # Count pods per node
    echo "Pods per Node:" >> "$output_dir/metrics/pod_distribution_summary.txt"
    $KUBECTL get pods -n workloads -o wide | grep -v "NAME" | awk '{print $7}' | sort | uniq -c | \
        while read count node; do
            echo "  $node: $count pods" >> "$output_dir/metrics/pod_distribution_summary.txt"
        done

    echo "" >> "$output_dir/metrics/pod_distribution_summary.txt"

    # Count pods per type (CPU/GPU/Memory)
    echo "Pods per Type:" >> "$output_dir/metrics/pod_distribution_summary.txt"

    # Count GPU pods
    GPU_COUNT=$($KUBECTL get pods -n workloads -l app=mobilenetv4-triton -o wide | grep -v "NAME" | wc -l)
    echo "  gpu: $GPU_COUNT pods" >> "$output_dir/metrics/pod_distribution_summary.txt"

    # Count CPU pods
    CPU_COUNT=$($KUBECTL get pods -n workloads -l app=mobilenetv4-triton-cpu -o wide | grep -v "NAME" | wc -l)
    echo "  cpu: $CPU_COUNT pods" >> "$output_dir/metrics/pod_distribution_summary.txt"

    # Count Memory pods
    MEMORY_COUNT=$($KUBECTL get pods -n workloads -l app=memory-intensive -o wide | grep -v "NAME" | wc -l)
    echo "  memory: $MEMORY_COUNT pods" >> "$output_dir/metrics/pod_distribution_summary.txt"

    # Collect scheduling latency information
    echo "Collecting scheduling latency information..."

    # Get pod creation timestamps and first scheduled timestamps
    echo "Pod Scheduling Latency:" > "$output_dir/metrics/scheduling_latency.txt"
    echo "======================" >> "$output_dir/metrics/scheduling_latency.txt"
    echo "" >> "$output_dir/metrics/scheduling_latency.txt"

    $KUBECTL get pods -n workloads -o wide | grep -v "NAME" | awk '{print $1}' | \
    while read pod; do
        CREATED_TIME=$($KUBECTL get pod $pod -n workloads -o jsonpath='{.metadata.creationTimestamp}')
        SCHEDULED_TIME=$($KUBECTL get pod $pod -n workloads -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].lastTransitionTime}')

        if [ ! -z "$SCHEDULED_TIME" ]; then
            echo "Pod: $pod" >> "$output_dir/metrics/scheduling_latency.txt"
            echo "  Created:   $CREATED_TIME" >> "$output_dir/metrics/scheduling_latency.txt"
            echo "  Scheduled: $SCHEDULED_TIME" >> "$output_dir/metrics/scheduling_latency.txt"
            echo "" >> "$output_dir/metrics/scheduling_latency.txt"
        fi
    done

    # Collect resource allocation information
    echo "Collecting resource allocation information..."

    # Get resource requests and limits for all pods
    echo "Pod Resource Allocation:" > "$output_dir/metrics/resource_allocation.txt"
    echo "======================" >> "$output_dir/metrics/resource_allocation.txt"
    echo "" >> "$output_dir/metrics/resource_allocation.txt"

    $KUBECTL get pods -n workloads -o wide | grep -v "NAME" | awk '{print $1}' | \
    while read pod; do
        echo "Pod: $pod" >> "$output_dir/metrics/resource_allocation.txt"

        # Get container resource requests and limits
        $KUBECTL get pod $pod -n workloads -o jsonpath='{range .spec.containers[*]}Container: {.name}{"\n"}  Requests: {.resources.requests}{"\n"}  Limits: {.resources.limits}{"\n"}{end}' >> "$output_dir/metrics/resource_allocation.txt"
        echo "" >> "$output_dir/metrics/resource_allocation.txt"
    done
}

# Generate comparison report
generate_comparison_report() {
    echo "Generating comparison report..."

    # Ensure results directory exists
    mkdir -p "$RESULTS_DIR"

    # Check if both GPU and CPU results exist
    if [[ -f "$GPU_RESULTS_DIR/metrics/synthetic_results.json" && -f "$CPU_RESULTS_DIR/metrics/synthetic_results.json" ]]; then
        python3 -c "
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('$GPU_RESULTS_DIR/metrics/synthetic_results.json', 'r') as f:
    gpu_results = json.load(f)
with open('$CPU_RESULTS_DIR/metrics/synthetic_results.json', 'r') as f:
    cpu_results = json.load(f)

# Extract metrics
gpu_latency_avg = gpu_results['latency_stats']['mean_ms']
gpu_latency_p95 = gpu_results['latency_stats']['p95_ms']
gpu_throughput = gpu_results['throughput_stats']['images_per_second']

cpu_latency_avg = cpu_results['latency_stats']['mean_ms']
cpu_latency_p95 = cpu_results['latency_stats']['p95_ms']
cpu_throughput = cpu_results['throughput_stats']['images_per_second']

# Calculate speedups
latency_speedup = cpu_latency_avg / gpu_latency_avg if gpu_latency_avg > 0 else float('inf')
throughput_speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else float('inf')

# Generate text report
report = f'CPU vs GPU Performance Comparison\\n'
report += f'=================================\\n'
report += f'CPU Average Latency: {cpu_latency_avg:.2f} ms\\n'
report += f'GPU Average Latency: {gpu_latency_avg:.2f} ms\\n'
report += f'Latency Speedup (CPU/GPU): {latency_speedup:.2f}x\\n\\n'
report += f'CPU P95 Latency: {cpu_latency_p95:.2f} ms\\n'
report += f'GPU P95 Latency: {gpu_latency_p95:.2f} ms\\n\\n'
report += f'CPU Throughput: {cpu_throughput:.2f} images/sec\\n'
report += f'GPU Throughput: {gpu_throughput:.2f} images/sec\\n'
report += f'Throughput Speedup (GPU/CPU): {throughput_speedup:.2f}x\\n'

# Save report
with open('$RESULTS_DIR/comparison_report.txt', 'w') as f:
    f.write(report)

print(report)

# Create bar charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Latency comparison
platforms = ['CPU', 'GPU']
avg_latencies = [cpu_latency_avg, gpu_latency_avg]
p95_latencies = [cpu_latency_p95, gpu_latency_p95]

x = np.arange(len(platforms))
width = 0.35

ax1.bar(x - width/2, avg_latencies, width, label='Avg Latency')
ax1.bar(x + width/2, p95_latencies, width, label='P95 Latency')
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Latency Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(platforms)
ax1.legend()

# Throughput comparison
throughputs = [cpu_throughput, gpu_throughput]
ax2.bar(platforms, throughputs)
ax2.set_ylabel('Throughput (images/sec)')
ax2.set_title('Throughput Comparison')

plt.tight_layout()
plt.savefig('$RESULTS_DIR/comparison_chart.png')
"
        echo "Comparison report saved to $RESULTS_DIR/comparison_report.txt"
        echo "Comparison chart saved to $RESULTS_DIR/comparison_chart.png"
    else
        echo "Cannot generate comparison report: missing results files"
    fi
}

# Main execution
echo "Starting comparative tests..."

# Ensure port 8089 is not in use
echo "Checking for existing port forwards on 8089..."
pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true
sleep 2

# Check dependencies
check_dependencies

# Run tests on GPU if requested
if $RUN_GPU; then
    echo "Running tests on GPU version..."

    # Clean up existing deployments
    echo "Cleaning up existing deployments..."
    cd "$PROJECT_ROOT/experiments" && make clean-baseline

    # Deploy GPU version
    echo "Deploying GPU version..."
    cd "$PROJECT_ROOT/experiments" && make prepare-model && make deploy-baseline

    # Wait for deployment to be ready
    echo "Waiting for GPU deployment to be ready..."
    sleep 10

    # Run the test
    run_pattern "$PATTERN" "$DURATION" "http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000" "$GPU_RESULTS_DIR" "GPU"

    # Collect metrics
    collect_metrics "$GPU_RESULTS_DIR" "mobilenetv4-triton" "GPU"
fi

# Run tests on CPU if requested
if $RUN_CPU; then
    echo "Running tests on CPU version..."

    # Clean up existing deployments
    echo "Cleaning up existing deployments..."
    cd "$PROJECT_ROOT/experiments" && make clean-baseline

    # Deploy CPU version
    echo "Deploying CPU version..."
    cd "$PROJECT_ROOT/experiments" && make deploy-baseline-cpu

    # Wait for deployment to be ready
    echo "Waiting for CPU deployment to be ready..."
    sleep 10

    # Run the test
    run_pattern "$PATTERN" "$DURATION" "http://mobilenetv4-triton-cpu-svc.workloads.svc.cluster.local:8000" "$CPU_RESULTS_DIR" "CPU"

    # Collect metrics
    collect_metrics "$CPU_RESULTS_DIR" "mobilenetv4-triton-cpu" "CPU"
fi

# Generate comparison report if both tests were run
if $RUN_GPU && $RUN_CPU; then
    generate_comparison_report
fi

echo "Comparative tests completed!"
echo "Results saved to:"
if $RUN_GPU; then
    echo "- GPU: $GPU_RESULTS_DIR"
fi
if $RUN_CPU; then
    echo "- CPU: $CPU_RESULTS_DIR"
fi
if $RUN_GPU && $RUN_CPU; then
    echo "- Comparison: $RESULTS_DIR/comparison_report.txt"
fi

# Final cleanup
echo "Performing final cleanup..."
pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true

echo "Done!"
