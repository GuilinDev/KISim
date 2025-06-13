#!/bin/bash
# Deploy and test CPU-only version of Triton server

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Determine the absolute path to the project's root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/experiments/scripts"
RESULTS_DIR="$PROJECT_ROOT/results/cpu_baseline"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Deploy and test CPU-only version of Triton server"
    echo ""
    echo "Options:"
    echo "  -u, --users COUNT       Number of users for load test (default: 10)"
    echo "  -d, --duration SECONDS  Test duration in seconds (default: 300)"
    echo "  -c, --clean             Clean up existing CPU deployment before starting"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --users 10 --duration 300"
    exit 1
}

# Default values
USERS=10
DURATION=300
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -u|--users)
            USERS="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
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

# Clean up existing CPU deployment if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning up existing CPU deployment..."
    $KUBECTL delete deployment mobilenetv4-triton-cpu-deployment -n workloads --ignore-not-found=true
    $KUBECTL delete service mobilenetv4-triton-cpu-svc -n workloads --ignore-not-found=true
    $KUBECTL delete configmap mobilenetv4-config-cpu-pbtxt-cm -n workloads --ignore-not-found=true
    $KUBECTL delete pvc mobilenetv4-model-cpu-pvc -n workloads --ignore-not-found=true
fi

# Create namespace if it doesn't exist
$KUBECTL create namespace workloads --dry-run=client -o yaml | $KUBECTL apply -f -

# Apply CPU Triton deployment
echo "Deploying CPU-only Triton server..."
$KUBECTL apply -f "$SCRIPTS_DIR/mobilenetv4-triton-cpu-deployment.yaml"

# Wait for Triton to be ready
echo "Waiting for CPU Triton server to be ready..."
$KUBECTL wait --for=condition=available deployment/mobilenetv4-triton-cpu-deployment -n workloads --timeout=300s || {
    echo "Error: CPU Triton server deployment failed or timed out"
    echo "Checking pod status..."
    $KUBECTL get pods -n workloads -l app=mobilenetv4-triton-cpu
    echo "Checking pod logs..."
    POD=$($KUBECTL get pods -n workloads -l app=mobilenetv4-triton-cpu -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$POD" ]; then
        $KUBECTL logs -n workloads $POD
    fi
    exit 1
}

# Copy model files from GPU PVC to CPU PVC
echo "Copying model files from GPU PVC to CPU PVC..."
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-copy-pod
  namespace: workloads
spec:
  containers:
  - name: copy-container
    image: busybox:latest
    command: ["/bin/sh", "-c"]
    args:
    - |
      echo "Copying model files from GPU PVC to CPU PVC..."
      mkdir -p /cpu_pvc/mobilenetv4/1
      cp /gpu_pvc/mobilenetv4/1/model.onnx /cpu_pvc/mobilenetv4/1/
      echo "Verifying copied files..."
      ls -la /cpu_pvc/mobilenetv4/1/
      echo "Copy complete."
      sleep 5
    volumeMounts:
    - name: gpu-pvc
      mountPath: /gpu_pvc
    - name: cpu-pvc
      mountPath: /cpu_pvc
  volumes:
  - name: gpu-pvc
    persistentVolumeClaim:
      claimName: mobilenetv4-model-pvc
  - name: cpu-pvc
    persistentVolumeClaim:
      claimName: mobilenetv4-model-cpu-pvc
  restartPolicy: Never
EOF

# Wait for model copy to complete
echo "Waiting for model copy to complete..."
$KUBECTL wait --for=condition=complete job/model-copy-pod -n workloads --timeout=60s || true
$KUBECTL logs -n workloads model-copy-pod
$KUBECTL delete pod model-copy-pod -n workloads

# Restart CPU Triton server to pick up the model
echo "Restarting CPU Triton server..."
$KUBECTL rollout restart deployment/mobilenetv4-triton-cpu-deployment -n workloads
$KUBECTL rollout status deployment/mobilenetv4-triton-cpu-deployment -n workloads --timeout=300s

# Run load test against CPU Triton server
echo "Running load test against CPU Triton server..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$RESULTS_DIR/cpu_baseline_$TIMESTAMP"
mkdir -p "$RESULT_DIR"

# Start port forwarding to Locust web interface
echo "Starting port forwarding to Locust web interface..."
$KUBECTL port-forward -n workloads svc/locust-master 8089:8089 &
PORT_FORWARD_PID=$!

# Wait for port forwarding to be established
echo "Waiting for port forwarding to be established..."
sleep 5

# Run Locust test using API
echo "Starting Locust test with $USERS users for $DURATION seconds..."
curl -X POST http://localhost:8089/swarm \
    -d "user_count=$USERS" \
    -d "spawn_rate=$USERS" \
    -d "host=http://mobilenetv4-triton-cpu-svc.workloads.svc.cluster.local:8000"

# Wait for test duration
echo "Test running for $DURATION seconds..."
sleep $DURATION

# Stop the test
echo "Stopping test..."
curl -X GET http://localhost:8089/stop

# Get test statistics
echo "Getting test statistics..."
curl -X GET http://localhost:8089/stats/requests -o "$RESULT_DIR/locust_stats.json"

# Stop port forwarding
echo "Stopping port forwarding..."
kill $PORT_FORWARD_PID

# Collect pod status and logs
echo "Collecting pod status and logs..."
$KUBECTL get pods -n workloads -o wide > "$RESULT_DIR/pod_status.txt"
$KUBECTL logs -n workloads deployment/mobilenetv4-triton-cpu-deployment --all-containers=true > "$RESULT_DIR/triton_cpu_logs.txt" 2>/dev/null || echo "No triton logs found or error collecting."

# Run a quick synthetic test to compare CPU vs GPU performance
echo "Running synthetic test to compare CPU vs GPU performance..."
python3 "$SCRIPTS_DIR/evaluate_synthetic.py" \
    --url http://$($KUBECTL get svc -n workloads mobilenetv4-triton-cpu-svc -o jsonpath='{.spec.clusterIP}'):8000 \
    --model-name mobilenetv4 \
    --num-samples 50 \
    --output-file "$RESULT_DIR/cpu_synthetic_results.json"

# Compare with GPU if it's running
if $KUBECTL get deployment mobilenetv4-triton-deployment -n workloads &>/dev/null; then
    echo "Running synthetic test on GPU for comparison..."
    python3 "$SCRIPTS_DIR/evaluate_synthetic.py" \
        --url http://$($KUBECTL get svc -n workloads mobilenetv4-triton-svc -o jsonpath='{.spec.clusterIP}'):8000 \
        --model-name mobilenetv4 \
        --num-samples 50 \
        --output-file "$RESULT_DIR/gpu_synthetic_results.json"
    
    # Generate comparison report
    echo "Generating comparison report..."
    python3 - <<EOF > "$RESULT_DIR/comparison_report.txt"
import json

# Load results
with open("$RESULT_DIR/cpu_synthetic_results.json", "r") as f:
    cpu_results = json.load(f)

with open("$RESULT_DIR/gpu_synthetic_results.json", "r") as f:
    gpu_results = json.load(f)

# Extract metrics
cpu_latency_avg = cpu_results["latency_stats"]["mean_ms"]
cpu_latency_p95 = cpu_results["latency_stats"]["p95_ms"]
cpu_throughput = cpu_results["throughput_stats"]["images_per_second"]

gpu_latency_avg = gpu_results["latency_stats"]["mean_ms"]
gpu_latency_p95 = gpu_results["latency_stats"]["p95_ms"]
gpu_throughput = gpu_results["throughput_stats"]["images_per_second"]

# Calculate speedup
latency_speedup = cpu_latency_avg / gpu_latency_avg if gpu_latency_avg > 0 else float('inf')
throughput_speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else float('inf')

# Generate report
print("CPU vs GPU Performance Comparison")
print("=================================")
print(f"CPU Average Latency: {cpu_latency_avg:.2f} ms")
print(f"GPU Average Latency: {gpu_latency_avg:.2f} ms")
print(f"Latency Speedup (CPU/GPU): {latency_speedup:.2f}x")
print()
print(f"CPU P95 Latency: {cpu_latency_p95:.2f} ms")
print(f"GPU P95 Latency: {gpu_latency_p95:.2f} ms")
print()
print(f"CPU Throughput: {cpu_throughput:.2f} images/sec")
print(f"GPU Throughput: {gpu_throughput:.2f} images/sec")
print(f"Throughput Speedup (GPU/CPU): {throughput_speedup:.2f}x")
EOF
fi

echo "CPU experiment completed!"
echo "Results saved to $RESULT_DIR"
