#!/bin/bash
# Run accuracy evaluation for MobileNetV4 model

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Determine the absolute path to the project's root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/experiments/scripts"
RESULTS_DIR="$PROJECT_ROOT/results/accuracy"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run accuracy evaluation for MobileNetV4 model"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL       Model to evaluate (gpu, cpu, both) (default: gpu)"
    echo "  -n, --num-samples NUM   Number of samples to evaluate (default: 1000)"
    echo "  -d, --dataset PATH      Path to dataset (default: /home/guilin/allProjects/ecrl/data/tiny-imagenet/tiny-imagenet-200)"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --model both --num-samples 500"
    exit 1
}

# Default values
MODEL="gpu"
NUM_SAMPLES=1000
DATASET="/home/guilin/allProjects/ecrl/data/tiny-imagenet/tiny-imagenet-200"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
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

# Validate model parameter
if [[ "$MODEL" != "gpu" && "$MODEL" != "cpu" && "$MODEL" != "both" ]]; then
    echo "Error: Invalid model '$MODEL'"
    usage
fi

# Check if dataset exists
if [ ! -d "$DATASET" ]; then
    echo "Error: Dataset directory not found: $DATASET"
    echo "Downloading dataset..."
    python3 "$SCRIPTS_DIR/download_tiny_imagenet.py" --output-dir "$(dirname "$DATASET")"
    if [ ! -d "$DATASET" ]; then
        echo "Error: Failed to download dataset"
        exit 1
    fi
fi

# Check if validation directory exists
VAL_DIR="$DATASET/val"
if [ ! -d "$VAL_DIR" ]; then
    echo "Error: Validation directory not found: $VAL_DIR"
    exit 1
fi

# Create PVC for dataset
echo "Creating PVC for dataset..."
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tiny-imagenet-pvc
  namespace: workloads
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

# Create PVC for results
echo "Creating PVC for results..."
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: accuracy-results-pvc
  namespace: workloads
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF

# Copy dataset to PVC
echo "Copying dataset to PVC..."
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: dataset-copy-pod
  namespace: workloads
spec:
  containers:
  - name: copy-container
    image: busybox:latest
    command: ["/bin/sh", "-c"]
    args:
    - |
      echo "Copying dataset to PVC..."
      mkdir -p /data/tiny-imagenet
      cp -r /host-data/val /data/tiny-imagenet/
      echo "Verifying copied files..."
      ls -la /data/tiny-imagenet/val/
      echo "Copy complete."
      sleep 5
    volumeMounts:
    - name: host-data
      mountPath: /host-data
    - name: pvc-data
      mountPath: /data
  volumes:
  - name: host-data
    hostPath:
      path: $DATASET
  - name: pvc-data
    persistentVolumeClaim:
      claimName: tiny-imagenet-pvc
  restartPolicy: Never
EOF

# Wait for dataset copy to complete
echo "Waiting for dataset copy to complete..."
$KUBECTL wait --for=condition=Ready pod/dataset-copy-pod -n workloads --timeout=300s || true
$KUBECTL logs -n workloads dataset-copy-pod
$KUBECTL delete pod dataset-copy-pod -n workloads

# Create ConfigMap with evaluation script
echo "Creating ConfigMap with evaluation script..."
$KUBECTL create configmap accuracy-evaluation-script \
  --from-file=evaluate_accuracy.py="$SCRIPTS_DIR/evaluate_accuracy.py" \
  -n workloads --dry-run=client -o yaml | $KUBECTL apply -f -

# Function to run evaluation for a specific model
run_evaluation() {
    local model_type=$1
    local service_name="mobilenetv4-triton-svc"
    local result_file="gpu_accuracy_results.json"
    
    if [ "$model_type" = "cpu" ]; then
        service_name="mobilenetv4-triton-cpu-svc"
        result_file="cpu_accuracy_results.json"
    fi
    
    echo "Running accuracy evaluation for $model_type model..."
    
    # Check if service exists
    if ! $KUBECTL get svc -n workloads $service_name &>/dev/null; then
        echo "Error: Service $service_name not found"
        echo "Please deploy the $model_type model first"
        return 1
    fi
    
    # Create and run evaluation job
    cat <<EOF | $KUBECTL apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: mobilenetv4-${model_type}-accuracy-evaluation
  namespace: workloads
spec:
  template:
    spec:
      containers:
      - name: accuracy-evaluator
        image: python:3.9-slim
        command:
        - "/bin/bash"
        - "-c"
        - |
          apt-get update && apt-get install -y wget unzip && \
          pip install numpy pillow tqdm tritonclient[http] requests && \
          
          # Run evaluation
          echo "Running evaluation..." && \
          python /scripts/evaluate_accuracy.py \
            --server-url ${service_name}.workloads.svc.cluster.local:8000 \
            --model-name mobilenetv4 \
            --dataset-path /data/tiny-imagenet/val \
            --output-file /results/${result_file} \
            --num-samples ${NUM_SAMPLES}
        volumeMounts:
        - name: evaluation-script
          mountPath: /scripts
        - name: dataset
          mountPath: /data
        - name: results
          mountPath: /results
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: evaluation-script
        configMap:
          name: accuracy-evaluation-script
      - name: dataset
        persistentVolumeClaim:
          claimName: tiny-imagenet-pvc
      - name: results
        persistentVolumeClaim:
          claimName: accuracy-results-pvc
      restartPolicy: Never
  backoffLimit: 2
EOF
    
    # Wait for job to complete
    echo "Waiting for evaluation job to complete..."
    $KUBECTL wait --for=condition=complete job/mobilenetv4-${model_type}-accuracy-evaluation -n workloads --timeout=1800s || {
        echo "Job did not complete within timeout. Checking logs..."
        $KUBECTL logs -n workloads job/mobilenetv4-${model_type}-accuracy-evaluation
        return 1
    }
    
    # Get logs
    echo "Job completed. Getting logs..."
    $KUBECTL logs -n workloads job/mobilenetv4-${model_type}-accuracy-evaluation
    
    # Copy results from PVC
    echo "Copying results from PVC..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_DIR="$RESULTS_DIR/${model_type}_${TIMESTAMP}"
    mkdir -p "$RESULT_DIR"
    
    # Create temporary pod to access results
    cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: results-copy-pod
  namespace: workloads
spec:
  containers:
  - name: copy-container
    image: busybox:latest
    command: ["/bin/sh", "-c", "echo 'Waiting for results to be copied...'; sleep 3600"]
    volumeMounts:
    - name: results-pvc
      mountPath: /results
  volumes:
  - name: results-pvc
    persistentVolumeClaim:
      claimName: accuracy-results-pvc
  restartPolicy: Never
EOF
    
    # Wait for pod to be ready
    echo "Waiting for results-copy-pod to be ready..."
    $KUBECTL wait --for=condition=Ready pod/results-copy-pod -n workloads --timeout=60s
    
    # Copy results to local directory
    echo "Copying results to local directory..."
    $KUBECTL cp workloads/results-copy-pod:/results/${result_file} "$RESULT_DIR/accuracy_results.json"
    
    # Delete results copy pod
    echo "Deleting results-copy-pod..."
    $KUBECTL delete pod results-copy-pod -n workloads
    
    echo "Accuracy evaluation for $model_type model complete. Results saved to $RESULT_DIR/accuracy_results.json"
    
    # Print summary of results
    echo "Summary of results:"
    cat "$RESULT_DIR/accuracy_results.json" | grep -E "overall_accuracy|correct_count|total_count|elapsed_time|images_per_second"
    
    # Clean up job
    echo "Cleaning up job..."
    $KUBECTL delete job mobilenetv4-${model_type}-accuracy-evaluation -n workloads
}

# Run evaluation for selected model(s)
if [ "$MODEL" = "gpu" ] || [ "$MODEL" = "both" ]; then
    run_evaluation "gpu"
fi

if [ "$MODEL" = "cpu" ] || [ "$MODEL" = "both" ]; then
    run_evaluation "cpu"
fi

# If both models were evaluated, generate comparison report
if [ "$MODEL" = "both" ]; then
    echo "Generating comparison report..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    COMPARISON_DIR="$RESULTS_DIR/comparison_${TIMESTAMP}"
    mkdir -p "$COMPARISON_DIR"
    
    # Find the most recent GPU and CPU result files
    GPU_RESULT=$(find "$RESULTS_DIR" -name "gpu_*" -type d | sort | tail -n 1)/accuracy_results.json
    CPU_RESULT=$(find "$RESULTS_DIR" -name "cpu_*" -type d | sort | tail -n 1)/accuracy_results.json
    
    if [ -f "$GPU_RESULT" ] && [ -f "$CPU_RESULT" ]; then
        # Generate comparison report
        python3 - <<EOF > "$COMPARISON_DIR/comparison_report.txt"
import json

# Load results
with open("$GPU_RESULT", "r") as f:
    gpu_results = json.load(f)

with open("$CPU_RESULT", "r") as f:
    cpu_results = json.load(f)

# Extract metrics
gpu_accuracy = gpu_results["overall_accuracy"]
gpu_correct = gpu_results["correct_count"]
gpu_total = gpu_results["total_count"]
gpu_throughput = gpu_results["images_per_second"]
gpu_time = gpu_results["elapsed_time"]

cpu_accuracy = cpu_results["overall_accuracy"]
cpu_correct = cpu_results["correct_count"]
cpu_total = cpu_results["total_count"]
cpu_throughput = cpu_results["images_per_second"]
cpu_time = cpu_results["elapsed_time"]

# Calculate speedup
throughput_speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else float('inf')
time_speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

# Generate report
print("GPU vs CPU Accuracy and Performance Comparison")
print("=============================================")
print(f"GPU Accuracy: {gpu_accuracy:.4f} ({gpu_correct}/{gpu_total})")
print(f"CPU Accuracy: {cpu_accuracy:.4f} ({cpu_correct}/{cpu_total})")
print(f"Accuracy Difference: {(gpu_accuracy - cpu_accuracy)*100:.2f}%")
print()
print(f"GPU Throughput: {gpu_throughput:.2f} images/sec")
print(f"CPU Throughput: {cpu_throughput:.2f} images/sec")
print(f"Throughput Speedup (GPU/CPU): {throughput_speedup:.2f}x")
print()
print(f"GPU Evaluation Time: {gpu_time:.2f} seconds")
print(f"CPU Evaluation Time: {cpu_time:.2f} seconds")
print(f"Time Speedup (CPU/GPU): {time_speedup:.2f}x")
EOF
        
        echo "Comparison report generated at $COMPARISON_DIR/comparison_report.txt"
        cat "$COMPARISON_DIR/comparison_report.txt"
    else
        echo "Error: Could not find both GPU and CPU result files for comparison"
    fi
fi

echo "Accuracy evaluation completed!"
