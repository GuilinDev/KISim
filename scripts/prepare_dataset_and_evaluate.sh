#!/bin/bash
# Prepare dataset and run accuracy evaluation

set -e

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Determine project root directory
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "Project root determined as: $PROJECT_ROOT"

# Create data directory if it doesn't exist
DATA_DIR="$PROJECT_ROOT/data/tiny-imagenet"
mkdir -p "$DATA_DIR"

# Download Tiny ImageNet dataset if not already downloaded
if [ ! -d "$DATA_DIR/val" ]; then
    echo "Downloading and preparing Tiny ImageNet dataset..."
    python "$PROJECT_ROOT/experiments/scripts/download_tiny_imagenet.py" --output-dir "$DATA_DIR"
else
    echo "Tiny ImageNet dataset already exists at $DATA_DIR"
fi

# Create namespace if it doesn't exist
$KUBECTL get namespace workloads >/dev/null 2>&1 || $KUBECTL create namespace workloads
echo "Ensured namespace 'workloads' exists."

# Create PVC for dataset
echo "Creating PVC for Tiny ImageNet dataset..."
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tiny-imagenet-pvc
  namespace: workloads
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: microk8s-hostpath
  resources:
    requests:
      storage: 1Gi
EOF

# Note: With WaitForFirstConsumer binding mode, PVC will only be bound when a pod uses it
echo "PVC 'tiny-imagenet-pvc' created. It will be bound when a pod uses it."

# Create temporary pod to copy dataset to PVC
echo "Creating pod to copy dataset to PVC..."
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: dataset-copy-pod
  namespace: workloads
spec:
  containers:
  - name: copy-container
    image: busybox
    command: ["/bin/sh", "-c", "mkdir -p /data/tiny-imagenet && echo 'Waiting for dataset to be copied...'; sleep 3600"]
    volumeMounts:
    - name: dataset-pvc
      mountPath: /data
  volumes:
  - name: dataset-pvc
    persistentVolumeClaim:
      claimName: tiny-imagenet-pvc
EOF

# Wait for pod to be ready
echo "Waiting for dataset-copy-pod to be ready..."
$KUBECTL wait --for=condition=Ready pod/dataset-copy-pod -n workloads --timeout=60s

# Copy dataset to PVC
echo "Copying dataset to PVC..."
# First, create a tar file of the dataset
echo "Creating tar file of the dataset..."
tar -cf /tmp/tiny-imagenet-val.tar -C "$DATA_DIR" val
# Then, copy the tar file to the pod
echo "Copying tar file to pod..."
$KUBECTL cp /tmp/tiny-imagenet-val.tar workloads/dataset-copy-pod:/data/tiny-imagenet-val.tar
# Finally, extract the tar file in the pod
echo "Extracting tar file in pod..."
$KUBECTL exec -n workloads dataset-copy-pod -- sh -c "cd /data && tar -xf tiny-imagenet-val.tar && rm tiny-imagenet-val.tar"

# Verify dataset was copied
echo "Verifying dataset was copied..."
$KUBECTL exec -n workloads dataset-copy-pod -- ls -la /data/tiny-imagenet/val

# Delete dataset copy pod
echo "Deleting dataset-copy-pod..."
$KUBECTL delete pod dataset-copy-pod -n workloads

# Create ConfigMap with evaluation script
echo "Creating ConfigMap with evaluation script..."
$KUBECTL create configmap accuracy-evaluation-script \
  --from-file=evaluate_accuracy.py="$PROJECT_ROOT/experiments/scripts/evaluate_accuracy.py" \
  -n workloads --dry-run=client -o yaml | $KUBECTL apply -f -

# Apply accuracy evaluation job
echo "Deploying accuracy evaluation job..."
$KUBECTL apply -f "$PROJECT_ROOT/experiments/scripts/accuracy-evaluation-job.yaml"

# Wait for job to complete
echo "Waiting for accuracy evaluation job to complete..."
echo "This may take some time depending on the dataset size..."
$KUBECTL wait --for=condition=complete job/mobilenetv4-accuracy-evaluation -n workloads --timeout=1800s || {
  echo "Job did not complete within timeout. Checking logs..."
  $KUBECTL logs -n workloads job/mobilenetv4-accuracy-evaluation
  # Don't exit with error, try to get logs anyway
}

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/results/baseline"
mkdir -p "$RESULTS_DIR"

# Copy results from PVC to local directory
echo "Copying results from PVC to local directory..."
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
    image: busybox
    command: ["/bin/sh", "-c", "echo 'Waiting for results to be copied...'; sleep 3600"]
    volumeMounts:
    - name: results-pvc
      mountPath: /results
  volumes:
  - name: results-pvc
    persistentVolumeClaim:
      claimName: accuracy-results-pvc
EOF

# Wait for pod to be ready
echo "Waiting for results-copy-pod to be ready..."
$KUBECTL wait --for=condition=Ready pod/results-copy-pod -n workloads --timeout=60s

# Copy results to local directory
echo "Copying results to local directory..."
$KUBECTL cp workloads/results-copy-pod:/results/accuracy_results.json "$RESULTS_DIR/accuracy_results.json"

# Delete results copy pod
echo "Deleting results-copy-pod..."
$KUBECTL delete pod results-copy-pod -n workloads

echo "Accuracy evaluation complete. Results saved to $RESULTS_DIR/accuracy_results.json"

# Print summary of results
echo "Summary of results:"
cat "$RESULTS_DIR/accuracy_results.json" | grep -E "overall_accuracy|correct_count|total_count|elapsed_time|images_per_second"
