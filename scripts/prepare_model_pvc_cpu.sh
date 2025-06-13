#!/bin/bash

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Determine the absolute path to the project's root directory
# This script is in experiments/scripts, so ../.. should be the project root.
PROJECT_ROOT_REL_TO_SCRIPT="../.."
# Convert to absolute path
PROJECT_ROOT_ABS="$(cd "$(dirname "$0")/$PROJECT_ROOT_REL_TO_SCRIPT" && pwd)"

LOCAL_MODEL_PATH="$PROJECT_ROOT_ABS/models/mobilenetv4/1/model.onnx"
TRITON_DEPLOYMENT_YAML_PATH="$(dirname "$0")/mobilenetv4-triton-cpu-deployment.yaml" # Current directory (experiments/scripts)
CONFIGMAP_NAME="mobilenetv4-cpu-config-pbtxt-cm"
NAMESPACE="workloads"

echo "Project root determined as: $PROJECT_ROOT_ABS"
echo "Expecting local ONNX model at: $LOCAL_MODEL_PATH"
echo "Expecting Triton deployment YAML (with PVC) at: $TRITON_DEPLOYMENT_YAML_PATH"


# Create a temporary directory for model preparation on the host
TEMP_DIR_HOST=$(mktemp -d) # e.g., /tmp/tmp.XXXXXXXX
MODEL_DIR_IN_TEMP_HOST="$TEMP_DIR_HOST/mobilenetv4/1" # Path on host inside TEMP_DIR_HOST
mkdir -p "$MODEL_DIR_IN_TEMP_HOST"
echo "Temporary host directory for model staging: $TEMP_DIR_HOST"


if [ -f "$LOCAL_MODEL_PATH" ]; then
    echo "Local ONNX model found. Copying to temporary host location: $MODEL_DIR_IN_TEMP_HOST/model.onnx"
    cp "$LOCAL_MODEL_PATH" "$MODEL_DIR_IN_TEMP_HOST/model.onnx"

    # Create CPU-optimized config.pbtxt for Triton
    echo "Creating CPU-optimized config.pbtxt for Triton..."
    cat > "$MODEL_DIR_IN_TEMP_HOST/config.pbtxt" << 'EOF'
name: "mobilenetv4"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 100
}
EOF
else
    echo "ERROR: Local ONNX model not found at $LOCAL_MODEL_PATH!"
    echo "Please ensure the model is downloaded first (e.g., via 'make download-hf-model')."
    rm -rf "$TEMP_DIR_HOST" # Clean up temp dir
    exit 1 # Exit if model is not found, as further steps will fail
fi

# Create namespace if it doesn't exist
$KUBECTL get ns $NAMESPACE > /dev/null 2>&1 || $KUBECTL create namespace $NAMESPACE
echo "Ensured namespace '$NAMESPACE' exists."

# Delete the ConfigMap if it exists, to ensure a fresh apply
echo "Attempting to delete existing ConfigMap '$CONFIGMAP_NAME' in namespace '$NAMESPACE' to ensure fresh apply..."
$KUBECTL delete configmap $CONFIGMAP_NAME -n $NAMESPACE --ignore-not-found=true

# Apply the Triton deployment YAML which should contain the PVC definition and the new ConfigMap.
echo "Applying Triton deployment manifest to create/ensure PVC and ConfigMap exists: $TRITON_DEPLOYMENT_YAML_PATH"
if [ ! -f "$TRITON_DEPLOYMENT_YAML_PATH" ]; then
    echo "ERROR: Triton deployment YAML not found at $TRITON_DEPLOYMENT_YAML_PATH"
    rm -rf "$TEMP_DIR_HOST"
    exit 1
fi
$KUBECTL apply -f "$TRITON_DEPLOYMENT_YAML_PATH"

# Wait for the PVC to be bound
PVC_NAME="mobilenetv4-cpu-model-pvc"
echo "Waiting for PVC '$PVC_NAME' in namespace '$NAMESPACE' to be Bound..."
timeout=120 # seconds
endtime=$(( $(date +%s) + timeout ))
pvc_status=""
while [ $(date +%s) -lt $endtime ]; do
    pvc_status=$($KUBECTL get pvc "$PVC_NAME" -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$pvc_status" == "Bound" ]; then
        echo "PVC '$PVC_NAME' is Bound."
        break
    fi
    echo "PVC '$PVC_NAME' status: $pvc_status (waiting...)"
    sleep 5
done

if [ "$pvc_status" != "Bound" ]; then
    echo "ERROR: PVC '$PVC_NAME' did not become Bound within $timeout seconds. Status: $pvc_status"
    $KUBECTL describe pvc "$PVC_NAME" -n $NAMESPACE
    rm -rf "$TEMP_DIR_HOST"
    exit 1
fi

# Now create the model-copy-pod, as PVC should be ready.
# The temporary model files are in TEMP_DIR_HOST on the node where this script runs.
# We mount TEMP_DIR_HOST into the copy-pod using hostPath.
echo "Creating model-copy-pod to transfer model from host path '$TEMP_DIR_HOST' to PVC '$PVC_NAME'..."
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-copy-pod
  namespace: $NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: copy-container
    image: busybox
    # Command to copy from the hostPath mount to the PVC mount
    # The source is /host_temp_model_files (mounted from TEMP_DIR_HOST)
    # The destination is /pvc_mount (mounted from PVC_NAME)
    command: ["/bin/sh", "-c", "echo 'Copying model files from host to PVC...'; mkdir -p /pvc_mount/mobilenetv4/1 && cp /host_temp_model_files/mobilenetv4/1/model.onnx /pvc_mount/mobilenetv4/1/model.onnx && cp /host_temp_model_files/mobilenetv4/1/config.pbtxt /pvc_mount/mobilenetv4/1/config.pbtxt && echo 'Copy complete. Verifying target on PVC...' && ls -lR /pvc_mount && echo 'Sleeping for a bit to allow volume to sync...' && sleep 5"]
    volumeMounts:
    - name: model-storage-on-pvc # PVC mount
      mountPath: /pvc_mount
    - name: model-storage-on-host # hostPath mount (where the model was temp copied)
      mountPath: /host_temp_model_files
  volumes:
  - name: model-storage-on-pvc
    persistentVolumeClaim:
      claimName: $PVC_NAME
  - name: model-storage-on-host
    hostPath:
      path: $TEMP_DIR_HOST # This is the critical part: path on the Kubernetes node
      type: DirectoryOrCreate # Ensure it's a directory
EOF

echo "Waiting for model-copy-pod to complete..."
if $KUBECTL wait --for=condition=Succeeded pod/model-copy-pod -n $NAMESPACE --timeout=180s; then
    echo "model-copy-pod completed successfully."
    echo "Logs from model-copy-pod:"
    $KUBECTL logs model-copy-pod -n $NAMESPACE
else
    echo "ERROR: model-copy-pod did not succeed within timeout."
    echo "Current status of model-copy-pod:"
    $KUBECTL describe pod model-copy-pod -n $NAMESPACE
    echo "Logs from model-copy-pod (if any):"
    $KUBECTL logs model-copy-pod -n $NAMESPACE
    # Attempt to clean up on failure, but the primary issue needs to be resolved.
fi

# Clean up the pod
echo "Deleting model-copy-pod..."
$KUBECTL delete pod model-copy-pod -n $NAMESPACE --ignore-not-found=true

# Clean up the temporary directory from host
echo "Cleaning up temporary host directory: $TEMP_DIR_HOST"
rm -rf "$TEMP_DIR_HOST"

echo "Model files preparation for PVC process complete."