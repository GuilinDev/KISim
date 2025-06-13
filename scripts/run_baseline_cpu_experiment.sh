#!/bin/bash
# Deploy CPU-only version of Triton server for baseline experiments

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Determine the absolute path to the project's root directory
PROJECT_ROOT_REL_TO_SCRIPT="../.."
PROJECT_ROOT_ABS="$(cd "$(dirname "$0")/$PROJECT_ROOT_REL_TO_SCRIPT" && pwd)"

LOCAL_MODEL_PATH="$PROJECT_ROOT_ABS/models/mobilenetv4/1/model.onnx"
TRITON_CPU_DEPLOYMENT_YAML_PATH="$(dirname "$0")/mobilenetv4-triton-cpu-deployment.yaml"
CONFIGMAP_NAME="mobilenetv4-config-cpu-pbtxt-cm"
NAMESPACE="workloads"

echo "Project root determined as: $PROJECT_ROOT_ABS"
echo "Expecting local ONNX model at: $LOCAL_MODEL_PATH"
echo "Expecting CPU Triton deployment YAML at: $TRITON_CPU_DEPLOYMENT_YAML_PATH"

# Create namespace if it doesn't exist
$KUBECTL create namespace $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

# Clean up existing CPU deployments if they exist
echo "Cleaning up existing CPU deployments..."
$KUBECTL delete deployment mobilenetv4-triton-cpu-deployment -n $NAMESPACE --ignore-not-found=true
$KUBECTL delete service mobilenetv4-triton-cpu-svc -n $NAMESPACE --ignore-not-found=true
$KUBECTL delete configmap $CONFIGMAP_NAME -n $NAMESPACE --ignore-not-found=true
$KUBECTL delete pvc mobilenetv4-cpu-model-pvc -n $NAMESPACE --ignore-not-found=true

# Use the CPU-specific model preparation script
echo "Preparing CPU model PVC..."
"$(dirname "$0")/prepare_model_pvc_cpu.sh"

# Wait for Triton CPU server to be ready
echo "Waiting for CPU Triton server to be ready..."
$KUBECTL rollout status deployment/mobilenetv4-triton-cpu-deployment -n $NAMESPACE --timeout=300s || {
    echo "Error: CPU Triton server deployment failed or timed out"
    echo "Checking pod status..."
    $KUBECTL get pods -n $NAMESPACE -l app=mobilenetv4-triton-cpu
    echo "Checking pod logs..."
    POD=$($KUBECTL get pods -n $NAMESPACE -l app=mobilenetv4-triton-cpu -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$POD" ]; then
        $KUBECTL logs -n $NAMESPACE $POD
    fi
    exit 1
}

# Delete existing Locust configmap if it exists, then create from file
echo "Updating Locust configuration..."
$KUBECTL delete configmap locustfile-config -n $NAMESPACE --ignore-not-found=true
$KUBECTL create configmap locustfile-config --from-file=locustfile.py="$(dirname "$0")/enhanced_locustfile.py" -n $NAMESPACE

# Deploy Locust
echo "Deploying Locust..."
$KUBECTL apply -f "$(dirname "$0")/locust-deployment.yaml"

# Wait for Locust to be ready
echo "Waiting for Locust to be ready..."
$KUBECTL wait --for=condition=available deployment/locust-master -n $NAMESPACE --timeout=300s
$KUBECTL wait --for=condition=available deployment/locust-worker -n $NAMESPACE --timeout=300s

# Print instructions
echo "CPU baseline experiment setup complete!"
echo "Access Locust web interface at http://localhost:8089"
echo "Configure your test with:"
echo "- Number of users: 10"
echo "- Spawn rate: 1"
echo "- Host: http://mobilenetv4-triton-cpu-svc.workloads.svc.cluster.local:8000"
