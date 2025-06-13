#!/bin/bash

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Clean up existing resources
echo "Cleaning up existing resources..."
$KUBECTL delete deployment mobilenetv4-triton-deployment -n workloads --ignore-not-found=true
$KUBECTL delete deployment locust-master -n workloads --ignore-not-found=true
$KUBECTL delete deployment locust-worker -n workloads --ignore-not-found=true
$KUBECTL delete configmap locustfile-config -n workloads --ignore-not-found=true
$KUBECTL delete configmap mobilenetv4-config-pbtxt-cm -n workloads --ignore-not-found=true

# Wait for resources to be deleted
echo "Waiting for resources to be deleted..."
sleep 5

# Apply the updated Triton deployment
echo "Applying updated Triton deployment..."
$KUBECTL apply -f "$(dirname "$0")/mobilenetv4-triton-deployment.yaml"

# Wait for Triton to be ready
echo "Waiting for Triton server to be ready..."
$KUBECTL wait --for=condition=available deployment/mobilenetv4-triton-deployment -n workloads --timeout=300s || echo "Triton deployment not ready, but continuing..."

# Create Locust configmap
echo "Creating Locust configmap..."
$KUBECTL create configmap locustfile-config --from-file=locustfile.py="$(dirname "$0")/locustfile.py" -n workloads

# Apply Locust deployment
echo "Deploying Locust..."
$KUBECTL apply -f "$(dirname "$0")/locust-deployment.yaml"

# Wait for Locust to be ready
echo "Waiting for Locust to be ready..."
$KUBECTL wait --for=condition=available deployment/locust-master -n workloads --timeout=300s || echo "Locust master not ready, but continuing..."
$KUBECTL wait --for=condition=available deployment/locust-worker -n workloads --timeout=300s || echo "Locust worker not ready, but continuing..."

# Port forward Locust web interface
echo "Starting port forward for Locust web interface..."
$KUBECTL port-forward -n workloads svc/locust-master 8089:8089 &

# Print instructions
echo "Experiment restarted!"
echo "Access Locust web interface at http://localhost:8089"
echo "Configure your test with:"
echo "- Number of users: 10"
echo "- Spawn rate: 1"
echo "- Host: http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000"
echo ""
echo "Press Ctrl+C to stop the port forward when done"
