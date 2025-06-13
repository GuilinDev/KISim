#!/bin/bash
# Create logical partitions using labels and taints in MicroK8s
# This script creates 3 logical partitions on a single physical node:
# 1. GPU partition - for GPU workloads
# 2. CPU-High partition - for CPU-intensive workloads
# 3. General partition - for general purpose workloads
#
# Note: This does NOT create actual separate nodes in Kubernetes.
# Instead, it uses labels and taints to guide the scheduler to
# place different types of workloads according to our partitioning scheme.

# Get the current node name
NODE_NAME=$(sudo microk8s kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
echo "Creating logical partitions on physical node: $NODE_NAME"

# Remove any existing partition labels
echo "Removing existing partition labels..."
sudo microk8s kubectl label node $NODE_NAME workload-gpu- workload-cpu-high- workload-general- node-type- partition-gpu- partition-cpu-high- partition-general- --overwrite || true
sudo microk8s kubectl taint node $NODE_NAME dedicated- --all || true

# Create partition 1: GPU Partition
echo "Creating GPU partition..."
sudo microk8s kubectl label node $NODE_NAME workload-gpu=true --overwrite

# Create partition 2: CPU-High Partition
echo "Creating CPU-High partition..."
sudo microk8s kubectl label node $NODE_NAME workload-cpu-high=true --overwrite

# Create partition 3: General Partition
echo "Creating General partition..."
sudo microk8s kubectl label node $NODE_NAME workload-general=true --overwrite

# Note: We're not using taints anymore as they would prevent pods from being scheduled
# unless they have specific tolerations. Instead, we'll rely on nodeSelectors to guide
# the scheduler's decisions.

# Verify the logical partitions
echo "Verifying logical partitions..."
echo "Node labels:"
sudo microk8s kubectl get node $NODE_NAME --show-labels

echo "Logical partitions created successfully!"
echo ""
echo "IMPORTANT: These are NOT separate physical or virtual nodes."
echo "They are logical partitions on the same physical node,"
echo "distinguished by labels to guide the Kubernetes scheduler."
