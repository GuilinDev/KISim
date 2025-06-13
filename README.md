# KISim: Kubernetes Intelligent Scheduling Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-1.24+-blue.svg)](https://kubernetes.io/)

**KISim** is a comprehensive simulation framework for evaluating Reinforcement Learning (RL) based scheduling algorithms in Kubernetes environments. It provides baseline experiments, RL training capabilities, and comprehensive performance analysis tools for GPU-accelerated workloads.

## 🎯 Overview

This project implements and evaluates RL-based resource management for Kubernetes clusters, specifically targeting GPU-accelerated inference workloads. It includes:

- **Baseline Experiments**: Traditional Kubernetes scheduling with comprehensive load testing
- **RL Training**: PPO-based agent for intelligent resource allocation
- **Performance Analysis**: Detailed metrics collection and visualization
- **Academic Research**: Publication-ready experimental framework

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Generator │    │  RL Agent       │    │  Kubernetes     │
│   (Locust)      │◄──►│  (PPO)          │◄──►│  Cluster        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Metrics       │    │  Training       │    │  GPU Workloads  │
│   Collection    │    │  Environment    │    │  (MobileNetV4)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Kubernetes Cluster**: MicroK8s or Kind with GPU support
- **Python 3.8+**: With pip and virtual environment
- **NVIDIA GPU**: With proper drivers and container toolkit
- **Docker**: For containerized workloads

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/KISim.git
cd KISim
```

2. **Set up Python environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Download ML model**:
```bash
make download-hf-model
```

### Running Experiments

#### Baseline Experiments
```bash
# Run complete baseline with all load patterns
make baseline

# Run CPU-only baseline for comparison
make baseline-cpu

# Run specific load pattern
make dynamic-load PATTERN=ramp
```

#### RL Training
```bash
# Install RL dependencies
make install-rl-deps

# Quick test (short episodes)
make rl-test

# Full training (100 episodes)
make rl-train

# Complete pipeline: baseline → train → evaluate
make rl-full-pipeline
```

#### Analysis and Visualization
```bash
# Generate comprehensive comparison
make analyze-scheduling

# Create academic figures
make rl-plots
make rl-individual
```

## 📊 Experimental Design

### Load Patterns
- **Ramp**: Gradual load increase (0→100 users)
- **Spike**: Sudden load bursts (20→100→20 users)
- **Periodic**: Cyclical load variations
- **Random**: Stochastic load patterns

### Metrics Collected
- **Performance**: Latency (P95), Throughput, Error rates
- **Resources**: CPU/Memory/GPU utilization
- **Scheduling**: Pod placement decisions, Resource allocation
- **RL Training**: Episode rewards, Policy gradients, Value functions

### Evaluation Criteria
- **Latency Improvement**: vs. baseline scheduling
- **Resource Efficiency**: Utilization optimization
- **System Stability**: Avoiding resource thrashing
- **Scalability**: Performance under varying loads

## 🔧 Configuration

### RL Agent Configuration (`configs/rl_production_config.json`)
```json
{
  "environment": {
    "episode_duration": 300,
    "action_interval": 30,
    "max_gpu_replicas": 5,
    "max_cpu_replicas": 10
  },
  "agent": {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "clip_epsilon": 0.2,
    "value_coef": 0.5
  }
}
```

### Environment Variables
```bash
export KUBECONFIG=/path/to/kubeconfig
export RESULTS_DIR=./results
export GPU_ENABLED=true
```

## 📈 Results Structure

```
results/
├── baseline/           # Traditional scheduling results
│   ├── dynamic/       # Load pattern experiments
│   └── metrics/       # Performance metrics
├── cpu_baseline/      # CPU-only comparison
├── rl/               # RL training results
│   ├── training_*/   # Individual training runs
│   └── analysis/     # Performance analysis
└── comparison/       # Comparative analysis
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Integration tests
make test-integration

# Performance benchmarks
make benchmark
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**KISim** - Making Kubernetes scheduling intelligent through reinforcement learning! 🚀
