# KISim Architecture

## Overview

KISim (Kubernetes Intelligent Scheduling Simulator) is a comprehensive framework for evaluating reinforcement learning-based scheduling algorithms in Kubernetes environments. The system is designed to provide reproducible, academic-quality experiments for GPU-accelerated workloads.

## System Components

### 1. Kubernetes Cluster
- **Platform**: MicroK8s or Kind with GPU support
- **Workloads**: MobileNetV4 inference service (GPU/CPU variants)
- **Monitoring**: Prometheus metrics collection
- **Scheduling**: Default Kubernetes scheduler vs. RL-based scheduler

### 2. Load Generation
- **Tool**: Locust-based load generator
- **Patterns**: Ramp, Spike, Periodic, Random
- **Metrics**: Response time, throughput, error rates
- **Scalability**: Dynamic user count adjustment

### 3. RL Environment
- **Framework**: OpenAI Gym compatible
- **Agent**: PPO (Proximal Policy Optimization)
- **State Space**: Resource utilization, performance metrics, load characteristics
- **Action Space**: Replica scaling, resource allocation, pod placement
- **Reward Function**: Multi-objective optimization (latency, throughput, efficiency)

### 4. Metrics Collection
- **Performance**: P95 latency, requests/second, error rates
- **Resources**: CPU/Memory/GPU utilization
- **Scheduling**: Pod placement decisions, scaling events
- **Training**: Episode rewards, policy gradients, value functions

## Data Flow

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

## Experimental Design

### Baseline Experiments
1. **Traditional Scheduling**: Kubernetes default scheduler
2. **Load Patterns**: Four distinct patterns (ramp, spike, periodic, random)
3. **Variants**: GPU-accelerated vs CPU-only deployments
4. **Metrics**: Comprehensive performance and resource utilization

### RL Training
1. **Environment Setup**: Kubernetes cluster with monitoring
2. **State Representation**: System metrics, load characteristics, episode progress
3. **Action Space**: Discrete actions for scaling and placement
4. **Training Process**: PPO with experience replay and policy updates
5. **Evaluation**: Performance comparison against baseline

### Analysis and Visualization
1. **Statistical Analysis**: Multiple runs with confidence intervals
2. **Performance Metrics**: Latency, throughput, resource efficiency
3. **Publication Figures**: High-quality plots and tables
4. **Comparative Studies**: RL vs baseline performance

## File Structure

```
KISim/
├── README.md                 # Project overview and quick start
├── LICENSE                   # MIT license
├── requirements.txt          # Python dependencies
├── Makefile                 # Automation commands
├── configs/                 # Configuration files
│   ├── rl_production_config.json
│   └── rl_test_config.json
├── scripts/                 # Core experiment scripts
│   ├── rl/                 # RL training and evaluation
│   ├── download_hf_onnx_model.py
│   ├── dynamic_load_controller.py
│   └── ...
├── kubernetes/             # Kubernetes manifests
│   ├── mobilenetv4-triton-deployment.yaml
│   ├── locust-deployment.yaml
│   └── ...
├── analysis/               # Analysis and visualization
│   └── generate_plots.py
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   └── API.md
├── results/                # Experiment results (created at runtime)
│   ├── baseline/
│   ├── cpu_baseline/
│   └── rl/
└── figures/                # Generated plots (created at runtime)
    ├── baseline/
    ├── rl/
    └── comparison/
```

## Key Design Principles

### 1. Reproducibility
- **Deterministic Seeds**: Fixed random seeds for consistent results
- **Version Control**: All configurations and code versioned
- **Environment Isolation**: Containerized workloads and dependencies
- **Documentation**: Comprehensive setup and usage instructions

### 2. Modularity
- **Pluggable Components**: Easy to swap RL algorithms, load patterns, workloads
- **Configuration-Driven**: JSON-based configuration for all parameters
- **Clean Interfaces**: Well-defined APIs between components
- **Extensibility**: Easy to add new metrics, algorithms, or workloads

### 3. Academic Quality
- **Statistical Rigor**: Multiple runs, confidence intervals, significance tests
- **Publication Ready**: High-quality figures, tables, and reports
- **Peer Review**: Code and methodology suitable for academic scrutiny
- **Open Source**: MIT license for community contribution

### 4. Practical Applicability
- **Real Workloads**: Actual ML inference services (MobileNetV4)
- **Production Metrics**: Industry-standard performance indicators
- **Scalable Design**: Supports various cluster sizes and configurations
- **Cloud Native**: Kubernetes-native implementation

## Performance Considerations

### Resource Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 1 GPU (optional)
- **Recommended**: 16GB RAM, 8 CPU cores, 1 GPU (RTX 3080 or better)
- **Storage**: 10GB for models, results, and logs
- **Network**: Stable internet for model downloads

### Scalability
- **Single Node**: Development and testing
- **Multi-Node**: Production-scale experiments
- **Cloud Deployment**: AWS, GCP, Azure compatibility
- **Edge Deployment**: Resource-constrained environments

## Security and Privacy

### Data Handling
- **Synthetic Data**: No real user data required
- **Local Processing**: All computation on local cluster
- **Configurable Logging**: Control over metrics collection
- **Clean Shutdown**: Proper resource cleanup

### Access Control
- **Kubernetes RBAC**: Role-based access control
- **Network Policies**: Isolated workload communication
- **Secret Management**: Secure credential handling
- **Audit Logging**: Comprehensive activity tracking

## Future Extensions

### Planned Features
- **Multi-Objective RL**: Pareto-optimal scheduling policies
- **Federated Learning**: Distributed RL training
- **Real-Time Adaptation**: Online policy updates
- **Advanced Workloads**: Multi-modal ML services

### Research Directions
- **Theoretical Analysis**: Convergence guarantees, sample complexity
- **Comparative Studies**: Different RL algorithms, baseline methods
- **Robustness Testing**: Adversarial scenarios, failure modes
- **Transfer Learning**: Cross-cluster policy adaptation

## Contributing

See the main README.md for contribution guidelines and development setup instructions.

## References

1. Kubernetes Documentation: https://kubernetes.io/docs/
2. OpenAI Gym: https://gym.openai.com/
3. Stable Baselines3: https://stable-baselines3.readthedocs.io/
4. Triton Inference Server: https://github.com/triton-inference-server/server
5. Prometheus Monitoring: https://prometheus.io/docs/
