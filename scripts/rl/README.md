# RL-based Kubernetes Resource Management

This module implements a Reinforcement Learning (RL) agent for intelligent Kubernetes resource management, specifically optimized for GPU/CPU scheduling decisions based on comprehensive baseline experimental findings.

## Overview

The RL system is designed to outperform static baseline approaches by learning optimal resource allocation strategies across different load patterns:

- **Random Load**: GPU baseline shows 1.96x speedup - RL should learn to prefer GPU
- **Spike Load**: GPU baseline shows 1.27x speedup - RL should learn burst handling
- **Ramp Load**: GPU baseline shows 1.16x speedup - RL should learn gradual scaling
- **Periodic Load**: Similar performance - RL should learn balanced allocation

## Architecture

### Core Components

1. **KubernetesRLEnvironment** (`kubernetes_env.py`)
   - Gym-compatible environment for Kubernetes resource management
   - Observation space: Resource metrics, performance data, load characteristics
   - Action space: GPU/CPU replica scaling + workload placement decisions
   - Reward function: Multi-objective optimization (latency, throughput, efficiency, stability)

2. **PPOAgent** (`ppo_agent.py`)
   - Proximal Policy Optimization agent optimized for resource management
   - Actor-Critic architecture with multi-dimensional action space
   - Experience buffer with Generalized Advantage Estimation (GAE)
   - Model checkpointing and evaluation capabilities

3. **RLLoadController** (`rl_load_controller.py`)
   - Integrates dynamic load generation with RL training
   - Coordinates load patterns with training episodes
   - Collects performance metrics for reward calculation

4. **Utility Modules** (`env_utils.py`)
   - MetricsCollector: Prometheus and Locust integration
   - LoadPatternGenerator: Dynamic load pattern creation
   - RewardCalculator: Advanced reward computation

## Quick Start

### 1. Install Dependencies

```bash
make install-rl-deps
```

### 2. Quick Test

```bash
make rl-test
```

### 3. Full Training

```bash
make rl-train
```

### 4. Evaluate Trained Model

```bash
make rl-evaluate MODEL=rl/training_YYYYMMDD_HHMMSS/best_model.pt
```

### 5. Compare with Baselines

```bash
make rl-compare MODEL=rl/training_YYYYMMDD_HHMMSS/best_model.pt
```

### 6. Complete Pipeline

```bash
make rl-full-pipeline
```

## Configuration

### Environment Configuration

```json
{
  "environment": {
    "namespace": "workloads",
    "prometheus_url": "http://localhost:9090",
    "locust_url": "http://localhost:8089",
    "episode_duration": 300,
    "action_interval": 30,
    "max_replicas": 10,
    "min_replicas": 1,
    "latency_weight": 0.4,
    "throughput_weight": 0.3,
    "resource_efficiency_weight": 0.2,
    "stability_weight": 0.1
  }
}
```

### Agent Configuration

```json
{
  "agent": {
    "learning_rate": 3e-4,
    "hidden_size": 256,
    "buffer_size": 2048,
    "batch_size": 64,
    "ppo_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2
  }
}
```

## Observation Space

The RL agent observes a 10-dimensional state vector:

1. **cpu_utilization** (0-100%): Current CPU usage
2. **memory_utilization** (0-100%): Current memory usage  
3. **gpu_utilization** (0-100%): Current GPU usage
4. **current_latency_p95** (0-10000ms): P95 response time
5. **current_throughput** (0-100 req/s): Request rate
6. **current_replicas_gpu** (0-10): GPU pod count
7. **current_replicas_cpu** (0-10): CPU pod count
8. **load_trend** (-1 to 1): Load increase/decrease trend
9. **load_variance** (0-1): Load volatility measure
10. **time_in_episode** (0-1): Episode progress

## Action Space

Multi-discrete action space with 3 dimensions:

1. **GPU Replica Action** (5 choices):
   - 0: Scale down by 2
   - 1: Scale down by 1
   - 2: Maintain current
   - 3: Scale up by 1
   - 4: Scale up by 2

2. **CPU Replica Action** (5 choices): Same as GPU

3. **Workload Placement** (3 choices):
   - 0: Prefer GPU
   - 1: Balanced
   - 2: Prefer CPU

## Reward Function

Multi-objective reward combining:

```python
total_reward = (
    latency_improvement * 0.4 +
    throughput_improvement * 0.3 + 
    resource_efficiency * 0.2 +
    stability_bonus * 0.1
)
```

### Reward Components

1. **Latency Improvement**: Compared to baseline performance for current load pattern
2. **Throughput Improvement**: Request rate improvement vs baseline
3. **Resource Efficiency**: Optimal GPU/CPU allocation based on pattern
4. **Stability Bonus**: Penalty for excessive scaling actions

## Training Process

### Episode Structure

1. **Reset**: Initialize environment, select load pattern
2. **Observation**: Collect system metrics
3. **Action**: Agent selects scaling/placement actions
4. **Step**: Execute actions, wait for effect
5. **Reward**: Calculate multi-objective reward
6. **Repeat**: Until episode duration reached

### Load Pattern Rotation

Training episodes cycle through load patterns to ensure robust learning:
- Episode 0: ramp
- Episode 1: spike  
- Episode 2: periodic
- Episode 3: random
- Episode 4: ramp (repeat)

### Model Checkpointing

- Save every 10 episodes
- Track best performing model
- Evaluation every 20 episodes

## Evaluation

### Comprehensive Evaluation

The evaluation process tests the trained agent across all load patterns:

1. **Pattern-Specific Testing**: 5 runs per pattern
2. **Baseline Comparison**: Compare with GPU/CPU baselines
3. **Statistical Analysis**: Mean, std, min, max performance
4. **Visualization**: Performance plots and radar charts

### Expected Improvements

Based on baseline findings, the RL agent should achieve:

- **Random Pattern**: >1.96x improvement (GPU advantage)
- **Spike Pattern**: >1.27x improvement (burst handling)
- **Ramp Pattern**: >1.16x improvement (gradual scaling)
- **Periodic Pattern**: Balanced performance optimization

## Results Structure

```
rl/
├── training_YYYYMMDD_HHMMSS/
│   ├── best_model.pt
│   ├── model_episode_*.pt
│   ├── training_progress.json
│   ├── training_progress.png
│   └── evaluation_episode_*.json
├── evaluation_YYYYMMDD_HHMMSS/
│   ├── comprehensive_evaluation.json
│   ├── evaluation_comparison.png
│   └── evaluation_YYYYMMDD_HHMMSS_report.txt
└── comparison_YYYYMMDD_HHMMSS/
    ├── rl_comparison_data.json
    ├── rl_comparison_plots.png
    └── rl_comparison_report.txt
```

## Troubleshooting

### Common Issues

1. **Environment Connection**: Ensure Kubernetes and Prometheus are accessible
2. **Load Generation**: Verify Locust is running and reachable
3. **GPU Availability**: Check CUDA installation for GPU training
4. **Memory Issues**: Reduce batch_size or buffer_size for limited memory

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Testing

Test individual components:

```python
from scripts.rl import KubernetesRLEnvironment, create_environment_config

config = create_environment_config()
env = KubernetesRLEnvironment(config)
obs = env.reset()
action = [2, 2, 1]  # Maintain, maintain, balanced
obs, reward, done, info = env.step(action)
```

## Academic Integration

This RL implementation is designed for academic research and publication:

- **Reproducible Results**: Comprehensive logging and checkpointing
- **Baseline Comparison**: Direct comparison with experimental baselines
- **Statistical Rigor**: Multiple evaluation runs with statistical analysis
- **Visualization**: Publication-ready plots and charts

The results can be directly integrated into academic papers targeting IPCCC and IEEE SEC conferences.
