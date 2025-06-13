# Sample Data Guide

## 🎯 Purpose

KISim includes comprehensive sample experimental results that allow researchers to:
- Generate publication-ready figures without running full experiments
- Understand data formats and analysis workflows
- Quickly evaluate the framework's capabilities
- Reproduce paper results for verification

## 📊 Included Sample Data

### Baseline Experiments (GPU)
- **4 Load Patterns**: ramp, spike, periodic, random
- **Performance Metrics**: P95 latency, throughput, error rates
- **Synthetic Tests**: Model accuracy and inference performance

### CPU Baseline (Comparison)
- **Same Load Patterns**: For GPU vs CPU comparison
- **Performance Degradation**: Shows CPU limitations
- **Resource Utilization**: CPU-intensive workload behavior

### RL Training Results
- **100 Episodes**: Complete training progression
- **Convergence Data**: Episode rewards, policy/value losses
- **Performance Metrics**: Training stability and final performance

## 🚀 Quick Commands

```bash
# Show all sample data files
make demo-data

# Generate all plots from sample data
make demo-plots

# Generate specific plot types
make demo-baseline-plots    # Workload pattern visualization
make demo-rl-plots         # RL training progress
```

## 📈 Generated Figures

### Baseline Workload Patterns
- **File**: `figures/baseline/workload_patterns.png`
- **Content**: 4 subplots showing user count and P95 latency over time
- **Usage**: Demonstrates different load characteristics

### RL Training Progress
- **File**: `figures/rl/training_progress.png`
- **Content**: Episode rewards and lengths over training
- **Usage**: Shows learning convergence and stability

## 🔍 Data Format Examples

### Dynamic Load Test Results
```json
{
  "timestamp": "2024-05-23T22:42:09Z",
  "user_count": 100,
  "stats": [{
    "name": "Aggregated",
    "num_requests": 26190,
    "num_failures": 45,
    "avg_response_time": 45.2,
    "response_time_percentile_0.95": 78.5,
    "current_rps": 87.3
  }]
}
```

### RL Training Metrics
```json
{
  "episode_rewards": [-45.2, -38.7, ..., 9.9],
  "episode_lengths": [120, 118, ..., 124],
  "policy_losses": [0.045, 0.042, ..., 0.001],
  "performance_metrics": {
    "final_avg_reward": 9.9,
    "convergence_episode": 85
  }
}
```

## 📚 Academic Usage

### For Paper Writing
1. **Run demo commands** to generate figures
2. **Copy figures** to your paper directory
3. **Reference data** in methodology sections
4. **Cite performance** numbers in results

### For Peer Review
- **Reproducible Results**: Reviewers can verify figures
- **Data Transparency**: Complete experimental data available
- **Method Validation**: Clear data format documentation

### For Follow-up Research
- **Baseline Comparison**: Use as performance reference
- **Data Format**: Follow established JSON schemas
- **Experimental Design**: Replicate load patterns and metrics

## 🛠 Customization

### Adding Your Own Data
1. **Follow JSON Format**: Match existing schema
2. **Place in results/**: Appropriate subdirectory
3. **Run Analysis**: Use existing plot generation scripts

### Modifying Plots
1. **Edit Scripts**: `analysis/generate_plots.py`
2. **Adjust Styling**: Colors, fonts, layout
3. **Add Metrics**: New performance indicators

## ✅ Verification

The sample data represents realistic experimental results:
- **GPU Performance**: ~12ms average latency, 80+ images/sec
- **CPU Performance**: ~90ms average latency, 11 images/sec  
- **RL Convergence**: 85 episodes to stable performance
- **Load Patterns**: Realistic user behavior simulation

## 🎓 Educational Value

This sample data demonstrates:
- **System Performance**: Real-world GPU vs CPU differences
- **RL Training**: Practical convergence patterns
- **Load Testing**: Various workload characteristics
- **Academic Standards**: Publication-quality data collection

Perfect for teaching, research, and development without requiring full experimental infrastructure!
