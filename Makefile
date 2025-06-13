.PHONY: baseline baseline-cpu clean-baseline clean-cpu download-hf-model prepare-model deploy-baseline deploy-baseline-cpu run-baseline run-baseline-cpu collect-results collect-results-cpu evaluate-accuracy clean dynamic-load cpu-baseline accuracy-evaluation run-tests run-tests-gpu run-tests-cpu run run-all-load-tests run-all-load-tests-gpu run-all-load-tests-cpu analyze-scheduling create-logical-partitions deploy-memory-intensive rl-train rl-evaluate rl-test rl-compare clean-rl install-rl-deps

# Directory for storing experiment results (relative to KISim root)
RESULTS_DIR := ./results
BASELINE_DIR := $(RESULTS_DIR)/baseline
CPU_BASELINE_DIR := $(RESULTS_DIR)/cpu_baseline
FIGURES_DIR := ./figures
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
BASELINE_RESULT := $(BASELINE_DIR)/baseline_$(TIMESTAMP)
CPU_BASELINE_RESULT := $(CPU_BASELINE_DIR)/cpu_baseline_$(TIMESTAMP)

# Use sudo with microk8s kubectl
KUBECTL := sudo microk8s kubectl

# Python interpreter (ensure it has torch and torchvision)
PYTHON := python3 # Ensure this python has huggingface_hub installed

# Create necessary directories
$(shell mkdir -p $(BASELINE_DIR) $(CPU_BASELINE_DIR) experiments/models/mobilenetv4/1)

download-hf-model:
	@echo "Downloading ONNX model from Hugging Face Hub..."
	@$(PYTHON) ./scripts/download_hf_onnx_model.py
	@echo "ONNX model download complete. Check experiments/models/mobilenetv4/1/model.onnx"

baseline: download-hf-model prepare-model deploy-baseline run-all-load-tests-gpu collect-results

# Clean CPU resources before running baseline-cpu to avoid conflicts
clean-cpu:
	@echo "Cleaning up CPU-only resources..."
	@$(KUBECTL) delete deployment mobilenetv4-triton-cpu-deployment -n workloads --ignore-not-found=true
	@$(KUBECTL) delete service mobilenetv4-triton-cpu-svc -n workloads --ignore-not-found=true
	@$(KUBECTL) delete configmap mobilenetv4-config-cpu-pbtxt-cm -n workloads --ignore-not-found=true
	@$(KUBECTL) delete pvc mobilenetv4-cpu-model-pvc -n workloads --ignore-not-found=true
	@$(KUBECTL) delete pod model-copy-pod-cpu -n workloads --ignore-not-found=true

baseline-cpu: download-hf-model clean-cpu deploy-baseline-cpu run-all-load-tests-cpu collect-results-cpu

prepare-model:
	@echo "Preparing model files for PVC..."
	@./scripts/prepare_model_pvc.sh

deploy-baseline:
	@echo "Deploying baseline components (GPU version)..."
	@./scripts/run_baseline_experiment.sh

deploy-baseline-cpu:
	@echo "Deploying baseline components (CPU-only version)..."
	@./scripts/run_baseline_cpu_experiment.sh

run-baseline:
	@echo "Running baseline experiment (GPU version)..."
	@echo "Starting port forward for Locust web interface..."
	@PF_PID=$$($(KUBECTL) port-forward -n workloads svc/locust-master 8089:8089 & echo $$!) && \
	echo "Port forwarding started with PID $${PF_PID}" && \
	echo "Please configure and start the test in Locust web interface (http://localhost:8089)" && \
	echo "Host: http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000" && \
	echo "Press Enter when the test is complete..." && \
	read -p "Press Enter to continue..." && \
	echo "Stopping port forward..." && \
	kill $${PF_PID} 2>/dev/null || true && \
	pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true

run-baseline-cpu:
	@echo "Running baseline experiment (CPU-only version)..."
	@echo "Starting port forward for Locust web interface..."
	@PF_PID=$$($(KUBECTL) port-forward -n workloads svc/locust-master 8089:8089 & echo $$!) && \
	echo "Port forwarding started with PID $${PF_PID}" && \
	echo "Please configure and start the test in Locust web interface (http://localhost:8089)" && \
	echo "Host: http://mobilenetv4-triton-cpu-svc.workloads.svc.cluster.local:8000" && \
	echo "Press Enter when the test is complete..." && \
	read -p "Press Enter to continue..." && \
	echo "Stopping port forward..." && \
	kill $${PF_PID} 2>/dev/null || true && \
	pkill -f "port-forward -n workloads svc/locust-master 8089:8089" || true

collect-results:
	@echo "Collecting experiment results (GPU version)..."
	@mkdir -p $(BASELINE_RESULT)
	@$(KUBECTL) get pods -n workloads -o wide > $(BASELINE_RESULT)/pod_status.txt
	@$(KUBECTL) logs -n workloads deployment/mobilenetv4-triton-deployment --all-containers=true > $(BASELINE_RESULT)/triton_logs.txt 2>/dev/null || echo "No triton logs found or error collecting."
	@$(KUBECTL) logs -n workloads deployment/locust-master > $(BASELINE_RESULT)/locust_master_logs.txt 2>/dev/null || echo "No locust-master logs found or error collecting."
	# Try to get logs from one of the locust workers if the deployment exists
	@WORKER_POD=$($(KUBECTL) get pods -n workloads -l app=locust-worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && \
	    if [ ! -z "$$WORKER_POD" ]; then \
	        $(KUBECTL) logs -n workloads $$WORKER_POD > $(BASELINE_RESULT)/locust_worker_logs.txt; \
	    else \
	        echo "No locust-worker pods found or error collecting logs."; \
	    fi
	@echo "Results collected in $(BASELINE_RESULT)"
	@echo "Generating GPU-only report..."
	@python3 generate_comprehensive_report.py --gpu-only

collect-results-cpu:
	@echo "Collecting experiment results (CPU-only version)..."
	@mkdir -p $(CPU_BASELINE_RESULT)
	@$(KUBECTL) get pods -n workloads -o wide > $(CPU_BASELINE_RESULT)/pod_status.txt
	@$(KUBECTL) logs -n workloads deployment/mobilenetv4-triton-cpu-deployment --all-containers=true > $(CPU_BASELINE_RESULT)/triton_cpu_logs.txt 2>/dev/null || echo "No triton CPU logs found or error collecting."
	@$(KUBECTL) logs -n workloads deployment/locust-master > $(CPU_BASELINE_RESULT)/locust_master_logs.txt 2>/dev/null || echo "No locust-master logs found or error collecting."
	# Try to get logs from one of the locust workers if the deployment exists
	@WORKER_POD=$($(KUBECTL) get pods -n workloads -l app=locust-worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && \
	    if [ ! -z "$$WORKER_POD" ]; then \
	        $(KUBECTL) logs -n workloads $$WORKER_POD > $(CPU_BASELINE_RESULT)/locust_worker_logs.txt; \
	    else \
	        echo "No locust-worker pods found or error collecting logs."; \
	    fi
	# Run a quick synthetic test to compare CPU vs GPU performance
	@echo "Running synthetic test to compare CPU vs GPU performance..."
	@TRITON_CPU_IP=$$($(KUBECTL) get svc -n workloads mobilenetv4-triton-cpu-svc -o jsonpath='{.spec.clusterIP}') && \
	./scripts/evaluate_synthetic.py \
		--url http://$$TRITON_CPU_IP:8000 \
		--model-name mobilenetv4 \
		--num-samples 50 \
		--output-file $(CPU_BASELINE_RESULT)/cpu_synthetic_results.json
	# Note: GPU comparison is skipped in baseline-cpu to avoid deploying GPU version
	# If you want to compare with GPU, run 'make baseline' first, then 'make baseline-cpu'
	@echo "Note: GPU comparison is skipped. To compare with GPU, run 'make baseline' first, then 'make baseline-cpu'"
	@echo "Results collected in $(CPU_BASELINE_RESULT)"
	@echo "Generating CPU-only report..."
	@python3 generate_comprehensive_report.py --cpu-only

evaluate-accuracy:
	@echo "Evaluating model accuracy with synthetic data..."
	@mkdir -p $(BASELINE_RESULT)
	@echo "Running synthetic data test to evaluate model responsiveness..."
	@chmod +x ./scripts/evaluate_synthetic.py
	@echo "Installing required packages..."
	@pip install numpy requests tqdm
	@TRITON_IP=$$($(KUBECTL) get svc -n workloads mobilenetv4-triton-svc -o jsonpath='{.spec.clusterIP}') && \
	./scripts/evaluate_synthetic.py \
		--url http://$$TRITON_IP:8000 \
		--model-name mobilenetv4 \
		--num-samples 100 \
		--output-file $(BASELINE_RESULT)/accuracy_results.json
	@echo "Model evaluation complete. Results saved to $(BASELINE_RESULT)/accuracy_results.json"

# Run dynamic load tests
dynamic-load:
	@echo "Running dynamic load tests..."
	@chmod +x ./scripts/run_dynamic_load_test.sh
	@./scripts/run_dynamic_load_test.sh

# Run CPU-only baseline
cpu-baseline:
	@echo "Running CPU-only baseline experiment..."
	@chmod +x ./scripts/run_cpu_experiment.sh
	@./scripts/run_cpu_experiment.sh

# Run accuracy evaluation on Tiny ImageNet
accuracy-evaluation:
	@echo "Running accuracy evaluation on Tiny ImageNet..."
	@chmod +x ./scripts/run_accuracy_evaluation.sh
	@./scripts/run_accuracy_evaluation.sh

# Run automated tests on GPU version with a specific load pattern
run-tests-gpu:
	@echo "Running automated tests on GPU version..."
	@chmod +x ./scripts/run_comparative_tests.sh
	@./scripts/run_comparative_tests.sh --pattern ramp --duration 5 --gpu-only

# Run automated tests on CPU version with a specific load pattern
run-tests-cpu:
	@echo "Running automated tests on CPU version..."
	@chmod +x ./scripts/run_comparative_tests.sh
	@./scripts/run_comparative_tests.sh --pattern ramp --duration 5 --cpu-only

# Run all tests (both GPU and CPU) with a specific load pattern and generate comparison
run-tests:
	@echo "Running all tests (both GPU and CPU)..."
	@chmod +x ./scripts/run_comparative_tests.sh
	@./scripts/run_comparative_tests.sh --pattern ramp --duration 5

# Run all load patterns on GPU version
run-all-load-tests-gpu:
	@echo "Running all load patterns on GPU version..."
	@chmod +x ./scripts/run_comparative_tests.sh
	@echo "Running ramp load pattern..."
	@./scripts/run_comparative_tests.sh --pattern ramp --duration 5 --gpu-only
	@echo "Running spike load pattern..."
	@./scripts/run_comparative_tests.sh --pattern spike --duration 5 --gpu-only
	@echo "Running periodic load pattern..."
	@./scripts/run_comparative_tests.sh --pattern periodic --duration 5 --gpu-only
	@echo "Running random load pattern..."
	@./scripts/run_comparative_tests.sh --pattern random --duration 5 --gpu-only
	@echo "All load patterns completed on GPU version."

# Run all load patterns on CPU version
run-all-load-tests-cpu:
	@echo "Running all load patterns on CPU version..."
	@chmod +x ./scripts/run_comparative_tests.sh
	@echo "Running ramp load pattern..."
	@./scripts/run_comparative_tests.sh --pattern ramp --duration 5 --cpu-only
	@echo "Running spike load pattern..."
	@./scripts/run_comparative_tests.sh --pattern spike --duration 5 --cpu-only
	@echo "Running periodic load pattern..."
	@./scripts/run_comparative_tests.sh --pattern periodic --duration 5 --cpu-only
	@echo "Running random load pattern..."
	@./scripts/run_comparative_tests.sh --pattern random --duration 5 --cpu-only
	@echo "All load patterns completed on CPU version."

# Run all load patterns on both GPU and CPU versions
run-all-load-tests:
	@echo "Running all load patterns on both GPU and CPU versions..."
	@chmod +x ./scripts/run_comparative_tests.sh
	@echo "Running ramp load pattern..."
	@./scripts/run_comparative_tests.sh --pattern ramp --duration 5
	@echo "Running spike load pattern..."
	@./scripts/run_comparative_tests.sh --pattern spike --duration 5
	@echo "Running periodic load pattern..."
	@./scripts/run_comparative_tests.sh --pattern periodic --duration 5
	@echo "Running random load pattern..."
	@./scripts/run_comparative_tests.sh --pattern random --duration 5
	@echo "All load patterns completed on both versions."

# Create logical partitions for workload separation
create-logical-partitions:
	@echo "Creating logical partitions for workload separation..."
	@chmod +x ./scripts/create-logical-partitions.sh
	@./scripts/create-logical-partitions.sh
	@echo "Logical partitions created successfully."

# Deploy memory-intensive workload
deploy-memory-intensive:
	@echo "Deploying memory-intensive workload..."
	@$(KUBECTL) apply -f ./scripts/memory-intensive-deployment.yaml
	@echo "Memory-intensive workload deployed successfully."

# Analyze scheduling decisions
analyze-scheduling:
	@echo "Analyzing Kubernetes scheduling decisions..."
	@mkdir -p $(RESULTS_DIR)/scheduling_analysis/gpu $(RESULTS_DIR)/scheduling_analysis/cpu
	@chmod +x ./scripts/analyze_scheduling.py
	@if [ -d "$(BASELINE_RESULT)/metrics" ]; then \
		./scripts/analyze_scheduling.py --metrics-dir $(BASELINE_RESULT)/metrics --output-dir $(RESULTS_DIR)/scheduling_analysis/gpu; \
	else \
		echo "Warning: GPU baseline metrics not found at $(BASELINE_RESULT)/metrics"; \
	fi
	@if [ -d "$(CPU_BASELINE_RESULT)/metrics" ]; then \
		./scripts/analyze_scheduling.py --metrics-dir $(CPU_BASELINE_RESULT)/metrics --output-dir $(RESULTS_DIR)/scheduling_analysis/cpu; \
	else \
		echo "Warning: CPU baseline metrics not found at $(CPU_BASELINE_RESULT)/metrics"; \
	fi
	@echo "Scheduling analysis complete. Results saved to $(RESULTS_DIR)/scheduling_analysis"

# Run complete test suite (clean, deploy both versions, run all load tests, compare)
run:
	@echo "Running complete test suite..."
	@make clean
	@make download-hf-model
	@make create-logical-partitions
	@make deploy-memory-intensive
	@make run-all-load-tests
	@python3 generate_comprehensive_report.py
	@make analyze-scheduling

clean-baseline:
	@echo "Cleaning up baseline experiments (both GPU and CPU versions)..."
	# Delete deployments
	@$(KUBECTL) delete deployment mobilenetv4-triton-deployment -n workloads --ignore-not-found=true
	@$(KUBECTL) delete deployment mobilenetv4-triton-cpu-deployment -n workloads --ignore-not-found=true
	@$(KUBECTL) delete deployment memory-intensive-deployment -n workloads --ignore-not-found=true
	@$(KUBECTL) delete deployment locust-master -n workloads --ignore-not-found=true
	@$(KUBECTL) delete deployment locust-worker -n workloads --ignore-not-found=true
	# Delete services
	@$(KUBECTL) delete service mobilenetv4-triton-svc -n workloads --ignore-not-found=true
	@$(KUBECTL) delete service mobilenetv4-triton-cpu-svc -n workloads --ignore-not-found=true
	@$(KUBECTL) delete service memory-intensive-svc -n workloads --ignore-not-found=true
	@$(KUBECTL) delete service locust-master -n workloads --ignore-not-found=true
	# Delete configmaps
	@$(KUBECTL) delete configmap locustfile-config -n workloads --ignore-not-found=true
	@$(KUBECTL) delete configmap mobilenetv4-config-pbtxt-cm -n workloads --ignore-not-found=true
	@$(KUBECTL) delete configmap mobilenetv4-config-cpu-pbtxt-cm -n workloads --ignore-not-found=true
	# Delete PVCs
	@$(KUBECTL) delete pvc mobilenetv4-model-pvc -n workloads --ignore-not-found=true
	@$(KUBECTL) delete pvc mobilenetv4-cpu-model-pvc -n workloads --ignore-not-found=true
	# Delete any leftover pods
	@$(KUBECTL) delete pod model-copy-pod -n workloads --ignore-not-found=true
	@$(KUBECTL) delete pod model-copy-pod-cpu -n workloads --ignore-not-found=true
	@echo "Cleanup complete"

clean:
	@echo "Cleaning all experiment artifacts..."
	@rm -rf $(RESULTS_DIR)
	@rm -f experiments/models/mobilenetv4/1/model.onnx
	@make clean-baseline

# RL Training and Evaluation Commands
# ===================================

# Install RL dependencies
install-rl-deps:
	@echo "Installing RL dependencies..."
	@pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@pip install gym numpy matplotlib prometheus-api-client kubernetes requests tqdm
	@echo "RL dependencies installed successfully"

# Train RL agent
rl-train:
	@echo "Starting RL agent training..."
	@mkdir -p rl/training_logs
	@cd scripts/rl && bash -c "source ../../../.venv/bin/activate && python3 train_rl_agent.py --episodes 100 --gpu" \
		2>&1 | tee ../../rl/training_logs/training_$(TIMESTAMP).log
	@echo "RL training completed. Check rl/ directory for results."

# Train RL agent with custom configuration
rl-train-custom:
	@echo "Starting RL agent training with custom configuration..."
	@mkdir -p rl/training_logs
	@cd scripts/rl && source ../../../.venv/bin/activate && python3 train_rl_agent.py \
		--episodes $(EPISODES) \
		--config $(CONFIG) \
		$(if $(GPU),--gpu,) \
		2>&1 | tee ../../rl/training_logs/training_$(TIMESTAMP).log
	@echo "RL training completed. Check rl/ directory for results."

# Evaluate trained RL agent
rl-evaluate:
	@echo "Evaluating trained RL agent..."
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL path required. Usage: make rl-evaluate MODEL=path/to/model.pt"; \
		exit 1; \
	fi
	@mkdir -p rl/evaluation_logs
	@cd scripts/rl && source ../../../.venv/bin/activate && python3 evaluate_rl_agent.py \
		--model $(MODEL) \
		--runs 5 \
		2>&1 | tee ../../rl/evaluation_logs/evaluation_$(TIMESTAMP).log
	@echo "RL evaluation completed. Check rl/ directory for results."

# Quick RL test with short episodes
rl-test:
	@echo "Running quick RL test (short episodes)..."
	@mkdir -p rl/test_logs
	@cd scripts/rl && source ../../../.venv/bin/activate && python3 train_rl_agent.py \
		--episodes 10 \
		--config ../../configs/rl_test_config.json \
		2>&1 | tee ../../rl/test_logs/test_$(TIMESTAMP).log
	@echo "RL test completed."

# Compare RL agent with baselines
rl-compare:
	@echo "Comparing RL agent performance with baselines..."
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL path required. Usage: make rl-compare MODEL=path/to/model.pt"; \
		exit 1; \
	fi
	@mkdir -p rl/comparison_logs
	@cd scripts/rl && source ../../../.venv/bin/activate && python3 evaluate_rl_agent.py \
		--model $(MODEL) \
		--runs 10 \
		2>&1 | tee ../../rl/comparison_logs/comparison_$(TIMESTAMP).log
	@echo "Generating comprehensive comparison report..."
	@source ../.venv/bin/activate && python3 scripts/rl/generate_rl_comparison.py \
		--rl-results rl/evaluation_$(TIMESTAMP) \
		--baseline-results $(BASELINE_DIR) \
		--cpu-baseline-results $(CPU_BASELINE_DIR) \
		--output rl/comprehensive_comparison_$(TIMESTAMP).json
	@echo "RL comparison completed. Check rl/ directory for results."

# Clean RL artifacts
clean-rl:
	@echo "Cleaning RL artifacts..."
	@rm -rf rl/training_* rl/evaluation_* rl/test_* rl/comparison_*
	@echo "RL cleanup complete"

# Full RL pipeline: baseline -> train -> evaluate -> compare
rl-full-pipeline:
	@echo "Running full RL pipeline..."
	@echo "Step 1: Ensure baselines are available..."
	@if [ ! -d "$(BASELINE_DIR)" ] || [ ! -d "$(CPU_BASELINE_DIR)" ]; then \
		echo "Baselines not found. Running baseline experiments first..."; \
		make run; \
	fi
	@echo "Step 2: Install RL dependencies..."
	@make install-rl-deps
	@echo "Step 3: Train RL agent..."
	@make rl-train
	@echo "Step 4: Find best model and evaluate..."
	@BEST_MODEL=$$(find rl/training_* -name "best_model.pt" | head -1) && \
	if [ ! -z "$$BEST_MODEL" ]; then \
		echo "Found best model: $$BEST_MODEL"; \
		make rl-compare MODEL=$$BEST_MODEL; \
	else \
		echo "No trained model found. Training may have failed."; \
		exit 1; \
	fi
	@echo "Full RL pipeline completed!"

# Generate plots and figures
# ==========================

# Generate all plots
plots:
	@echo "Generating all plots..."
	@mkdir -p $(FIGURES_DIR)
	@python3 analysis/generate_plots.py --results-dir $(RESULTS_DIR) --output-dir $(FIGURES_DIR) --type all
	@echo "All plots generated in $(FIGURES_DIR)"

# Generate baseline plots only
plots-baseline:
	@echo "Generating baseline plots..."
	@mkdir -p $(FIGURES_DIR)
	@python3 analysis/generate_plots.py --results-dir $(RESULTS_DIR) --output-dir $(FIGURES_DIR) --type baseline
	@echo "Baseline plots generated in $(FIGURES_DIR)/baseline"

# Generate RL plots only
plots-rl:
	@echo "Generating RL plots..."
	@mkdir -p $(FIGURES_DIR)
	@python3 analysis/generate_plots.py --results-dir $(RESULTS_DIR) --output-dir $(FIGURES_DIR) --type rl
	@echo "RL plots generated in $(FIGURES_DIR)/rl"

# Generate comparison plots
plots-comparison:
	@echo "Generating comparison plots..."
	@mkdir -p $(FIGURES_DIR)
	@python3 analysis/generate_plots.py --results-dir $(RESULTS_DIR) --output-dir $(FIGURES_DIR) --type comparison
	@echo "Comparison plots generated in $(FIGURES_DIR)/comparison"

# Demo commands using sample data
# ================================

# Generate comprehensive comparison using sample data
demo-analyze:
	@echo "Generating comprehensive comparison using sample data..."
	@python3 scripts/generate_comprehensive_comparison.py
	@echo "Sample analysis completed! Check $(RESULTS_DIR)/comparison/"

# Generate all plots using sample data
demo-plots:
	@echo "Generating all plots using sample data..."
	@python3 analysis/generate_plots.py --results-dir $(RESULTS_DIR) --output-dir $(FIGURES_DIR) --type all
	@echo "Sample plots generated! Check $(FIGURES_DIR)/"

# Generate baseline plots using sample data
demo-baseline-plots:
	@echo "Generating baseline plots using sample data..."
	@python3 analysis/generate_plots.py --results-dir $(RESULTS_DIR) --output-dir $(FIGURES_DIR) --type baseline
	@echo "Sample baseline plots generated! Check $(FIGURES_DIR)/baseline/"

# Generate RL plots using sample data
demo-rl-plots:
	@echo "Generating RL plots using sample data..."
	@python3 analysis/generate_plots.py --results-dir $(RESULTS_DIR) --output-dir $(FIGURES_DIR) --type rl
	@echo "Sample RL plots generated! Check $(FIGURES_DIR)/rl/"

# Show sample data structure
demo-data:
	@echo "Sample data structure:"
	@echo "======================"
	@find $(RESULTS_DIR) -name "*.json" | head -10
	@echo ""
	@echo "Total sample files: $$(find $(RESULTS_DIR) -name '*.json' | wc -l)"
	@echo ""
	@echo "Use 'make demo-plots' to generate figures from this sample data"

# Help for RL commands
rl-help:
	@echo "RL Training and Evaluation Commands:"
	@echo "===================================="
	@echo "install-rl-deps     - Install required RL dependencies"
	@echo "rl-train           - Train RL agent with default settings"
	@echo "rl-train-custom    - Train with custom config (EPISODES=N CONFIG=path GPU=1)"
	@echo "rl-evaluate        - Evaluate trained model (MODEL=path/to/model.pt)"
	@echo "rl-test            - Quick test with short episodes"
	@echo "rl-compare         - Compare RL with baselines (MODEL=path/to/model.pt)"
	@echo "rl-full-pipeline   - Complete pipeline: baseline -> train -> evaluate"
	@echo "clean-rl           - Clean RL artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make rl-train"
	@echo "  make rl-train-custom EPISODES=50 GPU=1"
	@echo "  make rl-evaluate MODEL=rl/training_20231201_120000/best_model.pt"
	@echo "  make rl-compare MODEL=rl/training_20231201_120000/best_model.pt"
	@echo "  make rl-full-pipeline"
