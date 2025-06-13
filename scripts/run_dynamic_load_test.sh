#!/bin/bash
# Run dynamic load tests with different patterns

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Determine the absolute path to the project's root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/experiments/scripts"
RESULTS_DIR="$PROJECT_ROOT/results/dynamic"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run dynamic load tests with different patterns"
    echo ""
    echo "Options:"
    echo "  -p, --pattern PATTERN   Load pattern to use (spike, ramp, periodic, random, all)"
    echo "  -d, --duration MINUTES  Test duration in minutes (default: varies by pattern)"
    echo "  -m, --min-users COUNT   Minimum number of users (default: 10)"
    echo "  -M, --max-users COUNT   Maximum number of users (default: 100)"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --pattern spike --duration 15"
    exit 1
}

# Default values
PATTERN="all"
MIN_USERS=10
MAX_USERS=100
SPIKE_DURATION=15
RAMP_DURATION=20
PERIODIC_DURATION=30
RANDOM_DURATION=25

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -d|--duration)
            SPIKE_DURATION="$2"
            RAMP_DURATION="$2"
            PERIODIC_DURATION="$2"
            RANDOM_DURATION="$2"
            shift 2
            ;;
        -m|--min-users)
            MIN_USERS="$2"
            shift 2
            ;;
        -M|--max-users)
            MAX_USERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate pattern
if [[ "$PATTERN" != "spike" && "$PATTERN" != "ramp" && "$PATTERN" != "periodic" && "$PATTERN" != "random" && "$PATTERN" != "all" ]]; then
    echo "Error: Invalid pattern '$PATTERN'"
    usage
fi

# Check if Python dependencies are installed
check_dependencies() {
    echo "Checking Python dependencies..."
    python3 -c "import numpy, matplotlib, requests" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing required Python packages..."
        pip install numpy matplotlib requests
    fi
}

# Update Locust configuration with enhanced locustfile
update_locust_config() {
    echo "Updating Locust configuration..."
    $KUBECTL delete configmap locustfile-config -n workloads --ignore-not-found=true
    $KUBECTL create configmap locustfile-config --from-file=locustfile.py="$SCRIPTS_DIR/enhanced_locustfile.py" -n workloads
    
    # Restart Locust pods to pick up the new configuration
    echo "Restarting Locust pods..."
    $KUBECTL rollout restart deployment/locust-master -n workloads
    $KUBECTL rollout restart deployment/locust-worker -n workloads
    
    # Wait for Locust to be ready
    echo "Waiting for Locust to be ready..."
    $KUBECTL rollout status deployment/locust-master -n workloads --timeout=300s
    $KUBECTL rollout status deployment/locust-worker -n workloads --timeout=300s
}

# Run a specific load pattern
run_pattern() {
    local pattern=$1
    local duration=$2
    
    echo "Running $pattern load pattern for $duration minutes..."
    
    # Start port forwarding to Locust web interface
    echo "Starting port forwarding to Locust web interface..."
    $KUBECTL port-forward -n workloads svc/locust-master 8089:8089 &
    PORT_FORWARD_PID=$!
    
    # Wait for port forwarding to be established
    echo "Waiting for port forwarding to be established..."
    sleep 5
    
    # Run the dynamic load controller
    echo "Starting dynamic load controller..."
    python3 "$SCRIPTS_DIR/dynamic_load_controller.py" \
        --pattern "$pattern" \
        --duration "$duration" \
        --min-users "$MIN_USERS" \
        --max-users "$MAX_USERS" \
        --host "http://localhost:8089" \
        --target-host "http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000" \
        --output-dir "$RESULTS_DIR"
    
    # Stop port forwarding
    echo "Stopping port forwarding..."
    kill $PORT_FORWARD_PID
    
    # Wait a moment before starting the next test
    echo "Waiting for resources to settle..."
    sleep 10
}

# Main execution
echo "Starting dynamic load testing..."

# Check dependencies
check_dependencies

# Make sure Triton server is running
echo "Checking if Triton server is running..."
TRITON_READY=$($KUBECTL get deployment mobilenetv4-triton-deployment -n workloads -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
if [ "$TRITON_READY" != "1" ]; then
    echo "Error: Triton server is not running or not ready"
    echo "Please deploy Triton server first using 'make deploy-baseline'"
    exit 1
fi

# Update Locust configuration
update_locust_config

# Run the selected pattern(s)
if [ "$PATTERN" = "all" ] || [ "$PATTERN" = "spike" ]; then
    run_pattern "spike" "$SPIKE_DURATION"
fi

if [ "$PATTERN" = "all" ] || [ "$PATTERN" = "ramp" ]; then
    run_pattern "ramp" "$RAMP_DURATION"
fi

if [ "$PATTERN" = "all" ] || [ "$PATTERN" = "periodic" ]; then
    run_pattern "periodic" "$PERIODIC_DURATION"
fi

if [ "$PATTERN" = "all" ] || [ "$PATTERN" = "random" ]; then
    run_pattern "random" "$RANDOM_DURATION"
fi

echo "Dynamic load testing completed!"
echo "Results saved to $RESULTS_DIR"

# Collect Prometheus metrics if available
if $KUBECTL get svc -n observability prometheus-operated >/dev/null 2>&1; then
    echo "Collecting Prometheus metrics..."
    # TODO: Add code to export Prometheus metrics for the test period
    echo "Prometheus metrics collection not implemented yet"
fi

echo "Done!"
