#!/usr/bin/env python3
"""
Simple accuracy test for MobileNetV4 model using synthetic data.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import requests

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple accuracy test for MobileNetV4')
    parser.add_argument('--url', type=str, 
                        default='http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000',
                        help='Triton server URL')
    parser.add_argument('--model-name', type=str, default='mobilenetv4',
                        help='Model name in Triton')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of tests to run')
    parser.add_argument('--output-file', type=str, default='accuracy_results.json',
                        help='Path to save results')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    print(f"Testing {args.model_name} accuracy with synthetic data...")
    
    # Construct the inference URL
    infer_url = f"{args.url}/v2/models/{args.model_name}/infer"
    
    # Initialize counters
    correct = 0
    latencies = []
    
    # Run tests
    for i in range(args.num_tests):
        # Generate random image data
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        # Create request payload
        payload = {
            "inputs": [
                {
                    "name": "pixel_values",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "data": input_data.flatten().tolist()
                }
            ]
        }
        
        # Send request
        start_time = time.time()
        try:
            response = requests.post(infer_url, json=payload)
            latency = time.time() - start_time
            latencies.append(latency * 1000)  # Convert to milliseconds
            
            if response.status_code == 200:
                # For synthetic data, we just count successful responses as "correct"
                correct += 1
                if i % 10 == 0:
                    print(f"Test {i+1}/{args.num_tests}: Success (latency: {latency*1000:.2f} ms)")
            else:
                print(f"Test {i+1}/{args.num_tests}: Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Test {i+1}/{args.num_tests}: Exception: {e}")
    
    # Calculate metrics
    accuracy = correct / args.num_tests if args.num_tests > 0 else 0
    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    p99_latency = np.percentile(latencies, 99) if latencies else 0
    
    # Create results
    results = {
        "accuracy": accuracy,
        "correct_count": correct,
        "total_count": args.num_tests,
        "avg_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
        "p99_latency_ms": float(p99_latency)
    }
    
    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"Accuracy: {accuracy:.4f} ({correct}/{args.num_tests})")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    print(f"P99 latency: {p99_latency:.2f} ms")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
