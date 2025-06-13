#!/usr/bin/env python3
"""
Evaluate MobileNetV4 using synthetic data to test model responsiveness.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import requests
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV4 with synthetic data')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                        help='Triton server URL')
    parser.add_argument('--model-name', type=str, default='mobilenetv4',
                        help='Model name in Triton')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--output-file', type=str, default='synthetic_results.json',
                        help='Path to save results')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    return parser.parse_args()

def generate_synthetic_image():
    """Generate a synthetic image tensor."""
    # Generate random data in the shape expected by MobileNetV4
    # [1, 3, 224, 224] - batch size 1, 3 channels, 224x224 pixels
    image_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    
    # Normalize to [0, 1]
    image_data = image_data / 255.0
    
    # Standardize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image_data = (image_data - mean) / std
    
    return image_data

def evaluate_model(args):
    """Evaluate model using synthetic data."""
    # Initialize counters
    total = 0
    successful = 0
    latencies = []
    all_results = []
    
    # Construct the inference URL
    infer_url = f"{args.url}/v2/models/{args.model_name}/infer"
    
    # Process synthetic images
    start_time = time.time()
    
    for i in tqdm(range(args.num_samples), desc="Evaluating"):
        try:
            # Generate synthetic image
            input_data = generate_synthetic_image()
            
            # Create request payload
            payload = {
                "inputs": [
                    {
                        "name": "pixel_values",
                        "shape": list(input_data.shape),
                        "datatype": "FP32",
                        "data": input_data.flatten().tolist()
                    }
                ]
            }
            
            # Send request
            request_start = time.time()
            response = requests.post(infer_url, json=payload)
            latency = (time.time() - request_start) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            if response.status_code == 200:
                successful += 1
                
                # Parse response
                response_data = response.json()
                output_data = None
                
                # Find the output tensor (usually named "logits" or similar)
                for output in response_data.get('outputs', []):
                    if output.get('name') in ['logits', 'output', 'predictions']:
                        output_data = np.array(output.get('data')).reshape(output.get('shape'))
                        break
                
                if output_data is None:
                    print(f"Error: Could not find output tensor in response")
                    continue
                
                # Get top-5 predicted classes
                top5_indices = np.argsort(output_data[0])[-5:][::-1]
                
                # Store result
                result = {
                    'sample_id': i,
                    'top5_indices': [int(idx) for idx in top5_indices],
                    'top5_values': [float(output_data[0][idx]) for idx in top5_indices],
                    'latency_ms': float(latency)
                }
                all_results.append(result)
                
                if args.debug:
                    print(f"Sample {i}: Top-5 classes: {top5_indices}")
                    print(f"Latency: {latency:.2f} ms")
            else:
                print(f"Error: {response.status_code} - {response.text}")
            
            total += 1
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{args.num_samples}: Success rate: {successful/(i+1):.4f}")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            total += 1
    
    # Calculate success rate
    success_rate = successful / total if total > 0 else 0
    
    # Calculate latency statistics
    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    p99_latency = np.percentile(latencies, 99) if latencies else 0
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Check if model is responsive
    is_responsive = (success_rate > 0.9)  # Consider model responsive if >90% requests succeed
    
    # Analyze output distribution
    class_distribution = {}
    if all_results:
        # Count occurrences of each class in top-1 predictions
        for result in all_results:
            top1_class = result['top5_indices'][0]
            class_distribution[top1_class] = class_distribution.get(top1_class, 0) + 1
        
        # Sort by frequency
        class_distribution = {k: v for k, v in sorted(class_distribution.items(), 
                                                     key=lambda item: item[1], 
                                                     reverse=True)}
    
    # Prepare results
    results = {
        'success_rate': float(success_rate),
        'successful_count': successful,
        'total_count': total,
        'is_responsive': is_responsive,
        'avg_latency_ms': float(avg_latency),
        'p95_latency_ms': float(p95_latency),
        'p99_latency_ms': float(p99_latency),
        'elapsed_time': float(elapsed_time),
        'samples_per_second': float(total / elapsed_time) if elapsed_time > 0 else 0,
        'top_classes': list(class_distribution.items())[:10],  # Top 10 most frequent classes
        'detailed_results': all_results[:20]  # Limit detailed results to first 20 to keep file size reasonable
    }
    
    # Print summary
    print(f"Evaluation complete: Success rate: {success_rate:.4f} ({successful}/{total})")
    print(f"Model is {'responsive' if is_responsive else 'not responsive'}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    print(f"P99 latency: {p99_latency:.2f} ms")
    print(f"Evaluation took {elapsed_time:.2f} seconds ({total/elapsed_time:.2f} samples/sec)")
    
    if class_distribution:
        print("Top 5 most frequent classes:")
        for i, (class_id, count) in enumerate(list(class_distribution.items())[:5]):
            print(f"  {i+1}. Class {class_id}: {count} occurrences ({count/len(all_results)*100:.1f}%)")
    
    return results

def save_results(results, output_file):
    """Save evaluation results to file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

def main():
    """Main function."""
    args = parse_args()
    results = evaluate_model(args)
    if results:
        save_results(results, args.output_file)
    else:
        print("Evaluation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
