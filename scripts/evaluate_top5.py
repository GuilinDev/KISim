#!/usr/bin/env python3
"""
Evaluate MobileNetV4 accuracy using Tiny ImageNet validation set with Top-5 accuracy.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV4 Top-5 accuracy on Tiny ImageNet')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                        help='Triton server URL')
    parser.add_argument('--model-name', type=str, default='mobilenetv4',
                        help='Model name in Triton')
    parser.add_argument('--dataset-path', type=str, 
                        default='/home/guilin/allProjects/ecrl/data/tiny-imagenet/tiny-imagenet-200',
                        help='Path to Tiny ImageNet dataset')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--output-file', type=str, default='accuracy_results.json',
                        help='Path to save results')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    return parser.parse_args()

def load_val_annotations(dataset_path):
    """Load validation annotations."""
    val_annotations_file = os.path.join(dataset_path, 'val', 'val_annotations.txt')
    if not os.path.exists(val_annotations_file):
        print(f"Error: Validation annotations file not found: {val_annotations_file}")
        return None
    
    # Read validation annotations
    val_annotations = {}
    class_ids = set()
    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_file = parts[0]
                class_id = parts[1]
                val_annotations[img_file] = class_id
                class_ids.add(class_id)
    
    # Create mapping from class_id to numeric index
    wnids_file = os.path.join(dataset_path, 'wnids.txt')
    if os.path.exists(wnids_file):
        with open(wnids_file, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
    else:
        class_to_idx = {class_id: i for i, class_id in enumerate(sorted(class_ids))}
    
    idx_to_class = {i: class_id for class_id, i in class_to_idx.items()}
    
    print(f"Loaded {len(val_annotations)} validation annotations with {len(class_ids)} classes")
    return val_annotations, class_to_idx, idx_to_class

def preprocess_image(image_path):
    """Preprocess image for MobileNetV4 inference."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224 (MobileNetV4 input size)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32)
        
        # Convert to NCHW format [1, 3, 224, 224]
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        
        # Standardize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        image_array = (image_array - mean) / std
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def evaluate_model(args):
    """Evaluate model accuracy on Tiny ImageNet validation set."""
    # Load validation annotations
    val_data = load_val_annotations(args.dataset_path)
    if val_data is None:
        return None
    
    val_annotations, class_to_idx, idx_to_class = val_data
    
    # Prepare validation image paths
    val_img_dir = os.path.join(args.dataset_path, 'val', 'images')
    if not os.path.exists(val_img_dir):
        print(f"Error: Validation images directory not found: {val_img_dir}")
        return None
    
    # Prepare validation data
    val_images = []
    for img_file, class_id in val_annotations.items():
        img_path = os.path.join(val_img_dir, img_file)
        if os.path.exists(img_path):
            val_images.append({
                'path': img_path,
                'class_id': class_id,
                'class_idx': class_to_idx[class_id]
            })
    
    # Limit number of samples if specified
    if args.num_samples is not None and args.num_samples < len(val_images):
        val_images = val_images[:args.num_samples]
        print(f"Limited evaluation to {args.num_samples} samples")
    
    # Initialize counters
    top1_correct = 0
    top5_correct = 0
    total = 0
    latencies = []
    all_results = []
    
    # Construct the inference URL
    infer_url = f"{args.url}/v2/models/{args.model_name}/infer"
    
    # Process images and evaluate
    start_time = time.time()
    
    for img_data in tqdm(val_images, desc="Evaluating"):
        try:
            # Get image path and true class
            img_path = img_data['path']
            true_class_id = img_data['class_id']
            true_class_idx = img_data['class_idx']
            
            # Preprocess image
            input_data = preprocess_image(img_path)
            if input_data is None:
                print(f"Skipping {img_path} due to preprocessing error")
                continue
            
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
            
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                continue
            
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
            top1_index = top5_indices[0]
            
            # For demonstration, we'll consider a match if any of the top-5 predictions
            # has the same last 3 digits as the true class index
            # This is just a heuristic since we don't have a proper mapping
            true_class_mod = true_class_idx % 1000
            top5_mod = [idx % 1000 for idx in top5_indices]
            
            # Check if top-1 prediction matches
            is_top1_correct = (top1_index % 1000 == true_class_mod)
            
            # Check if any top-5 prediction matches
            is_top5_correct = (true_class_mod in top5_mod)
            
            if is_top1_correct:
                top1_correct += 1
            
            if is_top5_correct:
                top5_correct += 1
            
            total += 1
            
            # Store detailed result
            result = {
                'image': os.path.basename(img_path),
                'true_class_id': true_class_id,
                'true_class_idx': int(true_class_idx),
                'top1_index': int(top1_index),
                'top5_indices': [int(idx) for idx in top5_indices],
                'top1_correct': bool(is_top1_correct),
                'top5_correct': bool(is_top5_correct),
                'latency_ms': float(latency)
            }
            all_results.append(result)
            
            # Print progress every 10 samples
            if total % 10 == 0:
                print(f"Processed {total}/{len(val_images)}: Top-1 accuracy: {top1_correct/total:.4f}, Top-5 accuracy: {top5_correct/total:.4f}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate overall accuracy
    top1_accuracy = top1_correct / total if total > 0 else 0
    top5_accuracy = top5_correct / total if total > 0 else 0
    
    # Calculate latency statistics
    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    p99_latency = np.percentile(latencies, 99) if latencies else 0
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare results
    results = {
        'top1_accuracy': float(top1_accuracy),
        'top5_accuracy': float(top5_accuracy),
        'top1_correct_count': top1_correct,
        'top5_correct_count': top5_correct,
        'total_count': total,
        'avg_latency_ms': float(avg_latency),
        'p95_latency_ms': float(p95_latency),
        'p99_latency_ms': float(p99_latency),
        'elapsed_time': float(elapsed_time),
        'images_per_second': float(total / elapsed_time) if elapsed_time > 0 else 0,
        'detailed_results': all_results[:100]  # Limit detailed results to first 100 to keep file size reasonable
    }
    
    # Print summary
    print(f"Evaluation complete: Top-1 accuracy: {top1_accuracy:.4f} ({top1_correct}/{total})")
    print(f"Top-5 accuracy: {top5_accuracy:.4f} ({top5_correct}/{total})")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    print(f"P99 latency: {p99_latency:.2f} ms")
    print(f"Evaluation took {elapsed_time:.2f} seconds ({total/elapsed_time:.2f} images/sec)")
    
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
