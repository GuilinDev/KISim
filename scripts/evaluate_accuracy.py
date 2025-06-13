#!/usr/bin/env python3
"""
Evaluate the accuracy of MobileNetV4 model using Triton Inference Server.
This script processes images from a validation dataset and computes the accuracy.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from PIL import Image
import tritonclient.http as httpclient
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("accuracy-evaluator")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV4 accuracy using Triton')
    parser.add_argument('--server-url', type=str, 
                        default='mobilenetv4-triton-svc.workloads.svc.cluster.local:8000',
                        help='Triton server URL')
    parser.add_argument('--model-name', type=str, default='mobilenetv4',
                        help='Model name in Triton')
    parser.add_argument('--dataset-path', type=str, default='/data/tiny-imagenet/val',
                        help='Path to validation dataset')
    parser.add_argument('--class-map', type=str, default='/data/tiny-imagenet/val/class_map.txt',
                        help='Path to class mapping file')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--output-file', type=str, default='/results/accuracy_results.json',
                        help='Path to save results')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    return parser.parse_args()

def load_class_map(class_map_file):
    """Load class ID to name mapping."""
    class_map = {}
    try:
        with open(class_map_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_id = parts[0]  # Could be string ID or integer
                    class_name = parts[1]
                    class_map[class_id] = class_name
        logger.info(f"Loaded {len(class_map)} classes from {class_map_file}")
    except Exception as e:
        logger.warning(f"Failed to load class map from {class_map_file}: {e}")
        logger.warning("Will use numeric class IDs only")
    return class_map

def preprocess_image(image_path):
    """Preprocess image for MobileNetV4 inference."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
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
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def evaluate_model(args):
    """Evaluate model accuracy on validation dataset."""
    # Initialize counters and results
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    all_results = []
    
    # Create Triton client
    try:
        client = httpclient.InferenceServerClient(url=args.server_url, verbose=False)
        logger.info(f"Connected to Triton server at {args.server_url}")
        
        # Check if model is ready
        if not client.is_model_ready(args.model_name):
            logger.error(f"Model {args.model_name} is not ready on the server")
            return None
        
        logger.info(f"Model {args.model_name} is ready")
    except Exception as e:
        logger.error(f"Failed to connect to Triton server: {e}")
        return None
    
    # Load class mapping
    class_map = load_class_map(args.class_map)
    
    # Get list of validation images
    val_images = []
    try:
        # Check if dataset_path is a directory with class folders or a directory with images
        if os.path.isdir(args.dataset_path):
            # Check if there are subdirectories (class folders)
            subdirs = [d for d in os.listdir(args.dataset_path) 
                      if os.path.isdir(os.path.join(args.dataset_path, d))]
            
            if subdirs:
                # Dataset with class folders
                logger.info(f"Found {len(subdirs)} class folders in {args.dataset_path}")
                for class_folder in subdirs:
                    class_path = os.path.join(args.dataset_path, class_folder)
                    class_id = class_folder  # Class ID is the folder name
                    
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            val_images.append({
                                'path': os.path.join(class_path, img_file),
                                'class_id': class_id
                            })
            else:
                # Flat directory with images
                logger.info(f"Using flat directory of images in {args.dataset_path}")
                for img_file in os.listdir(args.dataset_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Try to extract class from filename (assuming format like class_id_image.jpg)
                        parts = img_file.split('_')
                        class_id = parts[0] if len(parts) > 1 else "unknown"
                        val_images.append({
                            'path': os.path.join(args.dataset_path, img_file),
                            'class_id': class_id
                        })
        else:
            logger.error(f"Dataset path {args.dataset_path} is not a directory")
            return None
            
        logger.info(f"Found {len(val_images)} validation images")
        
        # Limit number of samples if specified
        if args.num_samples is not None and args.num_samples < len(val_images):
            val_images = val_images[:args.num_samples]
            logger.info(f"Limited evaluation to {args.num_samples} samples")
    except Exception as e:
        logger.error(f"Error scanning validation dataset: {e}")
        return None
    
    # Process images and evaluate
    start_time = time.time()
    
    for img_data in tqdm(val_images, desc="Evaluating"):
        try:
            # Get image path and true class
            img_path = img_data['path']
            true_class = img_data['class_id']
            
            # Initialize class counters if needed
            if true_class not in class_total:
                class_total[true_class] = 0
                class_correct[true_class] = 0
            
            # Preprocess image
            input_data = preprocess_image(img_path)
            if input_data is None:
                logger.warning(f"Skipping {img_path} due to preprocessing error")
                continue
            
            # Create input tensor
            input_tensor = httpclient.InferInput(
                "pixel_values",  # Input tensor name
                input_data.shape,
                "FP32"
            )
            input_tensor.set_data_from_numpy(input_data)
            
            # Send inference request
            response = client.infer(
                model_name=args.model_name,
                inputs=[input_tensor]
            )
            
            # Get output
            output = response.as_numpy("logits")
            
            # Get predicted class
            predicted_class = str(np.argmax(output))
            
            # Record result
            is_correct = (predicted_class == true_class)
            if is_correct:
                correct += 1
                class_correct[true_class] += 1
            total += 1
            class_total[true_class] += 1
            
            # Store detailed result
            result = {
                'image': os.path.basename(img_path),
                'true_class': true_class,
                'true_class_name': class_map.get(true_class, true_class),
                'predicted_class': predicted_class,
                'predicted_class_name': class_map.get(predicted_class, predicted_class),
                'correct': is_correct
            }
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for class_id in class_total:
        class_accuracy[class_id] = class_correct[class_id] / class_total[class_id] if class_total[class_id] > 0 else 0
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare results
    results = {
        'overall_accuracy': accuracy,
        'correct_count': correct,
        'total_count': total,
        'per_class_accuracy': class_accuracy,
        'elapsed_time': elapsed_time,
        'images_per_second': total / elapsed_time if elapsed_time > 0 else 0,
        'detailed_results': all_results
    }
    
    # Log summary
    logger.info(f"Evaluation complete: {correct}/{total} correct, accuracy: {accuracy:.4f}")
    logger.info(f"Evaluation took {elapsed_time:.2f} seconds ({total/elapsed_time:.2f} images/sec)")
    
    return results

def save_results(results, output_file):
    """Save evaluation results to file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")

def main():
    """Main function."""
    args = parse_args()
    results = evaluate_model(args)
    if results:
        save_results(results, args.output_file)
    else:
        logger.error("Evaluation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
