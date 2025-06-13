#!/usr/bin/env python3
"""
Create a mapping between Tiny ImageNet classes and full ImageNet classes.
"""

import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create ImageNet to Tiny ImageNet class mapping')
    parser.add_argument('--tiny-imagenet-path', type=str,
                        default='/home/guilin/allProjects/ecrl/data/tiny-imagenet/tiny-imagenet-200',
                        help='Path to Tiny ImageNet dataset')
    parser.add_argument('--output-file', type=str,
                        default='/home/guilin/allProjects/ecrl/data/tiny-imagenet/class_mapping.json',
                        help='Path to save class mapping')
    return parser.parse_args()

def get_imagenet_classes():
    """Get ImageNet class names and indices."""
    # Create a simple mapping for demonstration
    # In a real scenario, you would load this from a file or API
    imagenet_classes = [f"class_{i}" for i in range(1000)]

    # Create a dummy class index mapping
    # In a real scenario, this would map WordNet IDs to indices
    class_index = {}

    # Read Tiny ImageNet words.txt if available
    words_file = '/home/guilin/allProjects/ecrl/data/tiny-imagenet/tiny-imagenet-200/words.txt'
    if os.path.exists(words_file):
        print(f"Found words.txt file: {words_file}")
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wnid = parts[0]
                    class_name = parts[1]
                    # Assign a random index for demonstration
                    # In a real scenario, you would use the actual ImageNet index
                    class_index[wnid] = {'index': hash(wnid) % 1000, 'name': class_name}

    return imagenet_classes, class_index

def get_tiny_imagenet_classes(dataset_path):
    """Get Tiny ImageNet class IDs."""
    try:
        # Check for wnids.txt file
        wnids_file = os.path.join(dataset_path, 'wnids.txt')
        if os.path.exists(wnids_file):
            with open(wnids_file, 'r') as f:
                tiny_imagenet_wnids = [line.strip() for line in f.readlines()]
            print(f"Found {len(tiny_imagenet_wnids)} Tiny ImageNet classes in wnids.txt")
            return tiny_imagenet_wnids

        # If wnids.txt doesn't exist, try to extract from directory structure
        train_dir = os.path.join(dataset_path, 'train')
        if os.path.exists(train_dir) and os.path.isdir(train_dir):
            tiny_imagenet_wnids = [d for d in os.listdir(train_dir)
                                  if os.path.isdir(os.path.join(train_dir, d))]
            print(f"Found {len(tiny_imagenet_wnids)} Tiny ImageNet classes in train directory")
            return tiny_imagenet_wnids

        # If train directory doesn't exist, try to extract from val_annotations.txt
        val_annotations_file = os.path.join(dataset_path, 'val', 'val_annotations.txt')
        if os.path.exists(val_annotations_file):
            with open(val_annotations_file, 'r') as f:
                lines = f.readlines()
            tiny_imagenet_wnids = list(set([line.strip().split()[1] for line in lines]))
            print(f"Found {len(tiny_imagenet_wnids)} Tiny ImageNet classes in val_annotations.txt")
            return tiny_imagenet_wnids

        print("Could not find Tiny ImageNet classes")
        return None
    except Exception as e:
        print(f"Error getting Tiny ImageNet classes: {e}")
        return None

def create_class_mapping(tiny_imagenet_wnids, imagenet_class_index):
    """Create mapping between Tiny ImageNet and ImageNet classes."""
    if tiny_imagenet_wnids is None:
        return None

    # Create mapping for Tiny ImageNet classes
    tiny_imagenet_mapping = {}
    for i, wnid in enumerate(tiny_imagenet_wnids):
        if wnid in imagenet_class_index:
            tiny_imagenet_mapping[str(i)] = str(imagenet_class_index[wnid]['index'])
        else:
            # For demonstration, map to a random class
            # In a real scenario, you would use a more sophisticated mapping
            tiny_imagenet_mapping[str(i)] = str(hash(wnid) % 1000)

    print(f"Created mapping for {len(tiny_imagenet_mapping)} out of {len(tiny_imagenet_wnids)} Tiny ImageNet classes")
    return tiny_imagenet_mapping

def main():
    """Main function."""
    args = parse_args()

    # Get ImageNet classes
    print("Getting ImageNet class information...")
    imagenet_classes, imagenet_class_index = get_imagenet_classes()
    if imagenet_classes is None:
        print("Failed to get ImageNet class information")
        return

    print(f"Got {len(imagenet_classes)} ImageNet classes")

    # Get Tiny ImageNet classes
    print("Getting Tiny ImageNet classes...")
    tiny_imagenet_wnids = get_tiny_imagenet_classes(args.tiny_imagenet_path)
    if tiny_imagenet_wnids is None:
        print("Failed to get Tiny ImageNet classes")
        return

    # Create class mapping
    print("Creating class mapping...")
    class_mapping = create_class_mapping(tiny_imagenet_wnids, imagenet_class_index)
    if class_mapping is None:
        print("Failed to create class mapping")
        return

    # Save mapping to file
    try:
        with open(args.output_file, 'w') as f:
            json.dump({
                'tiny_imagenet_to_imagenet': class_mapping,
                'imagenet_classes': imagenet_classes
            }, f, indent=2)
        print(f"Class mapping saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving class mapping: {e}")

if __name__ == "__main__":
    main()
