#!/usr/bin/env python3
"""
Download and prepare Tiny ImageNet dataset for accuracy evaluation.
"""

import os
import sys
import logging
import argparse
import tarfile
import shutil
from urllib.request import urlretrieve
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tiny-imagenet-downloader")

# Constants
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download and prepare Tiny ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='./data/tiny-imagenet',
                        help='Directory to save the dataset')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download the dataset, do not extract or prepare')
    return parser.parse_args()

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return True
    
    try:
        logger.info(f"Downloading {url} to {output_path}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def extract_archive(archive_path, output_dir):
    """Extract a tar or zip archive."""
    try:
        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            logger.info(f"Extracting {archive_path} to {output_dir}")
            with tarfile.open(archive_path) as tar:
                # Get the total number of members
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting"):
                    tar.extract(member, path=output_dir)
        elif archive_path.endswith('.zip'):
            logger.info(f"Extracting {archive_path} to {output_dir}")
            import zipfile
            with zipfile.ZipFile(archive_path) as zip_ref:
                # Get the total number of members
                members = zip_ref.infolist()
                for member in tqdm(members, desc="Extracting"):
                    zip_ref.extract(member, path=output_dir)
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        return False

def prepare_validation_data(tiny_imagenet_dir):
    """Prepare validation data for easier use."""
    val_dir = os.path.join(tiny_imagenet_dir, 'val')
    if not os.path.exists(val_dir):
        logger.error(f"Validation directory not found: {val_dir}")
        return False
    
    # Check if val_annotations.txt exists
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.exists(val_annotations_file):
        logger.error(f"Validation annotations file not found: {val_annotations_file}")
        return False
    
    # Create class directories
    logger.info("Organizing validation images into class directories")
    val_img_dir = os.path.join(val_dir, 'images')
    if not os.path.exists(val_img_dir):
        logger.error(f"Validation images directory not found: {val_img_dir}")
        return False
    
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
    
    # Create class map file
    class_map_file = os.path.join(val_dir, 'class_map.txt')
    with open(class_map_file, 'w') as f:
        for i, class_id in enumerate(sorted(class_ids)):
            f.write(f"{i} {class_id}\n")
    
    # Create mapping from class_id to numeric index
    class_to_idx = {class_id: i for i, class_id in enumerate(sorted(class_ids))}
    
    # Create organized directory structure
    for img_file, class_id in tqdm(val_annotations.items(), desc="Organizing validation images"):
        src_path = os.path.join(val_img_dir, img_file)
        if not os.path.exists(src_path):
            logger.warning(f"Image file not found: {src_path}")
            continue
        
        # Get numeric class index
        class_idx = class_to_idx[class_id]
        
        # Create class directory
        class_dir = os.path.join(val_dir, str(class_idx))
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy image to class directory
        dst_path = os.path.join(class_dir, img_file)
        shutil.copy2(src_path, dst_path)
    
    logger.info(f"Validation data prepared with {len(class_ids)} classes")
    return True

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download Tiny ImageNet
    tiny_imagenet_zip = os.path.join(args.output_dir, 'tiny-imagenet-200.zip')
    if not download_file(TINY_IMAGENET_URL, tiny_imagenet_zip):
        logger.error("Failed to download Tiny ImageNet")
        sys.exit(1)
    
    if args.download_only:
        logger.info("Download completed. Exiting as requested.")
        sys.exit(0)
    
    # Extract archive
    if not extract_archive(tiny_imagenet_zip, args.output_dir):
        logger.error("Failed to extract Tiny ImageNet archive")
        sys.exit(1)
    
    # Prepare validation data
    tiny_imagenet_extracted_dir = os.path.join(args.output_dir, 'tiny-imagenet-200')
    if not prepare_validation_data(tiny_imagenet_extracted_dir):
        logger.error("Failed to prepare validation data")
        sys.exit(1)
    
    # Copy validation directory to the top level for easier access
    src_val_dir = os.path.join(tiny_imagenet_extracted_dir, 'val')
    dst_val_dir = os.path.join(args.output_dir, 'val')
    if os.path.exists(dst_val_dir):
        logger.info(f"Validation directory already exists at {dst_val_dir}")
    else:
        logger.info(f"Copying validation directory to {dst_val_dir}")
        shutil.copytree(src_val_dir, dst_val_dir)
    
    logger.info("Tiny ImageNet dataset prepared successfully")

if __name__ == "__main__":
    main()
