from huggingface_hub import hf_hub_download
import os
import shutil # Changed from os.rename to shutil.move for robustness
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_place_model(
    repo_id: str = "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
    filename_in_repo: str = "onnx/model.onnx", # Filename as it is in the HF repo
    output_dir: str = "../models",  # Relative to experiments/scripts
    model_subpath: str = "mobilenetv4/1",
    target_filename: str = "model.onnx" # Desired final filename
):
    """
    Downloads an ONNX model from Hugging Face Hub and places it in the desired directory structure.

    Args:
        repo_id (str): The Hugging Face Hub repository ID.
        filename_in_repo (str): The specific file to download from the repository (e.g., "onnx/model.onnx").
        output_dir (str): The base directory where the 'model_subpath' will be created.
                          This path is relative to the script's location if not absolute.
        model_subpath (str): The sub-directory structure (e.g., "mobilenetv4/1") where the model.onnx will be placed.
        target_filename (str): The desired final filename for the downloaded model.
    """
    try:
        logging.info(f"Downloading {filename_in_repo} from Hugging Face Hub repo: {repo_id}")

        # Construct the final directory path for the model (e.g., ../models/mobilenetv4/1)
        final_model_parent_dir = os.path.abspath(os.path.join(output_dir, model_subpath))
        os.makedirs(final_model_parent_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {final_model_parent_dir}")

        # Define the absolute final target path for the model (e.g., /abs/path/to/ecrl/experiments/models/mobilenetv4/1/model.onnx)
        absolute_target_model_path = os.path.join(final_model_parent_dir, target_filename)

        # hf_hub_download will save the file maintaining the structure from filename_in_repo
        # inside a temporary-like structure or directly if local_dir is specified carefully.
        # For robustness, let it download, then we move.
        # We'll use a temporary download directory within final_model_parent_dir to avoid polluting it if script aborts.
        temp_download_stage_dir = os.path.join(final_model_parent_dir, "temp_download_stage")
        os.makedirs(temp_download_stage_dir, exist_ok=True)

        actual_downloaded_path_in_stage = hf_hub_download(
            repo_id=repo_id,
            filename=filename_in_repo,
            local_dir=temp_download_stage_dir, # Download into a specific staging area
            local_dir_use_symlinks=False,
            # cache_dir=False # Consider if direct download without caching is better if space is an issue or for CI
        )
        logging.info(f"File initially downloaded to: {actual_downloaded_path_in_stage}")
        
        # Ensure the downloaded file exists before trying to move
        if not os.path.exists(actual_downloaded_path_in_stage):
            logging.error(f"Downloaded file not found at {actual_downloaded_path_in_stage}. Aborting.")
            if os.path.isdir(temp_download_stage_dir):
                shutil.rmtree(temp_download_stage_dir) # Clean up temp dir
            return

        # Move the downloaded file to the absolute_target_model_path
        # This handles cases where actual_downloaded_path_in_stage might be nested (e.g. temp_download_stage/onnx/model.onnx)
        logging.info(f"Moving file from {actual_downloaded_path_in_stage} to {absolute_target_model_path}")
        shutil.move(actual_downloaded_path_in_stage, absolute_target_model_path)
        
        # Clean up the temporary staging directory if it's empty (or the whole thing if it was just for the file)
        # If filename_in_repo had subdirs, those would be created under temp_download_stage_dir.
        # After moving the file, the original directory structure might be empty.
        if os.path.isdir(temp_download_stage_dir):
             try:
                # Attempt to remove the directory; fails if not empty (which is fine, means something else was there)
                os.rmdir(os.path.dirname(actual_downloaded_path_in_stage)) # remove parent of file if it was nested
                if os.path.dirname(actual_downloaded_path_in_stage) != temp_download_stage_dir: # if it was nested like 'onnx/'
                     os.rmdir(temp_download_stage_dir) # then remove the base temp dir
             except OSError:
                # If the directory is not empty or other issues, just leave it. Or be more aggressive:
                shutil.rmtree(temp_download_stage_dir) # This will remove it and any other leftover contents forcefully
                logging.info(f"Cleaned up temporary download stage directory: {temp_download_stage_dir}")

        if os.path.exists(absolute_target_model_path):
            logging.info(f"Model successfully downloaded and placed at: {absolute_target_model_path}")
        else:
            logging.error(f"Model processing failed. File not found at expected final path: {absolute_target_model_path}")

    except Exception as e:
        logging.error(f"An error occurred during model download: {e}", exc_info=True)
        # Clean up temp dir on error too
        if 'temp_download_stage_dir' in locals() and os.path.isdir(temp_download_stage_dir):
            shutil.rmtree(temp_download_stage_dir)

if __name__ == "__main__":
    # This script is intended to be run from experiments/scripts directory
    # The output_dir="../models" will correctly place the model in experiments/models/mobilenetv4/1/model.onnx
    download_and_place_model()
    logging.info("Ensure you have 'huggingface_hub' installed: pip install huggingface_hub") 