"""
Script to download the Adarsh Biradar C-programs dataset from Kaggle.
Source: https://www.kaggle.com/datasets/adarshbiradar/c-programs

This dataset contains ~270 real-world algorithm and data structure C programs.
Requires the 'kaggle' python package and a kaggle.json API token configured.
"""
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger("KaggleDownload")

KAGGLE_DATASET = "adarshbiradar/c-programs"
DEFAULT_TARGET_DIR = "data/raw/kaggle_c_programs"


def check_kaggle_auth():
    """Verify Kaggle authentication is configured."""
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    has_file = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    
    if not (has_env or has_file):
        logger.error("Kaggle authentication not found.")
        logger.error("Please place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env variables.")
        return False
    return True


def download_kaggle_dataset(target_dir: str = DEFAULT_TARGET_DIR, clean: bool = False):
    """
    Download the dataset via Kaggle API.

    Args:
        target_dir: Directory to extract the dataset into
        clean: If True, remove existing directory
    """
    try:
        import kaggle
    except ImportError:
        logger.error("The 'kaggle' package is not installed.")
        logger.info("Please install it: pip install kaggle")
        return False

    if not check_kaggle_auth():
        return False

    target_path = Path(target_dir)

    if clean and target_path.exists():
        logger.info(f"Cleaning existing directory: {target_dir}")
        shutil.rmtree(target_path)

    os.makedirs(target_path, exist_ok=True)
    
    logger.info(f"Downloading Kaggle dataset '{KAGGLE_DATASET}' to {target_dir}...")
    try:
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(target_path),
            unzip=True
        )
        logger.info("Download and extraction completed successfully.")
        
        # Some Kaggle datasets download as single large text files or flat structures.
        files = list(target_path.glob("*"))
        logger.info(f"Files found: {[f.name for f in files]}")
        
    except Exception as e:
        logger.error(f"Failed to download Kaggle dataset: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle C-Programs Dataset")
    parser.add_argument("--dir", type=str, default=DEFAULT_TARGET_DIR, help="Target directory")
    parser.add_argument("--clean", action="store_true", help="Clean existing directory")
    
    args = parser.parse_args()
    download_kaggle_dataset(args.dir, args.clean)
