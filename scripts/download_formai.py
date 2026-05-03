"""
Script to download and process the FormAI dataset.
Source: https://github.com/FormAI-Dataset/FormAI-dataset

The FormAI dataset contains ~112,000 AI-generated C programs.
This script clones the repository or downloads a subset if a limit is specified.
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

logger = setup_logger("FormAIDownload")

FORMAI_REPO_URL = "https://github.com/FormAI-Dataset/FormAI-dataset.git"
DEFAULT_TARGET_DIR = "data/raw/formai"


def download_formai(target_dir: str = DEFAULT_TARGET_DIR, limit: int = None, clean: bool = False):
    """
    Clone the FormAI dataset repository.

    Args:
        target_dir: Directory to store the dataset
        limit: Maximum number of programs to keep (if None, keep all)
        clean: If True, remove existing directory before cloning
    """
    target_path = Path(target_dir)

    if clean and target_path.exists():
        logger.info(f"Cleaning existing directory: {target_dir}")
        shutil.rmtree(target_path)

    if not target_path.exists():
        os.makedirs(target_path, exist_ok=True)
        logger.info(f"Cloning FormAI dataset into {target_dir}...")
        
        try:
            # Clone only history depth 1 to save time
            subprocess.run([
                "git", "clone", "--depth", "1", FORMAI_REPO_URL, str(target_path)
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Clone completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr.decode('utf-8')}")
            return False
    else:
        logger.info(f"Directory {target_dir} already exists. Skipping clone.")

    # Find all .c files
    c_files = list(target_path.rglob("*.c"))
    logger.info(f"Found {len(c_files)} C programs in the FormAI dataset.")

    if limit is not None and limit < len(c_files):
        logger.info(f"Limiting dataset to {limit} files. Removing excess files...")
        files_to_remove = c_files[limit:]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
                
        logger.info(f"Successfully truncated dataset to {limit} files.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FormAI Dataset")
    parser.add_argument("--dir", type=str, default=DEFAULT_TARGET_DIR, help="Target directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--clean", action="store_true", help="Clean existing directory")
    
    args = parser.parse_args()
    download_formai(args.dir, args.limit, args.clean)
