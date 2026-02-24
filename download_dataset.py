#!/usr/bin/env python3
"""
Download Health & Fitness Dataset from Kaggle using kagglehub

Usage:
    python download_dataset.py
    
This will:
1. Download the health-and-fitness-dataset from Kaggle
2. Show you the path where it's saved
3. You can then use that path in your .env file as DATASET_PATH
"""

import os
import sys

def download_dataset():
    """Download the health & fitness dataset from Kaggle"""
    try:
        import kagglehub
    except ImportError:
        print(" kagglehub not installed. Install with: pip install kagglehub")
        sys.exit(1)

    print("   Downloading health-and-fitness-dataset from Kaggle...")
    print("   Note: You need Kaggle API credentials. See https://www.kaggle.com/settings/account")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("evan65549/health-and-fitness-dataset")
        
        print(f"\n Dataset downloaded successfully!")
        print(f" Location: {path}")
        
        # List the files
        print(f"\n Files in dataset:")
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   • {file} ({size_mb:.2f} MB)")
        
        print(f"\n Add to your .env file:")
        print(f"   DATASET_PATH={path}")
        
        return path
        
    except Exception as e:
        print(f"\n Error downloading dataset: {e}")
        print("\n Make sure:")
        print("   1. You have kagglehub installed: pip install kagglehub")
        print("   2. You have Kaggle API credentials at ~/.kaggle/kaggle.json")
        print("   3. You accepted the dataset terms at https://www.kaggle.com/datasets/evan65549/health-and-fitness-dataset")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()