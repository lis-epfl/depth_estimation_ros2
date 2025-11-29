#!/usr/bin/env python3
import os
import sys
import urllib.request

# --- CONFIGURATION ---
# Replace with your actual username and release tag
VERSION = "v1.0.0"
BASE_URL = f"https://github.com/lis-epfl/depth_estimation_ros2/releases/download/{VERSION}/"

# The specific S2M2 files you requested
FILES_TO_DOWNLOAD = [
    # S2M2 Medium variants
    "S2M2_M_128_128_v2_torch29.onnx",
    "S2M2_M_128_128_v2_torch29.onnx.data",
    "S2M2_M_160_160_v2_torch29.onnx",
    "S2M2_M_160_160_v2_torch29.onnx.data",
    "S2M2_M_192_192_v2_torch29.onnx",
    "S2M2_M_192_192_v2_torch29.onnx.data",
    "S2M2_M_224_224_v2_torch29.onnx",
    "S2M2_M_224_224_v2_torch29.onnx.data",

    # S2M2 Small variants
    "S2M2_S_128_128_v2_torch29.onnx",
    "S2M2_S_128_128_v2_torch29.onnx.data",
    "S2M2_S_160_160_v2_torch29.onnx",
    "S2M2_S_160_160_v2_torch29.onnx.data",
    "S2M2_S_192_192_v2_torch29.onnx",
    "S2M2_S_192_192_v2_torch29.onnx.data",
    "S2M2_S_224_224_v2_torch29.onnx",
    "S2M2_S_224_224_v2_torch29.onnx.data"
]

def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()

def main():
    # 1. Determine paths based on THIS script's location
    # This ensures files go into 'src/depth_estimation_ros2/models'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir) # Go up one level from scripts/
    models_dir = os.path.join(package_root, 'models')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    print(f"Target Directory: {models_dir}")

    # 2. Download files
    for filename in FILES_TO_DOWNLOAD:
        dest_path = os.path.join(models_dir, filename)

        if os.path.exists(dest_path):
            print(f" - {filename} exists. Skipping.")
            continue

        url = BASE_URL + filename
        try:
            print(f" - Fetching {filename}...")
            urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
            print(" Done.")
        except Exception as e:
            print(f"\nError downloading {url}: {e}")
            # Optional: remove partial file on failure
            if os.path.exists(dest_path):
                os.remove(dest_path)

    print("\nAll models downloaded to source directory.")

if __name__ == "__main__":
    main()
