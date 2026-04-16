#!/usr/bin/env python3
"""
EgoLanes Inference Script
Runs EgoLanes model on images and saves predictions
"""

import sys
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "Models"))

from Models.inference.ego_lanes_infer import EgoLanesNetworkInfer

# Expected input size for EgoLanes model
# Based on visualization script FRAME_INF_SIZE
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 320


def parse_args():
    parser = argparse.ArgumentParser(description="Run EgoLanes inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()

    print("========================================")
    print("  EgoLanes Inference")
    print("========================================\n")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print(f"Loading model from: {args.checkpoint}")
    model = EgoLanesNetworkInfer(checkpoint_path=args.checkpoint)

    # Get input files
    input_path = Path(args.input)
    if input_path.is_dir():
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        image_files = [input_path]

    print(f"\nFound {len(image_files)} image(s) to process\n")

    # Process each image
    for img_file in image_files:
        print(f"Processing: {img_file.name}")

        # Load image
        image = Image.open(img_file).convert("RGB")

        # Resize to expected input size
        image = image.resize((EXPECTED_WIDTH, EXPECTED_HEIGHT), Image.BILINEAR)

        # Run inference
        prediction = model.inference(image)

        # Save prediction
        output_name = out_dir / f"{img_file.stem}_prediction.npy"
        np.save(output_name, prediction)

        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Saved to: {output_name}\n")

    print(f"Done! Processed {len(image_files)} image(s)")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
