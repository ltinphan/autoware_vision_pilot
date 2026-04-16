#!/usr/bin/env python3
"""CLI entrypoint for MVP perception pipeline (contract only).
Provides argument parsing and dispatch points; does not load model weights.
"""
import argparse
import os
import sys
from pathlib import Path
# Ensure repository root is on sys.path so `import Models...` works when executed as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def parse_args():
    p = argparse.ArgumentParser(prog="visionpilot-cli", description="Run model inference on images")
    p.add_argument("--model", required=True, help="Model id (e.g. AutoSpeed, SceneSeg)")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint file")
    p.add_argument("--input", required=True, help="Path to input image or directory")
    p.add_argument("--out_dir", required=True, help="Directory to write outputs")
    p.add_argument("--config", help="Optional YAML config to override flags")
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # Basic validation
    if not os.path.exists(args.input):
        print(f"Input path not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    try:
        os.makedirs(args.out_dir, exist_ok=True)
    except Exception as e:
        print(f"Unable to create out_dir {args.out_dir}: {e}", file=sys.stderr)
        sys.exit(3)

    # Dispatch and wiring
    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(4)

    # Minimal model map for dynamic dispatch
    model_map = {
        "SceneSeg": ("scene_seg_infer", "SceneSegNetworkInfer"),
        "AutoSpeed": ("auto_speed_infer", "AutoSpeedNetworkInfer"),
        "EgoLanes": ("ego_lanes_infer", "EgoLanesNetworkInfer"),
    }

    if args.model not in model_map:
        print(f"Unknown model id: {args.model}", file=sys.stderr)
        sys.exit(5)

    mod_file, cls_name = model_map[args.model]
    try:
        module = __import__(f"Models.inference.{mod_file}", fromlist=[cls_name])
        ModelClass = getattr(module, cls_name)
    except Exception as e:
        print(f"Failed to import model module Models.inference.{mod_file}: {e}", file=sys.stderr)
        sys.exit(6)

    try:
        model_instance = ModelClass(checkpoint_path=args.checkpoint)
    except Exception as e:
        print(f"Failed to initialize model with checkpoint: {e}", file=sys.stderr)
        sys.exit(7)

    print("Model initialized (dry-run).")

if __name__ == '__main__':
    main()
