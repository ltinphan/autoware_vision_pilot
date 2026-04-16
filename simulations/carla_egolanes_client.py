#!/usr/bin/env python3
"""
CARLA EgoLanes Simulation Client

This script connects to a running CARLA simulator, spawns a vehicle,
captures camera frames, and runs EgoLanes inference in real-time.

Usage:
    python3 simulations/carla_egolanes_client.py --checkpoint <path_to_checkpoint.pth>
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def parse_args():
    parser = argparse.ArgumentParser(description="CARLA EgoLanes Simulation Client")
    parser.add_argument("--checkpoint", required=True, help="Path to EgoLanes checkpoint file")
    parser.add_argument("--server-ip", default="localhost", help="CARLA server IP address")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to capture")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--output-dir", default="outputs/carla_simulation", help="Output directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device")
    return parser.parse_args()


def run_simulation(args):
    """Run CARLA simulation with EgoLanes inference."""

    try:
        import carla
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please install: pip install carla Pillow numpy")
        sys.exit(1)

    # Check for checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Create output directories
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    result_dir = output_dir / "results"
    frame_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to CARLA server at {args.server_ip}:{args.port}...")

    # Connect to CARLA
    client = carla.Client(args.server_ip, args.port)
    client.set_timeout(10.0)

    # Get world and spawn vehicle
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable synchronous mode for control
    world.apply_settings(settings)

    # Find vehicle blueprint
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.*")[0]

    # Find spawn point
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("Error: No spawn points available")
        sys.exit(1)

    # Spawn vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print(f"Spawned vehicle: {vehicle.id}")

    # Spawn camera
    camera_bp = bp_lib.filter("sensor.camera.rgb*")[0]
    camera_transform = carla.Transform(
        carla.Location(x=2.0, z=1.7),
        carla.Rotation(pitch=-15)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    print(f"Spawned camera attached to vehicle")

    # Import EgoLanes inference
    from Models.inference.ego_lanes_infer import EgoLanesNetworkInfer

    # Initialize EgoLanes model
    print(f"Loading EgoLanes model from: {args.checkpoint}")
    egolanes_infer = EgoLanesNetworkInfer(checkpoint_path=args.checkpoint)

    frame_count = 0

    def receive_frame(frame):
        """Callback for receiving camera frames."""
        nonlocal frame_count

        if frame_count >= args.frames:
            return

        frame_count += 1

        # Convert to PIL Image
        image_data = frame.raw_data
        np_image = np.frombuffer(image_data, dtype=np.uint8)
        np_image = np_image.reshape((frame.height, frame.width, 3))
        pil_image = Image.fromarray(np_image)

        # Save frame
        frame_path = frame_dir / f"frame_{frame_count:04d}.png"
        pil_image.save(frame_path)

        # Run EgoLanes inference
        print(f"Frame {frame_count}/{args.frames}: Running EgoLanes inference...")

        try:
            prediction = egolanes_infer.inference(pil_image)

            # Save prediction
            pred_path = result_dir / f"prediction_{frame_count:04d}.npy"
            np.save(pred_path, prediction)

            print(f"  Prediction shape: {prediction.shape}")

        except Exception as e:
            print(f"  Inference error: {e}")

    # Subscribe camera to callback
    camera.listen(receive_frame)

    print(f"Capturing {args.frames} frames...")

    # Tick simulation
    for _ in range(args.frames):
        world.tick()

    print(f"\nSimulation complete!")
    print(f"Frames saved to: {frame_dir}")
    print(f"Predictions saved to: {result_dir}")

    # Cleanup
    camera.destroy()
    vehicle.destroy()
    print("Actors cleaned up.")


def main():
    args = parse_args()

    print("===========================================")
    print("  CARLA EgoLanes Simulation")
    print("===========================================\n")

    run_simulation(args)


if __name__ == "__main__":
    main()
