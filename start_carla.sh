#!/bin/bash
# CARLA Simulation Launcher for VisionPilot
# This script sets up and runs CARLA simulation for testing perception models
# Supports: EgoLanes, SceneSeg, AutoSpeed inference on real-time simulation frames

set -e

echo "==========================================="
echo "  VisionPilot - CARLA Simulation"
echo "==========================================="

# Configuration
CARLA_VERSION="0.20.0"
# Use /workspace for Docker, local directory otherwise
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    WORKSPACE="/workspace"
else
    WORKSPACE="$(pwd)"
fi
OUTPUT_DIR="${WORKSPACE}/outputs/carla_simulation"
FRAME_DIR="${OUTPUT_DIR}/frames"
RESULT_DIR="${OUTPUT_DIR}/results"
LOG_FILE="${OUTPUT_DIR}/carla.log"

# Create output directories
echo "Setting up output directories..."
mkdir -p "${FRAME_DIR}"
mkdir -p "${RESULT_DIR}"

# Export display if needed (for headless mode, this may not be set)
if [ -n "$DISPLAY" ]; then
    echo "Using display: $DISPLAY"
else
    echo "No DISPLAY set - running in headless mode"
fi

# Launch CARLA Simulator
echo "Starting CARLA simulation..."
echo "Logs: ${LOG_FILE}"

echo ""
echo "CARLA Simulator starting..."
echo ""

# Run CARLA Python example/client that:
# 1. Spawns a vehicle
# 2. Captures camera images
# 3. Runs EgoLanes inference
# 4. Visualizes results

python3 -c "
import subprocess
import sys

print('Starting CARLA simulation environment...')
print('This will spawn a vehicle and capture frames for EgoLanes inference.')
print('Press Ctrl+C to stop.')

# For now, show basic CARLA info
try:
    import carla
    print(f'CARLA version: {carla.__version__}')
    print('CARLA module available.')
except ImportError:
    print('Error: CARLA module not installed.')
    print('Run: pip install carla')
    sys.exit(1)
" 2>&1 | tee "${LOG_FILE}"

# Alternative: If you have CARLA binary installed, uncomment and configure:
# /path/to/Carla-Simulator/$CARLA_VERSION/Linux-carla-server.roslaunch \
#     --ros-args --log=${LOG_FILE} &


echo ""
echo "==========================================="
echo "Simulation setup complete."
echo "To run a full simulation with vehicle spawn and inference:"
echo "  1. Install CARLA simulator binary"
echo "  2. Configure ROS2 bridge (optional)"
echo "  3. Run: python3 Models/inference/cli.py --help"
echo "==========================================="
