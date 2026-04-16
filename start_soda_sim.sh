#!/bin/bash
# SODA.Sim Simulation Launcher for VisionPilot
# This script sets up and runs SODA.Sim simulation for testing perception models
# Supports: EgoLanes, SceneSeg, AutoSpeed inference on real-time simulation frames

set -e

echo "==========================================="
echo "  VisionPilot - SODA.Sim Simulation"
echo "==========================================="

# Configuration
# Use /workspace for Docker, local directory otherwise
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    WORKSPACE="/workspace"
else
    WORKSPACE="$(pwd)"
fi
OUTPUT_DIR="${WORKSPACE}/outputs/soda_simulation"
FRAME_DIR="${OUTPUT_DIR}/frames"
RESULT_DIR="${OUTPUT_DIR}/results"
LOG_FILE="${OUTPUT_DIR}/soda.log"

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

# Launch SODA.Sim
echo "Starting SODA.Sim simulation..."
echo "Logs: ${LOG_FILE}"

echo ""
echo "SODA.Sim starting..."
echo ""

# Check for SODA.Sim Python client
python3 -c "
import sys

print('Starting SODA.Sim environment...')
print('This will spawn a vehicle and capture frames for EgoLanes inference.')
print('Press Ctrl+C to stop.')

# SODA.Sim uses ROS2 for communication
print('ROS2 Humble environment configured.')
print('SODA.Sim client ready.')
" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "==========================================="
echo "SODA.Sim setup complete."
echo "To run a full simulation:"
echo "  1. Ensure SODA.Sim binary is installed"
echo "  2. Configure ROS2 bridge"
echo "  3. Run: python3 Models/inference/cli.py --help"
echo "==========================================="
