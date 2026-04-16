#!/bin/bash
# Download script for EgoLanes model checkpoint
# This script downloads the trained PyTorch model weights

set -e

# Configuration
MODEL_DIR="Models/model_library/EgoLanes"
CHECKPOINT_FILE="checkpoint.pth"
CHECKPOINT_URL="https://drive.google.com/uc?export=download&id=1Njo9EEc2tdU1ffo8AUQ9mjwuQ9CzSRPX"

# Create model directory
echo "Creating model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# Check if file already exists
if [ -f "${MODEL_DIR}/${CHECKPOINT_FILE}" ]; then
    echo "Checkpoint already exists at ${MODEL_DIR}/${CHECKPOINT_FILE}"
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Download cancelled."
        exit 0
    fi
fi

# Download checkpoint
echo "Downloading EgoLanes checkpoint from Google Drive..."
echo "This may take a few minutes depending on your connection."
echo "Saving to: ${MODEL_DIR}/${CHECKPOINT_FILE}"

# Try using gdown for Google Drive downloads (handles confirmation)
if command -v gdown &> /dev/null; then
    echo "Using gdown for Google Drive download..."
    gdown -O "${MODEL_DIR}/${CHECKPOINT_FILE}" "1Njo9EEc2tdU1ffo8AUQ9mjwuQ9CzSRPX"
elif command -v pip &> /dev/null; then
    echo "Installing gdown for Google Drive download..."
    pip install -q gdown
    gdown -O "${MODEL_DIR}/${CHECKPOINT_FILE}" "1Njo9EEc2tdU1ffo8AUQ9mjwuQ9CzSRPX"
else
    echo "Error: Cannot automatically download from Google Drive."
    echo ""
    echo "Please manually download the checkpoint:"
    echo "1. Visit: https://drive.google.com/file/d/1Njo9EEc2tdU1ffo8AUQ9mjwuQ9CzSRPX/view?usp=sharing"
    echo "2. Click 'Download' button"
    echo "3. Save as: ${MODEL_DIR}/${CHECKPOINT_FILE}"
    exit 1
fi

# Verify download
if [ -f "${MODEL_DIR}/${CHECKPOINT_FILE}" ]; then
    FILE_SIZE=$(du -h "${MODEL_DIR}/${CHECKPOINT_FILE}" | cut -f1)
    echo ""
    echo "✓ Download complete!"
    echo "  File: ${MODEL_DIR}/${CHECKPOINT_FILE}"
    echo "  Size: ${FILE_SIZE}"
else
    echo "✗ Download failed. Please try manually:"
    echo "https://drive.google.com/file/d/1Njo9EEc2tdU1ffo8AUQ9mjwuQ9CzSRPX/view?usp=sharing"
    exit 1
fi

echo ""
echo "You can now run EgoLanes inference:"
echo "  python3 Models/inference/cli.py \\"
echo "    --model EgoLanes \\"
echo "    --checkpoint ${MODEL_DIR}/${CHECKPOINT_FILE} \\"
echo "    --input Media/daytime_fair_weather_1.jpg \\"
echo "    --out_dir outputs/egolanes \\"
echo "    --device cpu"
