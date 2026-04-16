# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Autoware Vision Pilot** is a free, open-source perception stack for ADAS and self-driving systems, built on an End-to-End AI Architecture. The system is designed for automotive integration and does not require HD maps.

### Key Components

- **Perception Models**: Scene segmentation, ego-lane detection, auto-steer, auto-speed
- **ONNX Export**: Models can be converted from PyTorch to ONNX for deployment
- **Simulation**: CARLA and SODA.Sim integration via Docker
- **Data Parsing**: Pipeline for processing multiple autonomous driving datasets

---

## Quick Reference

### Build and Run Commands

```bash
# Build Docker simulation container
make docker-sim

# Run Docker simulation (shows help)
docker run --rm -v "$(pwd)":/workspace -w /workspace visionpilot-sim --help

# Run model inference (local)
python3 Models/inference/cli.py --model SceneSeg \
  --checkpoint Models/model_library/SceneSeg/checkpoint.pth \
  --input Media/daytime_fair_weather_1.jpg \
  --out_dir outputs/scene_seg \
  --device cpu

# Run tests
python3 -m pytest tests/

# Run specific test
python3 -m pytest tests/test_cli_help.py
```

---

## Architecture

### Models Folder Structure

```
Models/
├── data_parsing/          # Dataset processing pipelines
│   ├── AutoDrive/        # Radar/ODD data
│   ├── AutoSpeed/        # Speed limit detection data
│   ├── AutoSteer/        # Steering prediction data
│   ├── EgoLanes/         # Lane detection data
│   ├── Scene3D/          # 3D scene understanding
│   ├── SceneSeg/         # Scene segmentation data
│   └── DomainSeg/        # Domain segmentation data
├── data_utils/           # Shared data utilities
│   ├── lite_models/      # Lightweight model datasets
│   │   ├── augmentation/
│   │   ├── dataloaders/
│   │   └── helpers/
│   └── load_data_*.py    # Data loading modules
├── inference/            # Model inference pipeline
│   ├── cli.py           # CLI entrypoint with dispatch
│   ├── auto_speed_infer.py
│   ├── auto_steer_infer.py
│   └── output_writer.py # Output serialization
├── exports/              # ONNX export and quantization
│   ├── convert_pytorch_to_onnx.py
│   ├── quantize_model_sceneseg.py
│   └── trace_pytorch_model.py
├── model_library/        # Trained model checkpoints (not committed)
└── requirements.txt      # Python dependencies
```

### Key Design Pattern: Dynamic Model Dispatch

The `cli.py` uses a model map for dispatching to different inference implementations:

```python
model_map = {
    "SceneSeg": ("scene_seg_infer", "SceneSegNetworkInfer"),
    "AutoSpeed": ("auto_speed_infer", "AutoSpeedNetworkInfer"),
}
```

This allows adding new models by:
1. Creating `<model>_infer.py` in `Models/inference/`
2. Adding entry to `model_map` in `cli.py`

### CLI Arguments Contract

All inference runs require:
- `--model`: Model ID (e.g., "SceneSeg", "AutoSpeed")
- `--checkpoint`: Path to `.pth` checkpoint file
- `--input`: Input image file or directory
- `--out_dir`: Output directory for results
- `--device`: "cpu" or "cuda" (default: "cpu")
- `--config`: Optional YAML config file (overrides flags)

---

## Development Guidelines

### Adding a New Model

1. **Create inference module**: `Models/inference/<model_name>_infer.py` with class `<ModelName>NetworkInfer`
2. **Register in CLI**: Add to `model_map` in `cli.py`
3. **Create data parsing** (if needed): Add to `Models/data_parsing/<category>/`
4. **Add tests**: Create test in `tests/` directory

### Docker/Simulation Setup

Three Docker configurations exist:
- `Dockerfile`: Main inference container (Python 3.10)
- `Dockerfile.carla`: CARLA simulation (Ubuntu 22.04 + ROS2 Humble)
- `Dockerfile.soda`: SODA.Sim simulation (Ubuntu 22.04 + ROS2 Humble)
- `docker-compose.yml`: Orchestrates simulation services

### Dependencies

See `Models/requirements.txt` for Python dependencies including:
- PyTorch, TorchVision
- ONNX, ONNX Runtime
- OpenCV, Pillow
- NumPy, Matplotlib

---

## Important Notes

- Model checkpoints are **not committed**; they must be downloaded per model README
- License: Apache 2.0
- Platform: ROS2 Humble, Python 3.10+
