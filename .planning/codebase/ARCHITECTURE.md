# Architecture

**Analysis Date:** 2026-04-11

## Pattern Overview

**Overall:** Modular, mixed-language layered architecture separating on-vehicle runtime (C++), model development/training (Python/PyTorch), and middleware/adaptation layers for simulation and vehicle integration.

**Key Characteristics:**
- Clear separation between: model development & training (`Models/`), vehicle runtime and control (`VisionPilot/production_release/`), and middleware/transport adapters (`VisionPilot/middleware_recipes/`).
- Models developed in Python/PyTorch and exported to runtimes (ONNX / libtorch / ORT) for onboard inference (`Models/exports/`, `build_ort-gpu.sh`).
- Multiple runtime targets supported: onboard C++ (real vehicle), simulation environments (CARLA, SODA.Sim), and research/training servers.

## Layers

Perception (Model + Inference):
- Purpose: Run neural nets for scene understanding, lanes, path, speed, steering percepts.
- Location: `Models/` (development & export) and `Models/inference/` (inference scripts)
- Contains: model architectures (`Models/model_components/*.py`), training pipelines (`Models/training/*.py`), inference wrappers (`Models/inference/*.py`), export utilities (`Models/exports/*`).
- Depends on: PyTorch, ONNX/ORT exporters (evidence: `Models/requirements.txt`, `Models/exports/`)
- Used by: Onboard runtime via exported model artifacts

Middleware / Integration:
- Purpose: Adapt transport layer (ROS2, Zenoh, IceOryx2, custom IPC) and provide recipes to connect sensors, inference, and actuators.
- Location: `VisionPilot/middleware_recipes/` with subfolders `ROS2/`, `Zenoh/`, `IceOryx2/` and `common/`.
- Contains: middleware examples, config and small adapters (`VisionPilot/middleware_recipes/*`).
- Used by: `VisionPilot/production_release/` for different deployment environments

Runtime / Control (Onboard C++):
- Purpose: Vehicle-side execution: camera acquisition, inference node orchestration, tracking, path planning, longitudinal and lateral control, and actuator publishing.
- Location: `VisionPilot/production_release/` (`main.cpp`, `src/*`, `include/*`)
- Contains: drivers, camera, inference integration, tracking, path_planning, steering_control, longitudinal, speed_planning, publisher.
- Entry point: `VisionPilot/production_release/main.cpp`
- Config: `VisionPilot/production_release/visionpilot.conf` and `VisionPilot/production_release/VisionPilot.conf.example`

Simulation / Testbeds:
- Purpose: Run the stack in simulated environments for validation (CARLA, SODA.Sim) and tooling for video pub/sub.
- Location: `VisionPilot/simulation/` (e.g., `VisionPilot/simulation/CARLA/`, `VisionPilot/simulation/SODA.Sim/`)
- Contains: middleware integration for simulation, media examples, ROS2 bridge recipes.

Model Development & Training (Server/Research):
- Purpose: Train, validate and export models. Jupyter tutorials and training scripts.
- Location: `Models/training/`, `Models/tutorials/`, `Models/model_library/`
- Entry points: training scripts such as `Models/training/train_scene_3d.py`, `Models/training/auto_steer_trainer.py`.

Assets & Media:
- Purpose: UI assets, logos, sample videos and images used by docs and demos.
- Location: `Media/` (e.g., `Media/VisionPilot_logo.png`, `Media/hero_GIF.gif`)

## Data Flow (high-level)

ASCII diagram (simplified):

[Camera/ Sensors]
   |
   v
[Frame Ingest / Driver]  -- (`VisionPilot/production_release/src/camera/*`)
   |
   v
[Inference Node(s)]  -- (calls exported models: `Models/exports/*`, `Models/inference/*.py`, or native ORT/libtorch `VisionPilot/production_release/src/inference/`)
   |
   v
[Perception Outputs] -> [Tracking] -> [Path Planner] -> [Control]  
   |                     |               |                
   v                     v               v                
[Visualization / Recording] (`VisionPilot/production_release/src/visualization/`, `VisionPilot/production_release/include/visualization/`)

Notes and evidence:
- Camera code and driver hooks: `VisionPilot/production_release/src/camera/` and `VisionPilot/production_release/include/camera/`.
- Inference integration and exported main: `VisionPilot/production_release/src/inference/` and `Models/exports/onnx_rt/`, `Models/exports/libtorch/`.
- Build script for ONNX Runtime GPU: `VisionPilot/production_release/build_ort-gpu.sh` (indicates GPU-accelerated inference target).

## Entry Points

- Onboard (vehicle) runtime: `VisionPilot/production_release/main.cpp` — compiled C++ application for deployment on vehicle hardware.
- Development release binaries: `VisionPilot/development_releases/*/main.cpp`.
- Model training (server): various Python entry scripts in `Models/training/` such as `Models/training/train_scene_3d.py`, `Models/training/auto_steer_trainer.py`.
- Model inference scripts for research/evaluation: `Models/inference/*_infer.py` (e.g., `Models/inference/scene_3d_infer.py`).
- Export/Runtime glue: `Models/exports/*` (ONNX, libtorch, onnx_rt/main.cpp)

## Key Abstractions

- Neural model components: `Models/model_components/*` — modular Python modules for backbone, heads, necks.
- Middleware recipes: `VisionPilot/middleware_recipes/*` — small apps and YAML/JSON configs to run the pipeline over different transports.
- Runtime modules: `VisionPilot/production_release/src/*` and `include/*` broken down by subsystem (e.g., `path_planning`, `steering_control`, `lane_filtering`).

## Where Models and Assets Live

- Models source & training: `Models/` (full repo of architectures, training code and data parsing: `Models/model_components/`, `Models/training/`, `Models/data_parsing/`).
- Exported model runtime code: `Models/exports/` contains stubs and examples to convert to ONNX/libtorch and C++ integration (`Models/exports/onnx_rt/main.cpp`, `Models/exports/libtorch/main.cpp`).
- Media assets: `Media/` (logos, GIFs, sample images/videos)

## Runtime Targets (likely)

- Onboard / Vehicle: compiled C++ binary from `VisionPilot/production_release/` (evidence: `main.cpp`, `CMakeLists.txt`, `visionpilot.conf`, `run_final.sh`).
- Edge/Server (GPU) for inference: ONNX Runtime with GPU (`build_ort-gpu.sh`, `Models/exports/onnx_rt/`).
- Training & Research: Python/PyTorch on servers/GPUs (`Models/training/`, `Models/requirements.txt`).
- Simulation: CARLA and SODA.Sim via `VisionPilot/simulation/` with ROS2/Zenoh bridges.

## Error Handling & Cross-Cutting Concerns (observed places)

- Configuration via `visionpilot.conf` and example configs in `VisionPilot/production_release/` and `VisionPilot/development_releases/*/`.
- Visualization & logging code located under `VisionPilot/production_release/src/visualization/` and `include/visualization/`.

---

*Architecture analysis: 2026-04-11*
