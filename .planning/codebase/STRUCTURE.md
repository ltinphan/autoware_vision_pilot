# Codebase Structure

**Analysis Date:** 2026-04-11

## Directory Layout

```
[project-root]/
├── VisionPilot/          # Vehicle runtime, middleware recipes, simulation and release artifacts
├── Models/               # Model architectures, training, inference, export utilities
├── Media/                # Images, GIFs, logos, demo media
├── .github/              # CI / workflows
├── .planning/            # Mapping and planning docs (this folder)
├── README.md             # Project overview and high-level docs
├── ONBOARDING.md         # Contributor onboarding and contact info
└── CONTRIBUTING.md      # Contribution guidelines
```

## Top-level Directory Purposes

**`VisionPilot/`**:
- Purpose: Vehicle runtime (production & development releases), middleware adapters and simulation recipes.
- Contains:
  - `VisionPilot/production_release/` - C++ production runtime: `main.cpp`, `CMakeLists.txt`, `src/`, `include/`, `visionpilot.conf`.
  - `VisionPilot/development_releases/` - earlier/dev release artifacts with `main.cpp` and `src/`.
  - `VisionPilot/middleware_recipes/` - adapters and examples for `ROS2`, `Zenoh`, `IceOryx2` and common backends.
  - `VisionPilot/simulation/` - CARLA and SODA.Sim integration and media.
- Key files:
  - `VisionPilot/production_release/main.cpp` (entry point)
  - `VisionPilot/production_release/visionpilot.conf` (runtime config)
  - `VisionPilot/production_release/src/*` and `include/*` (modules by subsystem)
- Where to add runtime C++ changes: `VisionPilot/production_release/src/<subsystem>/` and headers in `VisionPilot/production_release/include/<subsystem>/`.

**`Models/`**:
- Purpose: All ML-related code: model definitions, training pipelines, data parsers, export tools and inference helpers.
- Contains:
  - `Models/model_components/` - modular network components and lite-models (`*.py` files)
  - `Models/training/` - trainers and scripts (`train_*.py`)
  - `Models/inference/` - python inference wrappers (`*_infer.py`)
  - `Models/exports/` - ONNX/libtorch export and C++ runtime examples
  - `Models/model_library/` - per-model tutorials and examples
  - `Models/data_parsing/` and `Models/data_utils/` - dataset loaders and augmentation utilities
- Key files:
  - `Models/model_components/scene_3d_network.py` (example architecture)
  - `Models/training/train_scene_3d.py` (training entry)
  - `Models/exports/onnx_rt/main.cpp` (C++ runtime example for exported model)
- Where to add new models: `Models/model_components/<new_model>/` and training script in `Models/training/`.

**`Media/`**:
- Purpose: Documentation and website media assets, logos and demo images/videos.
- Key files: `Media/VisionPilot_logo.png`, `Media/hero_GIF.gif`.

**`.github/`**:
- Purpose: CI workflow definitions (`.github/workflows/`).

**Documentation & Onboarding**:
- `README.md` - high level project overview and badges
- `ONBOARDING.md` - contributor onboarding and contact (lists a Senior Tech Lead contact)
- `CONTRIBUTING.md` - contribution process and CI checks

## Key File Locations (by concern)

Entry points:
- Onboard C++ runtime: `VisionPilot/production_release/main.cpp`
- Training scripts: `Models/training/*.py` (e.g., `Models/training/auto_steer_trainer.py`)
- Model inference (research): `Models/inference/*_infer.py`
- Export/C++ runtime examples: `Models/exports/onnx_rt/main.cpp`, `Models/exports/libtorch/main.cpp`

Configuration:
- Runtime config: `VisionPilot/production_release/visionpilot.conf`, `VisionPilot/development_releases/*/visionpilot.conf`
- Model configs: `Models/config/*.yaml` (e.g., `Models/config/Scene3DLite.yaml`)

Core logic:
- C++ runtime subsystems: `VisionPilot/production_release/src/{camera,inference,tracking,path_planning,steering_control,longitudinal,visualization}`
- Model components and pipelines: `Models/model_components/`, `Models/training/`, `Models/data_utils/`

Testing & examples:
- Production tests: `VisionPilot/production_release/test/` (C++ tests)
- Model visualizations and tutorials: `Models/visualizations/`, `Models/tutorials/`

## Naming & Placement Guidelines (prescriptive)

- New onboard runtime modules: place implementation in `VisionPilot/production_release/src/<subsystem>/` and public headers in `VisionPilot/production_release/include/<subsystem>/`.
- New model architectures: place under `Models/model_components/<ModelName>/` with an `__init__.py` and a README when appropriate.
- Training scripts: co-locate with `Models/training/` and name `train_<feature>.py`.
- Export/runtime adaptation: add exporter scripts in `Models/exports/` and C++ integration examples under `Models/exports/onnx_rt/` or `Models/exports/libtorch/`.
- Middleware recipes: add new middleware bridges or adapters under `VisionPilot/middleware_recipes/<MiddlewareName>/` following existing folder patterns.

## Important Modules & Where to Look

- Camera & drivers: `VisionPilot/production_release/src/camera/` and `include/camera/`
- Inference integration (C++): `VisionPilot/production_release/src/inference/` and `include/inference/`
- Tracking & planning: `VisionPilot/production_release/src/tracking/`, `src/path_planning/`, `src/longitudinal/`, `src/steering_control/`
- Model architectures: `Models/model_components/` (many files e.g., `scene_3d_network.py`, `backbone.py`)
- Model exports: `Models/exports/onnx_rt/`, `Models/exports/libtorch/`
- Simulation: `VisionPilot/simulation/CARLA/` and `VisionPilot/simulation/SODA.Sim/`

## Likely Owners (best guess from docs)

- Project-level steward: Autoware Foundation Privately Owned Vehicle work group (see `CONTRIBUTING.md`, `ONBOARDING.md`).
- Contact / Tech lead listed in onboarding: Muhammad Zain Khawaja (see `ONBOARDING.md`).
- ML & model maintainers: teams contributing to `Models/` (place new model PRs as per `CONTRIBUTING.md`).
- Vehicle runtime maintainers: C++/embedded team owning `VisionPilot/production_release/` and middleware recipes.

## Where to Add New Code (short decision guide)

- New sensor driver (C++): `VisionPilot/production_release/src/drivers/` + headers in `include/drivers/`.
- New inference model (research): `Models/model_components/<ModelName>/` + `Models/training/train_<ModelName>.py` + exports under `Models/exports/`.
- New middleware integration: `VisionPilot/middleware_recipes/<NewMiddleware>/` following `Zenoh` or `ROS2` structure.

## Special Directories

- `Models/exports/` - Contains examples to convert models to runtime formats (committed) — generated artifacts (model weights) are not present in repo.
- `VisionPilot/production_release/` - Production C++ build with `CMakeLists.txt` and runtime config. Treated as committed, built artifacts are generated at build time.

---

*Structure analysis: 2026-04-11*
