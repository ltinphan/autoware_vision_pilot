# External Integrations

**Analysis Date:** 2026-04-11

## APIs & External Services

- Google Drive (used as host for model weight downloads and demo videos)
  - Evidence: many model READMEs and tutorials contain Drive links, e.g. `Models/model_library/AutoSpeed/README.md`, `Models/model_library/EgoLanes/README.md`, `Models/tutorials/SceneSeg.ipynb`
  - Evidence: Dockerfile installs `gdown` and model download usage: `VisionPilot/software_defined_vehicle/OpenADKit/Docker/Dockerfile` (line `pip install gdown`) and tutorial notebooks referencing `https://drive.google.com` links

- GitHub Releases (ONNX Runtime binaries)
  - Evidence: Dockerfile downloads ONNX Runtime from GitHub Releases: `VisionPilot/software_defined_vehicle/OpenADKit/Docker/Dockerfile` (wget https://github.com/microsoft/onnxruntime/releases/download/...)
  - CMake requires `ONNXRUNTIME_ROOT` environment variable: `VisionPilot/production_release/CMakeLists.txt` (checks for `ENV{ONNXRUNTIME_ROOT}`)

- GitHub Container Registry (GHCR) - CI pushes built images
  - Evidence: GitHub Actions workflow `/.github/workflows/build-visionpilot-container.yaml` logs `ghcr.io/...` push and login steps

## Data Sources & Datasets

Repository contains parsers and references for many public datasets (used in training and data parsing):

- BDD100K
  - Evidence: `Models/data_parsing/SceneSeg/BDD100K` (directory)
- KITTI
  - Evidence: `Models/data_parsing/Scene3D/KITTI` (directory)
- Argoverse
  - Evidence: `Models/data_parsing/Scene3D/Argoverse` (directory)
- TuSimple, CULane, Comma2k19
  - Evidence: `Models/data_parsing/EgoLanes/TuSimple`, `Models/data_parsing/EgoLanes/CULane`, `Models/data_parsing/EgoLanes/Comma2k19`
- Mapillary, Mapillary Vistas, MUSES, ACDC
  - Evidence: `Models/data_parsing/SceneSeg/Mapillary_Vistas`, `Models/data_parsing/SceneSeg/MUSES`, `Models/data_parsing/SceneSeg/ACDC`

(These directories contain data parsing scripts and dataset-specific preprocessing utilities.)

## Model Artifacts & Runtime Models

- On-disk model artifacts and model libraries
  - Evidence: `Models/` contains model architectures, training scripts and export tools (e.g., `Models/exports/convert_pytorch_to_onnx.py`, `Models/exports/trace_pytorch_model.py`)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` installs ONNX files from `models/` into package (`install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/" ... PATTERN "*.onnx")`)

- Model weight hosting and retrieval
  - Evidence: Drive links in `Models/model_library/*/README.md` and `Models/tutorials/*.ipynb`; Dockerfile uses `gdown` to facilitate downloads: `VisionPilot/software_defined_vehicle/OpenADKit/Docker/Dockerfile` (pip install gdown, COPY Models)

## Hardware Interfaces

- CAN bus (SocketCAN) - vehicle state and steering/speed retrieval
  - Evidence: C++ CAN driver implementation: `VisionPilot/production_release/src/drivers/can_interface.cpp` and header `VisionPilot/production_release/include/drivers/can_interface.hpp` (SocketCAN usage and Linux headers in code)

- GPU acceleration / inference hardware
  - CUDA + TensorRT (optional high-performance path)
    - Evidence: `VisionPilot/production_release/CMakeLists.txt` (SKIP_ORT option triggers TensorRT/CUDA search and compile flags), `VisionPilot/production_release/main.cpp` (includes `inference/tensorrt_engine.hpp` when SKIP_ORT defined)
  - ONNX Runtime (CPU/GPU builds supported)
    - Evidence: `VisionPilot/production_release/CMakeLists.txt` (ONNXRUNTIME_ROOT expectation), `Models/requirements.txt` (onnxruntime)

## Libraries & Middleware Integrations

- OpenCV (vision primitives)
  - Evidence: CMake find_package and includes: `VisionPilot/production_release/CMakeLists.txt`, `VisionPilot/production_release/main.cpp`

- Eigen3 (numerical linear algebra)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (find_package(Eigen3 REQUIRED))

- yaml-cpp (configuration reader)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (find_package(yaml-cpp REQUIRED))

- Rerun SDK (optional telemetry/visualization)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (ENABLE_RERUN option + RERUN_SDK_ROOT), conditional includes in `VisionPilot/production_release/main.cpp` (#ifdef ENABLE_RERUN)
  - Additional dependency: Arrow (configured when Rerun enabled): `VisionPilot/production_release/CMakeLists.txt` (find_package(Arrow REQUIRED))

## CI/CD & Packaging

- GitHub Actions for builds and container publishing
  - Evidence: `.github/workflows/build-visionpilot-container.yaml` (build and push to GHCR)

- Docker (container image for runtime & visualizer)
  - Evidence: `VisionPilot/software_defined_vehicle/OpenADKit/Docker/Dockerfile` and Docker-related files under `VisionPilot/software_defined_vehicle/OpenADKit/Docker/`

## Authentication & Credential Hints (no secrets inspected)

- ONNX Runtime location is supplied via environment: `ONNXRUNTIME_ROOT` (CMake check)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (checks ENV{ONNXRUNTIME_ROOT}) and Dockerfile sets `ENV ONNXRUNTIME_ROOT=...`

- TensorRT/CUDA detection may rely on `TENSORRT_ROOT`, `CUDA_HOME` environment variables
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (TENSORRT_SEARCH_PATHS includes `$ENV{TENSORRT_ROOT}` and `$ENV{CUDA_HOME}`)

- Rerun SDK location via `RERUN_SDK_ROOT` environment variable (optional)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (checks `ENV{RERUN_SDK_ROOT}`)

- GitHub Actions use `secrets.GITHUB_TOKEN` and push to GHCR using the Actions login (no repository secrets exposed here)
  - Evidence: `.github/workflows/build-visionpilot-container.yaml` (uses `secrets.GITHUB_TOKEN` and `docker/login-action`)

## Monitoring, Logging & Webhooks

- External error tracking / observability services (Sentry, Datadog, etc.): Not found
  - Evidence: repository scan did not reveal references to Sentry, Datadog, NewRelic or similar (no configuration files or SDK imports detected)

- Webhooks / external callbacks: Not found (no explicit webhook endpoints or integrations discovered in code)

## Additional Notes

- The repo relies on public dataset download links and Google Drive hosted model weights rather than a centralized model registry or S3 bucket. Evidence: `Models/model_library/*/README.md` and tutorial notebooks linking to `drive.google.com`.

- ONNX Runtime and model artifacts are integrated both at build time (CMake) and at runtime via container build (Dockerfile `wget` + extraction).

---

*Integration audit: 2026-04-11*
