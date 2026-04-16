# Technology Stack

**Analysis Date:** 2026-04-11

## Languages

**Primary:**
- C++ (C++17) - Production runtime, inference and vehicle integration code
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (sets CMAKE_CXX_STANDARD 17)
  - Evidence: `VisionPilot/production_release/main.cpp`

**Secondary:**
- Python (3.x) - Model training, exports and inference utilities
  - Evidence: `Models/requirements.txt` (PyTorch, torchvision, onnx, onnxruntime)
  - Evidence: many Python training/inference scripts under `Models/` (e.g., `Models/training/train_scene_seg.py`, `Models/inference/scene_seg_infer.py`)

**Scripting / Shell:**
- Bash/SH for helper scripts and build wrappers
  - Evidence: `VisionPilot/production_release/build_ort-gpu.sh`, `VisionPilot/production_release/run_final.sh`

## Runtime

**Environment:**
- Native Linux/C++ runtime (compiled with CMake) for the VisionPilot binary
  - Evidence: `VisionPilot/production_release/CMakeLists.txt`, `VisionPilot/production_release/main.cpp`
- Python runtime for model development and offline inference (pip / virtualenv)
  - Evidence: `Models/requirements.txt`

**Package Manager:**
- Python: pip / requirements.txt
  - Evidence: `Models/requirements.txt`
- C++: System packages discovered/managed through CMake (find_package) and OS package manager for dependencies (no top-level package manager file)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (find_package OpenCV, Eigen3, yaml-cpp, CUDA/TensorRT checks)

## Frameworks

**Core (C++):**
- CMake (build system) - primary build system for VisionPilot
  - Evidence: `VisionPilot/production_release/CMakeLists.txt`
- OpenCV (computer vision library) - used across C++ modules
  - Evidence: `VisionPilot/production_release/CMakeLists.txt`, includes in `main.cpp` (`#include <opencv2/opencv.hpp>`)

**ML / Model frameworks (Python & runtime):**
- PyTorch (training & model code)
  - Evidence: `Models/requirements.txt` (torch), `Models/training/*.py`
- ONNX / ONNX Runtime (inference runtime & model export)
  - Evidence: `Models/requirements.txt` (onnx, onnxruntime), `VisionPilot/production_release/CMakeLists.txt` (ONNXRUNTIME_ROOT required), `Models/exports/convert_pytorch_to_onnx.py`
- TensorRT (optional high-performance inference backend)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (TENSORRT handling and SKIP_ORT option), `VisionPilot/production_release/main.cpp` (tensorrt_engine includes)

**Visualization / Tools:**
- TensorBoard (training metrics visualization)
  - Evidence: `Models/requirements.txt` (tensorboard)
- Rerun (optional visualization/logging SDK) - optional C++ integration
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (ENABLE_RERUN option + RERUN_SDK_ROOT), `VisionPilot/production_release/main.cpp` (conditional includes for rerun)

## Key Dependencies / Libraries

**Critical (inference & vision):**
- OpenCV - `VisionPilot/production_release/CMakeLists.txt`, `VisionPilot/production_release/main.cpp`
- ONNX Runtime - `VisionPilot/production_release/CMakeLists.txt`, `Models/requirements.txt`
- PyTorch / torchvision - `Models/requirements.txt`, `Models/training/*.py`
- TensorRT & CUDA (optional high-perf GPU path) - `VisionPilot/production_release/CMakeLists.txt`

**Numerics & Utilities:**
- Eigen3 - `VisionPilot/production_release/CMakeLists.txt`
- yaml-cpp - `VisionPilot/production_release/CMakeLists.txt` (config parsing)
- Boost (used at least for circular_buffer) - `VisionPilot/production_release/main.cpp` (`#include <boost/circular_buffer.hpp>`)

## Configuration & Tooling

- CMake for C++ build configuration
  - Evidence: `VisionPilot/production_release/CMakeLists.txt`
- `.clang-format` and `CPPLINT.cfg` for C++ style and linting
  - Evidence: `.clang-format`, `CPPLINT.cfg`
- Pre-commit hooks present (formatting / checks)
  - Evidence: `.pre-commit-config.yaml`
- Python dependency list in `Models/requirements.txt`
  - Evidence: `Models/requirements.txt`

## Build / Dev Tools

- Build: CMake + make/ninja (CMakeLists and packaging via CPack)
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (CPack configuration)
- Scripts to build ONNX Runtime artifacts: `VisionPilot/production_release/build_ort-gpu.sh`
- Model export & benchmarking tools in `Models/exports/` (PyTorch -> ONNX, quantize, trace)
  - Evidence: `Models/exports/convert_pytorch_to_onnx.py`, `Models/exports/quantize_model_sceneseg.py`

## Notable Absences / "Not found"

- Node.js / npm / package.json - Not found (no evidence of a package.json or JS runtime package manager)
  - Evidence: root directory listing (no `package.json`) and no `package.json` file present
- Dockerfiles / Kubernetes manifests - Not found in repository root (no `Dockerfile` detected at project root)
  - Evidence: repo root listing; no top-level `Dockerfile` found

---

*Stack analysis: 2026-04-11*
