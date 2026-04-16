# Testing Patterns and Status

**Analysis Date:** 2026-04-11

## Summary
- There are no formal unit-test frameworks wired into CI. The repository contains several "test" scripts (Python and C/C++) that are runnable as utilities, but they are not unit tests (no pytest or GoogleTest integration detected).
- CI/workflows do not run unit tests. The main workflow builds and publishes a container image: `.github/workflows/build-visionpilot-container.yaml`.

## Test files and test-like scripts found (evidence)
- C++ test executable (built via CMake):
  - `VisionPilot/production_release/test/test_autosteer.cpp` — a standalone test/executable that loads models and runs inference (not a GoogleTest suite).
  - `VisionPilot/production_release/CMakeLists.txt` declares an executable `test_autosteer` (lines adding `test/test_autosteer.cpp`) but does not link against GTest or integrate with CTest. Evidence: `VisionPilot/production_release/CMakeLists.txt` (target added lines 386-393).

- Python test-like scripts (script-style, not pytest suites):
  - `Models/training/test_validate_scene_seg.py` — script that loads datasets and runs validation with printed metrics (procedural script with main()).
  - `VisionPilot/middleware_recipes/Calibration/test_with_gt.py` — interactive/script-style validation utility.
  - `Models/visualizations/EgoLanes/mass_test_vid_inference.py` — inference script named "mass_test_..." (script style).

Evidence paths: `Models/training/test_validate_scene_seg.py`, `VisionPilot/middleware_recipes/Calibration/test_with_gt.py`, `Models/visualizations/EgoLanes/mass_test_vid_inference.py`.

## Testing Frameworks detected
- No formal test runners detected in repository configuration:
  - No `pytest` config file or `tox.ini` found.
  - No `jest`, `vitest`, or other JS test runner files detected.
  - No `find_package(GTest)` nor `add_subdirectory` for GoogleTest in CMake. `CMakeLists.txt` adds a test executable but not a GTest target.

Evidence: `VisionPilot/production_release/CMakeLists.txt` (no GTest/CTest configuration).

## How to run the available test-like scripts locally

- Run C++ test executable (build and run):
  1. Create a build directory and configure with CMake from `VisionPilot/production_release` or repository root:
     ```bash
     mkdir -p VisionPilot/production_release/build && cd VisionPilot/production_release/build
     cmake ..
     cmake --build . --target test_autosteer -j
     ./test_autosteer <video_path> <egolanes_model.onnx> <autosteer_model.onnx> [cache_dir]
     ```
     Note: ONNX Runtime requirement is enforced by `CMakeLists.txt` via `ONNXRUNTIME_ROOT` environment variable when SKIP_ORT=OFF. See `VisionPilot/production_release/CMakeLists.txt` lines ~92-112.
     Evidence: `VisionPilot/production_release/CMakeLists.txt`.

- Run Python scripts (script-style):
  - These are executed directly and typically require project Python dependencies (torch, numpy, opencv, etc.). Example:
    ```bash
    python3 Models/training/test_validate_scene_seg.py -c <checkpoint> -r <data_root>
    python3 VisionPilot/middleware_recipes/Calibration/test_with_gt.py --filename <file> --num_frames 5
    ```
  - These are not pytest test cases; they are validation/inference scripts that print metrics.
  Evidence: `Models/training/test_validate_scene_seg.py`, `VisionPilot/middleware_recipes/Calibration/test_with_gt.py`.

- Run repository linters / checks (pre-commit):
  - Pre-commit configuration is present and enforces formatting/linting. To run all checks locally:
    ```bash
    pip install pre-commit
    pre-commit run --all-files
    ```
  - Evidence: `/.pre-commit-config.yaml`.

## CI/Workflow status related to tests
- `.github/workflows/build-visionpilot-container.yaml` builds and pushes container images for multiple platforms but does not run unit tests. The workflow is triggered on pushes affecting `Models/**`, `Modules/**`, `VisionPilot/**` and on workflow_dispatch.
  - Evidence: `.github/workflows/build-visionpilot-container.yaml` (file builds Docker image and pushes tags).
- There are other workflows for spell-check and semantic PR checks but none that indicate executing test suites.
  - Evidence: `.github/workflows/spell-check-differential.yaml`, `.github/workflows/semantic-pull-request.yaml`, `.github/workflows/spell-check-daily.yaml`.

## Gaps & Recommendations (prioritized)
1. Add unit test frameworks and CI integration (High priority)
   - C++: Adopt GoogleTest for unit tests of core libraries and integrate with CMake/CTest.
     - Add `find_package(GTest REQUIRED)` or use `FetchContent` to vendor GoogleTest in CMake for CI reproducibility.
     - Add test targets under `VisionPilot/production_release/test/` (e.g., `test_lane_filtering`, `test_path_planning`, `test_tracking`) and enable `ctest` so CI can run `ctest --output-on-failure`.
     - Suggested files to target first for unit tests (core logic, small-surface-area code):
       - `VisionPilot/production_release/src/lane_filtering/lane_filter.cpp`
       - `VisionPilot/production_release/src/path_planning/poly_fit.cpp`
       - `VisionPilot/production_release/src/tracking/kalman_filter.cpp`
       - `VisionPilot/production_release/src/inference/*` (small helper functions where pure logic exists)
     - Evidence: `VisionPilot/production_release/CMakeLists.txt` (source files listed for these libraries).

2. Python unit tests (Medium priority)
   - Add `pytest` and convert script-like validations into small pytest functions where feasible. Start by adding tests for data utilities and deterministic functions:
     - `Models/training/data_utils/*` (e.g., `data_utils/load_data_scene_seg.py` — add tests for `LoadDataSceneSeg` behavior)
     - Trainer logic that does not require GPU (or mock heavy dependencies).
   - Add `pytest.ini` or `pyproject.toml` for pytest configuration and a GitHub Actions job to run `pytest --maxfail=1 -q`.
   - Evidence candidates: `Models/training/test_validate_scene_seg.py` (contains how trainer and data loader are used; extract smaller units for tests).

3. Convert existing runnable test scripts into integration tests (Low-Medium)
   - Wrap heavy inference scripts into integration tests that run in CI only when models/large artifacts are available (or use small/synthetic models for CI).
   - Add a separate workflow/job (or matrix job) to run integration tests inside the `build-visionpilot-container` job or a separate `test` job that uses the built container.

4. Lint & format enforcement in CI (Medium)
   - Add CI steps to run `pre-commit` hooks (or run `black --check`, `clang-format --dry-run`, `cpplint`) to enforce code style on PRs. The repository already lists pre-commit hooks, but CI should run them as part of PR validation.
   - Evidence: `/.pre-commit-config.yaml` exists but CI currently doesn't run those hooks.

5. Test automation and flake: add coverage reporting and minimum coverage gates if desired (optional)
   - For Python, use `pytest --cov` and upload coverage to Codecov or use GitHub Actions artifacts.

## Suggested minimal steps to get CI-run unit tests
1. Add GoogleTest to `VisionPilot/production_release/CMakeLists.txt` (via `FetchContent` or system package). Create a `tests/` directory with a small gtest for `lane_filter.cpp`.
2. Update `.github/workflows/build-visionpilot-container.yaml` or add a new `ci-tests.yaml` workflow to build with tests enabled and run `ctest`.
3. Add `pytest` to Python dev requirements and a GitHub Actions job that sets up Python, installs dependencies, and runs `pytest`.

## Short-term local commands
- Run all pre-commit checks:
  ```bash
  pip install pre-commit
  pre-commit run --all-files
  ```
- Build and run the C++ test executable (example):
  ```bash
  cd VisionPilot/production_release
  mkdir -p build && cd build
  cmake ..
  cmake --build . --target test_autosteer -j
  ./test_autosteer <video> <egolanes.onnx> <autosteer.onnx>
  ```
- Run Python test scripts (non-pytest):
  ```bash
  python3 Models/training/test_validate_scene_seg.py -c <checkpoint> -r <data_root>
  python3 VisionPilot/middleware_recipes/Calibration/test_with_gt.py --filename <file> --num_frames 5
  ```

---

*Testing analysis: 2026-04-11*
