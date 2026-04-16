# Codebase Concerns

**Analysis Date:** 2026-04-11

This document lists security, safety, deployment, operational, and technical-debt concerns prioritized for a greenfield start or safe production deployment. Each entry includes evidence (file paths, TODO markers, config gaps) and pragmatic suggestions or blockers for onboarding/refactor.

---

## High priority (blocks greenfield start or safe production deployment)

1) Missing automated tests for safety-critical code
- Issue: No unit/integration test suite covering core perception and control algorithms.
- Evidence:
  - No test files matching common patterns were found during repository scan (no `*.test.*`, `*.spec.*`, `*_test.*` detected).
  - Safety-critical code locations with no associated tests: `VisionPilot/production_release/main.cpp`, `VisionPilot/production_release/src/lane_tracking/lane_tracking.cpp`, `VisionPilot/production_release/src/lane_filtering/lane_filter.cpp`, `VisionPilot/production_release/src/visualization/visualize.cpp`.
- Impact: Cannot validate algorithm correctness or safety regressions; blocks any safe deployment or refactor that touches perception/control logic.
- Fix approach: Add unit and integration tests under a test directory (co-locate tests next to implementations) and add CI jobs to run tests on PRs. Prioritize tests for `lane_tracking`, `lane_filtering`, and core `main.cpp` subsystems.

2) Monolithic, very large source files and duplicate release branches
- Issue: Multiple extremely large single-file implementations and duplicate code across release directories increase risk and maintenance burden.
- Evidence:
  - Large files: `VisionPilot/production_release/main.cpp` (≈1959 lines), `VisionPilot/development_releases/1.0/main.cpp` (≈2153 lines), `VisionPilot/development_releases/0.9/main.cpp` (≈1959 lines).
  - Duplicate visualizations and lane-filter implementations in parallel release folders: `VisionPilot/production_release/src/visualization/visualize.cpp`, `VisionPilot/development_releases/1.0/src/visualization/visualize.cpp`, `VisionPilot/development_releases/0.9/src/visualization/visualize.cpp` and similarly for `lane_filter.cpp`.
- Impact: Refactoring or bug fixes must be applied in multiple places; onboarding is slowed by file size and unclear separation of concerns.
- Fix approach: Create a single source-of-truth code tree under `VisionPilot/` (or `src/`) and migrate release-specific differences to small config/flag files. Break large files into focused modules (e.g., camera calibration, perception pipeline, visualization) and add tests per module.

3) Uncalibrated transforms and explicit TODOs in perception pipeline (safety risk)
- Issue: Multiple TODO comments point to missing calibration steps required for correct metric computations.
- Evidence:
  - `VisionPilot/production_release/main.cpp`: contains `// TODO: Calibrate transformPixelsToMeters() for your specific camera` and an informational message about BEV pixels→meters stating `(TODO: calibrate)`.
  - Same TODO repeated in `VisionPilot/development_releases/1.0/main.cpp` and `VisionPilot/development_releases/0.9/main.cpp`.
- Impact: Running with default/unverified transforms can produce incorrect distance/speed estimates; unsafe in physical deployments.
- Fix approach: Provide a formal calibration procedure and automated checks. Add a configuration file for camera intrinsics and a unit/integration test that verifies transform outputs on synthetic inputs.

4) Lack of formal CI security and static analysis
- Issue: CI workflows exist but there is no evidence of SAST, dependency scanning, or runtime security checks.
- Evidence:
  - Detected workflows in `.github/workflows/`: `spell-check-differential.yaml`, `semantic-pull-request.yaml`, `spell-check-daily.yaml`, `build-visionpilot-container.yaml`.
  - No workflows detected for security scanning, fuzzing, static analysis (e.g., clang-tidy, cppcheck, bandit) or dependency vulnerability scanning.
- Impact: Vulnerabilities and critical bugs can be merged without automated detection; higher risk for production deployment.
- Fix approach: Add CI steps for static analysis (clang-tidy/cppcheck for C++; bandit or flake8 for Python), dependency scanning, and enforce these on PRs. Add a security-policy file and enable secret scanning.

---

## Medium priority (operational or safety concerns that should be addressed before production)

5) Missing or incomplete package metadata in simulation and ROS packages
- Issue: Several ROS2 and Python package metadata use placeholder TODOs for description and license.
- Evidence:
  - `VisionPilot/simulation/CARLA/ROS2/src/camera_publisher/package.xml`: `<description>TODO: Package description</description>` and `<license>TODO: License declaration</license>`.
  - `VisionPilot/simulation/CARLA/ROS2/src/camera_publisher/setup.py`: `description='TODO: Package description', license='TODO: License declaration',`.
  - Similar TODO placeholders in `VisionPilot/simulation/CARLA/ROS2/src/camera_spectator/package.xml` and `setup.py`, and other `package.xml` under `middleware_recipes/ROS2`.
- Impact: Packaging for distribution, reproducible builds, and third-party reuse are hindered; license uncertainty is a compliance risk for deployment.
- Fix approach: Fill package metadata with accurate descriptions and licenses. Add a verification CI job to ensure no `TODO` remains in packaging files.

6) No top-level Dockerfile or clear container entrypoint
- Issue: CI references container build workflow (`build-visionpilot-container.yaml`) but no top-level `Dockerfile` was found.
- Evidence:
  - `.github/workflows/build-visionpilot-container.yaml` exists, but `Dockerfile` not present at repository root (no `Dockerfile` detected by scan).
- Impact: Reproducing container builds locally or in alternative CI providers will be difficult and may hide missing build steps. Deployment reproducibility is a blocker for production.
- Fix approach: Add a documented `Dockerfile` at repository root (or document path used by the workflow) and add a developer `make build-image` / `make run` convenience target in README or `ONBOARDING.md`.

7) Observability and error reporting not defined
- Issue: No centralized logging/telemetry or error-tracking integration is visible.
- Evidence:
  - No obvious logging/config files or observability SDKs detected in top-level configuration. Core runtime files like `VisionPilot/production_release/main.cpp` and `VisionPilot/.../visualize.cpp` are large but do not reference a centralized observability config in the repo scan.
- Impact: Hard to diagnose failures in deployed systems and to meet operational SLAs.
- Fix approach: Define logging conventions (e.g., structured logs) in `ONBOARDING.md`, add a logging/metrics initializer module, and integrate with a chosen observability backend. Add CI checks to ensure libraries are included where needed.

---

## Low priority (technical debt and hygiene items)

8) Many TODOs and future-improvements notes across codebase
- Issue: Numerous TODO comments and README TODO sections indicate planned improvements not implemented.
- Evidence (non-exhaustive):
  - `VisionPilot/production_release/main.cpp`: calibration TODOs.
  - `VisionPilot/simulation/CARLA/ROS2/src/*/package.xml` and `setup.py`: package TODOs.
  - `VisionPilot/development_releases/*/src/lane_tracking/lane_tracking.cpp`: `// Now using quadratic (TODO: add dynamic logic later - Tran)`.
  - `VisionPilot/.../src/inference/autospeed/README.md`: `## TODOs / Future Improvements`.
- Impact: Accumulating TODOs increase future cost of change and surprise during deployment.
- Fix approach: Triage TODOs into tracked issues with priorities and owners; convert high-risk TODOs into required tasks before production releases.

9) Model artifacts and storage strategy unclear
- Issue: `Models/` directory contains scripts that parse and export models (e.g., `Models/exports/benchmark_onnx_models.py`) and many large model/data parsing scripts, but the repository does not make storage strategy or LFS usage explicit.
- Evidence:
  - `Models/` directory present with many large scripts: `Models/exports/benchmark_onnx_models.py`, `Models/data_parsing/...`.
- Impact: Large binary model files, if committed to Git, will bloat the repo and complicate CI. Missing artifact handling is an operational risk.
- Fix approach: Ensure large binaries are stored in an artifact registry (S3, model registry) or Git LFS. Add rules to `.gitignore` and CI steps to fetch model artifacts from a reproducible source.

10) Duplicate directories for releases increase code-ownership ambiguity
- Issue: `production_release/` and `development_releases/*` directories contain parallel implementations.
- Evidence:
  - `VisionPilot/production_release/` and `VisionPilot/development_releases/1.0/` and `0.9/` hold similar files (e.g., `main.cpp`, `src/visualization/visualize.cpp`).
- Impact: Contributors must know which tree is authoritative; changes may be inconsistently applied.
- Fix approach: Consolidate release branches into a single tree with versioned artifacts or tags. Use build-time flags/configuration to select behaviors.

---

## Security considerations

11) No explicit secret-management or secret-scan evidence
- Issue: No `.env` files were detected during scan (good), but there is no evidence of secret scanning in CI or a documented secret-management policy.
- Evidence:
  - No `*.env` files detected in top-level scan.
  - CI workflows do not include secret scanning or mention of secret policies in repository docs.
- Impact: Risk of accidental secret commits or unmanaged secrets in deployment.
- Fix approach: Add `.github` secret scanning, document secret policies in `ONBOARDING.md`, and add pre-commit hooks to prevent large secrets from being committed.

12) Dependency and supply-chain risk for Python/C++ tooling
- Issue: No single manifest for dependencies detected at repository root; project uses C++ and Python and multiple package manifests may be distributed across subfolders.
- Evidence:
  - No `package.json`, `requirements.txt`, `go.mod` at top-level were detected during the scan. Python package manifests exist under subfolders (e.g., `VisionPilot/simulation/CARLA/ROS2/.../setup.py`).
- Impact: Hard to audit third-party dependencies across the entire project; increases supply-chain risk.
- Fix approach: Centralize dependency tracking or add a repository-level manifest and add dependency scanning CI jobs.

---

## Potential blockers for onboarding or refactor

- Large monolithic files and duplicate release trees (`VisionPilot/production_release/main.cpp`, `VisionPilot/development_releases/*/main.cpp`): block any refactor that touches core pipeline.
- No tests around `lane_tracking` and `lane_filtering` modules (`VisionPilot/.../src/lane_tracking/lane_tracking.cpp`, `VisionPilot/.../src/lane_filtering/lane_filter.cpp`): new contributors cannot safely change algorithms.
- Missing packaging metadata and license placeholders in simulation packages (`VisionPilot/simulation/CARLA/ROS2/.../package.xml`, `setup.py`): prevents building/running those packages in some environments until addressed.
- No documented container build artifact (`Dockerfile` not present at root) while CI references a container build workflow (`.github/workflows/build-visionpilot-container.yaml`): blocks reproducible local environment setup.
- Unclear model artifact management (`Models/` contents): onboarding requires clarity on how to obtain model weights and datasets.

---

## Recommended immediate next steps (minimum viable changes to unblock development)
1. Add a minimal test harness and CI job that runs unit tests for a small subset of critical modules (start with `lane_tracking` and `lane_filtering`).
2. Consolidate or document the canonical source tree (decide whether `production_release/` or another directory is authoritative) and add a short migration/cleanup plan.
3. Implement camera calibration configuration and a unit test verifying `transformPixelsToMeters()` behavior; remove or escalate calibration TODO comments.
4. Add a repo-level `Dockerfile` or document the build-image path used by `.github/workflows/build-visionpilot-container.yaml` and add a `make` shortcut in `ONBOARDING.md`.
5. Add static-analysis and dependency-scanning steps to CI; enforce as required checks for PRs.
6. Inventory `Models/` for large binaries and document where model artifacts are stored and how to fetch them.

---

## Appendix – scan evidence highlights (selected file paths)
- Todo/calibration markers:
  - `VisionPilot/production_release/main.cpp`
  - `VisionPilot/development_releases/1.0/main.cpp`
  - `VisionPilot/development_releases/0.9/main.cpp`
- Large and duplicated files:
  - `VisionPilot/development_releases/1.0/main.cpp` (~2153 lines)
  - `VisionPilot/production_release/main.cpp` (~1959 lines)
  - `VisionPilot/development_releases/0.9/main.cpp` (~1959 lines)
  - `VisionPilot/production_release/src/visualization/visualize.cpp`
  - `VisionPilot/development_releases/1.0/src/visualization/visualize.cpp`
- ROS2 / package TODOs:
  - `VisionPilot/simulation/CARLA/ROS2/src/camera_publisher/package.xml`
  - `VisionPilot/simulation/CARLA/ROS2/src/camera_publisher/setup.py`
  - `VisionPilot/simulation/CARLA/ROS2/src/camera_spectator/package.xml`
- CI workflows present:
  - `.github/workflows/build-visionpilot-container.yaml`
  - `.github/workflows/spell-check-differential.yaml`
  - `.github/workflows/semantic-pull-request.yaml`
  - `.github/workflows/spell-check-daily.yaml`
- Models directory (needs artifact policy):
  - `Models/exports/benchmark_onnx_models.py`
  - `Models/data_parsing/...` (multiple scripts)
- Onboarding doc:
  - `ONBOARDING.md` (present, but will require updates once above issues are addressed)

---

*End of concerns audit.*
