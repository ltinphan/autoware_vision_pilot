# Coding Conventions

**Analysis Date:** 2026-04-11

## Overview
This document records the coding and style conventions detected in the repository, the automated tooling that enforces them, and concrete examples (file paths) as evidence.

## Formatting & Linters (enforced via pre-commit)

- Pre-commit configuration: `/.pre-commit-config.yaml` registers multiple formatters and linters that are applied locally and can be run in CI. Key hooks present:
  - `black` (Python formatting) — args: `--line-length=100` (`.pre-commit-config.yaml` lines 67-71)
  - `isort` (Python import sorting) — args: `--profile=black, --line-length=100` (`.pre-commit-config.yaml` lines 61-66)
  - `clang-format` (C/C++ formatting) — configured for C/C++/CUDA files (`.pre-commit-config.yaml` lines 73-78)
  - `cpplint` (C++ linting) — `cpplint` hook is present (`.pre-commit-config.yaml` lines 79-84)
  - `prettier` (JS/JSON/YAML and with plugins for xml/svg) (`.pre-commit-config.yaml` lines 30-34 and local prettier-svg hook lines 86-95)
  - Shell and YAML linters: `shellcheck`, `shfmt`, `yamllint`, `markdownlint` (`.pre-commit-config.yaml` lines 8-22, 35-39, 50-59)

Evidence: `/.pre-commit-config.yaml` (hooks list and arguments).

### C++ formatting
- Project has a repository-level clang-format config: `/.clang-format` with:
  - `BasedOnStyle: Google`
  - `ColumnLimit: 100`
  - Custom include ordering and other Google-style overrides

Evidence: `/.clang-format` (see IncludeCategories and ColumnLimit).

- `cpplint` rules are customized in `CPPLINT.cfg` (top-level), including `linelength=100`, `includeorder=standardcfirst`, and many filter overrides (allow C++11/17, whitespace rules overrides).

Evidence: `/CPPLINT.cfg` (linelength and filters).

### JavaScript / XML / Misc formatting
- Prettier configuration: `/.prettierrc.yaml` with `printWidth: 100`, `tabWidth: 2`, and file-specific overrides for XML and package.xml.

Evidence: `/.prettierrc.yaml`.

## Style & Naming Patterns (evidence-based)

- C++ style:
  - Based on Google style via `.clang-format` (`BasedOnStyle: Google`) with a 100-column limit -> expect Google-style bracing and spacing.
    - File: `/.clang-format`
  - Functions and helper routines in C++ use camelCase names (examples):
    - `printTensorStats`, `compareTensors` in `VisionPilot/production_release/test/test_autosteer.cpp` (file uses camelCase for function identifiers).
    - Evidence: `VisionPilot/production_release/test/test_autosteer.cpp` lines 20-52, 51-82.
  - Local variables and some identifiers use snake_case in C++ sources (examples): `frame_number`, `steering_angle` in `VisionPilot/production_release/test/test_autosteer.cpp`.
    - Evidence: `VisionPilot/production_release/test/test_autosteer.cpp` lines 186-196, 238-246.
  - Types and classes follow PascalCase/CamelCase: classes like `EgoLanesOnnxEngine`, `AutoSteerOnnxEngine` (evidence: `VisionPilot/production_release/test/test_autosteer.cpp` lines 128-135).

- Python style:
  - Formatting for Python is enforced with `black` and `isort` (line-length 100). Expect Black's canonical formatting with 100-column limit.
    - Evidence: `/.pre-commit-config.yaml` (Black and isort hooks, args) and `/.prettierrc.yaml` for xml-like files.
  - Observed Python files use snake_case for function and variable names (examples): `get_manual_bbox`, `load_homography_from_yaml` in `VisionPilot/middleware_recipes/Calibration/test_with_gt.py` (function names and variables use snake_case).
    - Evidence: `VisionPilot/middleware_recipes/Calibration/test_with_gt.py` lines 15-22, 54-66.
  - Python test-like scripts in `Models/training/test_validate_scene_seg.py` are written as scripts with `main()` and no pytest-style asserts; they print results rather than using test runner asserts.
    - Evidence: `Models/training/test_validate_scene_seg.py` (top-level `main()` function and script-like structure).

## Import ordering and module organization
- Python import sorting is enforced with `isort --profile=black` via pre-commit.
  - Evidence: `/.pre-commit-config.yaml` lines 61-66.

- C++ include order is driven by `.clang-format` IncludeCategories and by `CPPLINT.cfg` includeorder settings.
  - Evidence: `/.clang-format` (IncludeCategories) and `/CPPLINT.cfg` (`includeorder=standardcfirst`).

## Comments, Logging & Debugging Patterns
- C++ test file uses `std::cout`/`std::cerr` for logging and debug information (no structured logging framework in test files).
  - Evidence: `VisionPilot/production_release/test/test_autosteer.cpp` showing many `std::cout` and `std::cerr` statements.

- Python scripts use `print()` statements for progress and debugging in test scripts.
  - Evidence: `Models/training/test_validate_scene_seg.py`, `VisionPilot/middleware_recipes/Calibration/test_with_gt.py`.

## Build & Lint Integration
- C++ build is CMake-based. The `CMakeLists.txt` defines targets and includes test executable `test_autosteer` (not a GoogleTest target). CMake does not show `find_package(GTest)` or `add_subdirectory` for unit tests.
  - Evidence: `VisionPilot/production_release/CMakeLists.txt` (test target `test_autosteer` added as executable lines 386-393, but no GTest integration).

- Pre-commit runs clang-format and cpplint locally; `cpplint` config is in `/CPPLINT.cfg`.
  - Evidence: `/.pre-commit-config.yaml` and `/CPPLINT.cfg`.

## Recommended Prescriptive Conventions (based on detected patterns)
- Follow `.clang-format` for all C/C++ code. Add CI job that runs `clang-format --dry-run` or `pre-commit` (already present locally) on pushed commits.
  - Files: `/.clang-format`, `/.pre-commit-config.yaml`.

- For Python, use Black + isort with `--line-length=100` (already configured). Use pytest for tests (see TESTING.md recommendations).
  - Files: `/.pre-commit-config.yaml` (Black/isort), `/.prettierrc.yaml`.

- Naming:
  - Types/Classes: PascalCase / CamelCase (e.g., `EgoLanesOnnxEngine`)
  - Functions: camelCase in C++. For Python use snake_case.
  - Variables: prefer snake_case for Python; C++ code is mixed — adopt consistent snake_case for variables and camelCase for function names (matches current examples).

## Quick references (evidence files)
- `/.pre-commit-config.yaml`
- `/.clang-format`
- `/CPPLINT.cfg`
- `/.prettierrc.yaml`
- `VisionPilot/production_release/test/test_autosteer.cpp`
- `Models/training/test_validate_scene_seg.py`
- `VisionPilot/production_release/CMakeLists.txt`

---

*Convention analysis: 2026-04-11*
