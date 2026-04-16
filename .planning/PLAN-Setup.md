PLAN-Setup: Docker simulation environment

Purpose

Provide a reproducible containerized environment to run the MVP perception CLI and verify model artifacts without requiring host Python/tooling installs.

Files to add

- docker/Dockerfile: builds image `visionpilot-sim` with Python 3.8+, pip, and dependencies from Models/requirements.txt (if present). Entrypoint: `python3 Models/inference/cli.py`.
- docker/docker-compose.yml: defines service `sim` that mounts the repository root into `/workspace` and an outputs volume for results. Does not expose network by default.
- Makefile target `docker-sim`: builds image and runs container with a convenience command to exercise `--help` or run a sample inference.

Build and run examples

1) Build:
   docker build -t visionpilot-sim docker/

2) Run help smoke test:
   docker run --rm -v "$(pwd)":/workspace -w /workspace visionpilot-sim python3 Models/inference/cli.py --help

3) Run a sample (checkpoint mounted from host):
   docker run --rm -v "$(pwd)":/workspace -v /path/to/checkpoints:/checkpoints -w /workspace visionpilot-sim \
     python3 Models/inference/cli.py --model SceneSeg --checkpoint /checkpoints/scene_seg_checkpoint.pth --input Media/daytime_fair_weather_1.jpg --out_dir /workspace/outputs/scene_seg --device cpu

Notes and recommendations

- Do not bake large model checkpoints into the image; mount them at runtime from host or object storage.
- Limit container resources when testing: e.g., `docker run --cpus="1" --memory="4g" ...` for CI smoke tests.
- If Models/requirements.txt is missing, document required packages in ONBOARDING.md (torch, torchvision, pillow, numpy).
- For CI, prefer the smoke test that runs `--help` (fast) rather than full inference.

Security

- Container runs as the invoking user by default. Avoid running as root in CI.
- Do not accept externally-provided checkpoints in CI unless checksums/verifications are enforced.

1. Manually Create Dockerfile.carla
Save the following content into a file named Dockerfile.carla:
# Dockerfile for CARLA Simulation
FROM ubuntu:22.04
# Base environment setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales curl gnupg lsb-release python3-pip build-essential \
    && locale-gen en_US.UTF-8
# Install ROS2 Humble
RUN apt update && apt install -y \
    ros-humble-desktop python3-colcon-common-extensions
# Install CARLA dependencies
RUN pip install carla
# Default entry
CMD ["/bin/bash"]
---
2. Manually Create Dockerfile.soda
Save the following content into Dockerfile.soda:
# Dockerfile for SODA.Sim Simulation
FROM ubuntu:22.04
# Base environment setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales curl gnupg lsb-release python3-pip build-essential \
    && locale-gen en_US.UTF-8
# Install ROS2 Humble
RUN apt update && apt install -y \
    ros-humble-desktop python3-colcon-common-extensions
# Add any SODA.Sim-specific dependencies here
# Example:
# RUN apt install -y soda-sim-specific-package
# Default entry
CMD ["/bin/bash"]
---
3. Manually Create the run_simulation.sh Script
Save the following content into run_simulation.sh:
#!/bin/bash
echo "Choose a simulator to run:"
echo "1. CARLA"
echo "2. SODA.Sim"
read -p "Enter your choice (1/2): " choice
if [[ $choice -eq 1 ]]; then
    echo "Starting CARLA simulation..."
    docker run -it --rm --name carla-simulation \
        -v $(pwd):/workspace \
        -e "DISPLAY=$DISPLAY" \
        carla-image ./start_carla.sh
elif [[ $choice -eq 2 ]]; then
    echo "Starting SODA.Sim simulation..."
    docker run -it --rm --name soda-simulation \
        -v $(pwd):/workspace \
        -e "DISPLAY=$DISPLAY" \
        soda-image ./start_soda_sim.sh
else
    echo "Invalid choice. Exiting."
    exit 1
fi
Make it executable:
chmod +x run_simulation.sh
---
4. Manually Create Supporting Scripts
start_carla.sh:
#!/bin/bash
echo "Initializing CARLA simulation..."
# Add CARLA-specific commands here
start_soda_sim.sh:
#!/bin/bash
echo "Initializing SODA.Sim simulation..."
# Add SODA.Sim-specific commands here
Make both scripts executable:
chmod +x start_carla.sh start_soda_sim.sh
---
### 5. Update `AGENTS.md`
Follow the structure I provided earlier to update `AGENTS.md`.. you can append this into PLAN-Setup.md then run smoke test

(Generated to satisfy PLAN task: docker-simulation-setup)