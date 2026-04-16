# Agents and Simulator Integrations

This file documents local helper scripts and simulator Dockerfiles used to run simulation environments for development and testing.

Simulators

- CARLA
  - Dockerfile: docker/Dockerfile.carla
  - Run script: ./run_simulation.sh (choice 1)
  - Start script (inside container): ./start_carla.sh
  - Notes: Mount checkpoints or assets from host into /workspace or a dedicated volume. CARLA often requires additional GPU and display configuration.

- SODA.Sim
  - Dockerfile: docker/Dockerfile.soda
  - Run script: ./run_simulation.sh (choice 2)
  - Start script (inside container): ./start_soda_sim.sh
  - Notes: SODA.Sim-specific packages may need to be added to the Dockerfile.soda.

Maintenance

- Do not include large simulator binaries in the repo. Instead, provide scripts to download or mount them from host paths.
