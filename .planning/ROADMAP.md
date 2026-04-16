# ROADMAP

## VisionPilot - Docker Simulation Setup

---

## Phase 1: MVP Perception Pipeline

**Goal:** Run prebuilt EgoLanes model on sample images and output lane detections.

**Requirements:**
- EgoLanes model checkpoint
- Sample input images
- Docker container with dependencies

**Success Criteria:**
- [ ] EgoLanes model downloads and loads successfully
- [ ] CLI can run inference on sample images
- [ ] Outputs are saved to disk in numpy format
- [ ] Docker container builds and runs

---

## Phase 2: Simulation Environment

**Goal:** Provide Docker environment and simulation scripts for CARLA/SODA.Sim.

**Requirements:**
- CARLA simulation support
- SODA.Sim simulation support
- Docker compose configuration

**Success Criteria:**
- [ ] CARLA simulation script launches successfully
- [ ] SODA.Sim simulation script launches successfully
- [ ] Docker compose can orchestrate simulation services
- [ ] EgoLanes client can connect to simulation

---

## Phase 3: Integration Testing

**Goal:** Test end-to-end simulation with EgoLanes inference.

**Requirements:**
- CARLA client implementation
- Real-time inference capability
- Visualization tools

**Success Criteria:**
- [ ] CARLA client spawns vehicle and captures frames
- [ ] Real-time inference runs on simulation frames
- [ ] Results are visualized and saved
- [ ] All tests pass

---

## Phase 4: Documentation and Polish

**Goal:** Complete documentation and developer onboarding.

**Requirements:**
- Updated README
- Developer guide
- Script documentation

**Success Criteria:**
- [ ] README updated with usage examples
- [ ] Developer guide created
- [ ] All scripts documented
- [ ] CI/CD ready (if applicable)
