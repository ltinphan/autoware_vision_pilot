#!/usr/bin/env python3
"""
Integration tests for CARLA simulation with EgoLanes.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCARLAIntegration(unittest.TestCase):
    """Test CARLA simulation integration."""

    def test_carla_client_imports(self):
        """Test that CARLA client can be imported."""
        try:
            from simulations.carla_egolanes_client import parse_args
            self.assertTrue(True, "CARLA client imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import CARLA client: {e}")

    def test_egolanes_file_exists(self):
        """Test that EgoLanes inference file exists."""
        egolanes_file = Path(__file__).resolve().parents[1] / "Models" / "inference" / "ego_lanes_infer.py"
        self.assertTrue(egolanes_file.exists(), "EgoLanes inference file not found")

    def test_checkpoint_exists(self):
        """Test that EgoLanes checkpoint exists."""
        checkpoint_path = Path(__file__).resolve().parents[1] / "Models" / "model_library" / "EgoLanes" / "checkpoint.pth"
        self.assertTrue(checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}")

    def test_simulation_scripts_exist(self):
        """Test that simulation scripts exist."""
        carla_script = Path(__file__).resolve().parents[1] / "start_carla.sh"
        soda_script = Path(__file__).resolve().parents[1] / "start_soda_sim.sh"

        self.assertTrue(carla_script.exists(), "CARLA script not found")
        self.assertTrue(soda_script.exists(), "SODA script not found")


if __name__ == '__main__':
    unittest.main()
