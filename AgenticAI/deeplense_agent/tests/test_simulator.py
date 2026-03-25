"""
Tests for the DeepLense simulator wrapper.
"""

import pytest
import numpy as np

from deeplense_agent.models import (
    DarkMatterType,
    ModelType,
    SimulationConfig,
    SubstructureParameters,
)
from deeplense_agent.simulator import DeepLenseSimulator, create_simulator


class TestDeepLenseSimulator:
    """Tests for DeepLenseSimulator class."""

    @pytest.fixture
    def mock_simulator(self):
        """Create a simulator in mock mode."""
        return create_simulator(mock_mode=True)

    def test_create_mock_simulator(self, mock_simulator):
        """Test simulator creation in mock mode."""
        assert mock_simulator._mock_mode is True

    def test_basic_simulation(self, mock_simulator):
        """Test basic simulation execution."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=2,
        )
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert output.num_images_generated == 2
        assert len(output.images) == 2
        assert output.metadata is not None
        assert output.metadata.simulation_id is not None

    def test_cdm_simulation(self, mock_simulator):
        """Test CDM substructure simulation."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=1,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.CDM,
                n_sub_mean=30,
            ),
        )
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert output.images[0].width == 150
        assert output.images[0].height == 150

    def test_axion_simulation(self, mock_simulator):
        """Test axion/vortex simulation."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_II,
            num_images=1,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.AXION,
                axion_mass=1e-23,
            ),
        )
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert output.images[0].width == 64

    def test_no_substructure_simulation(self, mock_simulator):
        """Test simulation without substructure."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=1,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.NO_SUBSTRUCTURE,
            ),
        )
        output = mock_simulator.run_simulation(config)

        assert output.success

    def test_image_statistics(self, mock_simulator):
        """Test that image statistics are computed correctly."""
        config = SimulationConfig(num_images=1)
        output = mock_simulator.run_simulation(config)

        assert output.success
        img = output.images[0]

        assert img.min_value >= 0
        assert img.max_value <= 1
        assert img.min_value <= img.mean_value <= img.max_value

    def test_reproducibility_with_seed(self, mock_simulator):
        """Test that same seed produces same results."""
        config = SimulationConfig(
            num_images=1,
            random_seed=12345,
        )

        output1 = mock_simulator.run_simulation(config)
        output2 = mock_simulator.run_simulation(config)

        assert output1.success and output2.success

        # With same seed, mean values should be identical
        assert output1.images[0].mean_value == pytest.approx(
            output2.images[0].mean_value, rel=1e-6
        )

    def test_metadata_generation(self, mock_simulator):
        """Test that metadata is properly generated."""
        config = SimulationConfig(
            num_images=3,
            random_seed=42,
        )
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert output.metadata is not None
        assert output.metadata.random_state_used == 42
        assert output.metadata.duration_seconds > 0
        assert output.metadata.config.num_images == 3

    def test_model_ii_resolution(self, mock_simulator):
        """Test Model II produces 64x64 images."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_II,
            num_images=1,
        )
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert output.images[0].width == 64
        assert output.images[0].height == 64

    def test_model_iii_resolution(self, mock_simulator):
        """Test Model III produces 64x64 images."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_III,
            num_images=1,
        )
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert output.images[0].width == 64
        assert output.images[0].height == 64

    def test_warnings_in_mock_mode(self, mock_simulator):
        """Test that mock mode adds appropriate warnings."""
        config = SimulationConfig(num_images=1)
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert len(output.warnings) > 0
        assert any("mock" in w.lower() for w in output.warnings)

    def test_batch_generation(self, mock_simulator):
        """Test generating multiple images in batch."""
        config = SimulationConfig(num_images=10)
        output = mock_simulator.run_simulation(config)

        assert output.success
        assert output.num_images_generated == 10
        assert len(output.images) == 10

        # All images should have valid data
        for img in output.images:
            assert img.width > 0
            assert img.height > 0
            assert img.base64_png is not None

    def test_png_encoding(self, mock_simulator):
        """Test that PNG encoding works correctly."""
        config = SimulationConfig(num_images=1)
        output = mock_simulator.run_simulation(config)

        assert output.success
        img = output.images[0]

        # Should have base64 PNG data
        assert img.base64_png is not None
        assert len(img.base64_png) > 0

        # Should be valid base64
        import base64
        decoded = base64.b64decode(img.base64_png)
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic bytes


class TestSimulatorFactory:
    """Tests for simulator factory function."""

    def test_create_auto_detect(self):
        """Test auto-detection mode."""
        simulator = create_simulator(mock_mode=None)
        # Should create a simulator (likely in mock mode if deps not installed)
        assert simulator is not None

    def test_create_explicit_mock(self):
        """Test explicit mock mode."""
        simulator = create_simulator(mock_mode=True)
        assert simulator._mock_mode is True

    def test_is_available(self):
        """Test is_available property."""
        mock_sim = create_simulator(mock_mode=True)
        # Mock mode means real simulation is not available
        assert mock_sim.is_available is False
