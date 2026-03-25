"""
Tests for Pydantic models.
"""

import pytest
from pydantic import ValidationError

from deeplense_agent.models import (
    CosmologicalParameters,
    DarkMatterType,
    ImageData,
    InstrumentConfig,
    InstrumentType,
    MainHaloParameters,
    ModelType,
    SimulationConfig,
    SimulationMetadata,
    SimulationOutput,
    SimulationRequest,
    SourceLightParameters,
    SubstructureParameters,
)
import numpy as np


class TestDarkMatterType:
    """Tests for DarkMatterType enum."""

    def test_from_natural_language_cdm(self):
        assert DarkMatterType.from_natural_language("CDM") == DarkMatterType.CDM
        assert DarkMatterType.from_natural_language("cold dark matter") == DarkMatterType.CDM
        assert DarkMatterType.from_natural_language("WIMP") == DarkMatterType.CDM

    def test_from_natural_language_axion(self):
        assert DarkMatterType.from_natural_language("axion") == DarkMatterType.AXION
        assert DarkMatterType.from_natural_language("vortex") == DarkMatterType.AXION
        assert DarkMatterType.from_natural_language("ultralight") == DarkMatterType.AXION
        assert DarkMatterType.from_natural_language("fuzzy") == DarkMatterType.AXION

    def test_from_natural_language_no_sub(self):
        assert DarkMatterType.from_natural_language("no sub") == DarkMatterType.NO_SUBSTRUCTURE
        assert DarkMatterType.from_natural_language("clean") == DarkMatterType.NO_SUBSTRUCTURE
        assert DarkMatterType.from_natural_language("smooth lens") == DarkMatterType.NO_SUBSTRUCTURE

    def test_from_natural_language_default(self):
        # Unknown text should default to CDM
        assert DarkMatterType.from_natural_language("unknown") == DarkMatterType.CDM


class TestModelType:
    """Tests for ModelType enum."""

    def test_resolution(self):
        assert ModelType.MODEL_I.resolution == 150
        assert ModelType.MODEL_II.resolution == 64
        assert ModelType.MODEL_III.resolution == 64
        assert ModelType.MODEL_IV.resolution == 64

    def test_num_channels(self):
        assert ModelType.MODEL_I.num_channels == 1
        assert ModelType.MODEL_II.num_channels == 1
        assert ModelType.MODEL_III.num_channels == 1
        assert ModelType.MODEL_IV.num_channels == 3

    def test_default_instrument(self):
        assert ModelType.MODEL_I.default_instrument == InstrumentType.GENERIC
        assert ModelType.MODEL_II.default_instrument == InstrumentType.EUCLID
        assert ModelType.MODEL_III.default_instrument == InstrumentType.HST


class TestCosmologicalParameters:
    """Tests for CosmologicalParameters model."""

    def test_default_values(self):
        params = CosmologicalParameters()
        assert params.H0 == 70.0
        assert params.Om0 == 0.3
        assert params.Ob0 == 0.05
        assert params.z_lens == 0.5
        assert params.z_source == 1.0

    def test_custom_values(self):
        params = CosmologicalParameters(
            H0=68.0,
            Om0=0.28,
            z_lens=0.3,
            z_source=1.5,
        )
        assert params.H0 == 68.0
        assert params.z_source == 1.5

    def test_redshift_validation(self):
        # Source must be behind lens
        with pytest.raises(ValidationError):
            CosmologicalParameters(z_lens=1.0, z_source=0.5)

    def test_h0_bounds(self):
        # H0 must be between 50 and 100
        with pytest.raises(ValidationError):
            CosmologicalParameters(H0=40.0)
        with pytest.raises(ValidationError):
            CosmologicalParameters(H0=110.0)


class TestSubstructureParameters:
    """Tests for SubstructureParameters model."""

    def test_default_cdm(self):
        params = SubstructureParameters()
        assert params.substructure_type == DarkMatterType.CDM
        assert params.n_sub_mean == 25

    def test_axion_default_mass(self):
        params = SubstructureParameters(substructure_type=DarkMatterType.AXION)
        assert params.axion_mass == 1e-23

    def test_de_broglie_wavelength(self):
        params = SubstructureParameters(
            substructure_type=DarkMatterType.AXION,
            axion_mass=1e-22,
        )
        # λ_dB ≈ 0.6 kpc * (10^-22 eV / m_a)
        expected = 0.6 * (1e-22 / 1e-22)
        assert params.de_broglie_wavelength_kpc == pytest.approx(expected, rel=0.01)

    def test_no_wavelength_for_cdm(self):
        params = SubstructureParameters(substructure_type=DarkMatterType.CDM)
        assert params.de_broglie_wavelength_kpc is None


class TestSimulationConfig:
    """Tests for SimulationConfig model."""

    def test_default_config(self):
        config = SimulationConfig()
        assert config.model_type == ModelType.MODEL_I
        assert config.num_images == 1
        assert config.instrument is not None

    def test_instrument_auto_set(self):
        config = SimulationConfig(model_type=ModelType.MODEL_II)
        assert config.instrument.instrument_type == InstrumentType.EUCLID

    def test_expected_resolution(self):
        config = SimulationConfig(model_type=ModelType.MODEL_I)
        assert config.expected_resolution == (150, 150)

        config = SimulationConfig(model_type=ModelType.MODEL_II)
        assert config.expected_resolution == (64, 64)

    def test_full_config(self):
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=100,
            random_seed=42,
            cosmology=CosmologicalParameters(z_lens=0.3, z_source=1.2),
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.AXION,
                axion_mass=1e-23,
            ),
        )
        assert config.num_images == 100
        assert config.random_seed == 42
        assert config.cosmology.z_lens == 0.3
        assert config.substructure.substructure_type == DarkMatterType.AXION


class TestInstrumentConfig:
    """Tests for InstrumentConfig model."""

    def test_for_model_i(self):
        config = InstrumentConfig.for_model_type(ModelType.MODEL_I)
        assert config.instrument_type == InstrumentType.GENERIC
        assert config.num_pixels == 150
        assert config.psf_type == "GAUSSIAN"

    def test_for_model_ii(self):
        config = InstrumentConfig.for_model_type(ModelType.MODEL_II)
        assert config.instrument_type == InstrumentType.EUCLID
        assert config.num_pixels == 64

    def test_for_model_iii(self):
        config = InstrumentConfig.for_model_type(ModelType.MODEL_III)
        assert config.instrument_type == InstrumentType.HST
        assert config.num_pixels == 64


class TestSimulationRequest:
    """Tests for SimulationRequest model."""

    def test_natural_language_request(self):
        request = SimulationRequest(
            natural_language_prompt="Generate 10 CDM lens images"
        )
        assert request.natural_language_prompt is not None
        assert request.config is None

    def test_config_request(self):
        config = SimulationConfig()
        request = SimulationRequest(config=config)
        assert request.config is not None

    def test_require_at_least_one(self):
        with pytest.raises(ValidationError):
            SimulationRequest()


class TestImageData:
    """Tests for ImageData model."""

    def test_from_numpy_2d(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        img = ImageData.from_numpy(arr, encode_png=True)

        assert img.width == 64
        assert img.height == 64
        assert img.channels == 1
        assert img.base64_png is not None
        assert img.min_value == pytest.approx(arr.min())
        assert img.max_value == pytest.approx(arr.max())

    def test_from_numpy_3d(self):
        arr = np.random.rand(64, 64, 3).astype(np.float32)
        img = ImageData.from_numpy(arr, encode_png=True)

        assert img.width == 64
        assert img.height == 64
        assert img.channels == 3

    def test_without_encoding(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        img = ImageData.from_numpy(arr, encode_png=False)

        assert img.base64_png is None
        assert img.width == 64


class TestSimulationOutput:
    """Tests for SimulationOutput model."""

    def test_success_output(self):
        output = SimulationOutput(
            success=True,
            images=[ImageData.from_numpy(np.random.rand(64, 64))],
        )
        assert output.success
        assert output.num_images_generated == 1

    def test_failed_output(self):
        output = SimulationOutput(
            success=False,
            error_message="Test error",
        )
        assert not output.success
        assert output.error_message == "Test error"
        assert output.num_images_generated == 0
