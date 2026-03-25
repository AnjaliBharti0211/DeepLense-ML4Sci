"""
Pydantic models for DeepLense simulation parameters and outputs.

This module defines strongly-typed models for:
- Cosmological parameters (H0, matter density, redshifts)
- Dark matter substructure types and configurations
- Instrument configurations (Euclid, HST)
- Source light profiles
- Simulation requests and outputs
"""

from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class DarkMatterType(str, Enum):
    """Types of dark matter substructure configurations."""

    NO_SUBSTRUCTURE = "no_substructure"
    CDM = "cdm"  # Cold Dark Matter with point-mass subhalos
    AXION = "axion"  # Ultralight axion/vortex substructure

    @classmethod
    def from_natural_language(cls, text: str) -> DarkMatterType:
        """Parse dark matter type from natural language description."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["no sub", "clean", "smooth", "without sub"]):
            return cls.NO_SUBSTRUCTURE
        elif any(kw in text_lower for kw in ["cdm", "cold dark matter", "point mass", "wimp"]):
            return cls.CDM
        elif any(kw in text_lower for kw in ["axion", "vortex", "uldm", "ultralight", "fuzzy"]):
            return cls.AXION
        else:
            return cls.CDM  # Default to CDM


class InstrumentType(str, Enum):
    """Supported telescope/instrument types."""

    GENERIC = "generic"  # Model I style with Gaussian PSF
    EUCLID = "euclid"    # Euclid VIS band (Model II)
    HST = "hst"          # Hubble Space Telescope (Model III)

    @classmethod
    def from_natural_language(cls, text: str) -> InstrumentType:
        """Parse instrument type from natural language description."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["euclid", "model ii", "model 2", "model_ii"]):
            return cls.EUCLID
        elif any(kw in text_lower for kw in ["hst", "hubble", "model iii", "model 3", "model_iii"]):
            return cls.HST
        elif any(kw in text_lower for kw in ["generic", "basic", "model i", "model 1", "model_i", "simple"]):
            return cls.GENERIC
        else:
            return cls.GENERIC  # Default


class ModelType(str, Enum):
    """DeepLenseSim model configurations (I through IV)."""

    MODEL_I = "model_i"      # 150x150, Gaussian PSF, Sersic source
    MODEL_II = "model_ii"    # 64x64, Euclid-like
    MODEL_III = "model_iii"  # 64x64, HST-like
    MODEL_IV = "model_iv"    # 64x64, Euclid + real galaxy images (RGB)

    @property
    def resolution(self) -> int:
        """Get the image resolution for this model type."""
        return 150 if self == ModelType.MODEL_I else 64

    @property
    def num_channels(self) -> int:
        """Get number of image channels."""
        return 3 if self == ModelType.MODEL_IV else 1

    @property
    def default_instrument(self) -> InstrumentType:
        """Get the default instrument for this model."""
        mapping = {
            ModelType.MODEL_I: InstrumentType.GENERIC,
            ModelType.MODEL_II: InstrumentType.EUCLID,
            ModelType.MODEL_III: InstrumentType.HST,
            ModelType.MODEL_IV: InstrumentType.EUCLID,
        }
        return mapping[self]


class CosmologicalParameters(BaseModel):
    """Cosmological parameters for the simulation."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "H0": 70.0,
                    "Om0": 0.3,
                    "Ob0": 0.05,
                    "z_lens": 0.5,
                    "z_source": 1.0,
                }
            ]
        }
    )

    H0: Annotated[
        float,
        Field(
            default=70.0,
            ge=50.0,
            le=100.0,
            description="Hubble constant in km/s/Mpc"
        )
    ]

    Om0: Annotated[
        float,
        Field(
            default=0.3,
            ge=0.1,
            le=0.5,
            description="Total matter density parameter"
        )
    ]

    Ob0: Annotated[
        float,
        Field(
            default=0.05,
            ge=0.01,
            le=0.1,
            description="Baryon density parameter"
        )
    ]

    z_lens: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.1,
            le=2.0,
            description="Lens (halo) redshift",
            alias="z_halo"
        )
    ]

    z_source: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.2,
            le=5.0,
            description="Source galaxy redshift",
            alias="z_gal"
        )
    ]

    @model_validator(mode="after")
    def validate_redshifts(self) -> CosmologicalParameters:
        """Ensure source is behind the lens."""
        if self.z_source <= self.z_lens:
            raise ValueError(
                f"Source redshift ({self.z_source}) must be greater than "
                f"lens redshift ({self.z_lens}) for gravitational lensing"
            )
        return self


class SubstructureParameters(BaseModel):
    """
    Parameters for dark matter substructure.

    Based on research from:
    - Alexander et al. 2021 (arXiv:2008.12731): CDM subhalo parameters
    - Alexander et al. 2020 (arXiv:1909.07346): Vortex parameters

    CDM Parameters:
    - Subhalo mass range: 10^6 to 10^10 M_sun
    - Number of subhalos: Poisson with μ = 25
    - Power law slope: β = -1.9
    - Mass fraction: ~1% of halo mass

    Axion/Vortex Parameters:
    - Axion mass: typically 10^-23 eV
    - Vortex length: uniform [0.5, 2.0]
    - Mass fraction: uniform [3%, 5.5%]
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "substructure_type": "cdm",
                    "m_sub_min": 1e6,
                    "m_sub_max": 1e10,
                    "n_sub_mean": 25,
                    "power_law_slope": -1.9,
                },
                {
                    "substructure_type": "axion",
                    "axion_mass": 1e-23,
                    "vortex_mass": 3e10,
                    "vortex_length_min": 0.5,
                    "vortex_length_max": 2.0,
                }
            ]
        }
    )

    substructure_type: DarkMatterType = Field(
        default=DarkMatterType.CDM,
        description="Type of dark matter substructure"
    )

    # CDM-specific parameters
    m_sub_min: Annotated[
        float,
        Field(
            default=1e6,
            ge=1e4,
            le=1e8,
            description="Minimum subhalo mass in solar masses (CDM only)"
        )
    ]

    m_sub_max: Annotated[
        float,
        Field(
            default=1e10,
            ge=1e8,
            le=1e12,
            description="Maximum subhalo mass in solar masses (CDM only)"
        )
    ]

    n_sub_mean: Annotated[
        int,
        Field(
            default=25,
            ge=1,
            le=100,
            description="Mean number of subhalos (CDM only)"
        )
    ]

    power_law_slope: Annotated[
        float,
        Field(
            default=-1.9,  # From arXiv:2008.12731 - subhalo mass function slope β
            ge=-2.5,
            le=0.0,
            description="Power-law slope for subhalo mass function β (CDM only). "
                        "Value of -1.9 from Alexander et al. 2021 (arXiv:2008.12731)"
        )
    ]

    # Axion-specific parameters
    axion_mass: Annotated[
        float | None,
        Field(
            default=None,
            ge=1e-25,
            le=1e-20,
            description="Axion mass in eV (axion/vortex only)"
        )
    ]

    vortex_mass: Annotated[
        float,
        Field(
            default=3e10,
            ge=1e9,
            le=1e12,
            description="Total vortex mass in solar masses (axion/vortex only)"
        )
    ]

    vortex_resolution: Annotated[
        int,
        Field(
            default=100,
            ge=50,
            le=500,
            description="Resolution for vortex representation (axion/vortex only)"
        )
    ]

    # Additional vortex parameters from arXiv:2008.12731
    vortex_length_min: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.1,
            le=2.0,
            description="Minimum vortex length (uniform distribution). "
                        "From arXiv:2008.12731: uniform [0.5, 2.0]"
        )
    ]

    vortex_length_max: Annotated[
        float,
        Field(
            default=2.0,
            ge=0.5,
            le=5.0,
            description="Maximum vortex length (uniform distribution). "
                        "From arXiv:2008.12731: uniform [0.5, 2.0]"
        )
    ]

    vortex_mass_fraction_min: Annotated[
        float,
        Field(
            default=0.03,
            ge=0.01,
            le=0.1,
            description="Minimum vortex mass fraction (3%). "
                        "From arXiv:2008.12731: uniform [3%, 5.5%]"
        )
    ]

    vortex_mass_fraction_max: Annotated[
        float,
        Field(
            default=0.055,
            ge=0.02,
            le=0.15,
            description="Maximum vortex mass fraction (5.5%). "
                        "From arXiv:2008.12731: uniform [3%, 5.5%]"
        )
    ]

    # CDM mass fraction from papers
    cdm_mass_fraction: Annotated[
        float,
        Field(
            default=0.01,
            ge=0.001,
            le=0.1,
            description="CDM substructure mass fraction (1% of halo mass). "
                        "From arXiv:2008.12731: O(1%)"
        )
    ]

    @model_validator(mode="after")
    def validate_axion_params(self) -> SubstructureParameters:
        """Validate axion parameters when axion type is selected."""
        if self.substructure_type == DarkMatterType.AXION:
            if self.axion_mass is None:
                # Set default axion mass if not provided
                object.__setattr__(self, "axion_mass", 1e-23)
        return self

    @computed_field
    @property
    def de_broglie_wavelength_kpc(self) -> float | None:
        """Compute de Broglie wavelength in kpc for axion mass."""
        if self.axion_mass is None:
            return None
        # λ_dB ≈ 0.6 kpc * (10^-22 eV / m_a)
        return 0.6 * (1e-22 / self.axion_mass)


class SourceLightParameters(BaseModel):
    """
    Parameters for the source galaxy light profile.

    Based on research from arXiv:2008.12731 and arXiv:1909.07346:
    - Sersic index n: 1.5 (fixed)
    - Axis ratio e: uniform [0.4, 1.0] or [0.7, 1.0]
    - Effective radius R: uniform [0.25, 1.0] arcsec
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "magnitude": 20.0,
                    "effective_radius": 0.25,
                    "sersic_index": 1.5,
                    "axis_ratio_min": 0.4,
                    "axis_ratio_max": 1.0,
                }
            ]
        }
    )

    magnitude: Annotated[
        float,
        Field(
            default=20.0,
            ge=15.0,
            le=28.0,
            description="Source apparent magnitude"
        )
    ]

    effective_radius: Annotated[
        float,
        Field(
            default=0.25,
            ge=0.05,
            le=2.0,
            description="Effective (half-light) radius in arcseconds. "
                        "From arXiv:2008.12731: uniform [0.25, 1.0]"
        )
    ]

    effective_radius_max: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.1,
            le=3.0,
            description="Maximum effective radius for sampling. "
                        "From arXiv:2008.12731: uniform [0.25, 1.0]"
        )
    ]

    sersic_index: Annotated[
        float,
        Field(
            default=1.5,  # Updated from 1.0 to match arXiv:2008.12731
            ge=0.5,
            le=6.0,
            description="Sersic index (n=1 exponential, n=4 de Vaucouleurs). "
                        "From arXiv:2008.12731: n=1.5 (fixed)"
        )
    ]

    # Axis ratio parameters from papers
    axis_ratio_min: Annotated[
        float,
        Field(
            default=0.4,
            ge=0.1,
            le=1.0,
            description="Minimum axis ratio (ellipticity). "
                        "From arXiv:2008.12731: uniform [0.4, 1.0]"
        )
    ]

    axis_ratio_max: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.5,
            le=1.0,
            description="Maximum axis ratio (1.0 = circular). "
                        "From arXiv:2008.12731: uniform [0.4, 1.0]"
        )
    ]

    ellipticity_e1: Annotated[
        float,
        Field(
            default=-0.1,
            ge=-0.9,
            le=0.9,
            description="First ellipticity component"
        )
    ]

    ellipticity_e2: Annotated[
        float,
        Field(
            default=0.1,
            ge=-0.9,
            le=0.9,
            description="Second ellipticity component"
        )
    ]

    center_x_offset: Annotated[
        float | None,
        Field(
            default=None,
            ge=-0.5,
            le=0.5,
            description="Source center x offset (random if None)"
        )
    ]

    center_y_offset: Annotated[
        float | None,
        Field(
            default=None,
            ge=-0.5,
            le=0.5,
            description="Source center y offset (random if None)"
        )
    ]


class MainHaloParameters(BaseModel):
    """
    Parameters for the main lens halo.

    Based on research from arXiv:2008.12731 and arXiv:1909.07346:
    - Total halo mass: M_TOT = 10^12 M_sun (fixed)
    - Einstein radius: θ_E = 1.2 arcsec
    - Axis ratio e: uniform [0.5, 1.0]
    - External shear: γ_ext uniform [0.0, 0.3]
    """

    halo_mass: Annotated[
        float,
        Field(
            default=1e12,  # From arXiv:2008.12731: M_TOT = 10^12 M_sun
            ge=1e10,
            le=1e14,
            description="Main halo mass in solar masses. "
                        "From arXiv:2008.12731: 10^12 M_sun (fixed)"
        )
    ]

    einstein_radius: Annotated[
        float,
        Field(
            default=1.2,  # From arXiv:1909.07346: θ_E = 1.2 arcsec
            ge=0.5,
            le=5.0,
            description="Einstein radius in arcseconds. "
                        "From arXiv:1909.07346: θ_E = 1.2 arcsec"
        )
    ]

    axis_ratio_min: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.1,
            le=1.0,
            description="Minimum axis ratio for lens. "
                        "From arXiv:1909.07346: uniform [0.5, 1.0]"
        )
    ]

    axis_ratio_max: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.5,
            le=1.0,
            description="Maximum axis ratio (1.0 = circular)"
        )
    ]

    ellipticity_e1: Annotated[
        float,
        Field(
            default=0.1,
            ge=-0.5,
            le=0.5,
            description="First ellipticity component of the lens"
        )
    ]

    ellipticity_e2: Annotated[
        float,
        Field(
            default=0.0,
            ge=-0.5,
            le=0.5,
            description="Second ellipticity component of the lens"
        )
    ]

    # External shear from papers: uniform [0.0, 0.3]
    external_shear_gamma1: Annotated[
        float,
        Field(
            default=0.05,
            ge=-0.3,
            le=0.3,
            description="First external shear component"
        )
    ]

    external_shear_gamma2: Annotated[
        float,
        Field(
            default=0.0,
            ge=-0.3,
            le=0.3,
            description="Second external shear component"
        )
    ]

    external_shear_max: Annotated[
        float,
        Field(
            default=0.3,
            ge=0.0,
            le=0.5,
            description="Maximum external shear magnitude for sampling. "
                        "From arXiv:2008.12731: uniform [0.0, 0.3]"
        )
    ]


class InstrumentConfig(BaseModel):
    """Instrument/observation configuration."""

    model_config = ConfigDict(use_enum_values=True)

    instrument_type: InstrumentType = Field(
        default=InstrumentType.GENERIC,
        description="Telescope/instrument type"
    )

    num_pixels: Annotated[
        int,
        Field(
            default=150,
            ge=32,
            le=512,
            description="Number of pixels per side"
        )
    ]

    pixel_scale: Annotated[
        float,
        Field(
            default=0.05,
            ge=0.01,
            le=0.2,
            description="Pixel scale in arcseconds per pixel"
        )
    ]

    psf_fwhm: Annotated[
        float,
        Field(
            default=0.087,
            ge=0.01,
            le=0.5,
            description="PSF FWHM in arcseconds"
        )
    ]

    psf_type: Literal["GAUSSIAN", "PIXEL"] = Field(
        default="GAUSSIAN",
        description="Type of point spread function"
    )

    background_rms: Annotated[
        float,
        Field(
            default=0.01,
            ge=0.001,
            le=0.1,
            description="Background noise RMS"
        )
    ]

    exposure_time_log_min: Annotated[
        float,
        Field(
            default=3.0,
            ge=1.0,
            le=5.0,
            description="Log10 of minimum exposure time"
        )
    ]

    exposure_time_log_max: Annotated[
        float,
        Field(
            default=3.5,
            ge=1.0,
            le=5.0,
            description="Log10 of maximum exposure time"
        )
    ]

    @classmethod
    def for_model_type(cls, model_type: ModelType) -> InstrumentConfig:
        """Create instrument config appropriate for a model type."""
        if model_type == ModelType.MODEL_I:
            return cls(
                instrument_type=InstrumentType.GENERIC,
                num_pixels=150,
                pixel_scale=0.05,
                psf_fwhm=0.087,
                psf_type="GAUSSIAN",
            )
        elif model_type == ModelType.MODEL_II:
            return cls(
                instrument_type=InstrumentType.EUCLID,
                num_pixels=64,
                pixel_scale=0.1,  # Euclid VIS
                psf_fwhm=0.15,
            )
        elif model_type == ModelType.MODEL_III:
            return cls(
                instrument_type=InstrumentType.HST,
                num_pixels=64,
                pixel_scale=0.05,  # HST WFC3
                psf_fwhm=0.08,
            )
        else:
            # Model IV - Euclid defaults
            return cls(
                instrument_type=InstrumentType.EUCLID,
                num_pixels=64,
                pixel_scale=0.1,
            )


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "model_type": "model_i",
                    "num_images": 10,
                    "random_seed": 42,
                }
            ]
        }
    )

    model_type: ModelType = Field(
        default=ModelType.MODEL_I,
        description="DeepLenseSim model configuration to use"
    )

    num_images: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            le=1000,
            description="Number of images to generate"
        )
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Random seed for reproducibility"
        )
    ]

    cosmology: CosmologicalParameters = Field(
        default_factory=CosmologicalParameters,
        description="Cosmological parameters"
    )

    substructure: SubstructureParameters = Field(
        default_factory=SubstructureParameters,
        description="Dark matter substructure parameters"
    )

    source_light: SourceLightParameters = Field(
        default_factory=SourceLightParameters,
        description="Source galaxy light profile parameters"
    )

    main_halo: MainHaloParameters = Field(
        default_factory=MainHaloParameters,
        description="Main lens halo parameters"
    )

    instrument: InstrumentConfig | None = Field(
        default=None,
        description="Instrument configuration (auto-set from model_type if None)"
    )

    @model_validator(mode="after")
    def set_instrument_defaults(self) -> SimulationConfig:
        """Set instrument config based on model type if not provided."""
        if self.instrument is None:
            object.__setattr__(
                self,
                "instrument",
                InstrumentConfig.for_model_type(self.model_type)
            )
        return self

    @computed_field
    @property
    def expected_resolution(self) -> tuple[int, int]:
        """Expected output image resolution."""
        size = self.model_type.resolution
        return (size, size)


class SimulationRequest(BaseModel):
    """A request to run a simulation, possibly from natural language."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "natural_language_prompt": "Generate 5 CDM lens images using Model I",
                    "config": None,
                },
                {
                    "natural_language_prompt": None,
                    "config": {"model_type": "model_i", "num_images": 5},
                }
            ]
        }
    )

    natural_language_prompt: str | None = Field(
        default=None,
        description="Natural language description of desired simulation"
    )

    config: SimulationConfig | None = Field(
        default=None,
        description="Structured simulation configuration"
    )

    clarification_responses: dict[str, str] = Field(
        default_factory=dict,
        description="User responses to clarification questions"
    )

    @model_validator(mode="after")
    def require_prompt_or_config(self) -> SimulationRequest:
        """Ensure at least one input method is provided."""
        if self.natural_language_prompt is None and self.config is None:
            raise ValueError(
                "Either natural_language_prompt or config must be provided"
            )
        return self


class ClarificationQuestion(BaseModel):
    """A question to ask the user for clarification."""

    question_id: str = Field(description="Unique identifier for this question")
    question_text: str = Field(description="The question to ask the user")
    category: Literal["model", "substructure", "cosmology", "instrument", "quantity"] = Field(
        description="Category of the question"
    )
    options: list[str] | None = Field(
        default=None,
        description="Predefined options, if applicable"
    )
    default_value: str | None = Field(
        default=None,
        description="Suggested default value"
    )
    required: bool = Field(
        default=False,
        description="Whether this question must be answered"
    )
    scientific_context: str | None = Field(
        default=None,
        description="Scientific context to help user understand the question"
    )


class ClarificationResponse(BaseModel):
    """Response containing clarification questions for the user."""

    needs_clarification: bool = Field(
        description="Whether clarification is needed before proceeding"
    )

    questions: list[ClarificationQuestion] = Field(
        default_factory=list,
        description="Questions to ask the user"
    )

    partial_config: SimulationConfig | None = Field(
        default=None,
        description="Partially filled configuration based on available info"
    )

    confidence_score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Confidence in interpretation")
    ] = 0.5

    interpretation_summary: str = Field(
        default="",
        description="Summary of how the request was interpreted"
    )


class ImageData(BaseModel):
    """Container for a single generated image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_array: Any = Field(
        description="The image as a numpy array",
        exclude=True,
    )

    base64_png: str | None = Field(
        default=None,
        description="Base64-encoded PNG representation"
    )

    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    channels: int = Field(default=1, description="Number of color channels")

    min_value: float = Field(description="Minimum pixel value")
    max_value: float = Field(description="Maximum pixel value")
    mean_value: float = Field(description="Mean pixel value")

    @classmethod
    def from_numpy(cls, arr: np.ndarray, encode_png: bool = True) -> ImageData:
        """Create ImageData from a numpy array."""
        base64_str = None

        if encode_png:
            # Normalize to 0-255 range for PNG encoding
            arr_norm = arr.copy()
            arr_min, arr_max = arr_norm.min(), arr_norm.max()
            if arr_max > arr_min:
                arr_norm = (arr_norm - arr_min) / (arr_max - arr_min) * 255
            arr_norm = arr_norm.astype(np.uint8)

            # Handle different array shapes
            from PIL import Image

            if arr_norm.ndim == 2:
                img = Image.fromarray(arr_norm, mode='L')
            elif arr_norm.ndim == 3 and arr_norm.shape[2] == 3:
                img = Image.fromarray(arr_norm, mode='RGB')
            else:
                img = Image.fromarray(arr_norm[:, :, 0], mode='L')

            buffer = BytesIO()
            img.save(buffer, format='PNG')
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        channels = 1 if arr.ndim == 2 else arr.shape[2]

        return cls(
            image_array=arr,
            base64_png=base64_str,
            width=arr.shape[1],
            height=arr.shape[0],
            channels=channels,
            min_value=float(arr.min()),
            max_value=float(arr.max()),
            mean_value=float(arr.mean()),
        )


class SimulationMetadata(BaseModel):
    """Metadata for a completed simulation."""

    simulation_id: str = Field(description="Unique identifier for this simulation")

    config: SimulationConfig = Field(description="Configuration used for simulation")

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the simulation was run"
    )

    duration_seconds: float = Field(
        ge=0,
        description="Time taken to run the simulation"
    )

    deeplense_version: str = Field(
        default="unknown",
        description="Version of DeepLenseSim used"
    )

    random_state_used: int | None = Field(
        default=None,
        description="Actual random seed used"
    )

    # Physical parameters computed during simulation
    einstein_radius: float | None = Field(
        default=None,
        description="Einstein radius in arcseconds"
    )

    effective_num_subhalos: int | None = Field(
        default=None,
        description="Actual number of subhalos generated (CDM)"
    )


class SimulationOutput(BaseModel):
    """Complete output from a simulation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(description="Whether simulation completed successfully")

    images: list[ImageData] = Field(
        default_factory=list,
        description="Generated lens images"
    )

    metadata: SimulationMetadata | None = Field(
        default=None,
        description="Simulation metadata"
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if simulation failed"
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings generated during simulation"
    )

    @computed_field
    @property
    def num_images_generated(self) -> int:
        """Number of images successfully generated."""
        return len(self.images)


class AgentState(BaseModel):
    """State maintained by the agent across interactions."""

    current_request: SimulationRequest | None = Field(
        default=None,
        description="The current simulation request being processed"
    )

    clarification_history: list[ClarificationResponse] = Field(
        default_factory=list,
        description="History of clarification exchanges"
    )

    completed_simulations: list[SimulationOutput] = Field(
        default_factory=list,
        description="History of completed simulations"
    )

    user_preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="Learned user preferences"
    )
