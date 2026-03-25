"""
Scientific presets based on DeepLense research papers.

References:
- Alexander et al. (2020) "Deep Learning the Morphology of Dark Matter Substructure"
  arXiv:1909.07346, Astrophys. J. 893 (2020) 15
- Alexander et al. (2021) "Decoding Dark Matter Substructure without Supervision"
  arXiv:2008.12731

These presets encode the exact simulation parameters used in the published research
for reproducibility and scientific accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Tuple


class PresetType(str, Enum):
    """Scientific preset configurations from published research."""

    ALEXANDER_2020 = "alexander_2020"  # arXiv:1909.07346
    ALEXANDER_2021 = "alexander_2021"  # arXiv:2008.12731
    DEEPLENSE_DEFAULT = "deeplense_default"
    HIGH_RESOLUTION = "high_resolution"
    EUCLID_SURVEY = "euclid_survey"
    HST_OBSERVATION = "hst_observation"


@dataclass(frozen=True)
class CosmologyPreset:
    """
    Cosmological parameters from research papers.

    From arXiv:1909.07346 and arXiv:2008.12731:
    - Lens redshift: z_lens = 0.5 (fixed) or uniform [0.4, 0.6]
    - Source redshift: z_source = 1.0 (fixed) or uniform [0.8, 1.2]
    """

    z_lens: float = 0.5
    z_lens_min: float = 0.4
    z_lens_max: float = 0.6
    z_source: float = 1.0
    z_source_min: float = 0.8
    z_source_max: float = 1.2
    H0: float = 70.0
    Om0: float = 0.3
    Ob0: float = 0.05

    @property
    def z_lens_range(self) -> Tuple[float, float]:
        return (self.z_lens_min, self.z_lens_max)

    @property
    def z_source_range(self) -> Tuple[float, float]:
        return (self.z_source_min, self.z_source_max)


@dataclass(frozen=True)
class HaloPreset:
    """
    Main halo parameters from research papers.

    From arXiv:2008.12731:
    - Total halo mass: M_TOT = 10^12 M_sun (fixed)
    - Einstein radius: θ_E = 1.2 arcsec
    - Axis ratio: e in uniform [0.5, 1.0]
    """

    mass: float = 1e12  # Solar masses
    einstein_radius: float = 1.2  # arcseconds
    axis_ratio_min: float = 0.5
    axis_ratio_max: float = 1.0
    ellipticity_e1: float = 0.1
    ellipticity_e2: float = 0.0

    @property
    def axis_ratio_range(self) -> Tuple[float, float]:
        return (self.axis_ratio_min, self.axis_ratio_max)


@dataclass(frozen=True)
class CDMSubstructurePreset:
    """
    Cold Dark Matter subhalo parameters from research papers.

    From arXiv:2008.12731:
    - Subhalo mass function power law index: β = -1.9
    - Subhalo mass range: 10^6 to 10^10 M_sun
    - Number of subhalos: Poisson distribution with μ = 25
    - Fraction of mass in substructure: O(1%)
    - Substructure mass: 0.01 × M_Halo (1% of halo mass)

    From arXiv:1909.07346:
    - Number of spherical subhalos N: 25 (fixed) or Poisson μ=25
    - Detection threshold: ~0.3% of halo mass minimum
    """

    m_sub_min: float = 1e6  # Minimum subhalo mass (M_sun)
    m_sub_max: float = 1e10  # Maximum subhalo mass (M_sun)
    n_sub_mean: int = 25  # Mean number of subhalos (Poisson μ)
    power_law_slope: float = -1.9  # Mass function slope β
    mass_fraction: float = 0.01  # 1% of halo mass
    detection_threshold: float = 0.003  # ~0.3% minimum for detection

    # WIMP-specific parameters
    wimp_mass_gev: float = 100.0  # ~100 GeV typical WIMP mass


@dataclass(frozen=True)
class AxionVortexPreset:
    """
    Axion/Vortex (ULDM) parameters from research papers.

    From arXiv:2008.12731:
    - Position (θx, θy): Normal distribution [0.0, 0.5]
    - Vortex length: uniform [0.5, 2.0]
    - Mass fraction: uniform [3.0, 5.5]%
    - Expected vortices in M31: 340 (for m = 10^-23 eV)
    - Expected vortices in typical halo: 10^23 (for m = 1 eV)

    From arXiv:1909.07346:
    - Example axion mass: m = 10^-23 eV (yielding ~340 vortices in M31)
    - Alternative mass: m = 1 eV (yielding N = 10^23 vortices)
    """

    # Axion mass (eV) - key parameter controlling de Broglie wavelength
    axion_mass_default: float = 1e-23  # eV
    axion_mass_min: float = 1e-25  # eV
    axion_mass_max: float = 1e-20  # eV

    # Vortex spatial parameters
    position_mean: float = 0.0
    position_std: float = 0.5
    length_min: float = 0.5
    length_max: float = 2.0

    # Vortex mass parameters
    mass_fraction_min: float = 0.03  # 3% minimum
    mass_fraction_max: float = 0.055  # 5.5% maximum
    vortex_mass_default: float = 3e10  # M_sun

    # Expected vortices (for reference calculations)
    vortices_m31_1e23ev: int = 340  # Expected in M31 for m=10^-23 eV

    @staticmethod
    def de_broglie_wavelength_kpc(axion_mass_ev: float) -> float:
        """
        Calculate de Broglie wavelength in kpc.

        λ_dB ≈ 0.6 kpc × (10^-22 eV / m_a)
        """
        return 0.6 * (1e-22 / axion_mass_ev)

    @staticmethod
    def expected_vortices(axion_mass_ev: float, halo_mass_msun: float = 1e12) -> int:
        """
        Estimate number of expected vortices in a halo.

        N_vortex ∝ (M_halo / m_axion) for ULDM condensates.
        Normalized to 340 vortices for M31 at m = 10^-23 eV.
        """
        # M31 mass ~ 10^12 M_sun
        m31_mass = 1e12
        reference_vortices = 340
        reference_mass = 1e-23

        # Scale by halo mass and inversely by axion mass
        return int(
            reference_vortices
            * (halo_mass_msun / m31_mass)
            * (reference_mass / axion_mass_ev)
        )


@dataclass(frozen=True)
class SourceLightPreset:
    """
    Source galaxy light profile parameters from research papers.

    From arXiv:2008.12731:
    - Sersic index n: 1.5 (fixed)
    - Axis ratio e: uniform [0.4, 1.0]
    - Effective radius R: uniform [0.25, 1.0] arcsec

    From arXiv:1909.07346:
    - Sersic index n: 1.5
    - Axis ratio e: uniform [0.7, 1.0]
    - Lensing galaxy Sersic index: 2.5
    """

    # Source galaxy (lensed background galaxy)
    sersic_index: float = 1.5  # Sersic index n
    axis_ratio_min: float = 0.4
    axis_ratio_max: float = 1.0
    effective_radius_min: float = 0.25  # arcsec
    effective_radius_max: float = 1.0  # arcsec

    # Lens galaxy (foreground) - from 1909.07346
    lens_sersic_index: float = 2.5
    lens_axis_ratio_min: float = 0.5
    lens_axis_ratio_max: float = 1.0
    lens_intensity: float = 1.2
    lens_effective_radius_min: float = 0.5  # arcsec
    lens_effective_radius_max: float = 2.0  # arcsec

    @property
    def axis_ratio_range(self) -> Tuple[float, float]:
        return (self.axis_ratio_min, self.axis_ratio_max)

    @property
    def effective_radius_range(self) -> Tuple[float, float]:
        return (self.effective_radius_min, self.effective_radius_max)


@dataclass(frozen=True)
class ImagePreset:
    """
    Image simulation parameters from research papers.

    From arXiv:2008.12731:
    - Dimensions: 150 × 150 pixels
    - Pixel scale: 0.5 arcsec/pixel
    - Maximum SNR: ~20
    - PSF (Airy disk): first zero-crossing at σ_psf ≲ 1 arcsec

    From arXiv:1909.07346:
    - PSF: Airy disk with σ_psf ≲ arcsec (Hubble/LSST-like)
    - Sub-arcsecond resolution
    """

    num_pixels: int = 150
    pixel_scale: float = 0.05  # arcsec/pixel (0.5" in papers may be typo)
    snr_max: float = 20.0

    # PSF parameters
    psf_type: str = "GAUSSIAN"  # or "AIRY"
    psf_fwhm: float = 0.087  # arcsec (for Gaussian)
    psf_airy_sigma: float = 1.0  # arcsec (first zero-crossing)

    # Noise parameters
    background_rms: float = 0.01
    exposure_time_log_min: float = 3.0  # log10(t_exp)
    exposure_time_log_max: float = 3.5  # log10(t_exp)


@dataclass(frozen=True)
class ExternalShearPreset:
    """
    External shear parameters from research papers.

    From arXiv:2008.12731 and arXiv:1909.07346:
    - Magnitude γ_ext: uniform [0.0, 0.3]
    """

    gamma_min: float = 0.0
    gamma_max: float = 0.3
    gamma1_default: float = 0.05
    gamma2_default: float = 0.0

    @property
    def gamma_range(self) -> Tuple[float, float]:
        return (self.gamma_min, self.gamma_max)


@dataclass(frozen=True)
class TrainingPreset:
    """
    Training dataset parameters from research papers.

    From arXiv:2008.12731:
    - Training samples: 25,000 per class
    - Validation samples: 2,500 per class
    - Batch size: 250
    - Epochs: 50 (supervised), 500 (unsupervised)
    - Initial learning rate: 1×10^-3
    - Learning rate decay: 1×10^-5 per epoch

    From arXiv:1909.07346:
    - Training images: 150,000 total
    - Validation images: 15,000
    - Batch size: 200
    - Maximum epochs: 20
    - Initial learning rate: 1×10^-4
    """

    # Dataset sizes
    train_per_class: int = 25000
    val_per_class: int = 2500
    test_per_class: int = 5000

    # Alternative total counts (from 1909.07346)
    train_total_alt: int = 150000
    val_total_alt: int = 15000

    # Training parameters
    batch_size: int = 250
    batch_size_alt: int = 200
    epochs_supervised: int = 50
    epochs_unsupervised: int = 500
    learning_rate: float = 1e-3
    learning_rate_alt: float = 1e-4
    lr_decay_per_epoch: float = 1e-5


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Expected classification performance from research papers.

    From arXiv:1909.07346:
    - No substructure AUC: 0.998
    - Spherical sub-halos AUC: 0.985
    - Vortices AUC: 0.968
    - Macro-averaged AUC: 0.969

    From arXiv:2008.12731:
    - ResNet-18 AUC: 0.99637
    - AlexNet AUC: 0.98931
    - AAE AUC: 0.93207
    - VAE AUC: 0.89910
    """

    # Per-class AUC from 1909.07346
    auc_no_sub: float = 0.998
    auc_cdm: float = 0.985
    auc_vortex: float = 0.968
    auc_macro_avg: float = 0.969

    # Architecture comparison from 2008.12731
    auc_resnet18: float = 0.99637
    auc_alexnet: float = 0.98931
    auc_aae: float = 0.93207
    auc_vae: float = 0.89910
    auc_dcae: float = 0.73034
    auc_rbm: float = 0.51054


@dataclass
class ScientificPreset:
    """
    Complete scientific preset combining all parameter categories.

    This provides a one-stop configuration matching published research.
    """

    name: str
    description: str
    reference: str  # arXiv reference

    cosmology: CosmologyPreset = field(default_factory=CosmologyPreset)
    halo: HaloPreset = field(default_factory=HaloPreset)
    cdm: CDMSubstructurePreset = field(default_factory=CDMSubstructurePreset)
    axion: AxionVortexPreset = field(default_factory=AxionVortexPreset)
    source: SourceLightPreset = field(default_factory=SourceLightPreset)
    image: ImagePreset = field(default_factory=ImagePreset)
    shear: ExternalShearPreset = field(default_factory=ExternalShearPreset)
    training: TrainingPreset = field(default_factory=TrainingPreset)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)


# Pre-defined scientific presets
PRESETS: dict[PresetType, ScientificPreset] = {
    PresetType.ALEXANDER_2020: ScientificPreset(
        name="Alexander et al. 2020",
        description="Parameters from 'Deep Learning the Morphology of Dark Matter Substructure'",
        reference="arXiv:1909.07346",
        image=ImagePreset(num_pixels=150, pixel_scale=0.05),
        training=TrainingPreset(
            train_total_alt=150000,
            val_total_alt=15000,
            batch_size=200,
            learning_rate=1e-4,
        ),
    ),
    PresetType.ALEXANDER_2021: ScientificPreset(
        name="Alexander et al. 2021",
        description="Parameters from 'Decoding Dark Matter Substructure without Supervision'",
        reference="arXiv:2008.12731",
        image=ImagePreset(num_pixels=150, pixel_scale=0.05, snr_max=20),
        cdm=CDMSubstructurePreset(power_law_slope=-1.9, n_sub_mean=25),
        training=TrainingPreset(
            train_per_class=25000,
            val_per_class=2500,
            batch_size=250,
            learning_rate=1e-3,
        ),
    ),
    PresetType.DEEPLENSE_DEFAULT: ScientificPreset(
        name="DeepLense Default",
        description="Default parameters for DeepLenseSim",
        reference="https://github.com/mwt5345/DeepLenseSim",
    ),
    PresetType.HIGH_RESOLUTION: ScientificPreset(
        name="High Resolution",
        description="Higher resolution simulation for detailed studies",
        reference="Custom",
        image=ImagePreset(num_pixels=256, pixel_scale=0.03),
    ),
    PresetType.EUCLID_SURVEY: ScientificPreset(
        name="Euclid Survey",
        description="Parameters matching Euclid VIS band observations",
        reference="Euclid Collaboration",
        image=ImagePreset(num_pixels=64, pixel_scale=0.1),
    ),
    PresetType.HST_OBSERVATION: ScientificPreset(
        name="HST Observation",
        description="Parameters matching Hubble Space Telescope observations",
        reference="HST",
        image=ImagePreset(num_pixels=64, pixel_scale=0.05, psf_fwhm=0.08),
    ),
}


def get_preset(preset_type: PresetType) -> ScientificPreset:
    """Get a scientific preset by type."""
    return PRESETS[preset_type]


def get_default_preset() -> ScientificPreset:
    """Get the default scientific preset (Alexander 2021)."""
    return PRESETS[PresetType.ALEXANDER_2021]
