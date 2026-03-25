"""
DeepLenseSim simulator wrapper.

This module provides a Pythonic interface to the DeepLenseSim simulation
pipeline, handling Model I and Model II/III configurations.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

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
    SourceLightParameters,
    SubstructureParameters,
)

if TYPE_CHECKING:
    from deeplense.lens import DeepLens

logger = logging.getLogger(__name__)


class DeepLenseSimNotAvailable(Exception):
    """Raised when DeepLenseSim dependencies are not installed."""

    pass


@dataclass
class SimulatorState:
    """Internal state for a single simulation run."""

    lens: Any = None
    einstein_radius: float | None = None
    num_subhalos: int | None = None
    exposure_time: float | None = None
    warnings: list[str] = field(default_factory=list)


class DeepLenseSimulator:
    """
    Wrapper around the DeepLenseSim library.

    This class provides a clean interface for generating gravitational
    lensing simulations with full parameter control.

    Example:
        >>> simulator = DeepLenseSimulator()
        >>> config = SimulationConfig(
        ...     model_type=ModelType.MODEL_I,
        ...     num_images=5,
        ...     substructure=SubstructureParameters(substructure_type=DarkMatterType.CDM)
        ... )
        >>> output = simulator.run_simulation(config)
        >>> print(f"Generated {output.num_images_generated} images")
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize the simulator.

        Args:
            mock_mode: If True, generate mock data instead of running
                       real simulations. Useful for testing.
        """
        self._mock_mode = mock_mode
        self._deeplense_available = False
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if DeepLenseSim dependencies are available."""
        if self._mock_mode:
            logger.info("Running in mock mode - skipping dependency check")
            return

        try:
            from deeplense.lens import DeepLens

            self._deeplense_available = True
            logger.info("DeepLenseSim dependencies available")
        except ImportError as e:
            logger.warning(f"DeepLenseSim not available: {e}")
            logger.info("Falling back to mock mode")
            self._mock_mode = True

    @property
    def is_available(self) -> bool:
        """Check if real simulation is available."""
        return self._deeplense_available and not self._mock_mode

    def run_simulation(self, config: SimulationConfig) -> SimulationOutput:
        """
        Run a simulation with the given configuration.

        Args:
            config: Complete simulation configuration.

        Returns:
            SimulationOutput containing generated images and metadata.
        """
        start_time = time.time()

        # Set random seed if specified
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            actual_seed = config.random_seed
        else:
            actual_seed = np.random.randint(0, 2**31)
            np.random.seed(actual_seed)

        try:
            if self._mock_mode:
                images, state = self._run_mock_simulation(config)
            else:
                images, state = self._run_real_simulation(config)

            duration = time.time() - start_time

            metadata = SimulationMetadata(
                simulation_id=str(uuid.uuid4()),
                config=config,
                duration_seconds=duration,
                random_state_used=actual_seed,
                einstein_radius=state.einstein_radius,
                effective_num_subhalos=state.num_subhalos,
            )

            return SimulationOutput(
                success=True,
                images=[
                    ImageData.from_numpy(img, encode_png=True) for img in images
                ],
                metadata=metadata,
                warnings=state.warnings,
            )

        except Exception as e:
            logger.exception("Simulation failed")
            return SimulationOutput(
                success=False,
                error_message=str(e),
            )

    def _run_real_simulation(
        self, config: SimulationConfig
    ) -> tuple[list[np.ndarray], SimulatorState]:
        """Run actual DeepLenseSim simulation."""
        from deeplense.lens import DeepLens

        images: list[np.ndarray] = []
        state = SimulatorState()

        for i in range(config.num_images):
            logger.debug(f"Generating image {i + 1}/{config.num_images}")

            # Create lens instance
            lens = self._create_lens(config)
            state.lens = lens

            # Configure main halo
            self._configure_main_halo(lens, config.main_halo)

            # Configure substructure
            self._configure_substructure(lens, config.substructure, state)

            # Configure instrument (for Model II/III)
            if config.model_type in (ModelType.MODEL_II, ModelType.MODEL_III):
                self._configure_instrument(lens, config.instrument)

            # Configure source light
            self._configure_source_light(lens, config)

            # Run simulation
            image = self._generate_image(lens, config.model_type)
            images.append(image)

            # Store computed values
            if hasattr(lens, "theta_E"):
                state.einstein_radius = lens.theta_E

        return images, state

    def _create_lens(self, config: SimulationConfig) -> DeepLens:
        """Create and initialize a DeepLens instance."""
        from deeplense.lens import DeepLens

        kwargs: dict[str, Any] = {
            "H0": config.cosmology.H0,
            "Om0": config.cosmology.Om0,
            "Ob0": config.cosmology.Ob0,
            "z_halo": config.cosmology.z_lens,
            "z_gal": config.cosmology.z_source,
        }

        # Add axion mass if using axion substructure
        if config.substructure.substructure_type == DarkMatterType.AXION:
            if config.substructure.axion_mass is not None:
                kwargs["axion_mass"] = config.substructure.axion_mass

        return DeepLens(**kwargs)

    def _configure_main_halo(
        self, lens: DeepLens, halo_params: MainHaloParameters
    ) -> None:
        """Configure the main lens halo."""
        lens.make_single_halo(halo_params.halo_mass)

        # Update ellipticity if supported
        if hasattr(lens, "kwargs_lens_model"):
            for kw in lens.kwargs_lens_model:
                if "e1" in kw:
                    kw["e1"] = halo_params.ellipticity_e1
                    kw["e2"] = halo_params.ellipticity_e2

        # Update shear if present
        if hasattr(lens, "kwargs_shear"):
            lens.kwargs_shear["gamma1"] = halo_params.external_shear_gamma1
            lens.kwargs_shear["gamma2"] = halo_params.external_shear_gamma2

    def _configure_substructure(
        self,
        lens: DeepLens,
        sub_params: SubstructureParameters,
        state: SimulatorState,
    ) -> None:
        """Configure dark matter substructure."""
        if sub_params.substructure_type == DarkMatterType.NO_SUBSTRUCTURE:
            lens.make_no_sub()
            state.num_subhalos = 0

        elif sub_params.substructure_type == DarkMatterType.CDM:
            lens.make_old_cdm()
            # The actual number of subhalos is drawn from Poisson distribution
            # We estimate it from the mean
            state.num_subhalos = sub_params.n_sub_mean

        elif sub_params.substructure_type == DarkMatterType.AXION:
            lens.make_vortex(
                sub_params.vortex_mass, res=sub_params.vortex_resolution
            )
            state.num_subhalos = None  # Vortex is continuous, not discrete

    def _configure_instrument(
        self, lens: DeepLens, instrument: InstrumentConfig | None
    ) -> None:
        """Configure telescope/instrument settings."""
        if instrument is None:
            return

        if instrument.instrument_type == InstrumentType.EUCLID:
            lens.set_instrument("Euclid")
        elif instrument.instrument_type == InstrumentType.HST:
            lens.set_instrument("hst")

    def _configure_source_light(
        self, lens: DeepLens, config: SimulationConfig
    ) -> None:
        """Configure source galaxy light profile."""
        # Choose appropriate method based on model type
        if config.model_type == ModelType.MODEL_I:
            lens.make_source_light()
        else:
            lens.make_source_light_mag()

    def _generate_image(
        self, lens: DeepLens, model_type: ModelType
    ) -> np.ndarray:
        """Generate the simulated image."""
        if model_type == ModelType.MODEL_I:
            lens.simple_sim()
        else:
            lens.simple_sim_2()

        return lens.image_real

    def _run_mock_simulation(
        self, config: SimulationConfig
    ) -> tuple[list[np.ndarray], SimulatorState]:
        """Generate mock simulation data for testing."""
        state = SimulatorState()
        images: list[np.ndarray] = []

        resolution = config.model_type.resolution
        channels = config.model_type.num_channels

        for i in range(config.num_images):
            # Generate a realistic-looking mock lens image
            image = self._generate_mock_lens_image(
                resolution=resolution,
                channels=channels,
                substructure_type=config.substructure.substructure_type,
            )
            images.append(image)

        # Mock state values
        state.einstein_radius = 1.5  # arcseconds
        state.num_subhalos = (
            config.substructure.n_sub_mean
            if config.substructure.substructure_type == DarkMatterType.CDM
            else None
        )
        state.warnings.append("Running in mock mode - images are simulated")

        return images, state

    def _generate_mock_lens_image(
        self,
        resolution: int,
        channels: int,
        substructure_type: DarkMatterType,
    ) -> np.ndarray:
        """Generate a mock gravitational lens image."""
        # Create coordinate grid
        x = np.linspace(-3, 3, resolution)
        y = np.linspace(-3, 3, resolution)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        # Generate Einstein ring-like structure
        einstein_radius = 1.5
        ring_width = 0.3
        ring = np.exp(-0.5 * ((R - einstein_radius) / ring_width) ** 2)

        # Add some ellipticity
        theta = np.random.uniform(0, np.pi)
        e = 0.2 + np.random.uniform(-0.1, 0.1)
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
        R_ell = np.sqrt(X_rot**2 + (Y_rot / (1 - e)) ** 2)
        ring_ell = np.exp(-0.5 * ((R_ell - einstein_radius) / ring_width) ** 2)

        # Add arcs (lensed source)
        num_arcs = np.random.randint(2, 5)
        arcs = np.zeros_like(R)
        for _ in range(num_arcs):
            arc_theta = np.random.uniform(0, 2 * np.pi)
            arc_r = einstein_radius + np.random.uniform(-0.2, 0.2)
            arc_x = arc_r * np.cos(arc_theta)
            arc_y = arc_r * np.sin(arc_theta)
            arc_dist = np.sqrt((X - arc_x) ** 2 + (Y - arc_y) ** 2)
            arc_intensity = np.random.uniform(0.5, 1.0)
            arcs += arc_intensity * np.exp(-0.5 * (arc_dist / 0.15) ** 2)

        # Combine components
        image = 0.3 * ring_ell + 0.7 * arcs

        # Add substructure effects
        if substructure_type == DarkMatterType.CDM:
            # Add point-like substructure perturbations
            num_subs = np.random.randint(10, 30)
            for _ in range(num_subs):
                sub_r = np.random.uniform(0.5, 2.0)
                sub_theta = np.random.uniform(0, 2 * np.pi)
                sub_x = sub_r * np.cos(sub_theta)
                sub_y = sub_r * np.sin(sub_theta)
                sub_dist = np.sqrt((X - sub_x) ** 2 + (Y - sub_y) ** 2)
                sub_effect = 0.05 * np.random.uniform(0.5, 1.5)
                image += sub_effect * np.exp(-0.5 * (sub_dist / 0.08) ** 2)

        elif substructure_type == DarkMatterType.AXION:
            # Add vortex-like patterns
            vortex_freq = np.random.uniform(2, 4)
            theta_grid = np.arctan2(Y, X)
            vortex = 0.1 * np.sin(vortex_freq * theta_grid) * ring_ell
            image += vortex

        # Add galaxy-like central lens light
        central_light = 0.2 * np.exp(-0.5 * (R / 0.5) ** 2)
        image += central_light

        # Add realistic noise
        noise_level = 0.02
        image += np.random.normal(0, noise_level, image.shape)

        # Normalize
        image = np.clip(image, 0, None)
        image /= image.max() + 1e-10

        # Add Poisson-like noise
        image = image + 0.01 * np.sqrt(np.abs(image)) * np.random.randn(
            *image.shape
        )
        image = np.clip(image, 0, 1)

        # Handle multi-channel output
        if channels > 1:
            image = np.stack([image] * channels, axis=-1)

        return image.astype(np.float32)


def create_simulator(mock_mode: bool | None = None) -> DeepLenseSimulator:
    """
    Factory function to create a simulator instance.

    Args:
        mock_mode: If True, always use mock mode. If False, always try real mode.
                   If None (default), auto-detect based on dependency availability.

    Returns:
        Configured DeepLenseSimulator instance.
    """
    if mock_mode is None:
        # Auto-detect
        simulator = DeepLenseSimulator(mock_mode=False)
        return simulator
    else:
        return DeepLenseSimulator(mock_mode=mock_mode)
