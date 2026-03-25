"""
Pydantic AI Agent for DeepLense gravitational lensing simulations.

This module implements the core agentic workflow using Pydantic AI,
providing natural language interaction for simulation generation
with human-in-the-loop clarification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

from deeplense_agent.clarification import ClarificationEngine, NaturalLanguageParser
from deeplense_agent.models import (
    AgentState,
    ClarificationQuestion,
    ClarificationResponse,
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
from deeplense_agent.simulator import DeepLenseSimulator, create_simulator

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Dependencies (Injected Context)
# =============================================================================


@dataclass
class AgentDependencies:
    """
    Dependencies injected into the agent at runtime.

    This provides access to the simulator, clarification engine,
    and any callbacks for human-in-the-loop interaction.
    """

    simulator: DeepLenseSimulator = field(default_factory=create_simulator)
    clarification_engine: ClarificationEngine = field(
        default_factory=ClarificationEngine
    )
    state: AgentState = field(default_factory=AgentState)

    # Callback for human-in-the-loop interaction
    # Takes questions and returns answers
    human_callback: Callable[[list[ClarificationQuestion]], dict[str, str]] | None = (
        None
    )

    # Configuration
    auto_approve_high_confidence: bool = True
    confidence_threshold: float = 0.85
    max_clarification_rounds: int = 3


# =============================================================================
# Response Models
# =============================================================================


class SimulationPlanResponse(BaseModel):
    """Response when planning a simulation."""

    plan_summary: str
    estimated_images: int
    model_configuration: str
    substructure_type: str
    needs_clarification: bool
    clarification_questions: list[dict[str, Any]] = []
    ready_to_execute: bool = False


class SimulationResultResponse(BaseModel):
    """Response when returning simulation results."""

    success: bool
    message: str
    num_images_generated: int
    simulation_id: str | None = None
    duration_seconds: float | None = None
    warnings: list[str] = []
    # Image data is returned separately due to size


class ParameterSuggestionResponse(BaseModel):
    """Response when suggesting parameter values."""

    parameter_name: str
    suggested_value: Any
    alternatives: list[Any]
    scientific_rationale: str


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a scientific assistant specialized in gravitational lensing simulations using the DeepLenseSim framework. Your role is to help researchers generate realistic strong gravitational lensing images for dark matter studies.

## Your Capabilities

1. **Simulation Generation**: You can generate gravitational lensing images with various configurations:
   - Model I: 150x150 pixels, basic Gaussian PSF, good for general studies
   - Model II: 64x64 pixels, Euclid survey characteristics
   - Model III: 64x64 pixels, HST (Hubble) characteristics

2. **Dark Matter Substructure**: You support three types:
   - CDM (Cold Dark Matter): Point-mass subhalos with power-law mass function
   - Axion/Vortex: Ultralight dark matter with wave-like interference patterns
   - No substructure: Clean lens for comparison/baseline studies

3. **Parameter Configuration**: You can help users configure:
   - Cosmological parameters (H0, matter density, redshifts)
   - Halo properties (mass, ellipticity, shear)
   - Source light profiles (magnitude, Sersic index, size)
   - Instrument settings (PSF, pixel scale, noise)

## Interaction Guidelines

1. **Be Precise**: When users describe simulations, extract exact parameters.
2. **Clarify Ambiguity**: If a request is unclear, ask targeted questions with scientific context.
3. **Suggest Defaults**: Provide sensible defaults when parameters aren't specified.
4. **Explain Trade-offs**: Help users understand the implications of their choices.
5. **Scientific Accuracy**: Ensure all parameters are physically reasonable.

## Available Tools

Use the provided tools to:
- Parse natural language requests into simulation configurations
- Validate parameter combinations
- Run simulations
- Generate parameter suggestions
- Answer scientific questions about gravitational lensing

Always confirm the simulation plan before execution when significant resources are required (>10 images)."""


# =============================================================================
# Create the Pydantic AI Agent
# =============================================================================

from deeplense_agent.config import get_model_config, ModelProvider

# Get model configuration (defaults to Groq with the provided API key)
_config = get_model_config()

# Create model based on provider
if _config.provider in (ModelProvider.GROQ, ModelProvider.OPENAI):
    # Both use OpenAI-compatible API
    # Create AsyncOpenAI client with custom base_url if needed
    _client = AsyncOpenAI(
        api_key=_config.api_key,
        base_url=_config.base_url,
    )
    _provider = OpenAIProvider(openai_client=_client)
    _model = OpenAIModel(
        model_name=_config.model_name,
        provider=_provider,
    )
elif _config.provider == ModelProvider.GOOGLE:
    from pydantic_ai.models.gemini import GeminiModel
    import os
    os.environ["GEMINI_API_KEY"] = _config.api_key
    _model = GeminiModel(
        model_name=_config.model_name
    )
else:
    # Fallback to Anthropic string format
    _model = f"anthropic:{_config.model_name}"

# The main agent instance
deeplense_agent = Agent(
    _model,
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
)


# =============================================================================
# Tool Functions
# =============================================================================


@deeplense_agent.tool
async def parse_simulation_request(
    ctx: RunContext[AgentDependencies],
    natural_language_prompt: str,
) -> dict[str, Any]:
    """
    Parse a natural language prompt into simulation parameters.

    This tool extracts simulation configuration from user descriptions
    and identifies any parameters that need clarification.

    Args:
        natural_language_prompt: The user's description of desired simulation.

    Returns:
        Dictionary containing extracted parameters, confidence score,
        and any clarification questions needed.
    """
    engine = ctx.deps.clarification_engine
    response = engine.analyze_request(natural_language_prompt)

    # Store in state
    ctx.deps.state.current_request = SimulationRequest(
        natural_language_prompt=natural_language_prompt,
        config=response.partial_config,
    )
    ctx.deps.state.clarification_history.append(response)

    return {
        "extracted_config": response.partial_config.model_dump(),
        "confidence_score": response.confidence_score,
        "needs_clarification": response.needs_clarification,
        "interpretation_summary": response.interpretation_summary,
        "questions": [q.model_dump() for q in response.questions],
    }


@deeplense_agent.tool
async def get_clarification_questions(
    ctx: RunContext[AgentDependencies],
    parameters_to_clarify: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate clarification questions for missing or ambiguous parameters.

    Use this when the user's request is incomplete or unclear.

    Args:
        parameters_to_clarify: Optional list of specific parameters to ask about.
                              If None, generates questions for all missing params.

    Returns:
        List of clarification questions with scientific context.
    """
    engine = ctx.deps.clarification_engine

    # Get current state
    current_request = ctx.deps.state.current_request
    if current_request is None:
        return [
            {
                "question_id": "general",
                "question_text": "What kind of gravitational lensing simulation would you like to generate?",
                "category": "general",
                "scientific_context": "Please describe the dark matter substructure type, number of images, and any specific parameters.",
            }
        ]

    # Parse the current request again to get fresh questions
    prompt = current_request.natural_language_prompt or ""
    response = engine.analyze_request(
        prompt, current_request.clarification_responses
    )

    # Filter to requested parameters if specified
    if parameters_to_clarify:
        questions = [
            q for q in response.questions if q.question_id in parameters_to_clarify
        ]
    else:
        questions = response.questions

    return [q.model_dump() for q in questions]


@deeplense_agent.tool
async def apply_clarification_response(
    ctx: RunContext[AgentDependencies],
    question_id: str,
    user_response: str,
) -> dict[str, Any]:
    """
    Apply a user's response to a clarification question.

    This updates the simulation configuration based on the user's answer.

    Args:
        question_id: The ID of the question being answered.
        user_response: The user's response text.

    Returns:
        Updated configuration and remaining questions.
    """
    if ctx.deps.state.current_request is None:
        return {"error": "No active request to update"}

    # Store the response
    ctx.deps.state.current_request.clarification_responses[question_id] = (
        user_response
    )

    # Re-analyze with the new response
    engine = ctx.deps.clarification_engine
    prompt = ctx.deps.state.current_request.natural_language_prompt or ""
    response = engine.analyze_request(
        prompt, ctx.deps.state.current_request.clarification_responses
    )

    # Update the config in state
    ctx.deps.state.current_request.config = response.partial_config

    return {
        "updated_config": response.partial_config.model_dump(),
        "remaining_questions": len(response.questions),
        "ready_to_execute": not response.needs_clarification,
        "confidence_score": response.confidence_score,
    }


@deeplense_agent.tool
async def validate_simulation_config(
    ctx: RunContext[AgentDependencies],
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate a simulation configuration for physical consistency.

    Checks that all parameters are within valid ranges and
    physically meaningful.

    Args:
        config_dict: Configuration dictionary to validate.
                    If None, validates the current request config.

    Returns:
        Validation result with any warnings or errors.
    """
    try:
        if config_dict is not None:
            config = SimulationConfig(**config_dict)
        elif ctx.deps.state.current_request and ctx.deps.state.current_request.config:
            config = ctx.deps.state.current_request.config
        else:
            return {"valid": False, "error": "No configuration to validate"}

        warnings: list[str] = []
        suggestions: list[str] = []

        # Check redshift ordering
        if config.cosmology.z_source <= config.cosmology.z_lens:
            return {
                "valid": False,
                "error": "Source must be behind lens (z_source > z_lens)",
            }

        # Check for extreme parameters
        if config.main_halo.halo_mass > 1e14:
            warnings.append(
                "Halo mass is cluster-scale (>10^14 M_sun). "
                "Consider if this is intended for galaxy-scale lensing."
            )

        if config.cosmology.z_lens > 1.5:
            warnings.append(
                "High lens redshift may reduce image quality and "
                "is less common in observed systems."
            )

        # Axion-specific checks
        if config.substructure.substructure_type == DarkMatterType.AXION:
            if config.substructure.axion_mass:
                if config.substructure.axion_mass < 1e-24:
                    warnings.append(
                        "Very low axion mass will produce large-scale structure "
                        "that may dominate the image."
                    )
                if config.substructure.axion_mass > 1e-21:
                    warnings.append(
                        "High axion mass produces small-scale structure "
                        "that may be below resolution limits."
                    )

        # Resolution checks
        if config.model_type == ModelType.MODEL_I and config.num_images > 100:
            suggestions.append(
                "Model I generates higher-resolution images. Consider Model II/III "
                "for faster generation of large datasets."
            )

        return {
            "valid": True,
            "config": config.model_dump(),
            "warnings": warnings,
            "suggestions": suggestions,
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


@deeplense_agent.tool
async def run_simulation(
    ctx: RunContext[AgentDependencies],
    config_dict: dict[str, Any] | None = None,
    confirm_execution: bool = True,
) -> dict[str, Any]:
    """
    Execute the gravitational lensing simulation.

    This runs the DeepLenseSim pipeline with the specified configuration
    and returns the generated images with metadata.

    Args:
        config_dict: Configuration dictionary. If None, uses current request.
        confirm_execution: If True, requires explicit confirmation for >10 images.

    Returns:
        Simulation results including image data and metadata.
    """
    try:
        # Get configuration
        if config_dict is not None:
            config = SimulationConfig(**config_dict)
        elif ctx.deps.state.current_request and ctx.deps.state.current_request.config:
            config = ctx.deps.state.current_request.config
        else:
            return {"success": False, "error": "No configuration provided"}

        # Check for confirmation on large jobs
        if confirm_execution and config.num_images > 10:
            return {
                "success": False,
                "needs_confirmation": True,
                "message": f"This will generate {config.num_images} images. "
                "Please confirm by calling with confirm_execution=False "
                "or reduce the number of images.",
                "config": config.model_dump(),
            }

        # Run the simulation
        simulator = ctx.deps.simulator
        output = simulator.run_simulation(config)

        # Store in history
        ctx.deps.state.completed_simulations.append(output)

        if output.success:
            return {
                "success": True,
                "num_images_generated": output.num_images_generated,
                "simulation_id": (
                    output.metadata.simulation_id if output.metadata else None
                ),
                "duration_seconds": (
                    output.metadata.duration_seconds if output.metadata else None
                ),
                "warnings": output.warnings,
                "images": [
                    {
                        "index": i,
                        "width": img.width,
                        "height": img.height,
                        "channels": img.channels,
                        "min_value": img.min_value,
                        "max_value": img.max_value,
                        "mean_value": img.mean_value,
                        "base64_png": img.base64_png,
                    }
                    for i, img in enumerate(output.images)
                ],
            }
        else:
            return {
                "success": False,
                "error": output.error_message,
            }

    except Exception as e:
        logger.exception("Simulation failed")
        return {"success": False, "error": str(e)}


@deeplense_agent.tool
async def get_parameter_suggestions(
    ctx: RunContext[AgentDependencies],
    parameter_name: str,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Get suggested values for a simulation parameter.

    Provides scientifically-motivated suggestions with rationale.

    Args:
        parameter_name: Name of the parameter to get suggestions for.
        context: Optional context about the user's goals.

    Returns:
        Suggested values with scientific rationale.
    """
    suggestions = {
        "model_type": {
            "suggested": "model_i",
            "alternatives": ["model_ii", "model_iii"],
            "rationale": "Model I provides highest resolution (150x150) with basic "
            "Gaussian PSF, suitable for detailed studies. Model II/III "
            "emulate specific survey characteristics.",
        },
        "substructure_type": {
            "suggested": "cdm",
            "alternatives": ["axion", "no_substructure"],
            "rationale": "CDM (Cold Dark Matter) is the standard model prediction. "
            "Axion/vortex models test ultralight dark matter hypotheses. "
            "No substructure provides baseline comparison.",
        },
        "halo_mass": {
            "suggested": 1e12,
            "alternatives": [1e11, 1e13],
            "rationale": "10^12 solar masses is typical for galaxy-scale strong "
            "lensing systems. Lower masses produce smaller Einstein radii, "
            "higher masses produce larger, more dramatic arcs.",
        },
        "z_lens": {
            "suggested": 0.5,
            "alternatives": [0.3, 0.8],
            "rationale": "z_lens=0.5 is typical for observed strong lens systems. "
            "Lower redshifts are easier to observe, higher redshifts "
            "probe larger volumes but with lower signal-to-noise.",
        },
        "z_source": {
            "suggested": 1.0,
            "alternatives": [0.8, 1.5, 2.0],
            "rationale": "z_source=1.0 provides good lensing geometry with typical "
            "lenses at z~0.5. Higher source redshifts increase magnification "
            "but sources become fainter.",
        },
        "axion_mass": {
            "suggested": 1e-23,
            "alternatives": [1e-24, 1e-22],
            "rationale": "10^-23 eV produces de Broglie wavelength ~0.6 kpc, "
            "creating observable substructure. Lower masses produce larger "
            "structures, higher masses produce finer detail.",
        },
        "num_images": {
            "suggested": 10,
            "alternatives": [1, 100, 1000],
            "rationale": "10 images provides quick visual inspection. 100-1000 images "
            "are typical for ML training. Single images for detailed analysis.",
        },
    }

    if parameter_name in suggestions:
        return {
            "parameter": parameter_name,
            **suggestions[parameter_name],
        }
    else:
        return {
            "parameter": parameter_name,
            "error": f"No suggestions available for '{parameter_name}'",
            "available_parameters": list(suggestions.keys()),
        }


@deeplense_agent.tool
async def explain_simulation_physics(
    ctx: RunContext[AgentDependencies],
    topic: str,
) -> str:
    """
    Explain the physics behind a simulation concept.

    Use this to help users understand gravitational lensing concepts.

    Args:
        topic: The physics topic to explain.

    Returns:
        Educational explanation of the topic.
    """
    explanations = {
        "einstein_radius": """
The Einstein radius (θ_E) is the characteristic angular scale of strong gravitational
lensing. For a point mass, it marks the radius where light rays from a source directly
behind the lens would form a perfect ring.

θ_E = sqrt(4GM D_ls / (c² D_l D_s))

where M is the lens mass, D_l, D_s, D_ls are angular diameter distances to the lens,
source, and between them respectively.

For galaxy-scale lenses, θ_E is typically 0.5-2 arcseconds.
""",
        "cdm_subhalos": """
In Cold Dark Matter (CDM) theory, dark matter halos contain a hierarchy of substructure
(subhalos) from previous merger events. These subhalos follow a mass function:

dN/dM ∝ M^β

where β ≈ -1.9 to -0.9 depending on the radial position and host mass.

In gravitational lensing, CDM subhalos create localized perturbations to the lensing
potential, causing flux ratio anomalies and surface brightness variations.
""",
        "axion_vortex": """
Ultralight axions (m_a ~ 10^-22 eV) have de Broglie wavelengths comparable to galaxy
scales. This creates wave-like interference patterns in the dark matter density:

λ_dB ≈ 0.6 kpc × (10^-22 eV / m_a)

In halos, this produces vortex-like structures where the density has nodes. These
patterns create distinctive signatures in gravitational lensing that differ from
CDM's point-like subhalos, potentially allowing dark matter mass constraints.
""",
        "strong_lensing": """
Strong gravitational lensing occurs when a massive object (lens) significantly bends
light from a background source, creating multiple images, arcs, or Einstein rings.

Requirements for strong lensing:
1. High surface mass density (Σ > Σ_crit)
2. Favorable geometric alignment
3. Source behind the lens (z_source > z_lens)

The critical surface density is:
Σ_crit = c²D_s / (4πG D_l D_ls)

Strong lensing is a powerful probe of dark matter distribution at sub-galactic scales.
""",
    }

    topic_lower = topic.lower().replace(" ", "_")

    for key, explanation in explanations.items():
        if key in topic_lower or topic_lower in key:
            return explanation.strip()

    return (
        f"I don't have a pre-written explanation for '{topic}'. "
        f"Available topics: {', '.join(explanations.keys())}. "
        "Please ask about one of these or rephrase your question."
    )


@deeplense_agent.tool
async def get_simulation_history(
    ctx: RunContext[AgentDependencies],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Get history of recent simulations.

    Args:
        limit: Maximum number of simulations to return.

    Returns:
        List of simulation summaries.
    """
    history = ctx.deps.state.completed_simulations[-limit:]

    return [
        {
            "simulation_id": (sim.metadata.simulation_id if sim.metadata else None),
            "success": sim.success,
            "num_images": sim.num_images_generated,
            "timestamp": (
                sim.metadata.timestamp.isoformat() if sim.metadata else None
            ),
            "model_type": (
                sim.metadata.config.model_type.value
                if sim.metadata
                else None
            ),
            "substructure": (
                sim.metadata.config.substructure.substructure_type.value
                if sim.metadata
                else None
            ),
            "duration_seconds": (
                sim.metadata.duration_seconds if sim.metadata else None
            ),
        }
        for sim in history
    ]


@deeplense_agent.tool
async def create_batch_configuration(
    ctx: RunContext[AgentDependencies],
    base_config_dict: dict[str, Any],
    vary_parameter: str,
    values: list[Any],
) -> list[dict[str, Any]]:
    """
    Create multiple configurations varying a single parameter.

    Useful for parameter sweeps and systematic studies.

    Args:
        base_config_dict: Base configuration to start from.
        vary_parameter: Parameter to vary (dot notation for nested, e.g., "cosmology.z_lens").
        values: List of values to sweep through.

    Returns:
        List of configuration dictionaries.
    """
    configs = []

    for value in values:
        config_dict = base_config_dict.copy()

        # Handle nested parameters
        parts = vary_parameter.split(".")
        target = config_dict

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

        # Validate
        try:
            config = SimulationConfig(**config_dict)
            configs.append({
                "config": config.model_dump(),
                "varied_value": value,
                "valid": True,
            })
        except Exception as e:
            configs.append({
                "config": config_dict,
                "varied_value": value,
                "valid": False,
                "error": str(e),
            })

    return configs


# =============================================================================
# High-Level Agent Interface
# =============================================================================


class DeepLenseAgent:
    """
    High-level interface for the DeepLense simulation agent.

    This class provides a convenient wrapper around the Pydantic AI agent
    for common use cases.

    Example:
        >>> agent = DeepLenseAgent()
        >>> result = await agent.generate_from_prompt(
        ...     "Generate 10 CDM lens images using Model I"
        ... )
        >>> print(f"Generated {result['num_images']} images")
    """

    def __init__(
        self,
        simulator: DeepLenseSimulator | None = None,
        mock_mode: bool = False,
        human_callback: Callable[[list[ClarificationQuestion]], dict[str, str]]
        | None = None,
    ):
        """
        Initialize the agent.

        Args:
            simulator: Custom simulator instance. If None, creates default.
            mock_mode: If True, use mock simulations.
            human_callback: Callback for human-in-the-loop clarification.
        """
        self.simulator = simulator or create_simulator(mock_mode=mock_mode)
        self.deps = AgentDependencies(
            simulator=self.simulator,
            human_callback=human_callback,
        )
        self._agent = deeplense_agent

    async def generate_from_prompt(
        self,
        prompt: str,
        auto_execute: bool = True,
    ) -> dict[str, Any]:
        """
        Generate simulations from a natural language prompt.

        Args:
            prompt: Natural language description of desired simulation.
            auto_execute: If True, automatically execute after clarification.

        Returns:
            Dictionary with simulation results or clarification needs.
        """
        # First, parse the request
        result = await self._agent.run(
            f"Parse this simulation request and determine if clarification is needed: {prompt}",
            deps=self.deps,
        )

        # Check if clarification is needed
        # The agent will use tools and return a structured response
        return {"response": result.data, "messages": result.all_messages()}

    async def generate_from_config(
        self, config: SimulationConfig
    ) -> SimulationOutput:
        """
        Generate simulations from a structured configuration.

        Args:
            config: Complete simulation configuration.

        Returns:
            SimulationOutput with generated images and metadata.
        """
        return self.simulator.run_simulation(config)

    async def interactive_session(
        self,
        initial_prompt: str | None = None,
    ) -> None:
        """
        Start an interactive session with the agent.

        This provides a REPL-like interface for multi-turn conversation.

        Args:
            initial_prompt: Optional initial prompt to start with.
        """
        print("DeepLense Simulation Agent")
        print("=" * 40)
        print("Type 'quit' or 'exit' to end the session.")
        print()

        messages: list[Any] = []

        if initial_prompt:
            print(f"You: {initial_prompt}")
            result = await self._agent.run(initial_prompt, deps=self.deps)
            print(f"Agent: {result.data}")
            messages = result.all_messages()
            print()

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                # Continue conversation
                result = await self._agent.run(
                    user_input,
                    deps=self.deps,
                    message_history=messages,
                )
                print(f"Agent: {result.data}")
                messages = result.all_messages()
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def create_agent(
    mock_mode: bool = False,
    human_callback: Callable[[list[ClarificationQuestion]], dict[str, str]]
    | None = None,
) -> DeepLenseAgent:
    """
    Factory function to create a DeepLense agent.

    Args:
        mock_mode: If True, use mock simulations.
        human_callback: Callback for human-in-the-loop clarification.

    Returns:
        Configured DeepLenseAgent instance.
    """
    return DeepLenseAgent(mock_mode=mock_mode, human_callback=human_callback)


# =============================================================================
# Synchronous Interface
# =============================================================================


def run_sync(coro):
    """Run a coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


class SyncDeepLenseAgent:
    """
    Synchronous wrapper for DeepLenseAgent.

    Provides the same interface but with synchronous methods
    for convenience in non-async contexts.
    """

    def __init__(self, **kwargs):
        self._async_agent = DeepLenseAgent(**kwargs)

    def generate_from_prompt(
        self, prompt: str, auto_execute: bool = True
    ) -> dict[str, Any]:
        """Synchronous version of generate_from_prompt."""
        return run_sync(
            self._async_agent.generate_from_prompt(prompt, auto_execute)
        )

    def generate_from_config(self, config: SimulationConfig) -> SimulationOutput:
        """Synchronous version of generate_from_config."""
        return run_sync(self._async_agent.generate_from_config(config))

    def interactive_session(self, initial_prompt: str | None = None) -> None:
        """Synchronous version of interactive_session."""
        run_sync(self._async_agent.interactive_session(initial_prompt))
