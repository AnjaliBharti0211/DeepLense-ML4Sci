"""
DeepLense Agent - Agentic workflow for gravitational lensing simulations.

This package provides a Pydantic AI-based agent that wraps the DeepLenseSim
simulation pipeline, enabling natural language interaction for generating
strong gravitational lensing images.

References:
- Alexander et al. 2020 (arXiv:1909.07346): Deep Learning the Morphology of Dark Matter Substructure
- Alexander et al. 2021 (arXiv:2008.12731): Decoding Dark Matter Substructure without Supervision
- Menzo et al. 2025 (arXiv:2512.15867): HEPTAPOD workflow orchestration patterns
"""

from deeplense_agent.models import (
    CosmologicalParameters,
    DarkMatterType,
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
from deeplense_agent.agent import DeepLenseAgent, create_agent
from deeplense_agent.simulator import DeepLenseSimulator
from deeplense_agent.config import ModelProvider, ProviderConfig, get_model_config
from deeplense_agent.presets import (
    PresetType,
    ScientificPreset,
    CosmologyPreset,
    HaloPreset,
    CDMSubstructurePreset,
    AxionVortexPreset,
    SourceLightPreset,
    ImagePreset,
    TrainingPreset,
    get_preset,
    get_default_preset,
)
from deeplense_agent.workflow import (
    TaskStatus,
    WorkflowMode,
    WorkflowTask,
    WorkflowState,
    StructuredError,
    StructuredResult,
    RunCardConfig,
    ApprovalCheckpoint,
    ToolRegistry,
    create_deeplense_tool_registry,
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "CosmologicalParameters",
    "DarkMatterType",
    "InstrumentConfig",
    "InstrumentType",
    "MainHaloParameters",
    "ModelType",
    "SimulationConfig",
    "SimulationMetadata",
    "SimulationOutput",
    "SimulationRequest",
    "SourceLightParameters",
    "SubstructureParameters",
    # Agent
    "DeepLenseAgent",
    "create_agent",
    # Simulator
    "DeepLenseSimulator",
    # Config
    "ModelProvider",
    "ProviderConfig",
    "get_model_config",
    # Presets (from research papers)
    "PresetType",
    "ScientificPreset",
    "CosmologyPreset",
    "HaloPreset",
    "CDMSubstructurePreset",
    "AxionVortexPreset",
    "SourceLightPreset",
    "ImagePreset",
    "TrainingPreset",
    "get_preset",
    "get_default_preset",
    # Workflow (HEPTAPOD-inspired)
    "TaskStatus",
    "WorkflowMode",
    "WorkflowTask",
    "WorkflowState",
    "StructuredError",
    "StructuredResult",
    "RunCardConfig",
    "ApprovalCheckpoint",
    "ToolRegistry",
    "create_deeplense_tool_registry",
]
