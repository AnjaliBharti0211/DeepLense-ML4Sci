# DeepLense Agent

> **Agentic workflow for gravitational lensing simulations using Pydantic AI**

An intelligent agent that wraps the [DeepLenseSim](https://github.com/mwt5345/DeepLenseSim) simulation pipeline, enabling natural language interaction for generating strong gravitational lensing images with structured metadata.

## Features

- **Natural Language Interface**: Describe simulations in plain English
- **Human-in-the-Loop**: Interactive clarification for ambiguous requests
- **Strong Typing**: Pydantic models for all simulation parameters
- **Multiple Models**: Support for Model I (150x150), Model II (Euclid), and Model III (HST)
- **Dark Matter Substructure**: CDM, Axion/Vortex, and no-substructure configurations
- **Batch Generation**: Parameter sweeps and dataset generation
- **Rich CLI**: Terminal interface with progress tracking
- **Groq Integration**: Uses Groq's fast LLM inference by default

## Architecture

```
deeplense_agent/
├── models.py          # Pydantic models for parameters & outputs
├── simulator.py       # DeepLenseSim wrapper
├── clarification.py   # Human-in-the-loop engine
├── agent.py           # Pydantic AI agent with tools
├── config.py          # API configuration (Groq/OpenAI/Anthropic)
└── cli.py             # Rich terminal interface
```

### Design Philosophy

1. **Layered Pydantic Models**: Strong typing ensures parameter validity at compile time
2. **Tool-based Agent**: Specialized tools for parsing, validation, and simulation
3. **Clarification Engine**: Scientific context helps users make informed decisions
4. **Mock Mode**: Full testing without DeepLenseSim dependencies

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd deeplense_agent

# Install with pip
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Dependencies

- **Required**: pydantic, pydantic-ai, openai, numpy, pillow, rich, typer
- **Simulation**: lenstronomy==1.9.2, colossus (auto-detected, falls back to mock mode)
- **Optional**: pyhalo (for Model IV)

### LLM Provider Configuration

The agent uses **Groq** by default for fast LLM inference. You can configure different providers:

```python
# Default: Groq (preconfigured API key)
from deeplense_agent.config import ProviderConfig

# Use a custom Groq API key
config = ProviderConfig.groq(api_key="your-groq-api-key")

# Switch to OpenAI
config = ProviderConfig.openai(api_key="your-openai-api-key")

# Switch to Anthropic
config = ProviderConfig.anthropic(api_key="your-anthropic-api-key")
```

Or via environment variables:

```bash
# Set provider (groq, openai, anthropic)
export DEEPLENSE_PROVIDER=groq

# Set API keys
export GROQ_API_KEY=your-groq-api-key
export OPENAI_API_KEY=your-openai-api-key
export ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Quick Start

### Python API

```python
from deeplense_agent import (
    create_agent,
    SimulationConfig,
    DarkMatterType,
    ModelType,
    SubstructureParameters,
)

# Create agent (uses mock mode if DeepLenseSim not installed)
agent = create_agent(mock_mode=True)

# Generate from configuration
config = SimulationConfig(
    model_type=ModelType.MODEL_I,
    num_images=10,
    substructure=SubstructureParameters(
        substructure_type=DarkMatterType.CDM,
    ),
)

import asyncio
output = asyncio.run(agent.generate_from_config(config))
print(f"Generated {output.num_images_generated} images")
```

### Natural Language

```python
from deeplense_agent.clarification import ClarificationEngine

engine = ClarificationEngine()
response = engine.analyze_request(
    "Generate 20 CDM lens images using Euclid model with z_lens=0.5"
)

print(f"Confidence: {response.confidence_score:.0%}")
print(f"Config: {response.partial_config.model_dump()}")
```

### CLI

```bash
# Generate from natural language
deeplense-agent generate "10 CDM lens images using Model I" --mock

# Interactive chat mode
deeplense-agent chat --mock

# Generate config template
deeplense-agent config -t cdm -o my_config.json

# Run from config file
deeplense-agent run my_config.json --mock

# Show available models
deeplense-agent info
```

## Model Configurations

| Model | Resolution | Instrument | Use Case |
|-------|------------|------------|----------|
| Model I | 150x150 | Generic (Gaussian PSF) | General studies, high detail |
| Model II | 64x64 | Euclid | Euclid survey simulations |
| Model III | 64x64 | HST | Hubble observations |

## Dark Matter Substructure

| Type | Description | Key Parameters |
|------|-------------|----------------|
| CDM | Cold Dark Matter with point-mass subhalos | m_sub_min, m_sub_max, n_sub_mean |
| Axion | Ultralight DM with wave-like patterns | axion_mass, vortex_mass |
| No Sub | Clean lens baseline | None |

## Human-in-the-Loop Workflow

The agent uses a clarification engine to handle ambiguous requests:

```python
from deeplense_agent.clarification import ClarificationEngine

engine = ClarificationEngine()

# Ambiguous request
response = engine.analyze_request("some lens images")

if response.needs_clarification:
    for question in response.questions:
        print(f"Q: {question.question_text}")
        print(f"   Context: {question.scientific_context}")
        print(f"   Options: {question.options}")

    # Apply user responses
    user_answers = {"model_type": "Model I (150x150, basic)"}
    final = engine.analyze_request("some lens images", user_answers)
```

## Agent Tools

The Pydantic AI agent provides these tools:

| Tool | Description |
|------|-------------|
| `parse_simulation_request` | Extract parameters from natural language |
| `get_clarification_questions` | Generate questions for missing params |
| `apply_clarification_response` | Apply user answers to config |
| `validate_simulation_config` | Validate physical consistency |
| `run_simulation` | Execute the simulation |
| `get_parameter_suggestions` | Get scientific suggestions |
| `explain_simulation_physics` | Educational explanations |
| `create_batch_configuration` | Generate parameter sweep configs |

## Examples

See the `examples/` directory:

- **basic_usage.py**: Simple generation workflows
- **human_in_the_loop.py**: Interactive clarification demo
- **batch_generation.py**: Dataset and parameter sweep generation

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=deeplense_agent

# Run specific test file
pytest tests/test_models.py

# Skip slow tests
pytest -m "not slow"
```

## Pydantic Models Reference

### SimulationConfig

```python
SimulationConfig(
    model_type: ModelType = MODEL_I,
    num_images: int = 1,
    random_seed: int | None = None,
    cosmology: CosmologicalParameters = ...,
    substructure: SubstructureParameters = ...,
    source_light: SourceLightParameters = ...,
    main_halo: MainHaloParameters = ...,
    instrument: InstrumentConfig | None = None,  # Auto-set from model_type
)
```

### CosmologicalParameters

```python
CosmologicalParameters(
    H0: float = 70.0,           # Hubble constant (50-100 km/s/Mpc)
    Om0: float = 0.3,           # Matter density (0.1-0.5)
    Ob0: float = 0.05,          # Baryon density (0.01-0.1)
    z_lens: float = 0.5,        # Lens redshift (0.1-2.0)
    z_source: float = 1.0,      # Source redshift (0.2-5.0)
)
```

### SubstructureParameters

```python
SubstructureParameters(
    substructure_type: DarkMatterType = CDM,
    # CDM parameters
    m_sub_min: float = 1e6,     # Min subhalo mass
    m_sub_max: float = 1e10,    # Max subhalo mass
    n_sub_mean: int = 25,       # Mean number of subhalos
    # Axion parameters
    axion_mass: float = 1e-23,  # Axion mass in eV
    vortex_mass: float = 3e10,  # Vortex mass in M_sun
)
```

## Scientific Background

This project simulates strong gravitational lensing for dark matter research:

- **Einstein Radius**: Characteristic angular scale of lensing
- **CDM Subhalos**: Point-mass concentrations from hierarchical merging
- **Axion Vortices**: Wave-like interference patterns from ultralight DM

See the referenced papers:
- arXiv:1909.07346
- arXiv:2008.12731
- arXiv:2112.12121

## License

MIT License - see LICENSE file.

## Acknowledgments

- [DeepLenseSim](https://github.com/mwt5345/DeepLenseSim) by Michael W. Toomey
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) for the agent framework
- [lenstronomy](https://github.com/lenstronomy/lenstronomy) for gravitational lensing
