#!/usr/bin/env python3
"""
Basic usage example for DeepLense Agent.

This example demonstrates how to use the agent to generate
gravitational lensing simulations from natural language prompts.
"""

import asyncio

from deeplense_agent import (
    CosmologicalParameters,
    DarkMatterType,
    DeepLenseAgent,
    ModelType,
    SimulationConfig,
    SubstructureParameters,
    create_agent,
)


async def basic_prompt_example():
    """Generate simulations from a natural language prompt."""
    print("=" * 60)
    print("Example 1: Natural Language Prompt")
    print("=" * 60)

    # Create agent in mock mode (for testing without DeepLenseSim installed)
    agent = create_agent(mock_mode=True)

    # Generate from natural language
    prompt = "Generate 5 CDM lens images using Model I with z_lens=0.5"

    print(f"Prompt: {prompt}")
    print()

    result = await agent.generate_from_prompt(prompt)

    print("Agent Response:")
    print(result["response"])
    print()


async def structured_config_example():
    """Generate simulations from a structured configuration."""
    print("=" * 60)
    print("Example 2: Structured Configuration")
    print("=" * 60)

    # Create a detailed configuration
    config = SimulationConfig(
        model_type=ModelType.MODEL_I,
        num_images=3,
        random_seed=42,  # For reproducibility
        cosmology=CosmologicalParameters(
            H0=70.0,
            Om0=0.3,
            z_lens=0.5,
            z_source=1.0,
        ),
        substructure=SubstructureParameters(
            substructure_type=DarkMatterType.CDM,
            m_sub_min=1e6,
            m_sub_max=1e10,
            n_sub_mean=25,
        ),
    )

    print("Configuration:")
    print(f"  Model: {config.model_type.value}")
    print(f"  Images: {config.num_images}")
    print(f"  Substructure: {config.substructure.substructure_type.value}")
    print(f"  z_lens: {config.cosmology.z_lens}")
    print(f"  z_source: {config.cosmology.z_source}")
    print()

    # Create agent and run simulation
    agent = create_agent(mock_mode=True)
    output = await agent.generate_from_config(config)

    if output.success:
        print(f"Success! Generated {output.num_images_generated} images")
        print(f"Duration: {output.metadata.duration_seconds:.2f}s")

        for i, img in enumerate(output.images):
            print(f"  Image {i}: {img.width}x{img.height}, "
                  f"range [{img.min_value:.3f}, {img.max_value:.3f}]")
    else:
        print(f"Failed: {output.error_message}")
    print()


async def axion_simulation_example():
    """Generate axion/vortex simulations."""
    print("=" * 60)
    print("Example 3: Axion/Vortex Substructure")
    print("=" * 60)

    config = SimulationConfig(
        model_type=ModelType.MODEL_II,  # Euclid-like
        num_images=2,
        substructure=SubstructureParameters(
            substructure_type=DarkMatterType.AXION,
            axion_mass=1e-23,  # eV
            vortex_mass=3e10,  # Solar masses
        ),
    )

    print("Axion Configuration:")
    print(f"  Axion mass: {config.substructure.axion_mass:.0e} eV")
    print(f"  de Broglie wavelength: {config.substructure.de_broglie_wavelength_kpc:.2f} kpc")
    print()

    agent = create_agent(mock_mode=True)
    output = await agent.generate_from_config(config)

    if output.success:
        print(f"Generated {output.num_images_generated} axion lens images")
    print()


async def comparison_study_example():
    """Generate comparison dataset across substructure types."""
    print("=" * 60)
    print("Example 4: Comparison Study (CDM vs Axion vs No-sub)")
    print("=" * 60)

    agent = create_agent(mock_mode=True)

    substructure_types = [
        DarkMatterType.NO_SUBSTRUCTURE,
        DarkMatterType.CDM,
        DarkMatterType.AXION,
    ]

    for sub_type in substructure_types:
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=2,
            random_seed=12345,  # Same seed for comparison
            substructure=SubstructureParameters(
                substructure_type=sub_type,
                axion_mass=1e-23 if sub_type == DarkMatterType.AXION else None,
            ),
        )

        output = await agent.generate_from_config(config)

        if output.success:
            print(f"{sub_type.value}: {output.num_images_generated} images, "
                  f"mean intensity = {output.images[0].mean_value:.4f}")
    print()


def main():
    """Run all examples."""
    print("\nDeepLense Agent - Usage Examples")
    print("=" * 60)
    print("Note: Running in mock mode (simulated images)")
    print()

    asyncio.run(basic_prompt_example())
    asyncio.run(structured_config_example())
    asyncio.run(axion_simulation_example())
    asyncio.run(comparison_study_example())

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
