#!/usr/bin/env python3
"""
Batch generation and parameter sweep example.

This example demonstrates how to:
1. Generate large batches of simulations for ML training
2. Perform parameter sweeps for systematic studies
3. Save results with structured metadata
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from deeplense_agent import (
    CosmologicalParameters,
    DarkMatterType,
    ModelType,
    SimulationConfig,
    SubstructureParameters,
)
from deeplense_agent.simulator import create_simulator


def generate_training_dataset():
    """Generate a balanced dataset for ML classification."""
    print("=" * 70)
    print("ML Training Dataset Generation")
    print("=" * 70)
    print()

    # Configuration for training data
    images_per_class = 10  # Use larger values (1000+) for real training
    substructure_types = [
        DarkMatterType.NO_SUBSTRUCTURE,
        DarkMatterType.CDM,
        DarkMatterType.AXION,
    ]

    simulator = create_simulator(mock_mode=True)
    output_dir = Path("./training_data")

    all_metadata: list[dict[str, Any]] = []

    for sub_type in substructure_types:
        print(f"Generating {images_per_class} {sub_type.value} images...")

        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=images_per_class,
            substructure=SubstructureParameters(
                substructure_type=sub_type,
                axion_mass=1e-23 if sub_type == DarkMatterType.AXION else None,
            ),
        )

        output = simulator.run_simulation(config)

        if output.success:
            # Save images and metadata
            class_dir = output_dir / sub_type.value
            class_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(output.images):
                # In real usage, save the actual image data
                metadata = {
                    "filename": f"{sub_type.value}_{i:04d}.npy",
                    "class": sub_type.value,
                    "model_type": config.model_type.value,
                    "z_lens": config.cosmology.z_lens,
                    "z_source": config.cosmology.z_source,
                    "halo_mass": config.main_halo.halo_mass,
                    "image_stats": {
                        "shape": (img.width, img.height, img.channels),
                        "min": img.min_value,
                        "max": img.max_value,
                        "mean": img.mean_value,
                    },
                }

                if sub_type == DarkMatterType.CDM:
                    metadata["n_sub_mean"] = config.substructure.n_sub_mean
                elif sub_type == DarkMatterType.AXION:
                    metadata["axion_mass"] = config.substructure.axion_mass

                all_metadata.append(metadata)

            print(f"  Saved {output.num_images_generated} images to {class_dir}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "total_images": len(all_metadata),
        "classes": [t.value for t in substructure_types],
        "images_per_class": images_per_class,
        "samples": all_metadata,
    }

    print()
    print(f"Dataset Summary:")
    print(f"  Total images: {len(all_metadata)}")
    print(f"  Classes: {', '.join(t.value for t in substructure_types)}")
    print(f"  Images per class: {images_per_class}")
    print()


def parameter_sweep_redshift():
    """Perform a parameter sweep over lens redshift."""
    print("=" * 70)
    print("Parameter Sweep: Lens Redshift")
    print("=" * 70)
    print()

    simulator = create_simulator(mock_mode=True)

    z_lens_values = np.linspace(0.2, 1.0, 5)
    z_source = 1.5  # Fixed source redshift

    results: list[dict[str, Any]] = []

    for z_lens in z_lens_values:
        print(f"  z_lens = {z_lens:.2f}...")

        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=3,
            cosmology=CosmologicalParameters(
                z_lens=z_lens,
                z_source=z_source,
            ),
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.CDM,
            ),
        )

        output = simulator.run_simulation(config)

        if output.success:
            # Compute mean statistics across images
            mean_intensities = [img.mean_value for img in output.images]

            results.append({
                "z_lens": z_lens,
                "z_source": z_source,
                "D_lens_source_ratio": z_lens / z_source,
                "mean_intensity": np.mean(mean_intensities),
                "std_intensity": np.std(mean_intensities),
                "num_images": output.num_images_generated,
            })

    print()
    print("Results:")
    print("-" * 60)
    print(f"{'z_lens':>8} {'z_src':>8} {'D ratio':>10} {'Mean I':>12} {'Std I':>10}")
    print("-" * 60)

    for r in results:
        print(f"{r['z_lens']:8.2f} {r['z_source']:8.2f} {r['D_lens_source_ratio']:10.3f} "
              f"{r['mean_intensity']:12.6f} {r['std_intensity']:10.6f}")
    print()


def parameter_sweep_axion_mass():
    """Perform a parameter sweep over axion mass."""
    print("=" * 70)
    print("Parameter Sweep: Axion Mass")
    print("=" * 70)
    print()

    simulator = create_simulator(mock_mode=True)

    # Log-spaced axion masses
    axion_masses = 10 ** np.linspace(-24, -22, 5)

    results: list[dict[str, Any]] = []

    for m_axion in axion_masses:
        print(f"  m_axion = {m_axion:.2e} eV...")

        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=3,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.AXION,
                axion_mass=m_axion,
            ),
        )

        output = simulator.run_simulation(config)

        if output.success:
            mean_intensities = [img.mean_value for img in output.images]

            # Compute de Broglie wavelength
            lambda_db = 0.6 * (1e-22 / m_axion)  # kpc

            results.append({
                "axion_mass": m_axion,
                "log10_mass": np.log10(m_axion),
                "de_broglie_wavelength_kpc": lambda_db,
                "mean_intensity": np.mean(mean_intensities),
                "num_images": output.num_images_generated,
            })

    print()
    print("Results:")
    print("-" * 70)
    print(f"{'m_axion (eV)':>14} {'log10(m)':>10} {'λ_dB (kpc)':>12} {'Mean I':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['axion_mass']:14.2e} {r['log10_mass']:10.1f} "
              f"{r['de_broglie_wavelength_kpc']:12.3f} {r['mean_intensity']:12.6f}")
    print()


def grid_search_cosmology():
    """Perform a 2D grid search over cosmological parameters."""
    print("=" * 70)
    print("Grid Search: H0 vs Om0")
    print("=" * 70)
    print()

    simulator = create_simulator(mock_mode=True)

    h0_values = [65, 70, 75]
    om0_values = [0.25, 0.30, 0.35]

    results: list[dict[str, Any]] = []

    total = len(h0_values) * len(om0_values)
    count = 0

    for h0 in h0_values:
        for om0 in om0_values:
            count += 1
            print(f"  [{count}/{total}] H0={h0}, Om0={om0:.2f}...")

            config = SimulationConfig(
                model_type=ModelType.MODEL_I,
                num_images=2,
                cosmology=CosmologicalParameters(
                    H0=h0,
                    Om0=om0,
                ),
            )

            output = simulator.run_simulation(config)

            if output.success:
                mean_intensities = [img.mean_value for img in output.images]

                results.append({
                    "H0": h0,
                    "Om0": om0,
                    "mean_intensity": np.mean(mean_intensities),
                    "duration": output.metadata.duration_seconds,
                })

    print()
    print("Results Grid (Mean Intensity):")
    print("-" * 45)

    # Print header
    header = "         " + " ".join(f"Om0={om0:.2f}" for om0 in om0_values)
    print(header)

    # Print rows
    for h0 in h0_values:
        row_results = [r for r in results if r["H0"] == h0]
        row_values = [r["mean_intensity"] for r in sorted(row_results, key=lambda x: x["Om0"])]
        row_str = f"H0={h0:3d}  " + " ".join(f"{v:8.5f}" for v in row_values)
        print(row_str)
    print()


def batch_with_progress():
    """Generate a batch with progress tracking."""
    print("=" * 70)
    print("Batch Generation with Progress")
    print("=" * 70)
    print()

    simulator = create_simulator(mock_mode=True)

    # Multiple configurations to run
    configs = [
        SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=5,
            substructure=SubstructureParameters(substructure_type=DarkMatterType.CDM),
        ),
        SimulationConfig(
            model_type=ModelType.MODEL_II,
            num_images=5,
            substructure=SubstructureParameters(substructure_type=DarkMatterType.AXION, axion_mass=1e-23),
        ),
        SimulationConfig(
            model_type=ModelType.MODEL_III,
            num_images=5,
            substructure=SubstructureParameters(substructure_type=DarkMatterType.NO_SUBSTRUCTURE),
        ),
    ]

    total_images = sum(c.num_images for c in configs)
    generated = 0
    successful = 0
    failed = 0

    print(f"Total configurations: {len(configs)}")
    print(f"Total images to generate: {total_images}")
    print()

    for i, config in enumerate(configs, 1):
        print(f"Config {i}/{len(configs)}: {config.model_type.value}, "
              f"{config.substructure.substructure_type.value}")

        output = simulator.run_simulation(config)

        if output.success:
            successful += output.num_images_generated
            generated += output.num_images_generated
            print(f"  Success: {output.num_images_generated} images in "
                  f"{output.metadata.duration_seconds:.2f}s")
        else:
            failed += config.num_images
            print(f"  Failed: {output.error_message}")

    print()
    print("Summary:")
    print(f"  Requested: {total_images}")
    print(f"  Generated: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {successful/total_images*100:.1f}%")


def main():
    """Run all batch and sweep examples."""
    print("\nDeepLense Agent - Batch Generation Examples")
    print("=" * 70)
    print("Note: Running in mock mode (simulated images)")
    print()

    generate_training_dataset()
    parameter_sweep_redshift()
    parameter_sweep_axion_mass()
    grid_search_cosmology()
    batch_with_progress()

    print("=" * 70)
    print("All batch examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
