"""
Command-line interface for the DeepLense Agent.

Provides a rich terminal interface for generating gravitational
lensing simulations through natural language interaction.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from deeplense_agent.agent import DeepLenseAgent, SyncDeepLenseAgent, create_agent
from deeplense_agent.clarification import ClarificationEngine
from deeplense_agent.models import (
    ClarificationQuestion,
    DarkMatterType,
    ModelType,
    SimulationConfig,
    SubstructureParameters,
)
from deeplense_agent.simulator import create_simulator

app = typer.Typer(
    name="deeplense-agent",
    help="Agentic workflow for DeepLense gravitational lensing simulations.",
    add_completion=False,
)
console = Console()


def print_banner() -> None:
    """Print the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                   DeepLense Agent                         ║
    ║         Gravitational Lensing Simulation Agent            ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def format_config_table(config: SimulationConfig) -> Table:
    """Format a simulation configuration as a rich table."""
    table = Table(
        title="Simulation Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="white")

    # Model info
    table.add_row("Model Type", config.model_type.value)
    table.add_row("Number of Images", str(config.num_images))
    table.add_row("Resolution", f"{config.model_type.resolution}x{config.model_type.resolution}")

    # Cosmology
    table.add_row("─" * 20, "─" * 30)
    table.add_row("[bold]Cosmology[/bold]", "")
    table.add_row("  H0", f"{config.cosmology.H0} km/s/Mpc")
    table.add_row("  Ωm", str(config.cosmology.Om0))
    table.add_row("  z_lens", str(config.cosmology.z_lens))
    table.add_row("  z_source", str(config.cosmology.z_source))

    # Substructure
    table.add_row("─" * 20, "─" * 30)
    table.add_row("[bold]Substructure[/bold]", "")
    table.add_row("  Type", config.substructure.substructure_type.value)
    if config.substructure.substructure_type == DarkMatterType.CDM:
        table.add_row("  N_sub (mean)", str(config.substructure.n_sub_mean))
        table.add_row("  M_sub range", f"{config.substructure.m_sub_min:.0e} - {config.substructure.m_sub_max:.0e} M☉")
    elif config.substructure.substructure_type == DarkMatterType.AXION:
        if config.substructure.axion_mass:
            table.add_row("  Axion mass", f"{config.substructure.axion_mass:.2e} eV")
        table.add_row("  Vortex mass", f"{config.substructure.vortex_mass:.2e} M☉")

    # Main halo
    table.add_row("─" * 20, "─" * 30)
    table.add_row("[bold]Main Halo[/bold]", "")
    table.add_row("  Mass", f"{config.main_halo.halo_mass:.2e} M☉")
    table.add_row("  Ellipticity", f"({config.main_halo.ellipticity_e1}, {config.main_halo.ellipticity_e2})")

    return table


def ask_clarification_questions(
    questions: list[ClarificationQuestion],
) -> dict[str, str]:
    """Interactively ask clarification questions."""
    responses: dict[str, str] = {}

    for question in questions:
        console.print()
        console.print(Panel(
            f"[bold cyan]{question.question_text}[/bold cyan]\n\n"
            f"[dim]{question.scientific_context or ''}[/dim]",
            title=f"Question: {question.category.upper()}",
            border_style="blue",
        ))

        if question.options:
            for i, option in enumerate(question.options, 1):
                default_marker = " [default]" if question.default_value and question.default_value in option else ""
                console.print(f"  [{i}] {option}{default_marker}")

            choice = Prompt.ask(
                "Select option",
                default="1" if question.default_value else None,
            )

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(question.options):
                    responses[question.question_id] = question.options[idx]
                else:
                    responses[question.question_id] = choice
            except ValueError:
                responses[question.question_id] = choice
        else:
            response = Prompt.ask(
                "Your answer",
                default=question.default_value,
            )
            responses[question.question_id] = response

    return responses


@app.command()
def generate(
    prompt: str = typer.Argument(
        ...,
        help="Natural language description of desired simulation",
    ),
    num_images: int = typer.Option(
        None,
        "--num", "-n",
        help="Number of images to generate",
    ),
    model: str = typer.Option(
        None,
        "--model", "-m",
        help="Model type (model_i, model_ii, model_iii)",
    ),
    substructure: str = typer.Option(
        None,
        "--substructure", "-s",
        help="Substructure type (cdm, axion, no_substructure)",
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Output directory for generated images",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock simulation mode (for testing)",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm", "-y",
        help="Skip confirmation prompt",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """
    Generate gravitational lensing simulations from natural language.

    Example:
        deeplense-agent generate "10 CDM lens images using Model I"
    """
    if not json_output:
        print_banner()

    # Create engine and parse request
    engine = ClarificationEngine()

    # Enhance prompt with CLI options
    enhanced_prompt = prompt
    if num_images:
        enhanced_prompt += f" {num_images} images"
    if model:
        enhanced_prompt += f" using {model}"
    if substructure:
        enhanced_prompt += f" with {substructure}"

    # Analyze request
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Parsing request...", total=None)
        response = engine.analyze_request(enhanced_prompt)

    if not json_output:
        console.print()
        console.print(f"[bold]Interpretation:[/bold] {response.interpretation_summary}")
        console.print(f"[bold]Confidence:[/bold] {response.confidence_score:.0%}")

    # Handle clarification
    if response.needs_clarification and not no_confirm:
        if not json_output:
            console.print()
            console.print("[yellow]Some parameters need clarification:[/yellow]")

        user_responses = ask_clarification_questions(response.questions)

        # Re-analyze with responses
        response = engine.analyze_request(enhanced_prompt, user_responses)

    config = response.partial_config

    # Show configuration and confirm
    if not json_output:
        console.print()
        console.print(format_config_table(config))

    if not no_confirm and not json_output:
        console.print()
        if not Confirm.ask("Proceed with simulation?", default=True):
            console.print("[yellow]Simulation cancelled.[/yellow]")
            raise typer.Exit(0)

    # Run simulation
    simulator = create_simulator(mock_mode=mock)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=not json_output,
    ) as progress:
        task = progress.add_task(
            f"Generating {config.num_images} image(s)...",
            total=None,
        )
        output = simulator.run_simulation(config)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if output.success:
        # Save images
        saved_files: list[str] = []
        for i, img_data in enumerate(output.images):
            if img_data.base64_png:
                import base64
                img_bytes = base64.b64decode(img_data.base64_png)
                filename = f"lens_{i:04d}.png"
                filepath = output_dir / filename
                filepath.write_bytes(img_bytes)
                saved_files.append(str(filepath))

        # Save metadata
        if output.metadata:
            metadata_file = output_dir / "metadata.json"
            metadata_dict = {
                "simulation_id": output.metadata.simulation_id,
                "timestamp": output.metadata.timestamp.isoformat(),
                "duration_seconds": output.metadata.duration_seconds,
                "config": output.metadata.config.model_dump(),
                "num_images": len(saved_files),
                "files": saved_files,
            }
            metadata_file.write_text(json.dumps(metadata_dict, indent=2))

        if json_output:
            result = {
                "success": True,
                "num_images": len(saved_files),
                "output_dir": str(output_dir),
                "files": saved_files,
                "metadata": metadata_dict if output.metadata else None,
            }
            console.print_json(json.dumps(result))
        else:
            console.print()
            console.print(Panel(
                f"[green]Successfully generated {len(saved_files)} image(s)[/green]\n"
                f"Output directory: {output_dir}\n"
                f"Duration: {output.metadata.duration_seconds:.2f}s" if output.metadata else "",
                title="Success",
                border_style="green",
            ))

            if output.warnings:
                for warning in output.warnings:
                    console.print(f"[yellow]Warning:[/yellow] {warning}")
    else:
        if json_output:
            console.print_json(json.dumps({
                "success": False,
                "error": output.error_message,
            }))
        else:
            console.print(Panel(
                f"[red]Simulation failed[/red]\n{output.error_message}",
                title="Error",
                border_style="red",
            ))
        raise typer.Exit(1)


@app.command()
def chat(
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock simulation mode",
    ),
) -> None:
    """
    Start an interactive chat session with the agent.

    This provides a conversational interface for exploring
    simulation options and generating images.
    """
    print_banner()
    console.print(
        "Welcome to the DeepLense Agent interactive mode.\n"
        "Describe your simulation needs in natural language.\n"
        "Type 'quit' or 'exit' to end the session.\n"
    )

    agent = SyncDeepLenseAgent(mock_mode=mock)
    agent.interactive_session()


@app.command()
def config(
    output: Path = typer.Option(
        Path("simulation_config.json"),
        "--output", "-o",
        help="Output file for configuration",
    ),
    template: str = typer.Option(
        "default",
        "--template", "-t",
        help="Configuration template (default, cdm, axion, comparison)",
    ),
) -> None:
    """
    Generate a simulation configuration template.

    Creates a JSON configuration file that can be edited
    and used with the 'run' command.
    """
    templates = {
        "default": SimulationConfig(),
        "cdm": SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=100,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.CDM,
            ),
        ),
        "axion": SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=100,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.AXION,
                axion_mass=1e-23,
            ),
        ),
        "comparison": SimulationConfig(
            model_type=ModelType.MODEL_II,
            num_images=50,
        ),
    }

    if template not in templates:
        console.print(f"[red]Unknown template: {template}[/red]")
        console.print(f"Available templates: {', '.join(templates.keys())}")
        raise typer.Exit(1)

    config = templates[template]
    config_dict = config.model_dump()

    output.write_text(json.dumps(config_dict, indent=2, default=str))
    console.print(f"[green]Configuration saved to {output}[/green]")


@app.command()
def run(
    config_file: Path = typer.Argument(
        ...,
        help="Path to JSON configuration file",
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Output directory for generated images",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock simulation mode",
    ),
) -> None:
    """
    Run a simulation from a configuration file.

    Use 'config' command to generate a template configuration.
    """
    print_banner()

    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)

    try:
        config_dict = json.loads(config_file.read_text())
        config = SimulationConfig(**config_dict)
    except Exception as e:
        console.print(f"[red]Error parsing configuration: {e}[/red]")
        raise typer.Exit(1)

    console.print(format_config_table(config))
    console.print()

    simulator = create_simulator(mock_mode=mock)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating {config.num_images} image(s)...",
            total=None,
        )
        output = simulator.run_simulation(config)

    if output.success:
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img_data in enumerate(output.images):
            if img_data.base64_png:
                import base64
                img_bytes = base64.b64decode(img_data.base64_png)
                filename = f"lens_{i:04d}.png"
                filepath = output_dir / filename
                filepath.write_bytes(img_bytes)

        console.print(Panel(
            f"[green]Successfully generated {output.num_images_generated} image(s)[/green]\n"
            f"Output directory: {output_dir}",
            title="Success",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[red]Simulation failed[/red]\n{output.error_message}",
            title="Error",
            border_style="red",
        ))
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """
    Display information about available models and parameters.
    """
    print_banner()

    # Models table
    models_table = Table(
        title="Available Model Configurations",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    models_table.add_column("Model", style="green")
    models_table.add_column("Resolution")
    models_table.add_column("Instrument")
    models_table.add_column("Description")

    models_table.add_row(
        "Model I",
        "150x150",
        "Generic (Gaussian PSF)",
        "Basic simulation for general studies",
    )
    models_table.add_row(
        "Model II",
        "64x64",
        "Euclid",
        "Euclid survey characteristics",
    )
    models_table.add_row(
        "Model III",
        "64x64",
        "HST",
        "Hubble Space Telescope characteristics",
    )

    console.print(models_table)
    console.print()

    # Substructure table
    sub_table = Table(
        title="Dark Matter Substructure Types",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    sub_table.add_column("Type", style="green")
    sub_table.add_column("Description")
    sub_table.add_column("Key Parameters")

    sub_table.add_row(
        "CDM",
        "Cold Dark Matter with point-mass subhalos",
        "m_sub_min, m_sub_max, n_sub_mean",
    )
    sub_table.add_row(
        "Axion/Vortex",
        "Ultralight dark matter with wave-like patterns",
        "axion_mass, vortex_mass",
    )
    sub_table.add_row(
        "No Substructure",
        "Clean lens without subhalos (baseline)",
        "N/A",
    )

    console.print(sub_table)


@app.command()
def version() -> None:
    """Display version information."""
    from deeplense_agent import __version__

    console.print(f"DeepLense Agent v{__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
