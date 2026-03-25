#!/usr/bin/env python3
"""
Advanced example demonstrating the human-in-the-loop workflow.

This example shows how to:
1. Parse ambiguous natural language requests
2. Generate and handle clarification questions
3. Apply user responses to refine the configuration
4. Execute the simulation with the final configuration
"""

from deeplense_agent.clarification import ClarificationEngine, NaturalLanguageParser
from deeplense_agent.models import (
    ClarificationQuestion,
    DarkMatterType,
    ModelType,
    SimulationConfig,
)
from deeplense_agent.simulator import create_simulator


def simulate_user_response(question: ClarificationQuestion) -> str:
    """Simulate a user response for demonstration purposes."""
    # In a real application, this would prompt the user
    responses = {
        "model_type": "Model I (150x150, basic)",
        "substructure": "CDM (Cold Dark Matter)",
        "num_images": "10 (quick test)",
        "axion_mass": "1e-23 eV (typical)",
    }
    return responses.get(question.question_id, question.default_value or "")


def demonstrate_clarification_workflow():
    """Demonstrate the full clarification workflow."""
    print("=" * 70)
    print("Human-in-the-Loop Clarification Workflow")
    print("=" * 70)
    print()

    # Create the clarification engine
    engine = ClarificationEngine()

    # Example 1: Ambiguous request
    print("SCENARIO 1: Ambiguous Request")
    print("-" * 40)
    prompt1 = "I want to generate some lens images"
    print(f"User: {prompt1}")
    print()

    response1 = engine.analyze_request(prompt1)

    print(f"Confidence: {response1.confidence_score:.0%}")
    print(f"Needs clarification: {response1.needs_clarification}")
    print(f"Interpretation: {response1.interpretation_summary}")
    print()

    if response1.needs_clarification:
        print("Clarification questions needed:")
        for q in response1.questions:
            print(f"\n  [{q.category.upper()}] {q.question_text}")
            if q.options:
                for i, opt in enumerate(q.options, 1):
                    print(f"    {i}. {opt}")
            if q.scientific_context:
                print(f"  Context: {q.scientific_context[:80]}...")
    print()

    # Example 2: More specific request
    print("\n" + "=" * 70)
    print("SCENARIO 2: More Specific Request")
    print("-" * 40)
    prompt2 = "Generate 20 CDM lens images using Model II with Euclid characteristics"
    print(f"User: {prompt2}")
    print()

    response2 = engine.analyze_request(prompt2)

    print(f"Confidence: {response2.confidence_score:.0%}")
    print(f"Needs clarification: {response2.needs_clarification}")
    print(f"Interpretation: {response2.interpretation_summary}")
    print()

    if response2.partial_config:
        config = response2.partial_config
        print("Extracted configuration:")
        print(f"  Model: {config.model_type.value}")
        print(f"  Images: {config.num_images}")
        print(f"  Substructure: {config.substructure.substructure_type.value}")
    print()

    # Example 3: Iterative clarification
    print("\n" + "=" * 70)
    print("SCENARIO 3: Iterative Clarification")
    print("-" * 40)
    prompt3 = "axion vortex lens simulation"
    print(f"User: {prompt3}")
    print()

    response3 = engine.analyze_request(prompt3)

    print(f"Initial confidence: {response3.confidence_score:.0%}")
    print(f"Questions: {len(response3.questions)}")
    print()

    # Simulate collecting user responses
    user_responses = {}
    for question in response3.questions:
        simulated_answer = simulate_user_response(question)
        user_responses[question.question_id] = simulated_answer
        print(f"Q: {question.question_text}")
        print(f"A: {simulated_answer}")
        print()

    # Re-analyze with responses
    final_response = engine.analyze_request(prompt3, user_responses)

    print(f"Final confidence: {final_response.confidence_score:.0%}")
    print(f"Needs clarification: {final_response.needs_clarification}")
    print()

    if final_response.partial_config:
        config = final_response.partial_config
        print("Final configuration:")
        print(f"  Model: {config.model_type.value}")
        print(f"  Images: {config.num_images}")
        print(f"  Substructure: {config.substructure.substructure_type.value}")


def demonstrate_parameter_parsing():
    """Demonstrate natural language parameter parsing."""
    print("\n" + "=" * 70)
    print("Natural Language Parameter Parsing")
    print("=" * 70)
    print()

    parser = NaturalLanguageParser()

    test_prompts = [
        "Generate 100 images",
        "Model III HST simulation",
        "CDM lens with z_lens=0.8 and z_source=1.5",
        "axion mass 1e-23 eV vortex simulation",
        "10^12 solar mass halo simulation",
        "64x64 pixel resolution Euclid-like lens",
        "Generate 50 samples with seed 42 for reproducibility",
    ]

    for prompt in test_prompts:
        print(f"Prompt: \"{prompt}\"")
        result = parser.parse(prompt)

        extracted = []
        if result.num_images:
            extracted.append(f"images={result.num_images}")
        if result.model_type:
            extracted.append(f"model={result.model_type.value}")
        if result.substructure_type:
            extracted.append(f"sub={result.substructure_type.value}")
        if result.z_lens:
            extracted.append(f"z_lens={result.z_lens}")
        if result.z_source:
            extracted.append(f"z_source={result.z_source}")
        if result.axion_mass:
            extracted.append(f"axion_mass={result.axion_mass:.0e}")
        if result.halo_mass:
            extracted.append(f"halo_mass={result.halo_mass:.0e}")
        if result.resolution:
            extracted.append(f"resolution={result.resolution}")
        if result.random_seed:
            extracted.append(f"seed={result.random_seed}")

        print(f"  Extracted: {', '.join(extracted) if extracted else 'none'}")
        print(f"  Confidence: {result.confidence:.0%}")
        print()


def demonstrate_full_workflow():
    """Demonstrate the complete end-to-end workflow."""
    print("\n" + "=" * 70)
    print("Complete End-to-End Workflow")
    print("=" * 70)
    print()

    # Step 1: Parse request
    print("Step 1: Parse natural language request")
    print("-" * 40)

    prompt = "Generate 5 cold dark matter lens images using the Euclid model"
    print(f"User: {prompt}")
    print()

    engine = ClarificationEngine()
    response = engine.analyze_request(prompt)

    print(f"Parsed with {response.confidence_score:.0%} confidence")
    print(f"Interpretation: {response.interpretation_summary}")
    print()

    # Step 2: Handle any clarifications
    print("Step 2: Clarification (if needed)")
    print("-" * 40)

    config = response.partial_config
    if response.needs_clarification:
        print("Clarification needed for:")
        for q in response.questions:
            answer = simulate_user_response(q)
            print(f"  - {q.question_id}: {answer}")

        # Apply responses
        user_responses = {
            q.question_id: simulate_user_response(q)
            for q in response.questions
        }
        response = engine.analyze_request(prompt, user_responses)
        config = response.partial_config
    else:
        print("No clarification needed!")
    print()

    # Step 3: Validate configuration
    print("Step 3: Validate configuration")
    print("-" * 40)

    print(f"Model: {config.model_type.value}")
    print(f"Images: {config.num_images}")
    print(f"Substructure: {config.substructure.substructure_type.value}")
    print(f"z_lens: {config.cosmology.z_lens}")
    print(f"z_source: {config.cosmology.z_source}")

    # Check validity
    is_valid = config.cosmology.z_source > config.cosmology.z_lens
    print(f"Valid configuration: {is_valid}")
    print()

    # Step 4: Run simulation
    print("Step 4: Run simulation")
    print("-" * 40)

    simulator = create_simulator(mock_mode=True)
    output = simulator.run_simulation(config)

    if output.success:
        print(f"SUCCESS: Generated {output.num_images_generated} images")
        print(f"Duration: {output.metadata.duration_seconds:.3f}s")
        print(f"Simulation ID: {output.metadata.simulation_id}")
        print()

        print("Image statistics:")
        for i, img in enumerate(output.images):
            print(f"  Image {i}: {img.width}x{img.height}, "
                  f"mean={img.mean_value:.4f}, range=[{img.min_value:.4f}, {img.max_value:.4f}]")
    else:
        print(f"FAILED: {output.error_message}")


def main():
    """Run all demonstrations."""
    demonstrate_clarification_workflow()
    demonstrate_parameter_parsing()
    demonstrate_full_workflow()

    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
