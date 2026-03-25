"""
Tests for the clarification engine and natural language parser.
"""

import pytest

from deeplense_agent.clarification import (
    ClarificationEngine,
    ExtractedParameters,
    NaturalLanguageParser,
)
from deeplense_agent.models import DarkMatterType, ModelType


class TestNaturalLanguageParser:
    """Tests for NaturalLanguageParser class."""

    @pytest.fixture
    def parser(self):
        return NaturalLanguageParser()

    def test_extract_num_images(self, parser):
        """Test extraction of number of images."""
        result = parser.parse("Generate 100 images")
        assert result.num_images == 100

        result = parser.parse("create 50 lens simulations")
        assert result.num_images == 50

        result = parser.parse("10 samples of gravitational lensing")
        assert result.num_images == 10

    def test_extract_model_type(self, parser):
        """Test extraction of model type."""
        result = parser.parse("Use Model I for this simulation")
        assert result.model_type == ModelType.MODEL_I

        result = parser.parse("Euclid model please")
        assert result.model_type == ModelType.MODEL_II

        result = parser.parse("HST-like simulation")
        assert result.model_type == ModelType.MODEL_III

        result = parser.parse("using model_ii configuration")
        assert result.model_type == ModelType.MODEL_II

    def test_extract_substructure(self, parser):
        """Test extraction of substructure type."""
        result = parser.parse("CDM lens simulation")
        assert result.substructure_type == DarkMatterType.CDM

        result = parser.parse("cold dark matter subhalos")
        assert result.substructure_type == DarkMatterType.CDM

        result = parser.parse("axion vortex lens")
        assert result.substructure_type == DarkMatterType.AXION

        result = parser.parse("ultralight dark matter")
        assert result.substructure_type == DarkMatterType.AXION

        result = parser.parse("clean lens without substructure")
        assert result.substructure_type == DarkMatterType.NO_SUBSTRUCTURE

    def test_extract_redshifts(self, parser):
        """Test extraction of redshift values."""
        result = parser.parse("lens redshift = 0.5")
        assert result.z_lens == pytest.approx(0.5)

        result = parser.parse("z_lens: 0.3 and z_source: 1.5")
        assert result.z_lens == pytest.approx(0.3)
        assert result.z_source == pytest.approx(1.5)

        result = parser.parse("halo at redshift 0.8")
        assert result.z_lens == pytest.approx(0.8)

        result = parser.parse("source galaxy at z=2.0")
        assert result.z_source == pytest.approx(2.0)

    def test_extract_halo_mass(self, parser):
        """Test extraction of halo mass."""
        result = parser.parse("mass = 1e12 solar masses")
        assert result.halo_mass == pytest.approx(1e12)

        result = parser.parse("10^13 M_sun halo")
        assert result.halo_mass == pytest.approx(1e13)

    def test_extract_axion_mass(self, parser):
        """Test extraction of axion mass."""
        result = parser.parse("axion mass = 1e-23 eV")
        assert result.axion_mass == pytest.approx(1e-23)

        result = parser.parse("m_axion: 1e-22 eV ultralight")
        assert result.axion_mass == pytest.approx(1e-22)

    def test_extract_resolution(self, parser):
        """Test extraction of resolution."""
        result = parser.parse("64x64 pixel resolution")
        assert result.resolution == 64

        result = parser.parse("resolution: 150")
        assert result.resolution == 150

    def test_extract_seed(self, parser):
        """Test extraction of random seed."""
        result = parser.parse("seed = 42")
        assert result.random_seed == 42

        result = parser.parse("reproducible with seed 12345")
        assert result.random_seed == 12345

    def test_confidence_calculation(self, parser):
        """Test confidence calculation."""
        # Specific request should have high confidence
        result = parser.parse("Generate 10 CDM lens images using Model I")
        assert result.confidence > 0.7

        # Vague request should have lower confidence
        result = parser.parse("some lens images")
        assert result.confidence < 0.5

    def test_complex_prompt(self, parser):
        """Test parsing of complex prompts."""
        prompt = (
            "Generate 100 CDM lens images using Model II (Euclid) "
            "with z_lens=0.5, z_source=1.5, seed=42"
        )
        result = parser.parse(prompt)

        assert result.num_images == 100
        assert result.substructure_type == DarkMatterType.CDM
        assert result.model_type == ModelType.MODEL_II
        assert result.z_lens == pytest.approx(0.5)
        assert result.z_source == pytest.approx(1.5)
        assert result.random_seed == 42
        assert result.confidence > 0.8


class TestClarificationEngine:
    """Tests for ClarificationEngine class."""

    @pytest.fixture
    def engine(self):
        return ClarificationEngine()

    def test_analyze_complete_request(self, engine):
        """Test analysis of complete, unambiguous request."""
        prompt = "Generate 10 CDM lens images using Model I"
        response = engine.analyze_request(prompt)

        assert response.partial_config is not None
        assert response.interpretation_summary != ""
        assert response.confidence_score > 0.5

    def test_analyze_ambiguous_request(self, engine):
        """Test analysis of ambiguous request requiring clarification."""
        prompt = "I want lens images"
        response = engine.analyze_request(prompt)

        assert response.needs_clarification
        assert len(response.questions) > 0

    def test_clarification_questions_structure(self, engine):
        """Test structure of clarification questions."""
        prompt = "generate some simulations"
        response = engine.analyze_request(prompt)

        for question in response.questions:
            assert question.question_id is not None
            assert question.question_text is not None
            assert question.category in ["model", "substructure", "cosmology", "instrument", "quantity"]

    def test_apply_responses(self, engine):
        """Test applying user responses to clarification questions."""
        prompt = "lens simulation"
        initial_response = engine.analyze_request(prompt)

        # Simulate user responses
        user_responses = {
            "model_type": "Model I (150x150, basic)",
            "substructure": "CDM (Cold Dark Matter)",
            "num_images": "10 (quick test)",
        }

        # Re-analyze with responses
        final_response = engine.analyze_request(prompt, user_responses)

        # Should be more confident now
        assert final_response.confidence_score >= initial_response.confidence_score
        assert final_response.partial_config is not None

    def test_partial_config_from_prompt(self, engine):
        """Test that partial config is correctly built from prompt."""
        prompt = "50 axion lens images with z_lens=0.3"
        response = engine.analyze_request(prompt)

        config = response.partial_config
        assert config is not None
        assert config.num_images == 50
        assert config.substructure.substructure_type == DarkMatterType.AXION
        assert config.cosmology.z_lens == pytest.approx(0.3)

    def test_scientific_context_provided(self, engine):
        """Test that scientific context is provided for questions."""
        prompt = "lens images"
        response = engine.analyze_request(prompt)

        # At least some questions should have scientific context
        questions_with_context = [
            q for q in response.questions if q.scientific_context
        ]
        assert len(questions_with_context) > 0

    def test_axion_specific_questions(self, engine):
        """Test that axion-specific questions are asked when needed."""
        prompt = "axion vortex simulation"
        response = engine.analyze_request(prompt)

        # May need to ask about axion mass
        question_ids = [q.question_id for q in response.questions]
        # Should either have axion_mass question or have default set
        config = response.partial_config
        assert (
            "axion_mass" in question_ids or
            config.substructure.axion_mass is not None
        )

    def test_interpretation_summary(self, engine):
        """Test that interpretation summary is generated."""
        prompt = "100 CDM images Model II z_lens=0.5"
        response = engine.analyze_request(prompt)

        summary = response.interpretation_summary
        assert "100" in summary or "Model II" in summary.lower()

    def test_question_options(self, engine):
        """Test that questions have appropriate options."""
        prompt = "generate lenses"
        response = engine.analyze_request(prompt)

        for question in response.questions:
            if question.options:
                assert len(question.options) >= 2
                # Options should be distinct
                assert len(question.options) == len(set(question.options))


class TestClarificationEngineEdgeCases:
    """Edge case tests for ClarificationEngine."""

    @pytest.fixture
    def engine(self):
        return ClarificationEngine()

    def test_empty_prompt(self, engine):
        """Test handling of empty prompt."""
        response = engine.analyze_request("")
        assert response.needs_clarification
        assert response.confidence_score < 0.3

    def test_very_long_prompt(self, engine):
        """Test handling of very long prompts."""
        prompt = "generate " + "CDM lens " * 100 + "images"
        response = engine.analyze_request(prompt)
        assert response.partial_config is not None

    def test_numeric_ranges(self, engine):
        """Test parsing of numeric ranges."""
        prompt = "z_lens between 0.3 and 0.5"
        response = engine.analyze_request(prompt)
        # Should extract at least one value
        assert response.partial_config is not None

    def test_case_insensitivity(self, engine):
        """Test case insensitive parsing."""
        prompts = [
            "CDM simulation",
            "cdm simulation",
            "CdM SIMULATION",
        ]
        results = [engine.analyze_request(p) for p in prompts]

        # All should extract CDM substructure
        for r in results:
            assert (
                r.partial_config.substructure.substructure_type ==
                DarkMatterType.CDM
            )

    def test_multiple_conflicting_values(self, engine):
        """Test handling of conflicting values."""
        # Two different model specifications - should pick one
        prompt = "Model I and Model II simulation"
        response = engine.analyze_request(prompt)
        # Should have some model type set
        assert response.partial_config.model_type is not None
