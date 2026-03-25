"""
Human-in-the-loop clarification engine.

This module provides intelligent parameter clarification for ambiguous
or incomplete simulation requests, with scientific context to help
users make informed decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from deeplense_agent.models import (
    ClarificationQuestion,
    ClarificationResponse,
    CosmologicalParameters,
    DarkMatterType,
    InstrumentConfig,
    InstrumentType,
    MainHaloParameters,
    ModelType,
    SimulationConfig,
    SourceLightParameters,
    SubstructureParameters,
)


@dataclass
class ExtractedParameters:
    """Parameters extracted from natural language."""

    model_type: ModelType | None = None
    num_images: int | None = None
    substructure_type: DarkMatterType | None = None
    instrument: InstrumentType | None = None
    z_lens: float | None = None
    z_source: float | None = None
    halo_mass: float | None = None
    axion_mass: float | None = None
    resolution: int | None = None
    random_seed: int | None = None
    confidence: float = 0.0
    raw_matches: dict[str, str] = field(default_factory=dict)


class NaturalLanguageParser:
    """
    Parser for extracting simulation parameters from natural language.

    This uses pattern matching and heuristics rather than ML models,
    making it fast and deterministic for common cases.
    """

    # Patterns for parameter extraction
    PATTERNS = {
        "num_images": [
            r"(\d+)\s*(?:images?|simulations?|samples?|lens(?:es)?)",
            r"generate\s+(\d+)",
            r"create\s+(\d+)",
            r"(\d+)\s*(?:of\s+them|total)",
        ],
        "model_type": [
            r"model\s*[_\s]?(i{1,3}v?|[1-4]|one|two|three|four)",
            r"(basic|simple|generic)\s*(?:model)?",
            r"(euclid|hst|hubble)\s*(?:model|style|like)?",
        ],
        "substructure": [
            r"\b(cdm|cold\s*dark\s*matter|wimp)\b",
            r"\b(axion|vortex|uldm|ultralight|fuzzy)\b",
            r"\b(no\s*sub(?:structure)?|clean|smooth|without\s*sub)\b",
        ],
        "z_lens": [
            r"lens\s*(?:redshift|z)\s*[=:]?\s*(\d*\.?\d+)",
            r"z_?(?:lens|halo)\s*[=:]?\s*(\d*\.?\d+)",
            r"halo\s*(?:at|redshift)\s*[=:]?\s*(\d*\.?\d+)",
        ],
        "z_source": [
            r"source\s*(?:redshift|z)\s*[=:]?\s*(\d*\.?\d+)",
            r"z_?(?:source|gal(?:axy)?)\s*[=:]?\s*(\d*\.?\d+)",
            r"galaxy\s*(?:at|redshift)\s*[=:]?\s*(\d*\.?\d+)",
        ],
        "halo_mass": [
            r"(?:halo\s*)?mass\s*[=:]?\s*(\d+(?:\.\d+)?)\s*[*x×]?\s*10\^?(\d+)",
            r"(\d+(?:\.\d+)?)\s*[*x×]?\s*10\^?(\d+)\s*(?:solar\s*masses?|m_?sun|msol)",
        ],
        "axion_mass": [
            r"axion\s*mass\s*[=:]?\s*(\d+(?:\.\d+)?)\s*[*x×]?\s*10\^?(-?\d+)",
            r"m_?a(?:xion)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*[*x×]?\s*10\^?(-?\d+)\s*ev",
        ],
        "resolution": [
            r"(\d+)\s*[x×]\s*\d+\s*(?:pixels?|resolution)?",
            r"resolution\s*[=:]?\s*(\d+)",
            r"(\d+)\s*(?:px|pixels?)\s*(?:resolution)?",
        ],
        "seed": [
            r"(?:random\s*)?seed\s*[=:]?\s*(\d+)",
            r"reproducib(?:le|ility).*?(\d+)",
        ],
    }

    MODEL_MAPPING = {
        "1": ModelType.MODEL_I,
        "i": ModelType.MODEL_I,
        "one": ModelType.MODEL_I,
        "basic": ModelType.MODEL_I,
        "simple": ModelType.MODEL_I,
        "generic": ModelType.MODEL_I,
        "2": ModelType.MODEL_II,
        "ii": ModelType.MODEL_II,
        "two": ModelType.MODEL_II,
        "euclid": ModelType.MODEL_II,
        "3": ModelType.MODEL_III,
        "iii": ModelType.MODEL_III,
        "three": ModelType.MODEL_III,
        "hst": ModelType.MODEL_III,
        "hubble": ModelType.MODEL_III,
        "4": ModelType.MODEL_IV,
        "iv": ModelType.MODEL_IV,
        "four": ModelType.MODEL_IV,
    }

    def parse(self, text: str) -> ExtractedParameters:
        """Parse natural language text to extract simulation parameters."""
        text_lower = text.lower()
        params = ExtractedParameters()
        confidence_factors: list[float] = []

        # Extract number of images
        for pattern in self.PATTERNS["num_images"]:
            match = re.search(pattern, text_lower)
            if match:
                params.num_images = int(match.group(1))
                params.raw_matches["num_images"] = match.group(0)
                confidence_factors.append(0.9)
                break

        # Extract model type
        for pattern in self.PATTERNS["model_type"]:
            match = re.search(pattern, text_lower)
            if match:
                model_key = match.group(1).lower().replace(" ", "")
                if model_key in self.MODEL_MAPPING:
                    params.model_type = self.MODEL_MAPPING[model_key]
                    params.raw_matches["model_type"] = match.group(0)
                    confidence_factors.append(0.95)
                break

        # Extract substructure type
        for pattern in self.PATTERNS["substructure"]:
            match = re.search(pattern, text_lower)
            if match:
                params.substructure_type = DarkMatterType.from_natural_language(
                    match.group(1)
                )
                params.raw_matches["substructure"] = match.group(0)
                confidence_factors.append(0.95)
                break

        # Extract redshifts
        for pattern in self.PATTERNS["z_lens"]:
            match = re.search(pattern, text_lower)
            if match:
                params.z_lens = float(match.group(1))
                params.raw_matches["z_lens"] = match.group(0)
                confidence_factors.append(0.9)
                break

        for pattern in self.PATTERNS["z_source"]:
            match = re.search(pattern, text_lower)
            if match:
                params.z_source = float(match.group(1))
                params.raw_matches["z_source"] = match.group(0)
                confidence_factors.append(0.9)
                break

        # Extract halo mass (scientific notation)
        for pattern in self.PATTERNS["halo_mass"]:
            match = re.search(pattern, text_lower)
            if match:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                params.halo_mass = mantissa * (10**exponent)
                params.raw_matches["halo_mass"] = match.group(0)
                confidence_factors.append(0.85)
                break

        # Extract axion mass
        for pattern in self.PATTERNS["axion_mass"]:
            match = re.search(pattern, text_lower)
            if match:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                params.axion_mass = mantissa * (10**exponent)
                params.raw_matches["axion_mass"] = match.group(0)
                confidence_factors.append(0.9)
                break

        # Extract resolution
        for pattern in self.PATTERNS["resolution"]:
            match = re.search(pattern, text_lower)
            if match:
                params.resolution = int(match.group(1))
                params.raw_matches["resolution"] = match.group(0)
                confidence_factors.append(0.85)
                break

        # Extract seed
        for pattern in self.PATTERNS["seed"]:
            match = re.search(pattern, text_lower)
            if match:
                params.random_seed = int(match.group(1))
                params.raw_matches["seed"] = match.group(0)
                confidence_factors.append(0.95)
                break

        # Infer instrument from model type if not explicitly set
        if params.model_type:
            if params.model_type == ModelType.MODEL_II:
                params.instrument = InstrumentType.EUCLID
            elif params.model_type == ModelType.MODEL_III:
                params.instrument = InstrumentType.HST
            elif params.model_type == ModelType.MODEL_I:
                params.instrument = InstrumentType.GENERIC

        # Calculate overall confidence
        if confidence_factors:
            params.confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            params.confidence = 0.1

        return params


class ClarificationEngine:
    """
    Engine for generating clarification questions based on ambiguous
    or incomplete simulation requests.

    This implements the human-in-the-loop component, asking targeted
    questions with scientific context to help users specify parameters.
    """

    SCIENTIFIC_CONTEXT = {
        "model_type": (
            "Model I uses 150x150 pixel simulation with Gaussian PSF (arXiv:2008.12731), "
            "suitable for detailed studies with SNR~20. Model II emulates Euclid survey "
            "characteristics (64x64), while Model III emulates HST observations. "
            "Model I achieved ResNet-18 AUC of 0.996 in Alexander et al. 2021."
        ),
        "substructure": (
            "From arXiv:2008.12731 and arXiv:1909.07346: CDM produces point-mass subhalos "
            "with Poisson(μ=25) distribution and mass range 10^6-10^10 M_sun (β=-1.9 slope). "
            "Axion/vortex (ULDM) creates wave-like interference patterns; for m=10^-23 eV, "
            "~340 vortices expected in M31-like halos. Classification AUC: CDM=0.985, vortex=0.968."
        ),
        "num_images": (
            "From arXiv:2008.12731: Training used 25,000 images per class (75,000 total). "
            "For quick validation, 10-100 images suffice. ML training requires 1000+ per class. "
            "Alexander et al. 2020 used 150,000 training + 15,000 validation images."
        ),
        "redshift": (
            "Standard configuration from papers: z_lens=0.5 (fixed) or uniform[0.4,0.6], "
            "z_source=1.0 (fixed) or uniform[0.8,1.2]. These values match typical strong lens "
            "systems and ensure adequate lensing geometry (z_source > z_lens required)."
        ),
        "halo_mass": (
            "From arXiv:2008.12731: M_TOT = 10^12 M_sun (fixed) for galaxy-scale lensing. "
            "This produces Einstein radius θ_E ≈ 1.2 arcsec (arXiv:1909.07346). "
            "Substructure comprises ~1% of total halo mass in CDM scenarios."
        ),
        "axion_mass": (
            "From arXiv:2008.12731: Axion mass m determines de Broglie wavelength "
            "λ_dB ≈ 0.6 kpc × (10^-22 eV / m). For m=10^-23 eV, expect ~340 vortices "
            "in M31. Detection threshold: substructure mass >0.3% of halo mass. "
            "Vortex mass fraction typically 3-5.5% with length uniform[0.5,2.0]."
        ),
        "einstein_radius": (
            "From arXiv:1909.07346: Einstein radius θ_E = 1.2 arcsec for M=10^12 M_sun "
            "at z_lens=0.5. This sets the characteristic scale of lensed arcs and "
            "multiple images in strong gravitational lensing systems."
        ),
        "external_shear": (
            "From papers: External shear γ_ext sampled uniformly from [0.0, 0.3]. "
            "Shear accounts for tidal gravitational effects from nearby mass "
            "concentrations and large-scale structure along the line of sight."
        ),
        "power_law_slope": (
            "From arXiv:2008.12731: Subhalo mass function follows dN/dM ∝ M^β with "
            "β = -1.9. This matches theoretical CDM predictions and N-body simulations. "
            "Steeper slopes (more negative β) produce more low-mass subhalos."
        ),
    }

    def __init__(self, parser: NaturalLanguageParser | None = None):
        self.parser = parser or NaturalLanguageParser()

    def analyze_request(
        self, prompt: str, previous_responses: dict[str, str] | None = None
    ) -> ClarificationResponse:
        """
        Analyze a natural language request and generate clarification questions.

        Args:
            prompt: The user's natural language request.
            previous_responses: Responses to previous clarification questions.

        Returns:
            ClarificationResponse with questions and partial configuration.
        """
        # Parse the request
        extracted = self.parser.parse(prompt)

        # Build partial configuration from extracted parameters
        partial_config = self._build_partial_config(extracted, previous_responses)

        # Generate clarification questions for missing/ambiguous parameters
        questions = self._generate_questions(extracted, previous_responses)

        # Determine if we need clarification
        needs_clarification = len(questions) > 0 and extracted.confidence < 0.8

        # Generate interpretation summary
        summary = self._generate_summary(extracted, partial_config)

        return ClarificationResponse(
            needs_clarification=needs_clarification,
            questions=questions,
            partial_config=partial_config,
            confidence_score=extracted.confidence,
            interpretation_summary=summary,
        )

    def _build_partial_config(
        self,
        extracted: ExtractedParameters,
        responses: dict[str, str] | None,
    ) -> SimulationConfig:
        """Build a partial configuration from extracted and clarified parameters."""
        # Start with defaults
        config_kwargs: dict[str, Any] = {}

        # Apply extracted values
        if extracted.model_type:
            config_kwargs["model_type"] = extracted.model_type

        if extracted.num_images:
            config_kwargs["num_images"] = extracted.num_images

        if extracted.random_seed:
            config_kwargs["random_seed"] = extracted.random_seed

        # Build cosmology parameters
        cosmo_kwargs = {}
        if extracted.z_lens:
            cosmo_kwargs["z_lens"] = extracted.z_lens
        if extracted.z_source:
            cosmo_kwargs["z_source"] = extracted.z_source

        if cosmo_kwargs:
            config_kwargs["cosmology"] = CosmologicalParameters(**cosmo_kwargs)

        # Build substructure parameters
        sub_kwargs = {}
        if extracted.substructure_type:
            sub_kwargs["substructure_type"] = extracted.substructure_type
        if extracted.axion_mass:
            sub_kwargs["axion_mass"] = extracted.axion_mass

        if sub_kwargs:
            config_kwargs["substructure"] = SubstructureParameters(**sub_kwargs)

        # Build main halo parameters
        if extracted.halo_mass:
            config_kwargs["main_halo"] = MainHaloParameters(
                halo_mass=extracted.halo_mass
            )

        # Apply responses to previous questions
        if responses:
            config_kwargs = self._apply_responses(config_kwargs, responses)

        return SimulationConfig(**config_kwargs)

    def _apply_responses(
        self, config_kwargs: dict[str, Any], responses: dict[str, str]
    ) -> dict[str, Any]:
        """Apply user responses to clarification questions."""
        for question_id, response in responses.items():
            if question_id == "model_type":
                model_map = {
                    "Model I": ModelType.MODEL_I,
                    "Model II": ModelType.MODEL_II,
                    "Model III": ModelType.MODEL_III,
                }
                if response in model_map:
                    config_kwargs["model_type"] = model_map[response]

            elif question_id == "substructure":
                sub_map = {
                    "CDM": DarkMatterType.CDM,
                    "Axion/Vortex": DarkMatterType.AXION,
                    "No substructure": DarkMatterType.NO_SUBSTRUCTURE,
                }
                if response in sub_map:
                    if "substructure" not in config_kwargs:
                        config_kwargs["substructure"] = SubstructureParameters()
                    config_kwargs["substructure"] = SubstructureParameters(
                        substructure_type=sub_map[response],
                        **(
                            config_kwargs.get("substructure", SubstructureParameters())
                            .model_dump(exclude={"substructure_type"})
                        ),
                    )

            elif question_id == "num_images":
                try:
                    config_kwargs["num_images"] = int(response)
                except ValueError:
                    pass

        return config_kwargs

    def _generate_questions(
        self,
        extracted: ExtractedParameters,
        responses: dict[str, str] | None,
    ) -> list[ClarificationQuestion]:
        """Generate clarification questions for missing/ambiguous parameters."""
        questions: list[ClarificationQuestion] = []
        already_answered = set(responses.keys()) if responses else set()

        # Check for missing essential parameters
        if extracted.model_type is None and "model_type" not in already_answered:
            questions.append(
                ClarificationQuestion(
                    question_id="model_type",
                    question_text="Which simulation model would you like to use?",
                    category="model",
                    options=["Model I (150x150, basic)", "Model II (Euclid-like)", "Model III (HST-like)"],
                    default_value="Model I (150x150, basic)",
                    required=True,
                    scientific_context=self.SCIENTIFIC_CONTEXT["model_type"],
                )
            )

        if (
            extracted.substructure_type is None
            and "substructure" not in already_answered
        ):
            questions.append(
                ClarificationQuestion(
                    question_id="substructure",
                    question_text="What type of dark matter substructure should be simulated?",
                    category="substructure",
                    options=["CDM (Cold Dark Matter)", "Axion/Vortex (Ultralight)", "No substructure (clean lens)"],
                    default_value="CDM (Cold Dark Matter)",
                    required=True,
                    scientific_context=self.SCIENTIFIC_CONTEXT["substructure"],
                )
            )

        if extracted.num_images is None and "num_images" not in already_answered:
            questions.append(
                ClarificationQuestion(
                    question_id="num_images",
                    question_text="How many lens images would you like to generate?",
                    category="quantity",
                    options=["1 (single sample)", "10 (quick test)", "100 (small dataset)", "1000 (training set)"],
                    default_value="10 (quick test)",
                    required=False,
                    scientific_context=self.SCIENTIFIC_CONTEXT["num_images"],
                )
            )

        # Axion-specific question
        if (
            extracted.substructure_type == DarkMatterType.AXION
            and extracted.axion_mass is None
            and "axion_mass" not in already_answered
        ):
            questions.append(
                ClarificationQuestion(
                    question_id="axion_mass",
                    question_text="What axion mass would you like to simulate?",
                    category="substructure",
                    options=["1e-24 eV (large structures)", "1e-23 eV (typical)", "1e-22 eV (small structures)"],
                    default_value="1e-23 eV (typical)",
                    required=False,
                    scientific_context=self.SCIENTIFIC_CONTEXT["axion_mass"],
                )
            )

        return questions

    def _generate_summary(
        self, extracted: ExtractedParameters, config: SimulationConfig
    ) -> str:
        """Generate a human-readable summary of the interpretation."""
        parts = []

        if config.num_images > 1:
            parts.append(f"Generate {config.num_images} images")
        else:
            parts.append("Generate 1 image")

        model_names = {
            ModelType.MODEL_I: "Model I (150x150, basic)",
            ModelType.MODEL_II: "Model II (64x64, Euclid-like)",
            ModelType.MODEL_III: "Model III (64x64, HST-like)",
            ModelType.MODEL_IV: "Model IV (64x64, RGB, real galaxies)",
        }
        parts.append(f"using {model_names.get(config.model_type, str(config.model_type))}")

        sub_names = {
            DarkMatterType.CDM: "CDM substructure",
            DarkMatterType.AXION: "axion/vortex substructure",
            DarkMatterType.NO_SUBSTRUCTURE: "no substructure",
        }
        parts.append(
            f"with {sub_names.get(config.substructure.substructure_type, 'default substructure')}"
        )

        parts.append(
            f"at z_lens={config.cosmology.z_lens:.1f}, z_source={config.cosmology.z_source:.1f}"
        )

        return ", ".join(parts) + "."


def create_clarification_engine() -> ClarificationEngine:
    """Factory function to create a clarification engine."""
    return ClarificationEngine()
