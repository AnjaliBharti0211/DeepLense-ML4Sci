"""
Configuration management for DeepLense Agent.

Handles API keys, model selection, and provider configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelProvider(str, Enum):
    """Supported model providers."""

    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""

    provider: ModelProvider
    api_key: str
    model_name: str
    base_url: str | None = None

    @classmethod
    def groq(
        cls,
        api_key: str | None = None,
        model_name: str = "mixtral-8x7b-32768",
    ) -> ProviderConfig:
        """Create Groq provider configuration."""
        return cls(
            provider=ModelProvider.GROQ,
            api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
            model_name=model_name,
            base_url="https://api.groq.com/openai/v1",
        )

    @classmethod
    def openai(
        cls,
        api_key: str | None = None,
        model_name: str = "gpt-4o-mini",
    ) -> ProviderConfig:
        """Create OpenAI provider configuration."""
        return cls(
            provider=ModelProvider.OPENAI,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            model_name=model_name,
            base_url=None,
        )

    @classmethod
    def anthropic(
        cls,
        api_key: str | None = None,
        model_name: str = "claude-sonnet-4-20250514",
    ) -> ProviderConfig:
        """Create Anthropic provider configuration."""
        return cls(
            provider=ModelProvider.ANTHROPIC,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            model_name=model_name,
            base_url=None,
        )

    @classmethod
    def google(
        cls,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
    ) -> ProviderConfig:
        """Create Google provider configuration."""
        return cls(
            provider=ModelProvider.GOOGLE,
            api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""),
            model_name=model_name,
            base_url=None,
        )


# Default configuration uses Google
DEFAULT_CONFIG = ProviderConfig.google()


def get_model_config() -> ProviderConfig:
    """Get the current model configuration."""
    # Check environment for provider override
    provider_name = os.environ.get("DEEPLENSE_PROVIDER", "google").lower()

    if provider_name == "openai":
        return ProviderConfig.openai()
    elif provider_name == "anthropic":
        return ProviderConfig.anthropic()
    elif provider_name == "groq":
        return ProviderConfig.groq()
    else:
        return ProviderConfig.google()
