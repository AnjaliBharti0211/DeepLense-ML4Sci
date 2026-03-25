"""
Tests for the Pydantic AI agent.
"""

import pytest
import asyncio

from deeplense_agent.agent import (
    AgentDependencies,
    DeepLenseAgent,
    SyncDeepLenseAgent,
    create_agent,
)
from deeplense_agent.models import (
    DarkMatterType,
    ModelType,
    SimulationConfig,
    SubstructureParameters,
)
from deeplense_agent.simulator import create_simulator


class TestDeepLenseAgent:
    """Tests for DeepLenseAgent class."""

    @pytest.fixture
    def mock_agent(self):
        """Create an agent in mock mode."""
        return create_agent(mock_mode=True)

    @pytest.mark.asyncio
    async def test_generate_from_config(self, mock_agent):
        """Test generating simulations from config."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=2,
        )
        output = await mock_agent.generate_from_config(config)

        assert output.success
        assert output.num_images_generated == 2

    @pytest.mark.asyncio
    async def test_generate_cdm(self, mock_agent):
        """Test generating CDM simulations."""
        config = SimulationConfig(
            num_images=1,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.CDM,
            ),
        )
        output = await mock_agent.generate_from_config(config)

        assert output.success

    @pytest.mark.asyncio
    async def test_generate_axion(self, mock_agent):
        """Test generating axion simulations."""
        config = SimulationConfig(
            num_images=1,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.AXION,
                axion_mass=1e-23,
            ),
        )
        output = await mock_agent.generate_from_config(config)

        assert output.success


class TestSyncDeepLenseAgent:
    """Tests for synchronous agent wrapper."""

    @pytest.fixture
    def sync_agent(self):
        """Create a synchronous agent in mock mode."""
        return SyncDeepLenseAgent(mock_mode=True)

    def test_generate_from_config_sync(self, sync_agent):
        """Test synchronous config-based generation."""
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=2,
        )
        output = sync_agent.generate_from_config(config)

        assert output.success
        assert output.num_images_generated == 2


class TestAgentDependencies:
    """Tests for AgentDependencies class."""

    def test_default_dependencies(self):
        """Test default dependency creation."""
        deps = AgentDependencies()

        assert deps.simulator is not None
        assert deps.clarification_engine is not None
        assert deps.state is not None
        assert deps.auto_approve_high_confidence is True
        assert deps.confidence_threshold == 0.85

    def test_custom_simulator(self):
        """Test using custom simulator."""
        simulator = create_simulator(mock_mode=True)
        deps = AgentDependencies(simulator=simulator)

        assert deps.simulator is simulator

    def test_human_callback(self):
        """Test setting human callback."""
        def mock_callback(questions):
            return {"question_id": "answer"}

        deps = AgentDependencies(human_callback=mock_callback)
        assert deps.human_callback is not None


class TestAgentFactory:
    """Tests for agent factory function."""

    def test_create_default_agent(self):
        """Test creating default agent."""
        agent = create_agent()
        assert agent is not None

    def test_create_mock_agent(self):
        """Test creating mock agent."""
        agent = create_agent(mock_mode=True)
        assert agent.simulator._mock_mode is True

    def test_create_with_callback(self):
        """Test creating agent with human callback."""
        def callback(questions):
            return {}

        agent = create_agent(human_callback=callback)
        assert agent.deps.human_callback is callback


class TestAgentIntegration:
    """Integration tests for the agent."""

    @pytest.fixture
    def mock_agent(self):
        return create_agent(mock_mode=True)

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_agent):
        """Test complete workflow from config to output."""
        # Create config
        config = SimulationConfig(
            model_type=ModelType.MODEL_I,
            num_images=3,
            random_seed=42,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.CDM,
            ),
        )

        # Generate
        output = await mock_agent.generate_from_config(config)

        # Verify
        assert output.success
        assert output.num_images_generated == 3
        assert output.metadata is not None
        assert output.metadata.random_state_used == 42

        # Check images
        for img in output.images:
            assert img.width == 150
            assert img.height == 150
            assert img.base64_png is not None

    @pytest.mark.asyncio
    async def test_multiple_model_types(self, mock_agent):
        """Test generating with different model types."""
        model_types = [ModelType.MODEL_I, ModelType.MODEL_II, ModelType.MODEL_III]
        expected_resolutions = [150, 64, 64]

        for model_type, expected_res in zip(model_types, expected_resolutions):
            config = SimulationConfig(
                model_type=model_type,
                num_images=1,
            )
            output = await mock_agent.generate_from_config(config)

            assert output.success
            assert output.images[0].width == expected_res
            assert output.images[0].height == expected_res

    @pytest.mark.asyncio
    async def test_batch_generation(self, mock_agent):
        """Test batch generation."""
        config = SimulationConfig(num_images=10)
        output = await mock_agent.generate_from_config(config)

        assert output.success
        assert len(output.images) == 10

        # All images should be distinct
        mean_values = [img.mean_value for img in output.images]
        # With random generation, mean values should vary
        assert len(set(mean_values)) > 1


class TestAgentState:
    """Tests for agent state management."""

    @pytest.fixture
    def mock_agent(self):
        return create_agent(mock_mode=True)

    @pytest.mark.asyncio
    async def test_simulation_history(self, mock_agent):
        """Test that simulations are recorded in history."""
        # Run first simulation
        config1 = SimulationConfig(num_images=1)
        await mock_agent.generate_from_config(config1)

        # Run second simulation
        config2 = SimulationConfig(
            num_images=2,
            substructure=SubstructureParameters(
                substructure_type=DarkMatterType.AXION
            ),
        )
        await mock_agent.generate_from_config(config2)

        # Check history
        history = mock_agent.deps.state.completed_simulations
        assert len(history) == 2
        assert history[0].num_images_generated == 1
        assert history[1].num_images_generated == 2
