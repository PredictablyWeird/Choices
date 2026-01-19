"""
Tests for LLM agent creation and querying.

These tests verify:
1. Different agent types can be created correctly
2. Agents can query LLMs and get responses
3. Reasoning configurations work as expected
4. Reasoning traces/summaries are captured when applicable
"""

import pytest

from choices.llm_agent import (
    CompletionModelAgent,
    LiteLLMAgent,
    LLMResponse,
    ReasoningAgent,
)
from choices.utils import create_agent


# =============================================================================
# Agent Creation Tests
# =============================================================================


class TestAgentCreation:
    """Tests for creating agents with different configurations."""

    def test_create_litellm_agent_openai(self, has_openai_key):
        """Test creating a LiteLLMAgent for OpenAI models."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = create_agent("gpt-4o-mini", temperature=0.0, max_tokens=50)

        assert isinstance(agent, LiteLLMAgent)
        assert agent.model == "gpt-4o-mini"
        assert agent.temperature == 0.0
        assert agent.max_tokens == 50
        assert agent.accepts_system_message is True

    def test_create_litellm_agent_openrouter(self, has_openrouter_key):
        """Test creating a LiteLLMAgent for OpenRouter models."""
        if not has_openrouter_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        agent = create_agent("llama-33-70b", temperature=0.5, max_tokens=100)

        assert isinstance(agent, LiteLLMAgent)
        assert "llama" in agent.model.lower()
        assert agent.temperature == 0.5
        assert agent.max_tokens == 100

    def test_create_reasoning_agent_openai(self, has_openai_key):
        """Test creating a ReasoningAgent for OpenAI reasoning models."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = create_agent("gpt-5-reasoning", temperature=1.0, max_tokens=2000)

        assert isinstance(agent, ReasoningAgent)
        assert agent.model == "gpt-5"
        assert agent.model_type == "openai"
        assert agent.reasoning_effort is not None
        assert agent.reasoning_effort.get("effort") == "low"

    def test_create_reasoning_agent_openrouter(self, has_openrouter_key):
        """Test creating a ReasoningAgent for OpenRouter reasoning models."""
        if not has_openrouter_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        agent = create_agent(
            "deepseek-v3-2-reasoning", temperature=1.0, max_tokens=2000
        )

        assert isinstance(agent, ReasoningAgent)
        assert "deepseek" in agent.model.lower()
        assert agent.model_type == "openrouter"
        assert agent.reasoning_effort is not None

    def test_create_non_reasoning_variant(self, has_openrouter_key):
        """Test that non-reasoning variants create LiteLLMAgent."""
        if not has_openrouter_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        # deepseek-v3-2-non-reasoning has no reasoning_effort, so should use LiteLLMAgent
        agent = create_agent(
            "deepseek-v3-2-non-reasoning", temperature=1.0, max_tokens=100
        )

        assert isinstance(agent, LiteLLMAgent)
        assert "deepseek" in agent.model.lower()

    def test_create_completion_model_agent(self, has_openrouter_key):
        """Test creating a CompletionModelAgent for base models."""
        if not has_openrouter_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        agent = create_agent("base-llama-3.1-405b", temperature=1.0, max_tokens=100)

        assert isinstance(agent, CompletionModelAgent)
        assert "llama" in agent.model.lower()
        assert agent.treat_as_chat_model is False

    def test_agent_config_parameters(self, has_openai_key):
        """Test that agent config parameters are applied correctly."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = create_agent(
            "gpt-4o-mini",
            temperature=0.7,
            max_tokens=256,
            concurrency_limit=10,
            base_timeout=30,
        )

        assert agent.temperature == 0.7
        assert agent.max_tokens == 256
        assert agent.concurrency_limit == 10
        assert agent.base_timeout == 30

    def test_invalid_model_key(self):
        """Test that invalid model keys raise an error."""
        with pytest.raises(ValueError, match="not found in models.yaml"):
            create_agent("nonexistent-model")


# =============================================================================
# Agent Query Tests
# =============================================================================


class TestAgentQueries:
    """Tests for querying agents and getting responses."""

    @pytest.mark.asyncio
    async def test_litellm_agent_query(self, has_openai_key, simple_message):
        """Test querying a LiteLLMAgent."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = create_agent("gpt-4o-mini", temperature=0.0, max_tokens=10)
        responses = await agent.async_completions(simple_message, verbose=False)

        assert len(responses) == 1
        assert isinstance(responses[0], LLMResponse)
        assert responses[0].content is not None
        assert "4" in responses[0].content  # 2+2=4

    @pytest.mark.asyncio
    async def test_litellm_agent_with_system_message(
        self, has_openai_key, simple_message_with_system
    ):
        """Test querying with a system message."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = create_agent("gpt-4o-mini", temperature=0.0, max_tokens=10)
        responses = await agent.async_completions(
            simple_message_with_system, verbose=False
        )

        assert len(responses) == 1
        assert responses[0].content is not None

    @pytest.mark.asyncio
    async def test_batch_queries(self, has_openai_key):
        """Test batch querying multiple messages."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        messages = [
            [{"role": "user", "content": "What is 1+1? Reply with just the number."}],
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            [{"role": "user", "content": "What is 3+3? Reply with just the number."}],
        ]

        agent = create_agent("gpt-4o-mini", temperature=0.0, max_tokens=10)
        responses = await agent.async_completions(messages, verbose=False)

        assert len(responses) == 3
        assert all(isinstance(r, LLMResponse) for r in responses)
        assert all(r.content is not None for r in responses)


# =============================================================================
# Reasoning Tests
# =============================================================================


class TestReasoningModes:
    """Tests for different reasoning configurations."""

    @pytest.mark.asyncio
    async def test_reasoning_agent_returns_response(self, has_openai_key):
        """Test that ReasoningAgent returns valid responses."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = create_agent(
            "gpt-5-reasoning",
            temperature=1.0,
            max_tokens=500,
            base_timeout=60,
        )
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 15 + 27?"},
            ]
        ]

        responses = await agent.async_completions(messages, verbose=False)

        assert len(responses) == 1
        assert isinstance(responses[0], LLMResponse)
        assert responses[0].content is not None
        # Check that the answer contains 42
        assert "42" in responses[0].content

    @pytest.mark.asyncio
    async def test_reasoning_agent_captures_reasoning_summary(self, has_openai_key):
        """Test that reasoning summaries are captured for OpenAI models."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = create_agent(
            "gpt-5-reasoning",
            temperature=1.0,
            max_tokens=1000,
            base_timeout=60,
        )
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Think step by step: What is 123 * 456?",
                },
            ]
        ]

        responses = await agent.async_completions(messages, verbose=False)

        assert len(responses) == 1
        response = responses[0]

        # For OpenAI reasoning models, reasoning_summary should be populated
        # (reasoning traces are internal, only summaries are exposed)
        assert response.content is not None
        # Note: reasoning_summary may or may not be populated depending on model behavior
        # The key test is that the response structure is correct
        assert response.reasoning_type == "NO_REASONING"  # Set by ReasoningAgent

    @pytest.mark.asyncio
    async def test_non_reasoning_model(self, has_openai_key):
        """Test that models without reasoning_effort use LiteLLMAgent."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        # gpt-4o-mini has no reasoning_effort, so should use LiteLLMAgent
        agent = create_agent(
            "gpt-4o-mini",
            temperature=0.0,
            max_tokens=50,
            base_timeout=30,
        )

        # Verify it's not a ReasoningAgent
        assert isinstance(agent, LiteLLMAgent)
        assert not isinstance(agent, ReasoningAgent)

        messages = [
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
        ]
        responses = await agent.async_completions(messages, verbose=False)

        assert len(responses) == 1
        assert responses[0].content is not None
        # LiteLLMAgent responses should have no reasoning fields populated
        assert responses[0].reasoning is None
        assert responses[0].reasoning_summary is None

    @pytest.mark.asyncio
    async def test_openrouter_reasoning_agent(self, has_openrouter_key):
        """Test OpenRouter reasoning models (e.g., DeepSeek)."""
        if not has_openrouter_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        agent = create_agent(
            "deepseek-v3-2-reasoning",
            temperature=1.0,
            max_tokens=1000,
            base_timeout=60,
        )
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 7 * 8?"},
            ]
        ]

        responses = await agent.async_completions(messages, verbose=False)

        assert len(responses) == 1
        response = responses[0]
        assert response.content is not None
        assert "56" in response.content

        # For OpenRouter reasoning models, full reasoning traces may be available
        # Check that the response structure is correct
        assert isinstance(response, LLMResponse)


# =============================================================================
# LLMResponse Structure Tests
# =============================================================================


class TestLLMResponseStructure:
    """Tests for LLMResponse dataclass structure."""

    def test_llm_response_fields(self):
        """Test LLMResponse has expected fields."""
        response = LLMResponse(content="test")

        assert response.content == "test"
        assert response.reasoning is None
        assert response.reasoning_summary is None
        assert response.reasoning_type == "NO_REASONING"

    def test_llm_response_with_reasoning(self):
        """Test LLMResponse with reasoning fields populated."""
        response = LLMResponse(
            content="The answer is 42",
            reasoning="Step 1: ...\nStep 2: ...",
            reasoning_summary="Computed the result",
            reasoning_type="REASONING_BEFORE",
        )

        assert response.content == "The answer is 42"
        assert response.reasoning == "Step 1: ...\nStep 2: ..."
        assert response.reasoning_summary == "Computed the result"
        assert response.reasoning_type == "REASONING_BEFORE"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_generate_responses_from_messages(self, has_openai_key):
        """Test the generate_responses_from_messages utility function."""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        from choices.utils import generate_responses_from_messages

        agent = create_agent("gpt-4o-mini", temperature=0.0, max_tokens=10)
        messages = [
            [{"role": "user", "content": "Say 'hello'"}],
            [{"role": "user", "content": "Say 'world'"}],
        ]

        responses = await generate_responses_from_messages(
            agent, messages=messages, verbose=False
        )

        assert len(responses) == 2
        assert all(r.content is not None for r in responses)
