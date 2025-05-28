"""
Simplified test fixtures.
"""
import pytest

from taskllm.ai import LLMConfig
from taskllm.optimizer.data import DataSet
from tests.simplified.test_models import create_minimal_test_dataset


@pytest.fixture
def mock_llm_config():
    """Create a mock LLM config."""
    return LLMConfig(
        model="test-model",
        temperature=0.1,
        max_tokens=100
    )


@pytest.fixture
def test_dataset():
    """Create a minimal test dataset."""
    return DataSet(
        name="test_dataset",
        rows=create_minimal_test_dataset(10)
    )


@pytest.fixture
def event_loop():
    """Create an event loop for testing."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()