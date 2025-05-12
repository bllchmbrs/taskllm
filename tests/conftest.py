import asyncio
import json
import random
from typing import Dict, List, Type
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from taskllm.ai import DEFAULT_LLM_CONFIG, LLMConfig
from taskllm.optimizer.data import DataSet, Row


class SampleResponse(BaseModel):
    """Sample structured output for testing."""
    
    value: str
    confidence: float


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_config():
    """Return test LLM configuration."""
    return LLMConfig(
        model="test-model",
        temperature=0.1,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


@pytest.fixture
def default_llm_config():
    """Return the default LLM configuration."""
    return DEFAULT_LLM_CONFIG


@pytest.fixture
def mock_llm_response():
    """Mock LLM responses to avoid actual API calls."""
    return AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": "This is a mock response from the LLM."
                    }
                }
            ]
        }
    )


@pytest.fixture
def mock_structured_response():
    """Mock structured response for testing."""
    return SampleResponse(value="test_value", confidence=0.95)


@pytest.fixture
def sample_row():
    """Create a sample Row for testing."""
    return Row[SampleResponse](
        input_variables={"prompt": "Test prompt", "context": "Test context"},
        expected_output=SampleResponse(value="Test output", confidence=0.9),
        is_labeled=True,
        task_name="test_task",
        timestamp="2023-01-01T00:00:00Z"
    )


@pytest.fixture
def sample_dataset():
    """Create a small reusable dataset with multiple rows."""
    rows = [
        Row[SampleResponse](
            input_variables={"prompt": f"Test prompt {i}", "context": f"Test context {i}"},
            expected_output=SampleResponse(value=f"Test output {i}", confidence=0.8),
            is_labeled=True,
            task_name="test_task",
            timestamp="2023-01-01T00:00:00Z"
        )
        for i in range(10)
    ]

    return DataSet(name="test_dataset", rows=rows)


@pytest.fixture
def mock_cache_dict():
    """Mocked cache dictionary for testing."""
    _cache = {}
    
    class MockCache:
        def get(self, key):
            return _cache.get(key)
            
        def set(self, key, value):
            _cache[key] = value
            
    return MockCache()