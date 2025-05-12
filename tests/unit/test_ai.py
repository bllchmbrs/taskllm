import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from taskllm.ai import LLMConfig, simple_llm_call, struct_llm_call, MaxRetriesExceeded


class TestOutput(BaseModel):
    """Test structured output model."""
    result: str
    score: float


@pytest.mark.asyncio
async def test_simple_llm_call(mock_llm_config, mock_cache_dict):
    """Test basic functionality of simple_llm_call."""
    messages = [{"role": "user", "content": "Hello, world!"}]
    
    # Test with mocked completion function
    with patch("taskllm.ai.completion") as mock_completion, \
         patch("taskllm.ai.cache", mock_cache_dict):
        mock_completion.return_value.choices[0].message.content = "Test response"
        
        result = await simple_llm_call(messages, mock_llm_config)
        
        # Verify the call was made correctly
        mock_completion.assert_called_once_with(
            model=mock_llm_config.model, messages=messages
        )
        
        # Verify the result
        assert result == "Test response"


@pytest.mark.asyncio
async def test_simple_llm_call_with_cache(mock_llm_config, mock_cache_dict):
    """Test caching behavior of simple_llm_call."""
    messages = [{"role": "user", "content": "Hello, world!"}]
    
    # Create the expected cache key
    key = hashlib.sha256(
        json.dumps(messages + [mock_llm_config.model_dump()]).encode()
    ).hexdigest()
    
    # Set up the cache
    cached_response = "Cached response"
    mock_cache_dict.set(key, cached_response)
    
    # Test with mocked completion function and cache
    with patch("taskllm.ai.completion") as mock_completion, \
         patch("taskllm.ai.cache", mock_cache_dict):
        # This should return the cached response without calling the API
        result = await simple_llm_call(messages, mock_llm_config)
        
        # Verify the completion function was not called
        mock_completion.assert_not_called()
        
        # Verify we got the cached response
        assert result == cached_response


@pytest.mark.asyncio
async def test_simple_llm_call_no_cache(mock_llm_config, mock_cache_dict):
    """Test simple_llm_call with caching disabled."""
    messages = [{"role": "user", "content": "Hello, world!"}]
    expected_response = "Fresh response"
    
    # Test with mocked completion function and cache
    with patch("taskllm.ai.completion") as mock_completion, \
         patch("taskllm.ai.cache", mock_cache_dict):
        mock_completion.return_value.choices[0].message.content = expected_response
        
        # Call with use_cache=False to bypass cache
        result = await simple_llm_call(messages, mock_llm_config, use_cache=False)
        
        # Verify the completion function was called
        mock_completion.assert_called_once()
        
        # Verify we got the fresh response
        assert result == expected_response


@pytest.mark.asyncio
async def test_struct_llm_call(mock_llm_config, mock_cache_dict):
    """Test basic functionality of struct_llm_call."""
    messages = [{"role": "user", "content": "Analyze this: example"}]
    expected_output = TestOutput(result="example_analyzed", score=0.95)
    
    # Test with mocked instructor client
    with patch("taskllm.ai.ins_client.chat.completions.create") as mock_create, \
         patch("taskllm.ai.cache", mock_cache_dict):
        mock_create.return_value = expected_output
        
        result = await struct_llm_call(messages, mock_llm_config, TestOutput)
        
        # Verify the call was made correctly
        mock_create.assert_called_once_with(
            model=mock_llm_config.model, 
            messages=messages,
            response_model=TestOutput
        )
        
        # Verify the result
        assert isinstance(result, TestOutput)
        assert result.result == "example_analyzed"
        assert result.score == 0.95


@pytest.mark.asyncio
async def test_struct_llm_call_with_cache(mock_llm_config, mock_cache_dict):
    """Test caching behavior of struct_llm_call."""
    messages = [{"role": "user", "content": "Analyze this: cached example"}]
    
    # Create the expected cache key
    key = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
    
    # Set up the cache with model data
    cached_data = {"result": "cached_result", "score": 0.85}
    mock_cache_dict.set(key, cached_data)
    
    # Test with mocked instructor client and cache
    with patch("taskllm.ai.ins_client.chat.completions.create") as mock_create, \
         patch("taskllm.ai.cache", mock_cache_dict):
        # This should return the cached response without calling the API
        result = await struct_llm_call(messages, mock_llm_config, TestOutput)
        
        # Verify the API was not called
        mock_create.assert_not_called()
        
        # Verify we got a proper model instance from cache
        assert isinstance(result, TestOutput)
        assert result.result == "cached_result"
        assert result.score == 0.85


@pytest.mark.asyncio
async def test_struct_llm_call_no_cache(mock_llm_config, mock_cache_dict):
    """Test struct_llm_call with caching disabled."""
    messages = [{"role": "user", "content": "Analyze this: fresh example"}]
    expected_output = TestOutput(result="fresh_result", score=0.99)
    
    # Test with mocked instructor client
    with patch("taskllm.ai.ins_client.chat.completions.create") as mock_create, \
         patch("taskllm.ai.cache", mock_cache_dict):
        mock_create.return_value = expected_output
        
        # Call with use_cache=False to bypass cache
        result = await struct_llm_call(messages, mock_llm_config, TestOutput, use_cache=False)
        
        # Verify the API was called
        mock_create.assert_called_once()
        
        # Verify we got the fresh response
        assert result.result == "fresh_result"
        assert result.score == 0.99


@pytest.mark.asyncio
async def test_struct_llm_call_cache_validation(mock_llm_config, mock_cache_dict):
    """Test that struct_llm_call properly validates cached data against the model."""
    messages = [{"role": "user", "content": "Analyze this: example"}]
    
    # Create the cache key
    key = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
    
    # Test with an invalid cache value (wrong type)
    mock_cache_dict.set(key, "not_a_valid_model_data")
    
    with patch("taskllm.ai.ins_client.chat.completions.create") as mock_create, \
         patch("taskllm.ai.cache", mock_cache_dict), \
         patch("taskllm.ai.logger.error") as mock_logger:
        
        # Set up the mock to return valid data when called
        mock_create.return_value = TestOutput(result="fresh_data", score=0.75)
        
        # This should fail to use the cache and make a real call
        result = await struct_llm_call(messages, mock_llm_config, TestOutput)
        
        # Verify logger.error was called (invalid cache data)
        mock_logger.assert_called_once()
        
        # API should be called since cache validation failed
        mock_create.assert_called_once()
        
        # Should get the fresh result
        assert result.result == "fresh_data"