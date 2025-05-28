import asyncio

import pyro
import pytest
import torch
from pydantic import BaseModel

from taskllm.optimizer.methods.bayesian import BayesianOptimizer, BayesianParams


class DummyOutputType(BaseModel):
    """Dummy output type for testing purposes."""

    result: str = "test"


def test_init_bayesian_optimizer():
    """Test initialization of BayesianOptimizer with Pyro backend."""
    # Setup test parameters
    task_guidance = "Test task guidance"
    variable_keys = ["key1", "key2"]
    
    def row_scoring_function(row, output):
        return 0.5  # Simple dummy function

    # Create optimizer
    optimizer = BayesianOptimizer(
        task_guidance=task_guidance,
        variable_keys=variable_keys,
        expected_output_type=DummyOutputType,
        row_scoring_function=row_scoring_function,
    )

    # Verify initialization
    assert optimizer.task_guidance == task_guidance
    assert optimizer.variable_keys == variable_keys
    assert optimizer._surrogate_model is None
    assert isinstance(optimizer.params, BayesianParams)
    assert isinstance(optimizer._torch_device, torch.device)


@pytest.mark.asyncio
async def test_feature_extraction():
    """Test feature extraction from prompts."""
    # Setup test parameters
    task_guidance = "Test task guidance"
    variable_keys = ["key1", "key2"]
    
    def row_scoring_function(row, output):
        return 0.5  # Simple dummy function

    # Create optimizer (not used in this test)
    BayesianOptimizer(
        task_guidance=task_guidance,
        variable_keys=variable_keys,
        expected_output_type=DummyOutputType,
        row_scoring_function=row_scoring_function,
    )

    # Create a dummy MetaPrompt to test feature extraction
    class DummySpec:
        def get_content(self):
            return "This is a test prompt with some instructions. You should do this. Consider that."

        def variation_types(self):
            return ["type1", "type2"]

        async def variation_weights(self):
            return [0.5, 0.5]

        async def vary(self, variation_type):
            return self

    class DummyMetaPrompt:
        def __init__(self):
            self.spec = DummySpec()
            self.generate_messages = "test"
            # Import LLMConfig to create a proper config
            from taskllm.optimizer.prompt.meta import DEFAULT_LLM_CONFIG
            self.config = DEFAULT_LLM_CONFIG

        def get_user_message_content(self):
            return "This is a test prompt with some instructions. You should do this. Consider that."

    # Extract features
    prompt = DummyMetaPrompt()
    # Import the function since it's module-level, not a method
    from taskllm.optimizer.methods.bayesian import _extract_single_prompt_features
    features = _extract_single_prompt_features(prompt)

    # Verify feature extraction
    # The feature vector should have at least 5 base features + provider features
    # Base features: length, instruction_count, keyword_count, question_count, model_complexity
    # Plus one-hot encoded providers (anthropic, openai, groq = 3 providers)
    expected_dim = 5 + 3  # 5 base features + 3 provider features
    assert features.shape == (expected_dim,)
    assert 0 <= features[0] <= 1  # Prompt length feature
    assert 0 <= features[2] <= 1  # Keyword presence feature


# Run tests if script is executed directly
if __name__ == "__main__":
    # Initialize Pyro
    pyro.clear_param_store()

    # Run tests
    test_init_bayesian_optimizer()
    asyncio.run(test_feature_extraction())

    print("All tests passed!")
