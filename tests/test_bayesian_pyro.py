import asyncio

import pyro
import pytest
import torch

from taskllm.optimizer.methods.bayesian import BayesianOptimizer, BayesianParams


class DummyOutputType:
    """Dummy output type for testing purposes."""

    pass


def test_init_bayesian_optimizer():
    """Test initialization of BayesianOptimizer with Pyro backend."""
    # Setup test parameters
    task_guidance = "Test task guidance"
    variable_keys = ["key1", "key2"]
    row_scoring_function = lambda row, output: 0.5  # Simple dummy function

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
    row_scoring_function = lambda row, output: 0.5  # Simple dummy function

    # Create optimizer
    optimizer = BayesianOptimizer(
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
            self.config = None

        def get_user_message_content(self):
            return "This is a test prompt with some instructions. You should do this. Consider that."

    # Extract features
    prompt = DummyMetaPrompt()
    features = optimizer._extract_single_prompt_features(prompt)

    # Verify feature extraction
    assert features.shape == (optimizer.params.feature_dimension,)
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
