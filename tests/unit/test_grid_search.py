from typing import List, Type
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from taskllm.ai import LLMConfig
from taskllm.optimizer.methods.grid import GridSearchOptimizer, GridSearchParams
from taskllm.optimizer.prompt.meta import (
    MetaPrompt,
    ModelsEnum,
    PromptMode,
    SimpleMetaPromptSpec,
)


class TestOutput(BaseModel):
    result: bool


def create_test_meta_prompt(expected_output_type=TestOutput):
    """Helper to create test MetaPrompt instances with valid values"""
    spec = SimpleMetaPromptSpec(
        input_user_task_goal="Test task",
        input_variable_keys=["input1", "input2"],
        input_expected_output_type=expected_output_type,
        instructions_and_context="Test instructions",
        model=ModelsEnum.GPT_4_1_MINI,
        mode=PromptMode.SIMPLE,
    )

    config = LLMConfig(
        model="test_model",
        temperature=0.5,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return MetaPrompt(
        spec=spec,
        config=config,
        expected_output_type=expected_output_type,
    )


# Mock function with correct signature
async def mock_generate_prompts(
    guidance: str,
    keys: List[str],
    expected_output_type: Type[BaseModel] | bool | str,
    num_prompts: int,
    mode: PromptMode = PromptMode.ADVANCED,
    failures: List | None = None,
) -> List[MetaPrompt]:
    """Mock implementation that returns exactly num_prompts prompts"""
    return [create_test_meta_prompt(expected_output_type) for _ in range(num_prompts)]


@pytest.fixture
def grid_optimizer():
    # Create a basic grid optimizer for testing
    optimizer = GridSearchOptimizer(
        task_guidance="Test task",
        variable_keys=["input1", "input2"],
        expected_output_type=TestOutput,
        row_scoring_function=lambda row, output: 1.0 if output else 0.0,
        grid_dimensions={"temperature": [0.1, 0.7], "model": ["model1", "model2"]},
    )
    return optimizer


def test_grid_params_default():
    # Test that default parameters are set correctly
    params = GridSearchParams()
    assert "temperature" in params.grid_dimensions
    assert "model" in params.grid_dimensions
    assert params.prompt_variations == 3


def test_generate_grid_combinations(grid_optimizer):
    # Test that all combinations are generated correctly
    grid_optimizer._generate_grid_combinations()

    assert len(grid_optimizer._grid_combinations) == 4  # 2 temp values × 2 models

    # Check that all combinations are present
    expected_combinations = [
        {"temperature": 0.1, "model": "model1"},
        {"temperature": 0.1, "model": "model2"},
        {"temperature": 0.7, "model": "model1"},
        {"temperature": 0.7, "model": "model2"},
    ]

    for combo in expected_combinations:
        assert combo in grid_optimizer._grid_combinations


# Directly patch select_next_prompts method for simpler testing
@pytest.mark.asyncio
async def test_select_next_prompts(grid_optimizer):
    # Modify the grid search optimizer to have a predictable response
    with patch.object(grid_optimizer, "_generate_grid_combinations"):
        grid_optimizer._grid_combinations = [
            {"temperature": 0.1, "model": "model1"},
            {"temperature": 0.1, "model": "model2"},
            {"temperature": 0.7, "model": "model1"},
            {"temperature": 0.7, "model": "model2"},
        ]
        grid_optimizer._current_grid_position = 0

        # Mock the generate_prompts call
        with patch(
            "taskllm.optimizer.methods.grid.generate_prompts", mock_generate_prompts
        ):
            # Get the first 2 prompts
            prompts1 = await grid_optimizer.select_next_prompts(2)
            assert len(prompts1) == 2

            # Check that grid position advances
            assert grid_optimizer._current_grid_position == 2

            # Get the next 3 prompts (only 2 remain)
            prompts2 = await grid_optimizer.select_next_prompts(3)
            assert len(prompts2) == 2  # Only 2 combinations remain

            # Check that grid position wraps around
            assert grid_optimizer._current_grid_position == 0


@pytest.mark.asyncio
async def test_select_next_prompts_with_empty_grid(grid_optimizer):
    # Test behavior when grid is empty
    with patch(
        "taskllm.optimizer.methods.grid.generate_prompts", mock_generate_prompts
    ):
        grid_optimizer._grid_combinations = []

        # Request 3 prompts
        prompts = await grid_optimizer.select_next_prompts(3)
        assert len(prompts) == 3


@pytest.mark.asyncio
async def test_prompt_config_application(grid_optimizer):
    # Create a test setup with a more controlled environment
    with patch.object(grid_optimizer, "_generate_grid_combinations"):
        # Set a controlled grid
        grid_optimizer._grid_combinations = [
            {"temperature": 0.1, "model": "model1"},
            {"temperature": 0.1, "model": "model2"},
        ]
        grid_optimizer._current_grid_position = 0

        # Create a mock for generate_prompts with specific return value
        mock_prompts = [create_test_meta_prompt() for _ in range(2)]

        # Set initial values to verify they change
        for prompt in mock_prompts:
            prompt.config.model = "initial_model"
            prompt.config.temperature = 0.0

        # Patch generate_prompts to return our mock prompts
        with patch(
            "taskllm.optimizer.methods.grid.generate_prompts", return_value=mock_prompts
        ):
            # Get the prompts
            prompts = await grid_optimizer.select_next_prompts(2)

            assert len(prompts) == 2

            # Check that the config values were correctly updated
            assert prompts[0].config.model == "model1"
            assert prompts[0].config.temperature == 0.1

            assert prompts[1].config.model == "model2"
            assert prompts[1].config.temperature == 0.1
