import asyncio
import itertools
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

from ...ai import LLMConfig
from ..data import DataSet, Row
from ..prompt.meta import MetaPrompt, PromptMode, generate_prompts
from .base import OUTPUT_TYPE, BaseOptimizer, PromptWithType, Trainer


class GridSearchParams(BaseModel):
    """Parameters for grid search optimization."""

    grid_dimensions: Dict[str, List[Any]] = Field(
        default_factory=lambda: {
            "temperature": [0.1, 0.5, 0.9],
            "model": ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
        }
    )
    prompt_variations: int = 3  # Number of prompt variations to test


class GridSearchOptimizer(BaseOptimizer[OUTPUT_TYPE]):
    """Grid search optimizer for exploring prompt variations."""

    params: GridSearchParams = GridSearchParams()
    _current_grid_position: int = 0
    _grid_combinations: List[Dict[str, Any]] = []
    prompt_mode: PromptMode = PromptMode.SIMPLE

    def __init__(
        self,
        task_guidance: str,
        variable_keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        grid_dimensions: Optional[Dict[str, List[Any]]] = None,
        prompt_mode: PromptMode = PromptMode.SIMPLE,
    ):
        super().__init__(
            task_guidance=task_guidance,
            variable_keys=variable_keys,
            expected_output_type=expected_output_type,
            row_scoring_function=row_scoring_function,
            prompt_history=[],
        )

        if grid_dimensions:
            self.params.grid_dimensions = grid_dimensions

        self.prompt_mode = prompt_mode

        # Generate all grid combinations upfront
        self._generate_grid_combinations()

    def _generate_grid_combinations(self) -> None:
        """Generate all combinations of grid parameters."""
        # Use itertools.product to generate all combinations
        keys = list(self.params.grid_dimensions.keys())
        values = [self.params.grid_dimensions[k] for k in keys]

        self._grid_combinations = [
            dict(zip(keys, combo)) for combo in itertools.product(*values)
        ]

        logger.info(f"Generated {len(self._grid_combinations)} grid combinations")

    async def select_next_prompts(
        self, num_variations: int = 3, rows: Optional[List[Row]] = None
    ) -> List[MetaPrompt]:
        """Select the next prompts based on grid search."""
        # If we don't have any combinations, return base prompts
        if not self._grid_combinations:
            logger.info("No grid combinations available, generating default prompts")
            return await generate_prompts(
                self.task_guidance,
                self.variable_keys,
                self.expected_output_type,  # type: ignore
                num_variations,
                mode=self.prompt_mode,
            )

        # Get current grid positions
        start_idx = self._current_grid_position
        end_idx = min(start_idx + num_variations, len(self._grid_combinations))
        current_combinations = self._grid_combinations[start_idx:end_idx]

        # Update position for next call
        self._current_grid_position = end_idx
        if self._current_grid_position >= len(self._grid_combinations):
            # Reset grid position to enable cycling through the grid if needed
            self._current_grid_position = 0
            logger.info("Completed one full grid search cycle")

        # Generate prompts with these configurations
        base_prompts = await generate_prompts(
            self.task_guidance,
            self.variable_keys,
            self.expected_output_type,  # type: ignore
            len(current_combinations),
            mode=self.prompt_mode,
        )

        logger.info(f"Applying grid configurations to {len(base_prompts)} prompts")

        # Apply grid configurations to prompts
        for i, (prompt, config) in enumerate(zip(base_prompts, current_combinations)):
            for key, value in config.items():
                if key == "model":
                    # Special case for model
                    prompt.config.model = value
                else:
                    # Update other config params
                    setattr(prompt.config, key, value)

            logger.debug(f"Created prompt with config: {prompt.config}")

        return base_prompts


class GridSearchTrainer(Trainer[OUTPUT_TYPE]):
    """Trainer using grid search optimization for prompt learning."""

    def __init__(
        self,
        all_rows: List[Row] | DataSet,
        task_guidance: str,
        keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        num_iterations: int = 5,
        candidates_per_iteration: int = 3,
        grid_dimensions: Optional[Dict[str, List[Any]]] = None,
        prompt_mode: PromptMode = PromptMode.SIMPLE,
        models: Optional[List[str]] = None,
        failure_analysis_enabled: bool = False,
        failure_threshold: int = 2,
    ):
        super().__init__(
            all_rows=all_rows,
            task_guidance=task_guidance,
            keys=keys,
            expected_output_type=expected_output_type,
            scoring_function=scoring_function,
            num_iterations=num_iterations,
            candidates_per_iteration=candidates_per_iteration,
            print_iteration_summary=True,
            models=models,
            failure_analysis_enabled=failure_analysis_enabled,
            failure_threshold=failure_threshold,
        )

        # Create the grid search optimizer with our parameters
        self.optimizer = GridSearchOptimizer(
            task_guidance=task_guidance,
            variable_keys=keys,
            expected_output_type=expected_output_type,
            row_scoring_function=scoring_function,
            grid_dimensions=grid_dimensions,
            prompt_mode=prompt_mode,
        )

        # If models are provided, update the grid dimensions to include them
        if models and "model" not in (grid_dimensions or {}):
            self.optimizer.params.grid_dimensions["model"] = models
            # Regenerate combinations with the new models
            self.optimizer._generate_grid_combinations()

    async def train(self) -> None:
        """Run grid search training process to find optimal prompt."""
        logger.info(
            f"Starting grid search training with {self.num_iterations} iterations, "
            f"{self.candidates_per_iteration} candidates per iteration"
        )

        # For each iteration
        for i in range(self.num_iterations):
            logger.info(f"Starting iteration {i + 1}/{self.num_iterations}")

            # Generate candidate prompts from the grid
            candidate_prompts = await self.optimizer.generate_candidate_prompts(
                self.candidates_per_iteration
            )

            if not candidate_prompts:
                logger.warning("No candidate prompts generated, ending training early")
                break

            logger.info(f"Generated {len(candidate_prompts)} candidate prompts")

            # Evaluate each candidate on training data
            perf_tasks = []
            for j, prompt in enumerate(candidate_prompts):
                logger.info(f"Testing candidate {j + 1}/{len(candidate_prompts)}")
                perf_tasks.append(self.run_for_prompt(prompt, self.train_rows))

            # Run evaluations concurrently
            performances = await asyncio.gather(*perf_tasks)

            # Log results and update history
            for perf in performances:
                await self.log_performance(perf)

            if self.print_iteration_summary:
                logger.info(f"Completed iteration {i + 1}/{self.num_iterations}")

        logger.success("Grid search training completed")
