import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Generic, List, Literal, Type, TypeVar

from loguru import logger
from pydantic import BaseModel

from ...ai import DEFAULT_LLM_CONFIG
from ...utils import get_cache
from ..data import DataSet, Row
from ..prompt.meta import MetaPrompt, generate_prompts

OUTPUT_TYPE = TypeVar("OUTPUT_TYPE", bound=BaseModel | bool | str | None)

cache = get_cache("evaluation")
DEFAULT_SEMAPHORE = asyncio.Semaphore(20)


class PromptWithType(BaseModel, Generic[OUTPUT_TYPE]):
    """Tracks performance metrics for a specific prompt template"""

    meta_prompt: MetaPrompt

    async def get_output(
        self, row: Row, semaphore: asyncio.Semaphore = DEFAULT_SEMAPHORE
    ) -> OUTPUT_TYPE | None:
        async with semaphore:
            logger.trace(f"Getting output for row")
            return await self.meta_prompt.execute(
                row.get_variables(), DEFAULT_LLM_CONFIG
            )

    async def get_outputs(self, rows: List[Row]) -> List[OUTPUT_TYPE | None]:
        """Get the outputs for a list of rows"""
        logger.debug(f"Getting outputs for {len(rows)} rows")
        output_tasks: List[Coroutine[Any, Any, OUTPUT_TYPE | None]] = [
            self.get_output(row, DEFAULT_SEMAPHORE) for row in rows
        ]
        return await asyncio.gather(*output_tasks)

    async def get_scores(
        self,
        rows: List[Row],
        row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
    ) -> List[float]:
        outputs = await self.get_outputs(rows)
        scores = [
            row_scoring_function(row, output) for row, output in zip(rows, outputs)
        ]
        logger.debug(f"Calculated {len(scores)} scores")
        return scores

    async def calculate_scores(
        self,
        rows: List[Row],
        row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
    ) -> float:
        scores = await self.get_scores(rows, row_scoring_function)

        correct = 0
        incorrect = 0
        unlabelled = 0
        for row, score in zip(rows, scores):
            if row.is_labeled:
                if score > 0:
                    correct += 1
                else:
                    incorrect += 1
            else:
                unlabelled += 1

        logger.info(
            f"Correct: {correct}, Incorrect: {incorrect}, Unlabelled: {unlabelled} out of {len(rows)}"
        )

        return sum(scores)


class BaseOptimizer(BaseModel, Generic[OUTPUT_TYPE], ABC):
    """Base class for optimizers"""

    task_guidance: str
    variable_keys: List[str]
    expected_output_type: Type[OUTPUT_TYPE]
    prompt_history: List[PromptWithType] = []
    row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float]
    print_iteration_summary: bool = True

    async def select_best_prompt(self, rows: List[Row]) -> MetaPrompt | None:
        """Get the best prompt from the performance history"""
        logger.info(f"Selecting best prompt from {len(self.prompt_history)} prompts")
        if not self.prompt_history:
            logger.warning("No performance history available to select best prompt")
            return None

        scores = await asyncio.gather(
            *[
                prompt.calculate_scores(rows, self.row_scoring_function)
                for prompt in self.prompt_history
            ]
        )
        best_performance = max(zip(self.prompt_history, scores), key=lambda x: x[1])
        return best_performance[0].meta_prompt

    async def log_prompt_to_history(self, performance: PromptWithType) -> None:
        """Add performance data for a prompt template"""
        self.prompt_history.append(performance)

    @abstractmethod
    async def select_next_prompts(self, num_variations: int = 3) -> List[MetaPrompt]:
        """Select the next prompts to evaluate"""
        pass

    async def generate_candidate_prompts(
        self, num_variations: int = 3
    ) -> List[MetaPrompt]:
        """Generate candidate prompts"""
        if not self.prompt_history:
            return await generate_prompts(
                self.task_guidance,
                self.variable_keys,
                self.expected_output_type,  # type: ignore
                num_variations,
            )
        return await self.select_next_prompts(num_variations)


class Trainer(Generic[OUTPUT_TYPE], ABC):
    """Base class for trainers"""

    def __init__(
        self,
        all_rows: List[Row] | DataSet,
        task_guidance: str,
        keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        optimizer: BaseOptimizer,
        scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        num_iterations: int = 5,
        candidates_per_iteration: int = 3,
        print_iteration_summary: bool = True,
    ):
        if isinstance(all_rows, List):
            self.dataset = DataSet(rows=all_rows, name="dataset")
        else:
            self.dataset = all_rows
        logger.info(f"All rows: {len(self.dataset.rows)}")
        self.task_guidance = task_guidance
        self.keys = keys
        self.expected_output_type = expected_output_type
        self.scoring_function = scoring_function
        self.num_iterations = num_iterations
        self.candidates_per_iteration = candidates_per_iteration
        self.optimizer = optimizer
        self.print_iteration_summary = print_iteration_summary

    async def select_best_prompt(
        self, rows: List[Row] | None = None
    ) -> MetaPrompt | None:
        """Select the best prompt from the performance history"""
        if rows is None:
            rows = self.dataset.training_rows
        return await self.optimizer.select_best_prompt(rows)

    async def log_performance(self, performance: PromptWithType) -> None:
        """Log performance data for a prompt template"""
        await self.optimizer.log_prompt_to_history(performance)

    async def generate_candidate_prompts(
        self, num_variations: int = 3
    ) -> List[MetaPrompt]:
        """Generate candidate prompts"""
        return await self.optimizer.select_next_prompts(num_variations)

    async def run_for_prompt(
        self, prompt: MetaPrompt, rows: List[Row] | None = None
    ) -> PromptWithType[OUTPUT_TYPE]:
        """Run the optimizer for a given prompt"""
        logger.info(
            f"Running for prompt {prompt.spec.get_content()[:100]}... against {len(rows)} rows"
        )
        if rows is None:
            rows = self.dataset.training_rows
        prompt_with_type: PromptWithType[OUTPUT_TYPE] = PromptWithType(
            meta_prompt=prompt
        )
        _ = await prompt_with_type.get_outputs(rows)
        return prompt_with_type

    def dump(self):
        """Save the best prompt to disk"""
        with open("best_prompt.json", "w") as f:
            f.write(
                self.select_best_prompt(
                    self.dataset.training_rows
                ).get_user_message_content()
            )

        with open("best_config.json", "w") as f:
            f.write(
                self.select_best_prompt(
                    self.dataset.training_rows
                ).config.model_dump_json()
            )

    @abstractmethod
    async def train(self) -> None:
        """Train the optimizer"""
        pass

    async def eval_prompt_on_training_set(self, prompt: MetaPrompt) -> float:
        """Evaluate a prompt on the training set"""
        pwt: PromptWithType[OUTPUT_TYPE] = PromptWithType(meta_prompt=prompt)
        return await pwt.calculate_scores(
            self.dataset.training_rows, self.scoring_function
        )

    async def eval_prompt_on_test_set(self, prompt: MetaPrompt) -> float:
        """Evaluate a prompt on the test set"""
        pwt: PromptWithType[OUTPUT_TYPE] = PromptWithType(meta_prompt=prompt)
        return await pwt.calculate_scores(self.dataset.test_rows, self.scoring_function)

    async def eval_prompt_on_all_data(self, prompt: MetaPrompt) -> float:
        """Evaluate a prompt on all data"""
        pwt: PromptWithType[OUTPUT_TYPE] = PromptWithType(meta_prompt=prompt)
        return await pwt.calculate_scores(self.dataset.rows, self.scoring_function)

    async def get_best_prompt(
        self, dev_or_test_dataset: Literal["dev", "test"] = "dev"
    ) -> MetaPrompt:
        """Get the best prompt from the performance history"""
        if dev_or_test_dataset == "dev":
            rows = self.dataset.training_rows
        else:
            rows = self.dataset.test_rows
        best_prompt = await self.optimizer.select_best_prompt(rows)
        if best_prompt is None:
            raise ValueError("No best prompt found")
        return best_prompt
