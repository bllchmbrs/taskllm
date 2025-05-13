import asyncio
import datetime
import json
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from loguru import logger
from pydantic import BaseModel, Field

from ...ai import DEFAULT_LLM_CONFIG, LLMConfig
from ...utils import get_cache
from ..data import DataSet, Row
from ..prompt.meta import MetaPrompt, generate_prompts

OUTPUT_TYPE = TypeVar("OUTPUT_TYPE", bound=BaseModel | bool | str | None)

cache = get_cache("evaluation")
DEFAULT_SEMAPHORE = asyncio.Semaphore(10)


class PromptWithType(BaseModel, Generic[OUTPUT_TYPE]):
    """Tracks performance metrics for a specific prompt template"""

    meta_prompt: MetaPrompt

    async def get_output(
        self, row: Row, semaphore: asyncio.Semaphore = DEFAULT_SEMAPHORE
    ) -> OUTPUT_TYPE | None:
        """Get output for a row using the meta prompt"""
        async with semaphore:
            logger.trace(
                f"Getting output for row using {self.meta_prompt.config.model}"
            )
            try:
                result = await self.meta_prompt.execute(  # type: ignore
                    row.get_variables(), self.meta_prompt.config
                )
                return result
            except Exception as e:
                logger.error(f"Error executing prompt: {e}")
                return None

    async def get_outputs(self, rows: List[Row]) -> List[OUTPUT_TYPE | None]:
        """Get the outputs for a list of rows"""
        logger.debug(
            f"Getting outputs for {len(rows)} rows using {self.meta_prompt.config.model}"
        )
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

        total_score = sum(scores)

        # Log with model info
        logger.info(
            f"Model {self.meta_prompt.config.model} with prompt achieved score: {total_score:.4f}, "
            f"Correct: {correct}, Incorrect: {incorrect}, Unlabelled: {unlabelled} out of {len(rows)}"
        )

        return total_score


class BaseOptimizer(BaseModel, Generic[OUTPUT_TYPE], ABC):
    """Base class for optimizers"""

    task_guidance: str
    variable_keys: List[str]
    expected_output_type: Type[OUTPUT_TYPE]
    prompt_history: List[PromptWithType] = []
    row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float]
    print_iteration_summary: bool = True
    # Multi-model testing parameters
    models_to_test: List[str] = Field(
        default_factory=lambda: [DEFAULT_LLM_CONFIG.model]
    )
    model_exploration_rate: float = 0.3  # How often to try different models
    prioritize_best_model: bool = True  # Whether to find best model per prompt

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

    async def select_best_prompt_by_model(
        self, rows: List[Row]
    ) -> Dict[str, Optional[MetaPrompt]]:
        """Find the best prompt for each model."""
        if not self.prompt_history:
            return {}

        # Group prompts by model
        model_prompts: Dict[str, List[PromptWithType]] = {}
        for pwt in self.prompt_history:
            model = pwt.meta_prompt.config.model
            if model not in model_prompts:
                model_prompts[model] = []
            model_prompts[model].append(pwt)

        # Find best prompt per model
        best_prompts: Dict[str, Optional[MetaPrompt]] = {}
        for model, prompts in model_prompts.items():
            if not prompts:
                best_prompts[model] = None
                continue

            scores = await asyncio.gather(
                *[
                    prompt.calculate_scores(rows, self.row_scoring_function)
                    for prompt in prompts
                ]
            )

            best_idx = scores.index(max(scores)) if scores else 0
            best_prompts[model] = prompts[best_idx].meta_prompt

        return best_prompts

    async def select_best_model_and_prompt(
        self, rows: List[Row]
    ) -> Tuple[str, Optional[MetaPrompt]]:
        """Find the best model and its best prompt."""
        model_prompts = await self.select_best_prompt_by_model(rows)

        if not model_prompts:
            return DEFAULT_LLM_CONFIG.model, None

        # Score each model's best prompt
        model_scores = {}
        for model, prompt in model_prompts.items():
            if prompt is None:
                continue

            pwt: PromptWithType[OUTPUT_TYPE] = PromptWithType(meta_prompt=prompt)
            score = await pwt.calculate_scores(rows, self.row_scoring_function)
            model_scores[model] = (score, prompt)

        if not model_scores:
            return DEFAULT_LLM_CONFIG.model, None

        # Find best model
        best_model = max(model_scores.items(), key=lambda x: x[1][0])[0]
        return best_model, model_scores[best_model][1]

    @abstractmethod
    async def select_next_prompts(
        self, num_variations: int = 3, rows: List[Row] | None = None
    ) -> List[MetaPrompt]:
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
        models: Optional[List[str]] = None,
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
        self.training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"prompt_history_{self.training_start_time}.jsonl"
        # Create empty file
        with open(self.log_file, "w") as f:
            pass

        # If models specified, pass to optimizer
        if models:
            self.optimizer.models_to_test = models

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

        # Log prompt and score
        score = await performance.calculate_scores(
            self.dataset.training_rows, self.scoring_function
        )

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_content": performance.meta_prompt.get_user_message_content(),
            "score": float(score),
            "model": performance.meta_prompt.config.model,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def generate_candidate_prompts(
        self, num_variations: int = 3
    ) -> List[MetaPrompt]:
        """Generate candidate prompts"""
        return await self.optimizer.select_next_prompts(num_variations)

    async def run_for_prompt(
        self, prompt: MetaPrompt, rows: List[Row] | None = None
    ) -> PromptWithType[OUTPUT_TYPE]:
        """Run the optimizer for a given prompt"""
        if rows is None:
            rows = self.dataset.training_rows
        logger.info(
            f"Running for prompt {prompt.spec.get_content()[:100]}... against {len(rows)} rows"
        )
        prompt_with_type: PromptWithType[OUTPUT_TYPE] = PromptWithType(
            meta_prompt=prompt
        )
        _ = await prompt_with_type.get_outputs(rows)
        return prompt_with_type

    def dump(self) -> None:
        """Save the best prompt to disk"""
        # First get the best prompt to avoid calling select_best_prompt twice
        best_prompt = asyncio.run(self.select_best_prompt(self.dataset.training_rows))
        if best_prompt is None:
            logger.error("Cannot dump best prompt: no best prompt found")
            return

        with open("best_prompt.json", "w") as f:
            f.write(best_prompt.get_user_message_content())

        with open("best_config.json", "w") as f:
            f.write(best_prompt.config.model_dump_json())

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

    async def get_best_llm_config(self) -> LLMConfig:
        """Get the best LLM config from the performance history"""
        best_prompt = await self.get_best_prompt()
        return best_prompt.config

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
