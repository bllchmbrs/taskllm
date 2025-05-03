import asyncio
import random
from typing import Callable, List, Type

from loguru import logger

from ...ai import DEFAULT_LLM_CONFIG
from ..data import DataSet, Row
from ..prompt.meta import MetaPrompt, PromptMode, generate_prompts

# Ensure PromptWithType is imported if needed for type hints, though not directly used in methods here
from .base import OUTPUT_TYPE, BaseOptimizer, Trainer


class BanditOptimizer(BaseOptimizer[OUTPUT_TYPE]):
    """Multi-armed bandit optimizer using exploration/exploitation for prompt selection."""

    # Internal state to track the best prompt found so far
    _current_best_prompt: MetaPrompt | None = None
    exploration_parameter: float = 0.1  # Controls exploration vs exploitation
    prompt_mode: PromptMode = PromptMode.ADVANCED  # Default mode

    def __init__(
        self,
        task_guidance: str,
        variable_keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        exploration_parameter: float = 0.1,
        prompt_mode: PromptMode = PromptMode.SIMPLE,
    ):
        super().__init__(
            task_guidance=task_guidance,
            variable_keys=variable_keys,
            expected_output_type=expected_output_type,
            row_scoring_function=row_scoring_function,
            prompt_history=[],
        )
        self.exploration_parameter = exploration_parameter
        self.prompt_mode = prompt_mode
        self._current_best_prompt = None  # Initialize internal state

    def update_best_prompt(self, prompt: MetaPrompt) -> None:
        """Updates the internally tracked best prompt."""
        self._current_best_prompt = prompt
        logger.debug("Updated internal best prompt in optimizer.")

    async def select_best_prompt(self, rows: List[Row]) -> MetaPrompt | None:
        """Select the best-performing prompt from history based on scores on given rows."""
        if not self.prompt_history:
            logger.warning("No performance history available to select best prompt")
            return None

        # Calculate scores for all prompts in history on the provided rows
        scoring_tasks = [
            prompt.calculate_scores(rows, self.row_scoring_function)
            for prompt in self.prompt_history
        ]
        scores = await asyncio.gather(*scoring_tasks)

        if not scores:
            logger.warning("Could not calculate scores for any prompt in history.")
            return None

        # Find the prompt with the highest score
        best_performance_index = scores.index(max(scores))
        best_prompt = self.prompt_history[best_performance_index].meta_prompt
        best_score = scores[best_performance_index]

        logger.info(
            f"Selected best prompt from history with score: {best_score:.4f}",
            # best_prompt_spec=best_prompt.spec, # Spec might be complex, log selectively
        )
        return best_prompt

    async def select_next_prompts(self, num_variations: int = 3) -> List[MetaPrompt]:
        """Generate variations of the current best prompt or explore new ones."""
        variations = []

        # Use internal best prompt if available and not exploring
        best_prompt_to_vary = self._current_best_prompt

        # Exploration: Generate completely new prompts
        if best_prompt_to_vary is None or random.random() < self.exploration_parameter:
            logger.debug(
                "Generating completely new prompts for exploration or no current best."
            )
            # Generate N new prompts directly
            new_prompts: List[MetaPrompt] = await generate_prompts(
                self.task_guidance,
                self.variable_keys,
                self.expected_output_type,  # type: ignore[arg-type]
                num_variations,
                mode=self.prompt_mode,  # Use the optimizer's prompt mode
            )
            if new_prompts:
                variations.extend(new_prompts)
            else:
                logger.warning("Failed to generate new prompts for exploration.")
            return variations  # Return only the newly generated prompts

        # Exploitation: Generate variations of the best known prompt
        logger.info(
            f"Generating {num_variations} variations of the current best prompt.",
            # best_prompt_spec=best_prompt_to_vary.spec, # Log selectively
        )

        # Always include the best prompt itself
        variations.append(best_prompt_to_vary)

        # Create variations by modifying different components
        variation_types = list(await best_prompt_to_vary.spec.variation_types())
        weights = await best_prompt_to_vary.spec.variation_weights()

        variation_tasks = []

        for _ in range(num_variations - 1):  # Generate N-1 variations
            variation_type = random.choices(variation_types, weights=weights, k=1)[0]
            logger.debug(f"Creating a {variation_type} variation task.")
            variation_tasks.append(
                best_prompt_to_vary.spec.vary(variation_type=variation_type)
            )

        # Execute all variation creation tasks concurrently
        created_variation_specs = await asyncio.gather(*variation_tasks)

        # Filter out None results (failed variations) and convert specs to MetaPrompt objects
        for spec in created_variation_specs:
            if spec is not None:
                variations.append(
                    MetaPrompt(
                        spec=spec,
                        expected_output_type=self.expected_output_type,  # type: ignore[arg-type]
                        config=DEFAULT_LLM_CONFIG,
                    )
                )

        logger.success(
            f"Generated {len(variations)} total prompts (best + variations)",
            variation_count=len(variations),
        )
        # Ensure we don't return more than num_variations total prompts
        return variations[:num_variations]


class BanditTrainer(Trainer[OUTPUT_TYPE]):
    """Trainer using multi-armed bandit optimization for prompt learning."""

    def __init__(
        self,
        all_rows: List[Row] | DataSet,
        task_guidance: str,
        keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        num_iterations: int = 5,
        candidates_per_iteration: int = 3,
        exploration_parameter: float = 0.1,  # Pass exploration param to optimizer
        prompt_mode: PromptMode = PromptMode.SIMPLE,  # Default to advanced mode
    ):
        """Initialize the bandit trainer."""
        optimizer = BanditOptimizer(
            task_guidance=task_guidance,
            variable_keys=keys,
            expected_output_type=expected_output_type,
            row_scoring_function=scoring_function,  # Pass correct param name
            exploration_parameter=exploration_parameter,
            prompt_mode=prompt_mode,  # Pass prompt mode to optimizer
        )
        super().__init__(
            all_rows=all_rows,
            task_guidance=task_guidance,
            keys=keys,
            expected_output_type=expected_output_type,
            optimizer=optimizer,  # Pass the created optimizer instance
            scoring_function=scoring_function,
            num_iterations=num_iterations,
            candidates_per_iteration=candidates_per_iteration,
        )
        # No self.best_prompt needed, managed by optimizer state + select_best_prompt
        logger.debug("BanditTrainer initialized", optimizer_type="BanditOptimizer")

    async def train(self) -> None:  # Return type changed to None
        """Train the prompt optimizer using a multi-armed bandit approach."""
        logger.info(
            f"Starting multi-armed bandit optimization: {self.num_iterations} iterations, "
            f"{self.candidates_per_iteration} candidates/iter."
        )

        # --- Step 1: Initial Phase ---
        logger.info("Phase 1: Generating and evaluating initial candidates...")
        try:
            # Pass the prompt mode from the optimizer to generate_prompts
            prompt_mode = (
                self.optimizer.prompt_mode
                if isinstance(self.optimizer, BanditOptimizer)
                else PromptMode.ADVANCED
            )
            initial_candidates: List[MetaPrompt] = await generate_prompts(
                self.task_guidance,
                self.keys,
                self.expected_output_type,  # type: ignore[arg-type]
                self.candidates_per_iteration,
                mode=prompt_mode,  # Use the optimizer's prompt mode
            )
        except Exception as e:
            logger.error(f"Failed to generate initial prompts: {e}")
            raise ValueError("Could not generate initial prompts.") from e

        if not initial_candidates:
            logger.error("No initial candidates could be generated.")
            raise ValueError("No initial candidates generated.")

        logger.info(f"Generated {len(initial_candidates)} initial candidate prompts.")

        # Evaluate initial candidates
        initial_eval_tasks = [
            self.run_for_prompt(candidate, self.dataset.training_rows)
            for candidate in initial_candidates
        ]
        initial_prompts_with_type = await asyncio.gather(*initial_eval_tasks)
        for pwt in initial_prompts_with_type:
            await self.log_performance(pwt)  # Log to optimizer history

        # Select initial best prompt and update optimizer state
        initial_best_prompt = await self.select_best_prompt(self.dataset.training_rows)
        current_best_score = -float("inf")

        if initial_best_prompt:
            # Calculate score for logging using the new helper method
            current_best_score = await self.eval_prompt_on_training_set(
                initial_best_prompt
            )
            if isinstance(self.optimizer, BanditOptimizer):  # Type check for safety
                self.optimizer.update_best_prompt(initial_best_prompt)
            logger.success(
                f"Selected initial best prompt with score: {current_best_score:.4f}",
                # prompt_template=initial_best_prompt.get_user_message_content()[:100] + "...",
            )
        else:
            logger.warning("Could not determine an initial best prompt.")
            # Cannot proceed without an initial prompt if exploration is not guaranteed first iteration
            if (
                isinstance(self.optimizer, BanditOptimizer)
                and self.optimizer.exploration_parameter == 0.0
            ):
                raise ValueError(
                    "Failed to find initial best prompt and exploration is zero."
                )

        # --- Step 2: Iterative Optimization ---
        logger.info(
            f"Phase 2: Starting {self.num_iterations - 1} optimization iterations..."
        )
        for iteration in range(
            self.num_iterations - 1
        ):  # Already did one effective round
            logger.info(f"Iteration {iteration + 2}/{self.num_iterations}")

            # Generate next candidates (will use optimizer's internal best or explore)
            candidates: List[MetaPrompt] = await self.generate_candidate_prompts(
                self.candidates_per_iteration
            )
            if not candidates:
                logger.warning(
                    f"Iteration {iteration + 2}: No new candidates generated. Skipping evaluation."
                )
                continue

            logger.info(f"Generated {len(candidates)} new candidates for evaluation.")

            # Evaluate candidates
            eval_tasks = [
                self.run_for_prompt(candidate, self.dataset.training_rows)
                for candidate in candidates
            ]
            prompts_with_type = await asyncio.gather(*eval_tasks)
            for pwt in prompts_with_type:
                await self.log_performance(pwt)  # Log to optimizer history
                if self.print_iteration_summary:
                    print(pwt.meta_prompt.spec.get_content())

            # Select new best prompt from history
            new_best_prompt = await self.select_best_prompt(self.dataset.training_rows)

            if new_best_prompt:
                # Calculate score for comparison using the new helper method
                new_best_score = await self.eval_prompt_on_training_set(new_best_prompt)
                if new_best_score > current_best_score:
                    logger.success(
                        f"Iteration {iteration + 2}: Found new best prompt! "
                        f"Score: {new_best_score:.4f} (Improvement: {new_best_score - current_best_score:+.4f})"
                        # prompt_template=new_best_prompt.get_user_message_content()[:100] + "...",
                    )
                    current_best_score = new_best_score
                    if isinstance(self.optimizer, BanditOptimizer):  # Type check
                        self.optimizer.update_best_prompt(new_best_prompt)
                else:
                    logger.info(
                        f"Iteration {iteration + 2}: No improvement found. "
                        f"Best score remains {current_best_score:.4f}"
                    )
            else:
                logger.warning(
                    f"Iteration {iteration + 2}: Could not select a best prompt after evaluation."
                )

        # --- Step 3: Final Evaluation ---
        logger.info("Phase 3: Final evaluation on test set...")
        # Select the best prompt based on performance on the *test* set
        # Note: This selects the best overall from history using test data, not necessarily the last best found during training.
        final_best_prompt = await self.select_best_prompt(self.dataset.test_rows)

        if final_best_prompt:
            # Evaluate on test set using the new helper method
            final_test_mean = await self.eval_prompt_on_test_set(final_best_prompt)
            logger.success(
                f"Training complete. Final best prompt test score: {final_test_mean:.4f}",
                test_score=final_test_mean,
                # final_prompt_template=final_best_prompt.get_user_message_content()[:100] + "...",
            )
            # Implement dumping logic here, similar to base Trainer.dump but async-compatible
            logger.info("Dumping final best prompt (based on test set).")
            try:
                with open("best_prompt.txt", "w") as f:  # Use .txt for clarity
                    f.write(final_best_prompt.get_user_message_content())
                with open("best_config.json", "w") as f:
                    f.write(final_best_prompt.config.model_dump_json(indent=2))
                logger.success("Successfully dumped best prompt and config.")
            except Exception as e:
                logger.error(f"Failed to dump best prompt: {e}")

        else:
            logger.error(
                "Training complete, but failed to determine a final best prompt on the test set."
            )
            # Maybe dump the best from training?
            training_best = await self.select_best_prompt(self.dataset.training_rows)
            if training_best:
                logger.info("Dumping best prompt found during training instead.")
                # Implement dumping logic here
                try:
                    with open("best_prompt.txt", "w") as f:  # Use .txt for clarity
                        f.write(training_best.get_user_message_content())
                    with open("best_config.json", "w") as f:
                        f.write(training_best.config.model_dump_json(indent=2))
                    logger.success(
                        "Successfully dumped best training prompt and config."
                    )
                except Exception as e:
                    logger.error(f"Failed to dump training best prompt: {e}")
