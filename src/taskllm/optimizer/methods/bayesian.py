import asyncio
import random
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, cast
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

# For Bayesian optimization components
import pymc as pm
import arviz as az
from scipy import stats

from ...ai import DEFAULT_LLM_CONFIG
from ..data import DataSet, Row
from ..prompt.meta import MetaPrompt, PromptMode, generate_prompts

# Base optimizer components
from .base import OUTPUT_TYPE, BaseOptimizer, PromptWithType, Trainer


class BayesianParams(BaseModel):
    """Parameters for the Bayesian optimizer."""
    
    exploration_weight: float = 0.1
    kernel_lengthscale: float = 1.0
    kernel_variance: float = 1.0
    noise_variance: float = 0.1
    num_samples: int = 1000
    num_tune: int = 500
    num_chains: int = 2
    random_seed: int = 42
    feature_dimension: int = 5  # Default feature dimension


class BayesianOptimizer(BaseOptimizer[OUTPUT_TYPE]):
    """Bayesian optimizer using Gaussian Processes for prompt selection."""
    
    params: BayesianParams = BayesianParams()
    _surrogate_model: Optional[Tuple[pm.Model, az.InferenceData]] = None
    _feature_cache: Dict[int, np.ndarray] = {}  # Cache for prompt features
    prompt_mode: PromptMode = PromptMode.ADVANCED  # Default mode
    acquisition_function: Literal["ei", "ucb"] = "ei"
    
    def __init__(
        self,
        task_guidance: str,
        variable_keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        acquisition_function: Literal["ei", "ucb"] = "ei",
        exploration_weight: float = 0.1,
        prompt_mode: PromptMode = PromptMode.ADVANCED,
    ):
        super().__init__(
            task_guidance=task_guidance,
            variable_keys=variable_keys,
            expected_output_type=expected_output_type,
            row_scoring_function=row_scoring_function,
            prompt_history=[],
        )
        self.acquisition_function = acquisition_function
        self.params.exploration_weight = exploration_weight
        self.prompt_mode = prompt_mode
        self._surrogate_model = None
        self._feature_cache = {}

    def _extract_prompt_features(self) -> np.ndarray:
        """Extract features from all prompts in history for model training.

        Returns:
            Feature matrix with shape (n_prompts, feature_dimension)
        """
        if not self.prompt_history:
            logger.warning("No prompt history available for feature extraction")
            return np.array([]).reshape(0, self.params.feature_dimension)

        # First check cache for all prompts
        features = []
        for prompt_with_type in self.prompt_history:
            prompt = prompt_with_type.meta_prompt
            prompt_hash = hash(prompt.spec)

            # Use cached features if available
            if prompt_hash in self._feature_cache:
                features.append(self._feature_cache[prompt_hash])
            else:
                # Extract and cache features
                prompt_features = self._extract_single_prompt_features(prompt)
                self._feature_cache[prompt_hash] = prompt_features
                features.append(prompt_features)

        return np.vstack(features)

    def _extract_single_prompt_features(self, prompt: MetaPrompt) -> np.ndarray:
        """Extract features for a single prompt.

        Args:
            prompt: The prompt to extract features from

        Returns:
            Feature vector with shape (feature_dimension,)
        """
        # Get the prompt content
        content = prompt.get_user_message_content()

        # Extract simple text features
        features = np.zeros(self.params.feature_dimension)

        # Feature 1: Prompt length (normalized)
        features[0] = min(len(content) / 1000, 1.0)  # Cap at 1.0

        # Feature 2: Number of instructions or steps (count by lines or paragraph breaks)
        instruction_count = content.count("\n\n") + 1
        features[1] = min(instruction_count / 10, 1.0)  # Cap at 1.0

        # Feature 3: Keyword presence - look for instructional keywords
        instruction_keywords = ["must", "should", "avoid", "ensure", "consider"]
        keyword_count = sum(1 for keyword in instruction_keywords if keyword.lower() in content.lower())
        features[2] = min(keyword_count / len(instruction_keywords), 1.0)

        # Feature 4: Question marks - indicates more interactive prompt
        question_count = content.count("?")
        features[3] = min(question_count / 5, 1.0)  # Cap at 1.0

        # Feature 5: Presence of examples (code blocks, etc.)
        example_indicators = ["example", "for instance", "e.g.", "```", "example:"]
        example_count = sum(1 for indicator in example_indicators if indicator.lower() in content.lower())
        features[4] = min(example_count / len(example_indicators), 1.0)

        # Add random noise for exploration
        random_noise = np.random.normal(0, 0.01, self.params.feature_dimension)
        features = np.clip(features + random_noise, 0, 1)

        return features

    async def fit_surrogate_model(self) -> Tuple[Optional[pm.Model], Optional[az.InferenceData]]:
        """Fit a GP surrogate model to the performance data and return model + trace.

        Returns:
            Tuple of (model, inference_data) or (None, None) if fitting fails
        """
        if not self.prompt_history:
            logger.warning("No prompt history available for model fitting")
            return None, None

        try:
            # Extract features from prompts
            X = self._extract_prompt_features()

            # Get the scores for each prompt
            score_tasks = []
            for pwt in self.prompt_history:
                # Use training rows from dataset for scoring
                # This assumes we have access to the dataset as in BanditTrainer
                # For now we'll use a small subset of rows for efficiency
                rows = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: random.sample(list(self.prompt_history),
                                                 min(5, len(self.prompt_history)))
                )
                score_tasks.append(pwt.calculate_scores(rows, self.row_scoring_function))

            scores_list = await asyncio.gather(*score_tasks)
            y = np.array(scores_list)

            if len(y) < 2:
                logger.warning("Not enough scored prompts for model fitting")
                return None, None

            # Normalize scores for better GP stability
            y_mean = np.mean(y)
            y_std = np.std(y) if np.std(y) > 0 else 1.0
            y_normalized = (y - y_mean) / y_std

            # Create and fit surrogate model using PyMC's Gaussian Process
            with pm.Model() as model:
                # Set priors for GP hyperparameters
                ls = pm.HalfNormal("ls", sigma=self.params.kernel_lengthscale)
                η = pm.HalfNormal("η", sigma=self.params.kernel_variance)
                σ = pm.HalfNormal("σ", sigma=self.params.noise_variance)

                # Define the covariance function (RBF/squared exponential kernel)
                cov_func = η * pm.gp.cov.ExpQuad(X.shape[1], ls=ls)

                # Create the GP and add the observations
                gp = pm.gp.Marginal(cov_func=cov_func)
                gp.marginal_likelihood("y", X=X, y=y_normalized, sigma=σ)

                # Sample from the posterior
                inference_data = pm.sample(
                    self.params.num_samples,
                    tune=self.params.num_tune,
                    chains=self.params.num_chains,
                    random_seed=self.params.random_seed,
                    progressbar=False,
                    return_inferencedata=True,
                )

            # Store normalization constants for prediction
            model.y_mean = y_mean  # type: ignore
            model.y_std = y_std  # type: ignore

            # Store the model and trace for future predictions
            self._surrogate_model = (model, inference_data)
            logger.info(f"Successfully fit GP surrogate model with {len(y)} data points")

            return model, inference_data

        except Exception as e:
            logger.error(f"Failed to fit surrogate model: {e}")
            return None, None

    async def predict_performance(self, prompt: MetaPrompt) -> Tuple[float, float]:
        """Predict performance (mean and std) for a new prompt.

        Args:
            prompt: The prompt to predict performance for

        Returns:
            Tuple of (mean, std) for the predicted performance
        """
        # If no model is available, return a default uncertainty-based prediction
        if self._surrogate_model is None:
            logger.warning("No surrogate model available for prediction - using default values")
            return 0.5, 1.0  # Default mean and high uncertainty

        try:
            # Extract features for the new prompt
            x_new = self._extract_single_prompt_features(prompt).reshape(1, -1)

            model, inference_data = self._surrogate_model

            # Use functional predictions with the GP model
            with model:  # type: ignore
                # Extract required variables from trace
                ls_trace = inference_data.posterior["ls"].mean().item()  # type: ignore
                η_trace = inference_data.posterior["η"].mean().item()  # type: ignore
                σ_trace = inference_data.posterior["σ"].mean().item()  # type: ignore

                # Recreate the covariance function with trace values
                cov_func = η_trace * pm.gp.cov.ExpQuad(x_new.shape[1], ls=ls_trace)

                # Create the GP with the fitted hyperparameters
                gp = pm.gp.Marginal(cov_func=cov_func)

                # Get training data from model
                X = self._extract_prompt_features()

                # Get scores
                score_tasks = []
                for pwt in self.prompt_history:
                    rows = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: random.sample(list(self.prompt_history),
                                                     min(5, len(self.prompt_history)))
                    )
                    score_tasks.append(pwt.calculate_scores(rows, self.row_scoring_function))

                scores_list = await asyncio.gather(*score_tasks)
                y = np.array(scores_list)

                # Normalize scores using stored constants
                y_normalized = (y - model.y_mean) / model.y_std  # type: ignore

                # Compute predictive distribution using fitted hyperparameters
                mu, var = gp.predict(X, y_normalized, x_new, predictive=True, sigma=σ_trace)

                # Convert standard deviation from variance
                std = np.sqrt(var).item()

                # Denormalize predictions
                mu_original = mu.item() * model.y_std + model.y_mean  # type: ignore
                std_original = std * model.y_std  # type: ignore

                logger.debug(f"Predicted performance: mean={mu_original:.4f}, std={std_original:.4f}")
                return mu_original, std_original

        except Exception as e:
            logger.error(f"Error in performance prediction: {e}")
            return 0.0, 1.0  # Default in case of error

    async def calculate_acquisition(
        self,
        prompt: MetaPrompt,
        best_score: float
    ) -> float:
        """Calculate acquisition function value based on selected function.

        Args:
            prompt: The prompt to evaluate
            best_score: The best score observed so far

        Returns:
            Acquisition function value
        """
        mean, std = await self.predict_performance(prompt)

        if self.acquisition_function == "ei":
            return self._expected_improvement(mean, std, best_score)
        elif self.acquisition_function == "ucb":
            return self._upper_confidence_bound(mean, std)
        else:
            # Default to expected improvement
            return self._expected_improvement(mean, std, best_score)

    def _expected_improvement(self, mean: float, std: float, best_score: float) -> float:
        """Expected improvement acquisition function.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            best_score: Best observed score so far

        Returns:
            Expected improvement value
        """
        # Handle numerical instabilities
        if std <= 1e-6:
            # If uncertainty is too small, we either:
            # - Return 0 (no expected improvement) if mean is worse than best
            # - Return a small positive value if mean is better than best
            return max(0, mean - best_score)

        # Calculate z-score for the improvement
        z = (mean - best_score) / std

        # Calculate expected improvement using the formula:
        # EI(x) = (μ(x) - f_best) * Φ(z) + σ(x) * φ(z)
        # Where Φ is the CDF and φ is the PDF of the standard normal distribution
        improvement = (mean - best_score) * stats.norm.cdf(z) + std * stats.norm.pdf(z)

        # Ensure we don't return invalid values
        if np.isnan(improvement) or np.isinf(improvement):
            logger.warning(f"EI calculation produced invalid value: {improvement}")
            return 0.0

        # Return max of 0 and improvement to avoid negative values
        return max(0, improvement)

    def _upper_confidence_bound(self, mean: float, std: float) -> float:
        """Upper confidence bound acquisition function.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation

        Returns:
            UCB value
        """
        # UCB = mean + exploration_weight * std
        ucb = mean + self.params.exploration_weight * std

        # Handle numerical instabilities
        if np.isnan(ucb) or np.isinf(ucb):
            logger.warning(f"UCB calculation produced invalid value: {ucb}")
            return mean  # Fall back to just using the mean

        return ucb

    async def select_next_prompts(self, num_variations: int = 3) -> List[MetaPrompt]:
        """Generate variations of prompts using Bayesian optimization.

        Args:
            num_variations: Number of prompt variations to generate

        Returns:
            List of selected prompts
        """
        # If we don't have enough history, generate random prompts
        if len(self.prompt_history) < 2:
            logger.info(f"Not enough prompt history ({len(self.prompt_history)}), generating initial prompts")
            return await generate_prompts(
                self.task_guidance,
                self.variable_keys,
                self.expected_output_type,  # type: ignore
                num_variations,
                mode=self.prompt_mode,
            )

        try:
            # Fit or update the surrogate model
            if self._surrogate_model is None:
                logger.info("Fitting new surrogate model")
                await self.fit_surrogate_model()

            # Find current best score
            logger.info("Calculating best score from history")
            best_prompt = await self.select_best_prompt([])  # Use default logic from base class
            if best_prompt is None:
                logger.warning("No best prompt found, using default score")
                best_score = 0.0
            else:
                # Get score for best prompt to use in acquisition function
                # Here we use the first prompt's score as a proxy since calculating
                # the actual best score might be expensive
                if self.prompt_history:
                    pwt = self.prompt_history[0]
                    rows = self.prompt_history[0:min(5, len(self.prompt_history))]  # Small subset
                    scores = await pwt.get_scores(rows, self.row_scoring_function)
                    best_score = max(scores) if scores else 0.0
                else:
                    best_score = 0.0

            # Generate a pool of candidate prompts
            logger.info(f"Generating candidate pool for Bayesian selection")

            # Always include the current best prompt
            variations = []
            if best_prompt is not None:
                variations.append(best_prompt)

            # Generate a larger pool of candidates to select from using base variation methods
            candidate_pool: List[MetaPrompt] = []

            # Try to use the best prompt for variations first
            prompt_to_vary = best_prompt

            # If no best prompt is available, use the most recent one
            if prompt_to_vary is None and self.prompt_history:
                prompt_to_vary = self.prompt_history[-1].meta_prompt

            # If we have a prompt to vary, generate variations
            if prompt_to_vary is not None:
                # Get variation methods specific to the prompt spec type
                variation_types = list(await prompt_to_vary.spec.variation_types())
                weights = await prompt_to_vary.spec.variation_weights()

                # Create multiple variation tasks
                variation_tasks = []
                for _ in range(num_variations * 2):  # Generate more candidates than needed
                    # Select a random variation type with weights
                    variation_type = random.choices(variation_types, weights=weights, k=1)[0]
                    variation_tasks.append(prompt_to_vary.spec.vary(variation_type=variation_type))

                # Execute all variation tasks concurrently
                variation_specs = await asyncio.gather(*variation_tasks)

                # Filter out None results and convert to MetaPrompt objects
                for spec in variation_specs:
                    if spec is not None:
                        candidate_pool.append(
                            MetaPrompt(
                                spec=spec,
                                expected_output_type=self.expected_output_type,  # type: ignore
                                config=DEFAULT_LLM_CONFIG,
                            )
                        )

            # If we couldn't generate variations, create new prompts
            if not candidate_pool:
                logger.warning("Could not generate variations, creating new prompts")
                candidate_pool = await generate_prompts(
                    self.task_guidance,
                    self.variable_keys,
                    self.expected_output_type,  # type: ignore
                    num_variations,
                    mode=self.prompt_mode,
                )

            # Calculate acquisition function values for all candidates
            acquisition_values = []
            for candidate in candidate_pool:
                acq_value = await self.calculate_acquisition(candidate, best_score)
                acquisition_values.append((candidate, acq_value))

            # Sort by acquisition value (descending) and select top candidates
            acquisition_values.sort(key=lambda x: x[1], reverse=True)

            # Get top candidates
            top_candidates = [candidate for candidate, _ in acquisition_values[:num_variations]]

            # Always include best prompt if we have one and there's room
            if best_prompt is not None and best_prompt not in top_candidates and len(top_candidates) < num_variations:
                top_candidates.append(best_prompt)

            # Limit to requested number of variations
            selected_candidates = top_candidates[:num_variations]

            logger.info(f"Selected {len(selected_candidates)} prompts using Bayesian optimization")
            return selected_candidates

        except Exception as e:
            logger.error(f"Error in Bayesian prompt selection: {e}")
            # Fallback to simpler generation method
            logger.warning("Falling back to default prompt generation")
            return await generate_prompts(
                self.task_guidance,
                self.variable_keys,
                self.expected_output_type,  # type: ignore
                num_variations,
                mode=self.prompt_mode,
            )

    async def select_best_prompt(self, rows: List[Row]) -> Optional[MetaPrompt]:
        """Select the best prompt from history based on performance on rows.

        Args:
            rows: List of rows to evaluate prompts on

        Returns:
            Best performing prompt or None if no history
        """
        if not self.prompt_history:
            logger.warning("No performance history available to select best prompt")
            return None

        # If no specific rows provided, use a subset of prompt history for evaluation
        if not rows and self.prompt_history:
            # Create a small set of rows from the first few prompts
            test_prompts = self.prompt_history[:min(5, len(self.prompt_history))]
            logger.info(f"No rows provided, using {len(test_prompts)} prompts for evaluation")
            # We'll calculate scores directly using our surrogate model

            best_prompt = None
            best_score = float('-inf')

            for pwt in self.prompt_history:
                prompt = pwt.meta_prompt
                # Predict performance using our surrogate model
                mean, _ = await self.predict_performance(prompt)

                if mean > best_score:
                    best_score = mean
                    best_prompt = prompt

            logger.info(f"Selected best prompt using surrogate model with predicted score: {best_score:.4f}")
            return best_prompt

        # If rows are provided, use the standard method from the base class
        logger.info(f"Selecting best prompt from {len(self.prompt_history)} prompts using {len(rows)} evaluation rows")
        return await super().select_best_prompt(rows)


class BayesianTrainer(Trainer[OUTPUT_TYPE]):
    """Trainer using Bayesian optimization for prompt learning."""

    def __init__(
        self,
        all_rows: List[Row] | DataSet,
        task_guidance: str,
        keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        num_iterations: int = 5,
        candidates_per_iteration: int = 3,
        acquisition_function: Literal["ei", "ucb"] = "ei",
        exploration_weight: float = 0.1,
        prompt_mode: PromptMode = PromptMode.ADVANCED,
    ):
        """Initialize the BayesianTrainer with appropriate configuration."""
        # Create the optimizer
        optimizer = BayesianOptimizer(
            task_guidance=task_guidance,
            variable_keys=keys,
            expected_output_type=expected_output_type,
            row_scoring_function=scoring_function,
            acquisition_function=acquisition_function,
            exploration_weight=exploration_weight,
            prompt_mode=prompt_mode,
        )
        super().__init__(
            all_rows=all_rows,
            task_guidance=task_guidance,
            keys=keys,
            expected_output_type=expected_output_type,
            optimizer=optimizer,
            scoring_function=scoring_function,
            num_iterations=num_iterations,
            candidates_per_iteration=candidates_per_iteration,
        )
        self.bayesian_optimizer = cast(BayesianOptimizer, self.optimizer)
        logger.debug("BayesianTrainer initialized", optimizer_type="BayesianOptimizer")

    async def train(self) -> None:
        """Train the prompt optimizer using Bayesian optimization."""
        logger.info(
            f"Starting Bayesian optimization: {self.num_iterations} iterations, "
            f"{self.candidates_per_iteration} candidates/iter."
        )

        # --- Step 1: Initial Phase - Generate and evaluate initial candidates ---
        logger.info("Phase 1: Generating and evaluating initial candidates...")
        try:
            # Generate initial prompts
            initial_candidates: List[MetaPrompt] = await generate_prompts(
                self.task_guidance,
                self.keys,
                self.expected_output_type,  # type: ignore
                self.candidates_per_iteration,
                mode=self.bayesian_optimizer.prompt_mode,
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

        # Fit initial surrogate model
        logger.info("Fitting initial surrogate model")
        await self.bayesian_optimizer.fit_surrogate_model()

        # --- Step 2: Iterative Optimization ---
        logger.info(f"Phase 2: Starting {self.num_iterations - 1} optimization iterations...")
        current_best_score = float('-inf')

        for iteration in range(self.num_iterations - 1):  # Already did one effective round
            logger.info(f"Iteration {iteration + 2}/{self.num_iterations}")

            # Get best prompt so far for reference
            best_prompt = await self.optimizer.select_best_prompt(self.dataset.training_rows)
            if best_prompt:
                # Get score for comparison
                best_score = await self.eval_prompt_on_training_set(best_prompt)
                if best_score > current_best_score:
                    current_best_score = best_score
                    logger.info(f"Current best score: {current_best_score:.4f}")

            # Generate next candidates using Bayesian optimization
            candidates: List[MetaPrompt] = await self.optimizer.select_next_prompts(
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
                    logger.info(f"Prompt content: {pwt.meta_prompt.spec.get_content()[:100]}...")

            # Update surrogate model with new data
            logger.info("Updating surrogate model with new data")
            await self.bayesian_optimizer.fit_surrogate_model()

            # Select new best prompt from history
            new_best_prompt = await self.select_best_prompt(self.dataset.training_rows)

            if new_best_prompt:
                # Calculate score for comparison
                new_best_score = await self.eval_prompt_on_training_set(new_best_prompt)
                if new_best_score > current_best_score:
                    logger.success(
                        f"Iteration {iteration + 2}: Found new best prompt! "
                        f"Score: {new_best_score:.4f} (Improvement: {new_best_score - current_best_score:+.4f})"
                    )
                    current_best_score = new_best_score
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

        # Final model update with all data
        await self.bayesian_optimizer.fit_surrogate_model()

        # Select the best prompt based on performance on the test set
        final_best_prompt = await self.select_best_prompt(self.dataset.test_rows)

        if final_best_prompt:
            # Evaluate on test set
            final_test_mean = await self.eval_prompt_on_test_set(final_best_prompt)
            logger.success(
                f"Training complete. Final best prompt test score: {final_test_mean:.4f}",
                test_score=final_test_mean,
            )
            # Save the best prompt
            logger.info("Saving final best prompt (based on test set).")
            try:
                with open("best_prompt.txt", "w") as f:
                    f.write(final_best_prompt.get_user_message_content())
                with open("best_config.json", "w") as f:
                    f.write(final_best_prompt.config.model_dump_json(indent=2))
                logger.success("Successfully saved best prompt and config.")
            except Exception as e:
                logger.error(f"Failed to save best prompt: {e}")
        else:
            logger.error(
                "Training complete, but failed to determine a final best prompt on the test set."
            )
            # Try to save the best prompt from training instead
            training_best = await self.select_best_prompt(self.dataset.training_rows)
            if training_best:
                logger.info("Saving best prompt found during training instead.")
                try:
                    with open("best_prompt.txt", "w") as f:
                        f.write(training_best.get_user_message_content())
                    with open("best_config.json", "w") as f:
                        f.write(training_best.config.model_dump_json(indent=2))
                    logger.success("Successfully saved best training prompt and config.")
                except Exception as e:
                    logger.error(f"Failed to save training best prompt: {e}")