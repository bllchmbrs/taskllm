import asyncio
import math
import random
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Type, cast

import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.optim
import torch
from loguru import logger
from pydantic import BaseModel, Field
from pyro.infer import SVI, Trace_ELBO
from scipy import stats  # type: ignore

from ..data import DataSet, Row
from ..prompt.meta import (
    DEFAULT_LLM_CONFIG,
    LLMConfig,
    MetaPrompt,
    MetaPromptSpecBase,
    ModelsEnum,
    PromptMode,
    generate_prompts,
)

# Base optimizer components
from .base import OUTPUT_TYPE, BaseOptimizer, FailureTracker, PromptWithType, Trainer


def calculate_model_complexity(model: str) -> float:
    """Calculate the complexity of a model.

    Args:
        model: The model to calculate the complexity of

    """
    _model = model.lower()
    if "nano" in _model:
        return 0.1
    elif "mini" in _model:
        return 0.3
    elif "haiku" in _model:
        return 0.3
    elif "sonnet" in _model:
        return 0.7
    elif "large" in _model:
        return 0.9

    match = re.search(r"(\d+)b", _model)
    if match:
        num = int(match.group(1)) / 100.0
        return num

    return 0.5


def calculate_model_provider(model: str, candidate_providers: List[str]) -> List[float]:
    """Calculate the provider of a model using one-hot encoding.

    Args:
        model: The model to calculate the provider of.
        candidate_providers: A list of possible providers.

    Returns:
        A one-hot encoded list representing the model's provider.
    """
    providers = [c.split("/")[0] for c in candidate_providers]
    unique_providers = list(set(providers))
    model_provider = model.split("/")[0]

    encoding = [0.0] * len(unique_providers)
    if model_provider in unique_providers:
        index = unique_providers.index(model_provider)
        encoding[index] = 1.0
    return encoding


class BayesianParams(BaseModel):
    """Parameters for the Bayesian optimizer."""

    exploration_weight: float = 0.1
    kernel_lengthscale: float = 1.0
    kernel_variance: float = 1.0
    noise_variance: float = 0.1
    num_iterations: int = 1000  # For SVI optimization
    num_warmup: int = 100  # Warmup iterations
    random_seed: int = 42
    learning_rate: float = 0.01  # Learning rate for Adam optimizer


def _extract_single_prompt_features(prompt: MetaPrompt) -> np.ndarray:
    """Extract features for a single prompt.

    Args:
        prompt: The prompt to extract features from

    Returns:
        Feature vector with shape (feature_dimension,)
    """
    # Get the prompt content
    content = prompt.get_user_message_content()

    # Initialize feature list
    features = []

    # Feature 1: Prompt length (normalized)
    features.append(min(len(content) / 1000, 1.0))

    # Feature 2: Number of instructions or steps (count by lines or paragraph breaks)
    instruction_count = content.count("\n\n") + 1
    features.append(min(instruction_count / 10, 1.0))

    # Feature 3: Keyword presence - look for instructional keywords
    instruction_keywords = ["must", "should", "avoid", "ensure", "consider"]
    keyword_count = sum(
        1 for keyword in instruction_keywords if keyword.lower() in content.lower()
    )
    features.append(min(keyword_count / len(instruction_keywords), 1.0))

    # Feature 4: Question marks - indicates more interactive prompt
    question_count = content.count("?")
    features.append(min(question_count / 5, 1.0))

    # Feature 5: Model complexity
    model_complexity = calculate_model_complexity(prompt.config.model)
    features.append(model_complexity)

    model_provider = calculate_model_provider(prompt.config.model, list(ModelsEnum))
    features.extend(model_provider)

    # Convert to numpy array
    feature_array = np.array(features)
    feature_dimension = len(feature_array)

    # Add random noise for exploration
    random_noise = np.random.normal(0, 0.01, feature_dimension)
    feature_array = np.clip(feature_array + random_noise, 0, 1).astype(np.float64)

    return feature_array


def _extract_prompt_features(prompts: List[MetaPrompt]) -> torch.Tensor | None:
    """Extract features from a list of prompts for model training.

    Args:
        prompts: List of MetaPrompt objects.

    Returns:
        Feature tensor with shape (n_prompts, feature_dimension)
    """
    if not prompts:
        logger.warning("No prompts provided for feature extraction")
        return None
    features = []
    for prompt in prompts:
        prompt_features = _extract_single_prompt_features(prompt)
        features.append(prompt_features)

    return torch.tensor(np.vstack(features), dtype=torch.float32, device="cpu")


class BayesianOptimizer(BaseOptimizer[OUTPUT_TYPE]):
    """Bayesian optimizer using Gaussian Processes for prompt selection."""

    params: BayesianParams = BayesianParams()
    _surrogate_model: Optional[gp.models.GPRegression] = None
    _feature_cache: Dict[int, np.ndarray] = {}  # Cache for prompt features
    prompt_mode: PromptMode = PromptMode.ADVANCED  # Default mode
    acquisition_function: Literal["ei", "ucb"] = "ei"
    _torch_device: torch.device = torch.device("cpu")
    _normalization_params: Optional[Tuple[float, float]] = None  # (y_mean, y_std)

    def __init__(
        self,
        task_guidance: str,
        variable_keys: List[str],
        expected_output_type: Type[OUTPUT_TYPE],
        row_scoring_function: Callable[[Row, OUTPUT_TYPE | None], float],
        acquisition_function: Literal["ei", "ucb"] = "ucb",
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

        # Initialize Pyro
        setup_pyro(self.params.random_seed)

        # Check for GPU availability
        if torch.cuda.is_available():
            self._torch_device = torch.device("cuda")
            logger.info("Using GPU for Pyro/Torch computations")
        else:
            self._torch_device = torch.device("cpu")
            logger.info("Using CPU for Pyro/Torch computations")

    def _create_gp_model(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> gp.models.GPRegression:
        """Create a Gaussian Process model using Pyro.

        Args:
            X: Input features tensor of shape (n_samples, n_features)
            y: Target values tensor of shape (n_samples,)

        Returns:
            Configured GP regression model
        """
        # Create kernel with priors
        kernel = gp.kernels.RBF(
            input_dim=X.shape[1],
            variance=torch.tensor(
                self.params.kernel_variance, device=self._torch_device
            ),
            lengthscale=torch.tensor(
                self.params.kernel_lengthscale, device=self._torch_device
            ),
        )

        # Create the GP model
        gp_model = gp.models.GPRegression(
            X=X,
            y=y,
            kernel=kernel,
            noise=torch.tensor(self.params.noise_variance, device=self._torch_device),
            jitter=1e-5,
        )

        return gp_model

    async def fit_surrogate_model(
        self, rows: List[Row]
    ) -> Optional[gp.models.GPRegression]:
        """Fit a GP surrogate model to the performance data.

        Args:
            rows: Data rows to evaluate prompt performance on

        Returns:
            Fitted GP model or None if fitting fails
        """
        if not self.prompt_history:
            logger.warning("No prompt history available for model fitting")
            return None

        try:
            # Extract features and convert to torch tensor
            X = _extract_prompt_features(
                cast(List[MetaPrompt], [pwt.meta_prompt for pwt in self.prompt_history])
            )
            if X is None:
                logger.error("No features extracted, skipping model fitting")
                return None
            logger.success(f"Extracted features for {X.shape[0]} prompts")
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

        # Get the scores for each prompt
        score_tasks = []
        for pwt in self.prompt_history:
            score_tasks.append(pwt.calculate_scores(rows, self.row_scoring_function))

        scores_list = await asyncio.gather(*score_tasks)
        y_np = np.array(scores_list)
        logger.info(f"Fitting GP surrogate model with {len(y_np)} data points")

        if len(y_np) < 2:
            logger.warning("Not enough scored prompts for model fitting")
            return None

        # Normalize scores for better GP stability
        y_mean = float(np.mean(y_np))
        y_std = float(np.std(y_np)) if np.std(y_np) > 0 else 1.0
        y_normalized = (y_np - y_mean) / y_std

        # Store normalization constants
        self._normalization_params = (y_mean, y_std)

        # Convert to torch tensor
        y = torch.tensor(y_normalized, dtype=torch.float32, device=self._torch_device)

        # Create the GP model
        gp_model = self._create_gp_model(X, y)

        # Setup optimizer
        optimizer = pyro.optim.Adam({"lr": self.params.learning_rate})  # type: ignore
        elbo = Trace_ELBO()

        # Create SVI for training
        svi = SVI(gp_model.model, gp_model.guide, optimizer, loss=elbo)

        # Run optimization
        losses = []
        for i in range(self.params.num_iterations):
            loss = svi.step()
            losses.append(loss)
            if i % 100 == 0:
                logger.debug(
                    f"SVI iteration {i}/{self.params.num_iterations}, Loss: {loss:.4f}"
                )

        # Set the fitted model
        self._surrogate_model = gp_model
        logger.info(f"Successfully fit Pyro GP model with final loss: {losses[-1]:.4f}")

        return gp_model

    async def predict_performance(self, prompt: MetaPrompt) -> Tuple[float, float]:
        """Predict performance (mean and std) for a new prompt.

        Args:
            prompt: The prompt to predict performance for

        Returns:
            Tuple of (mean, std) for the predicted performance
        """
        # If no model is available, return a default uncertainty-based prediction
        if self._surrogate_model is None or self._normalization_params is None:
            logger.warning(
                "No surrogate model available for prediction - using default values"
            )
            return 0.5, 1.0  # Default mean and high uncertainty

        try:
            # Extract features for the new prompt
            x_new_np = _extract_single_prompt_features(prompt).reshape(1, -1)
            x_new = torch.tensor(
                x_new_np, dtype=torch.float32, device=self._torch_device
            )

            # Get predictions from GP model
            with torch.no_grad():
                mean, variance = self._surrogate_model(x_new)

                # Extract values from tensors
                mu = mean.item()
                std = torch.sqrt(variance).item()

                # Denormalize predictions
                y_mean, y_std = self._normalization_params
                mu_original = mu * y_std + y_mean
                std_original = std * y_std

                logger.debug(
                    f"Predicted performance: mean={mu_original:.4f}, std={std_original:.4f}"
                )
                return mu_original, std_original

        except Exception as e:
            logger.error(f"Error in performance prediction: {e}")
            return 0.0, 1.0  # Default in case of error

    async def calculate_acquisition(
        self, prompt: MetaPrompt, best_score: float
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

    def _expected_improvement(
        self, mean: float, std: float, best_score: float
    ) -> float:
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
            return float(max(0, mean - best_score))

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
        return float(max(0, improvement))

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

    async def select_next_prompts(
        self, num_variations: int = 3, rows: Optional[List[Row]] = None
    ) -> List[MetaPrompt]:
        """Generate variations of prompts using Bayesian optimization.

        Args:
            num_variations: Number of prompt variations to generate
            rows: List of rows to evaluate prompts on

        Returns:
            List of selected prompts
        """
        # If we don't have enough history, generate random prompts
        if len(self.prompt_history) < 2:
            logger.info(
                f"Not enough prompt history ({len(self.prompt_history)}), generating initial prompts"
            )
            starting_prompts = await generate_prompts(
                self.task_guidance,
                self.variable_keys,
                self.expected_output_type,  # type: ignore
                num_variations,
                mode=self.prompt_mode,
            )
            logger.info(
                f"Starting with {len(starting_prompts)} prompts, now randomizing models"
            )
            for sp in starting_prompts:
                sp.config = DEFAULT_LLM_CONFIG.model_copy(
                    update={"model": random.choice(list(self.models_to_test))}
                )
            return starting_prompts
        try:
            # Fit or update the surrogate model
            if self._surrogate_model is None and rows is not None:
                logger.info("Fitting new surrogate model")
                await self.fit_surrogate_model(rows)

            # Find current best score
            logger.info("Calculating best score from history")
            best_prompt = await self.select_best_prompt(
                []
            )  # Use default logic from base class
            if best_prompt is None:
                logger.warning("No best prompt found, using default score")
                best_score = 0.0
            else:
                # Get score for best prompt to use in acquisition function
                if (
                    hasattr(self, "dataset")
                    and self.dataset
                    and hasattr(self.dataset, "training_rows")
                    and rows is None
                ):
                    # Use a subset of the actual training rows
                    eval_rows = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: random.sample(
                            list(self.dataset.training_rows),
                            min(5, len(self.dataset.training_rows)),
                        )
                        if self.dataset and hasattr(self.dataset, "training_rows")
                        else [],
                    )
                elif rows:
                    eval_rows = rows
                else:
                    eval_rows = []

                if self.prompt_history and eval_rows:
                    pwt = self.prompt_history[0]
                    scores = await pwt.get_scores(eval_rows, self.row_scoring_function)
                    best_score = max(scores) if scores else 0.0
                else:
                    best_score = 0.0

            # Generate a pool of candidate prompts
            logger.info("Generating candidate pool for Bayesian selection")

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
                for _ in range(
                    num_variations * 2
                ):  # Generate more candidates than needed
                    # Select a random variation type with weights
                    variation_type = random.choices(
                        variation_types, weights=weights, k=1
                    )[0]
                    variation_tasks.append(
                        prompt_to_vary.spec.vary(variation_type=variation_type)
                    )

                # Execute all variation tasks concurrently
                variation_specs: List[MetaPromptSpecBase | None] = await asyncio.gather(
                    *variation_tasks
                )

                # Filter out None results and convert to MetaPrompt objects
                for spec in variation_specs:
                    if spec is not None:
                        # Check if spec has config attribute, otherwise use default
                        model_config = DEFAULT_LLM_CONFIG.model_copy(
                            update={"model": spec.model}
                        )

                        candidate_pool.append(
                            MetaPrompt(
                                spec=spec,
                                expected_output_type=self.expected_output_type,  # type: ignore
                                config=model_config,
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
            top_candidates = [
                candidate for candidate, _ in acquisition_values[:num_variations]
            ]

            # Always include best prompt if we have one and there's room
            if (
                best_prompt is not None
                and best_prompt not in top_candidates
                and len(top_candidates) < num_variations
            ):
                top_candidates.append(best_prompt)

            # Limit to requested number of variations
            selected_candidates = top_candidates[:num_variations]

            logger.info(
                f"Selected {len(selected_candidates)} prompts using Bayesian optimization"
            )
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
            test_prompts = self.prompt_history[: min(5, len(self.prompt_history))]
            logger.info(
                f"No rows provided, using {len(test_prompts)} prompts for evaluation"
            )
            # We'll calculate scores directly using our surrogate model

            best_prompt = None
            best_score = float("-inf")

            for pwt in self.prompt_history:
                prompt = pwt.meta_prompt
                # Predict performance using our surrogate model
                mean, _ = await self.predict_performance(prompt)

                if mean > best_score:
                    best_score = mean
                    best_prompt = prompt

            logger.info(
                f"Selected best prompt using surrogate model with predicted score: {best_score:.4f}"
            )
            return best_prompt

        # If rows are provided, use the standard method from the base class
        logger.info(
            f"Selecting best prompt from {len(self.prompt_history)} prompts using {len(rows)} evaluation rows"
        )
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
        models: Optional[List[str]] = None,
        failure_analysis_enabled: bool = False,
        failure_threshold: int = 2,
        print_iteration_summary: bool = True,
    ):
        """Initialize the BayesianTrainer with appropriate configuration."""
        # Create the optimizer directly since we don't need to create the dataset first

        # Create the optimizer with the dataset
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
            models=models,
            failure_analysis_enabled=failure_analysis_enabled,
            failure_threshold=failure_threshold,
            print_iteration_summary=print_iteration_summary,
        )
        self.bayesian_optimizer = cast(BayesianOptimizer, self.optimizer)
        self.models_to_test = models
        logger.debug("BayesianTrainer initialized", optimizer_type="BayesianOptimizer")

    async def train(self) -> None:
        """Train the prompt optimizer using Bayesian optimization."""
        logger.info(
            f"Starting Bayesian optimization with Pyro: {self.num_iterations} iterations, "
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
            if self.models_to_test:
                for candidate in initial_candidates:
                    candidate.config = candidate.config.model_copy(
                        update={"model": random.choice(list(self.models_to_test))}
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
        await self.bayesian_optimizer.fit_surrogate_model(self.dataset.training_rows)

        # --- Step 2: Iterative Optimization ---
        logger.info(
            f"Phase 2: Starting {self.num_iterations - 1} optimization iterations..."
        )
        current_best_score = float("-inf")

        for iteration in range(
            self.num_iterations - 1
        ):  # Already did one effective round
            logger.info(f"Iteration {iteration + 2}/{self.num_iterations}")

            # Get best prompt so far for reference
            best_prompt = await self.optimizer.select_best_prompt(
                self.dataset.training_rows
            )
            if best_prompt:
                # Get score for comparison
                best_score = await self.eval_prompt_on_training_set(best_prompt)
                if best_score > current_best_score:
                    current_best_score = best_score
                    logger.info(f"Current best score: {current_best_score:.4f}")

            # Generate next candidates using Bayesian optimization or failure analysis
            if self.failure_analysis_enabled:
                logger.info("Performing failure analysis")
                consistent_failures = await self.get_consistent_failures()
                if consistent_failures:
                    logger.info(
                        f"Found {len(consistent_failures)} consistently failing examples"
                    )
                    # Use failures in next iteration's prompt generation
                    candidates = await generate_prompts(
                        self.task_guidance,
                        self.keys,
                        self.expected_output_type,  # type: ignore
                        self.candidates_per_iteration,
                        mode=self.bayesian_optimizer.prompt_mode,
                        failures=consistent_failures,  # Pass failures here
                    )
                    # Apply model selection if needed
                    if self.models_to_test:
                        for candidate in candidates:
                            candidate.config = candidate.config.model_copy(
                                update={
                                    "model": random.choice(list(self.models_to_test))
                                }
                            )
                else:
                    # Regular candidate generation using Bayesian optimization
                    candidates = await self.optimizer.select_next_prompts(
                        self.candidates_per_iteration, self.dataset.training_rows
                    )
            else:
                # Regular candidate generation without failure analysis
                candidates = await self.optimizer.select_next_prompts(
                    self.candidates_per_iteration, self.dataset.training_rows
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
                    logger.info(
                        f"Prompt content: {pwt.meta_prompt.spec.get_content()[:100]}..."
                    )

            # Update surrogate model with new data
            logger.info("Updating surrogate model with new data")
            await self.bayesian_optimizer.fit_surrogate_model(
                self.dataset.training_rows
            )

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

        # Update pyproject.toml to include Pyro dependencies
        try:
            with open("pyproject.toml", "a") as f:
                f.write("\n# Pyro dependencies for Bayesian optimization\n")
                f.write("# pyro-ppl>=1.9.0\n")
                f.write("# torch>=2.0.0\n")
            logger.info("Added Pyro dependencies to pyproject.toml (commented)")
        except Exception as e:
            logger.warning(
                f"Could not update pyproject.toml with Pyro dependencies: {e}"
            )

        # --- Model-specific analysis ---
        logger.info("Analyzing performance by model...")
        model_best_prompts = await self.optimizer.select_best_prompt_by_model(
            self.dataset.test_rows
        )

        for model, prompt in model_best_prompts.items():
            if prompt:
                pwt = PromptWithType(meta_prompt=prompt)
                score = await pwt.calculate_scores(
                    self.dataset.test_rows, self.scoring_function
                )
                logger.success(
                    f"Best prompt for model {model} achieved test score: {score:.4f}"
                )

                # Save model-specific best prompts
                safe_model_name = model.replace("/", "_")
                with open(f"best_prompt_{safe_model_name}.txt", "w") as f:
                    f.write(prompt.get_user_message_content())
            else:
                logger.warning(f"No valid prompt found for model {model}")

        # --- Overall best model+prompt ---
        # Find overall best model+prompt combination
        best_model, best_prompt = await self.optimizer.select_best_model_and_prompt(
            self.dataset.test_rows
        )

        if best_prompt:
            # Evaluate on test set
            final_test_mean = await self.eval_prompt_on_test_set(best_prompt)
            logger.success(
                f"Training complete. Best overall model is {best_model}. Final test score: {final_test_mean:.4f}",
                test_score=final_test_mean,
            )
            # Save the best prompt
            logger.info("Saving final best prompt (based on test set).")
            try:
                with open("best_prompt.txt", "w") as f:
                    f.write(best_prompt.get_user_message_content())
                with open("best_config.json", "w") as f:
                    f.write(best_prompt.config.model_dump_json(indent=2))
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
                    logger.success(
                        "Successfully saved best training prompt and config."
                    )
                except Exception as e:
                    logger.error(f"Failed to save training best prompt: {e}")


# Initialize Pyro for reproducibility
def setup_pyro(seed: int = 42):
    """Setup Pyro with specified random seed for reproducibility."""
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    return None
