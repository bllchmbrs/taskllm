# Grid Search Example

This example demonstrates how to use the Grid Search optimizer to find the optimal prompt and configuration for a given task.

## Overview

Grid Search is a simpler optimization method that systematically explores predefined prompt variations. It's useful for:

- More deterministic exploration of the prompt space
- Methodically testing different combinations of parameters
- Reproducibility in experiments

## How it Works

The Grid Search optimizer:

1. Takes a set of parameters and their possible values (grid dimensions)
2. Creates all possible combinations of these parameters
3. Systematically tests each combination to find the optimal configuration

This example uses a joke rating task to demonstrate grid search. It tries different combinations of:
- Temperature values (0.1, 0.5, 0.9)
- Different LLM models

## Running the Example

To run the example:

```bash
cd examples/grid_search_example
python run.py
```

The script will:
1. Load a dataset of jokes with ratings
2. Run grid search optimization to find the best parameters
3. Save the best prompt and configuration to files

## Advantages over Other Methods

Compared to Bandit or Bayesian optimization:

- **Simpler implementation**: No complex statistical models
- **Deterministic exploration**: Useful for reproducibility
- **Exhaustive coverage**: Tests every parameter combination within defined ranges

## Limitations

- Less efficient than Bandit/Bayesian for large search spaces
- Requires more iterations for large parameter grids
- Computationally expensive as grid size increases 