# TaskLLM

A library for optimizing LLM tasks, including prompt engineering and bandit-based training.

## Project Overview

TaskLLM is a Python library designed to help developers optimize their interactions with Large Language Models (LLMs). It provides tools for:

- **Instrumenting LLM tasks** to track inputs, outputs, and quality metrics
- **Making LLM calls** with both simple text responses and structured outputs
- **Optimizing prompts** through bandit-based training algorithms

The library is particularly useful for developers who want to:
- Systematically improve prompt performance
- Track and analyze LLM interactions
- Convert unstructured LLM outputs into structured data
- Implement quality assessment for LLM outputs

## Installation

TaskLLM requires Python 3.11 or higher.

```bash
# Install from PyPI, not yet available.
pip install taskllm

# Or install from source
git clone https://github.com/your-username/taskllm.git
cd taskllm
pip install -e .
```

## Quick Start

Here's a simple example of how to use TaskLLM to instrument a function. This shouldn't be used in production, but is helpful for prototyping and testing.

```python
from taskllm.instrument import instrument_task

# For synchronous functions
@instrument_task("add_numbers")
def add_numbers(a: int, b: int) -> int:
    return a + b

# For asynchronous functions
@instrument_task("add_numbers_async", enable_quality_labeling=True)
async def add_numbers_async(a: int, b: int) -> int:
    return a + b

# Call the functions
result = add_numbers(1, 2)  # Will log inputs and outputs
async_result = await add_numbers_async(3, 4)  # Will log and prompt for quality assessment
```

## Core Components


### Optimization

The `BanditTrainer` class helps you find the best prompts for your tasks:

```python
from taskllm.optimizer.methods import BanditTrainer
from taskllm.optimizer.data import DataSet, Row

trainer = BanditTrainer(
    all_rows=your_dataset,
    task_guidance="your task description",
    keys=["input_field1", "input_field2"],
    expected_output_type=YourOutputModel,
    scoring_function=your_scoring_function
)

await trainer.train()
best_prompt = await trainer.get_best_prompt()
```

## Examples

The repository includes several examples demonstrating how to use TaskLLM in different scenarios:

### Jokes Example

Determines whether jokes are funny using a bandit-based prompt optimizer:

```python
# From examples/jokes/run.py
trainer = BanditTrainer(
    all_rows=dataset,
    task_guidance="write a prompt that determines whether a joke is funny based on the category of joke",
    keys=["joke"],
    expected_output_type=IsJokeFunny,
    scoring_function=funny_scoring_function
)
```

### Tweet Sentiment Analysis

Analyzes the sentiment of tweets (positive, negative, or neutral):

```python
# From examples/tweet_sentiment/run.py
trainer = BanditTrainer(
    all_rows=dataset,
    task_guidance="what is the sentiment of this tweet?",
    keys=["tweet"],
    expected_output_type=TweetSentiment,
    scoring_function=sentiment_scoring_function
)
```

### Starbucks Reviews

Rates Starbucks reviews on a scale of 1-5:

```python
# From examples/starbucks/run.py
trainer = BanditTrainer(
    all_rows=dataset,
    task_guidance="determine the rating of this review",
    keys=["review", "name", "location", "date"],
    expected_output_type=StarbucksReviewRating,
    scoring_function=sentiment_scoring_function,
    prompt_mode=PromptMode.ADVANCED
)
```

To run any of these examples:

```bash
cd examples/[example_directory]
python run.py
```

## Advanced Usage

### Configuration Options

You can customize the LLM configuration:

```python
from taskllm.ai import LLMConfig

custom_config = LLMConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2000,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5
)
```

### Quality Labeling

Enable interactive quality assessment for your tasks:

```python
@instrument_task("your_task", enable_quality_labeling=True)
def your_function():
    # After execution, you'll be prompted to rate the quality
```

### Caching Strategies

TaskLLM automatically caches LLM responses to save time and costs:

```python
# Disable caching for specific calls
response = await simple_llm_call(
    messages=[...],
    config=config,
    use_cache=False
)
```

### Custom Prompt Modes

Use advanced prompt modes for more sophisticated optimization:

```python
from taskllm.optimizer.prompt.meta import PromptMode

trainer = BanditTrainer(
    # ...other parameters
    prompt_mode=PromptMode.ADVANCED
)
```

## API Reference

### Key Modules

- `taskllm.instrument`: Functions for tracking and logging LLM tasks
- `taskllm.ai`: Interface for making LLM calls
- `taskllm.optimizer`: Tools for optimizing prompts
  - `taskllm.optimizer.data`: Data structures for optimization
  - `taskllm.optimizer.methods`: Optimization algorithms
  - `taskllm.optimizer.prompt`: Prompt management and templates

## Contributing

### Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/taskllm.git
   cd taskllm
   ```

2. Create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies
   ```bash
   pip install -e ".[dev]"
   ```

### Testing

Run tests using:

```bash
python -m pytest
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests
5. Submit a pull request

## License

MIT License

---

For more information, check out the examples directory or open an issue on GitHub.