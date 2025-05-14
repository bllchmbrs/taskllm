import asyncio
import json
import os

from pydantic import BaseModel

from taskllm.optimizer.data import DataSet, Row
from taskllm.optimizer.methods import GridSearchTrainer


class IsJokeFunny(BaseModel):
    is_funny: bool


def load_joke_data(path: str) -> DataSet:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            rows.append(
                Row.create(
                    input_dictionary={"joke": data["inputs"]["joke"]},
                    output=IsJokeFunny(is_funny=data["outputs"]),
                )
            )
    return DataSet(rows=rows, name="joke_ratings")


def funny_scoring_function(row: Row[IsJokeFunny], output: IsJokeFunny | None) -> float:
    if output is None:
        return -10
    if not row.expected_output:
        return 0
    if row.expected_output.is_funny == output.is_funny:  # true positive
        return 2
    elif (
        row.expected_output.is_funny is True and output.is_funny is False
    ):  # false negative
        return -1
    else:  # false positive
        return -2


def main():
    # Make sure we can access the joke dataset
    path = "../jokes/joke_rating.jsonl"
    if not os.path.exists(path):
        print(f"Error: Dataset file not found at {path}")
        return

    dataset = load_joke_data(path)
    print("Starting grid search training")

    # Define grid dimensions for the search
    grid_dimensions = {
        "temperature": [0.1, 0.5, 0.9],
        "model": [
            "anthropic/claude-3-haiku-20240307",
            "openai/gpt-4.1-mini-2025-04-14",
        ],
    }

    trainer = GridSearchTrainer(
        all_rows=dataset,
        task_guidance="write a prompt that determines whether a joke is funny based on the category of joke",
        keys=["joke"],
        expected_output_type=IsJokeFunny,
        scoring_function=funny_scoring_function,
        num_iterations=2,  # Start with fewer iterations for testing
        candidates_per_iteration=3,  # Grid search examines combinations systematically
        grid_dimensions=grid_dimensions,
    )

    # Run training
    asyncio.run(trainer.train())

    # Get results
    best_prompt = asyncio.run(trainer.get_best_prompt())
    best_score = asyncio.run(trainer.eval_prompt_on_all_data(best_prompt))
    best_config = asyncio.run(trainer.get_best_llm_config())

    # Print results
    print(f"Best score: {best_score}")
    print(f"Best config: {best_config}")
    print(f"Best prompt: {best_prompt.get_user_message_content()}")

    # Save best prompt to file
    with open("best_prompt.txt", "w") as f:
        f.write(best_prompt.get_user_message_content())

    # Save best config to file
    with open("best_config.json", "w") as f:
        f.write(json.dumps(best_config.model_dump(), indent=2))


if __name__ == "__main__":
    main()
