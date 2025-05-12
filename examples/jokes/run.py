import asyncio
import json

from pydantic import BaseModel

from taskllm.optimizer.data import DataSet, Row
from taskllm.optimizer.methods import BanditTrainer


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
    path = "./joke_rating.jsonl"
    dataset = load_joke_data(path)
    print("Starting training")
    trainer = BanditTrainer(
        all_rows=dataset,
        task_guidance="write a prompt that determines whether a joke is funny based on the category of joke",
        keys=["joke"],
        expected_output_type=IsJokeFunny,
        scoring_function=funny_scoring_function,
        num_iterations=2,  # Start with fewer iterations for testing
        candidates_per_iteration=2,  # Start with fewer candidates for testing
    )
    asyncio.run(trainer.train())
    best_prompt = asyncio.run(trainer.get_best_prompt())
    best_score = asyncio.run(trainer.eval_prompt_on_all_data(best_prompt))

    print(f"Best score: {best_score}")
    print(best_prompt.get_user_message_content())


if __name__ == "__main__":
    main()
