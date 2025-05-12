import asyncio
import csv
from enum import Enum

from loguru import logger
from pydantic import BaseModel

from taskllm.optimizer.data import DataSet, Row
from taskllm.optimizer.methods import BanditTrainer
from taskllm.optimizer.prompt.meta import PromptMode

# logger.remove()  # remove the old handler. Else, the old one will work (and continue printing DEBUG logs) along with the new handler added below'
# logger.add(sys.stdout, level="TRACE")  # add a new handler which has INFO as the default


class Ratings(Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    NA = "N/A"


class StarbucksReviewRating(BaseModel):
    rating: Ratings


def sentiment_scoring_function(
    row: Row[StarbucksReviewRating], output: StarbucksReviewRating | None
) -> float:
    if output is None:
        return -10
    if not row.expected_output:
        return 0
    logger.trace(f"Expected: {row.expected_output.rating}, Output: {output.rating}")
    if row.expected_output.rating == output.rating:
        return 1

    return 0


def load_file_as_dataset(path: str) -> DataSet:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = StarbucksReviewRating(rating=Ratings(row["Rating"]))
            rows.append(
                Row.create(
                    input_dictionary={
                        "review": row["Review"],
                        "name": row["name"],
                        "location": row["location"],
                        "date": row["Date"],
                    },
                    output=rating,
                )
            )
    return DataSet(rows=rows[:100], name="starbucks_reviews")


def main():
    csv_path = "./starbucks_reviews.csv"
    dataset = load_file_as_dataset(csv_path)

    print("Starting training")
    trainer = BanditTrainer(
        all_rows=dataset,
        task_guidance="determine the rating of this review",
        keys=["review", "name", "location", "date"],
        expected_output_type=StarbucksReviewRating,
        scoring_function=sentiment_scoring_function,
        num_iterations=3,  # Start with fewer iterations for testing
        candidates_per_iteration=3,  # Start with fewer candidates for testing
        prompt_mode=PromptMode.ADVANCED,
        models=[
            "anthropic/claude-3-haiku-20240307",
            "openai/gpt-4.1-nano-2025-04-14",
            "openai/gpt-4.1-mini-2025-04-14",
        ],
    )
    asyncio.run(trainer.train())
    best_prompt = asyncio.run(trainer.get_best_prompt())
    best_score = asyncio.run(trainer.eval_prompt_on_all_data(best_prompt))
    best_config = asyncio.run(trainer.get_best_llm_config())
    print(f"Best score: {best_score}")
    print(f"Best config: {best_config}")
    print(best_prompt.get_user_message_content())


if __name__ == "__main__":
    main()
