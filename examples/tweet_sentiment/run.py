import asyncio
import csv
import sys
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from taskllm.optimizer.data import DataSet, Row
from taskllm.optimizer.methods import BanditTrainer

logger.remove()  # remove the old handler. Else, the old one will work (and continue printing DEBUG logs) along with the new handler added below'
logger.add(sys.stdout, level="TRACE")  # add a new handler which has INFO as the default


class TweetSentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]


def sentiment_scoring_function(
    row: Row[TweetSentiment], output: TweetSentiment | None
) -> float:
    if output is None:
        return -10
    logger.trace(
        f"Expected: {row.expected_output.sentiment}, Output: {output.sentiment}"
    )
    if row.expected_output.sentiment == output.sentiment:
        return 1

    return 0


def load_file_as_dataset(path: str) -> DataSet:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentiment = TweetSentiment(sentiment=row["sentiment"].lower())
            rows.append(
                Row.create(
                    input_dictionary={"tweet": row["tweet"]},
                    output=sentiment,
                    expected_output_type=TweetSentiment,
                )
            )
    return DataSet(rows=rows[:20], name="tweets")


def main():
    csv_path = "./tweet_sentiment.csv"
    dataset = load_file_as_dataset(csv_path)

    print("Starting training")
    trainer = BanditTrainer(
        all_rows=dataset,
        task_guidance="what is the sentiment of this tweet?",
        keys=["tweet"],
        expected_output_type=TweetSentiment,
        scoring_function=sentiment_scoring_function,
        num_iterations=3,  # Start with fewer iterations for testing
        candidates_per_iteration=3,  # Start with fewer candidates for testing
    )
    asyncio.run(trainer.train())
    best_prompt = asyncio.run(trainer.get_best_prompt())
    best_score = asyncio.run(trainer.eval_prompt_on_all_data(best_prompt))

    print(f"Best score: {best_score}")
    print(best_prompt.get_user_message_content())


if __name__ == "__main__":
    main()
