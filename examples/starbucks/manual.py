import asyncio

from run import StarbucksReviewRating, load_file_as_dataset, sentiment_scoring_function

from taskllm.ai import DEFAULT_LLM_CONFIG, struct_llm_call
from taskllm.optimizer.data import Row


def prompt_render_func(row: Row[StarbucksReviewRating]) -> str:
    prompt = "Consider whether this review is positive or negative and then give it a number from 1-5.\n"
    return f"""
    {prompt}
    <review>
    {row.input_variables["review"]}
    </review>
    """


async def main():
    dataset = load_file_as_dataset("./starbucks_reviews.csv")
    total_score = 0
    for row in dataset.training_rows[:100]:
        output = await struct_llm_call(
            messages=[
                {"role": "user", "content": prompt_render_func(row)},
            ],
            config=DEFAULT_LLM_CONFIG,
            response_model=StarbucksReviewRating,
        )
        score = sentiment_scoring_function(row, output)
        total_score += score
    print(total_score)


if __name__ == "__main__":
    asyncio.run(main())
