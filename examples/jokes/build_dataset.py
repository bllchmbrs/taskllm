import os
from typing import List

from pydantic import BaseModel

from taskllm.ai import DEFAULT_LLM_CONFIG, struct_llm_call
from taskllm.instrument import instrument_task


class StartingJokes(BaseModel):
    jokes: List[str]


@instrument_task(name="joke_rating", enable_quality_labeling=True)
def joke_judge(joke: str):
    pass


async def get_jokes() -> List[str]:
    res = await struct_llm_call(
        messages=[
            {"role": "system", "content": "You are a joke teller."},
            {
                "role": "user",
                "content": "Give me a list of 10 jokes, make them a variety of different types of jokes.",
            },
        ],
        config=DEFAULT_LLM_CONFIG,
        response_model=StartingJokes,
    )

    return res.jokes


async def build_dataset(path: str):
    jokes = await get_jokes()

    if not os.path.exists(path):
        for joke in jokes:
            joke_judge(joke)
