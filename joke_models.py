from typing import List

from pydantic import BaseModel


class StartingJokes(BaseModel):
    jokes: List[str]


class IsJokeFunny(BaseModel):
    is_funny: bool
