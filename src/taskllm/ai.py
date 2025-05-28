import hashlib
import json
from typing import Any, Dict, List, ParamSpec, Type, TypeVar, cast

import instructor
from litellm import completion
from loguru import logger
from pydantic import BaseModel

from .utils import get_cache

cache = get_cache("ai_responses")

ins_client = instructor.from_litellm(completion)

P = ParamSpec("P")
T = TypeVar("T")

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class LLMConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float


DEFAULT_LLM_CONFIG = LLMConfig(
    model="openai/gpt-4o-mini",
    temperature=0.5,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)


async def simple_llm_call(
    messages: List[Dict[str, Any]], config: LLMConfig, use_cache: bool = True
) -> str:
    key = hashlib.sha256(
        json.dumps(messages + [config.model_dump()]).encode()
    ).hexdigest()
    cached_response = cache.get(key)

    if use_cache and cached_response:
        logger.trace("Cache hit when calling simple LLM", key=key)
        return str(cached_response)

    logger.trace("Calling simple LLM", config=config, messages=messages)
    response = completion(model=config.model, messages=messages)
    content = str(response.choices[0].message.content)

    cache.set(key, content)
    return content


async def struct_llm_call(
    messages: List[Dict[str, Any]],
    config: LLMConfig,
    response_model: Type[BaseModelT],
    use_cache: bool = True,
) -> BaseModelT:
    key = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
    if use_cache and cache.get(key):
        logger.trace("Cache hit when calling struct LLM", key=key)
        # Ensure the cached data is validated against the expected model
        cached_data = cache.get(key)
        if isinstance(cached_data, dict):
            return response_model.model_validate(cached_data)
        elif isinstance(cached_data, response_model):  # If already model instance
            return cached_data
        else:
            # Handle unexpected cache type or log error
            logger.error(
                f"Unexpected data type found in cache for key {key}: {type(cached_data)}"
            )

    logger.trace("Calling struct LLM", config=config, messages=messages)
    response = ins_client.chat.completions.create(
        model=config.model, messages=messages, response_model=response_model
    )
    cache.set(key, response.model_dump())
    return cast(BaseModelT, response)


class MaxRetriesExceeded(Exception):
    pass
