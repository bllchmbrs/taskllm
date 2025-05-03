import os
from typing import ParamSpec, TypeVar

import diskcache

CACHE_DIR = os.path.expanduser("~/.cache/taskllm")

P = ParamSpec("P")
T = TypeVar("T")


def get_cache(name: str) -> diskcache.Cache:
    """
    Get a diskcache.Cache instance with the given name.

    Args:
        name: The name for the cache directory.

    Returns:
        A diskcache.Cache instance.
    """
    cache_path = os.path.join(CACHE_DIR, name)
    os.makedirs(cache_path, exist_ok=True)
    return diskcache.Cache(cache_path)
