import datetime
import functools
import inspect
import json
import os
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Literal,
    ParamSpec,
    TypeVar,
    Union,
    cast,
    overload,
)

import aioconsole  # type: ignore
from rich.console import Console
from rich.pretty import Pretty

T = TypeVar("T")
P = ParamSpec("P")

# Type Aliases for clarity
SyncFunc = Callable[P, T]
AsyncFunc = Callable[P, Awaitable[T]]
SyncWrapper = Callable[P, T]
AsyncWrapper = Callable[P, Coroutine[Any, Any, T]]

# We need to make the overloads clearly distinct
# Add a literal type to distinguish between them


@overload
def instrument_task(
    name: str,
    log_dir: str = "llm_logs",
    enable_quality_labeling: bool = False,
    *,
    is_async: Literal[False] = False,
) -> Callable[[SyncFunc[P, T]], SyncWrapper[P, T]]: ...


@overload
def instrument_task(
    name: str,
    log_dir: str = "llm_logs",
    enable_quality_labeling: bool = False,
    *,
    is_async: Literal[True],
) -> Callable[[AsyncFunc[P, T]], AsyncWrapper[P, T]]: ...


def instrument_task(
    name: str,
    log_dir: str = "llm_logs",
    enable_quality_labeling: bool = False,
    *,
    is_async: bool = False,
) -> Union[
    Callable[[SyncFunc[P, T]], SyncWrapper[P, T]],
    Callable[[AsyncFunc[P, T]], AsyncWrapper[P, T]],
]:
    """
    Decorator for LLM task functions, capturing inputs and outputs,
    storing them in JSON files in JSONL format.
    Works with both synchronous and asynchronous functions.

    Args:
        name: Name of the LLM task.
        log_dir: Directory to store log files (default: "llm_logs").
        enable_quality_labeling: Enable asking for quality judgment (default: False).
    """

    def decorator(
        func: Union[SyncFunc[P, T], AsyncFunc[P, T]],
    ) -> Union[SyncWrapper[P, T], AsyncWrapper[P, T]]:
        os.makedirs(log_dir, exist_ok=True)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timestamp = datetime.datetime.now().isoformat()
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Build inputs dictionary for logging
            inputs = dict(bound_args.arguments)

            try:
                # We cast here because inside this wrapper, func is guaranteed to be sync
                sync_func = cast(SyncFunc[P, T], func)
                result = sync_func(*args, **kwargs)
                outputs = result  # Capture output
                is_good = None

                if enable_quality_labeling:
                    console = Console()
                    console.print(f"Task: {name}")
                    console.print("Inputs:", Pretty(inputs))
                    console.print("Output:", Pretty(outputs))

                    quality = input(
                        f"Was the output of '{name}' good? (yes/no): "
                    ).lower()
                    is_good = quality == "yes"

                log_data = {
                    "task_name": name,
                    "timestamp": timestamp,
                    "inputs": inputs,
                    "outputs": outputs or is_good,
                    "quality": is_good,
                }

                log_filename = os.path.join(log_dir, f"{name}.jsonl")
                with open(log_filename, "a") as f:
                    json.dump(log_data, f)
                    f.write("\n")

                return result
            except Exception as e:
                print(f"Error during function execution: {e}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timestamp = datetime.datetime.now().isoformat()
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Build inputs dictionary for logging
            inputs = dict(bound_args.arguments)

            try:
                # We cast here because inside this wrapper, func is guaranteed to be async
                async_func = cast(AsyncFunc[P, T], func)
                result = await async_func(*args, **kwargs)
                outputs = result
                is_good = None
                if enable_quality_labeling:
                    console = Console()
                    console.print(f"Task: {name}")
                    console.print("Inputs:", Pretty(inputs))
                    console.print("Output:", Pretty(outputs))

                    quality = await aioconsole.ainput(
                        f"Was the output of '{name}' good? (yes/no): "
                    )
                    is_good = quality.lower() == "yes"

                log_data = {
                    "task_name": name,
                    "timestamp": timestamp,
                    "inputs": inputs,
                    "outputs": outputs or is_good,
                    "quality": is_good,
                }

                log_filename = os.path.join(log_dir, f"{name}.jsonl")
                with open(log_filename, "a") as f:
                    json.dump(log_data, f)
                    f.write("\n")

                return result
            except Exception as e:
                print(f"Error during function execution: {e}")
                raise

        if inspect.iscoroutinefunction(func):
            # Cast the return type to satisfy the outer function's Union return hint
            return cast(AsyncWrapper[P, T], async_wrapper)
        else:
            # Cast the return type to satisfy the outer function's Union return hint
            return cast(SyncWrapper[P, T], sync_wrapper)

    return decorator  # type: ignore[return-value]
