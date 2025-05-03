import asyncio

from src.taskllm.instrument import instrument_task


@instrument_task("test")
def test(a: int, b: int) -> int:
    return a + b


@instrument_task("test2", enable_quality_labeling=True)
async def test2(a: int, b: int) -> int:
    return a + b


def main():
    print("Hello from fcsmode!")
    print(test(1, 2))
    print(asyncio.run(test2(1, 2)))


if __name__ == "__main__":
    main()
