import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from taskllm.instrument import instrument_task


class TestInstrumentTask(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Create a temporary directory for logs before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.test_dir, "test_logs")

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.test_dir)

    def test_sync_function(self):
        """Test instrumentation of a synchronous function."""
        task_name = "sync_test_task"

        @instrument_task(task_name, log_dir=self.log_dir)
        def sync_func(x: int, y: int = 5) -> int:
            return x + y

        result = sync_func(10, y=20)
        self.assertEqual(result, 30)

        log_file = os.path.join(self.log_dir, f"{task_name}.jsonl")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r") as f:
            log_data = json.loads(f.readline())

        self.assertEqual(log_data["task_name"], task_name)
        self.assertEqual(log_data["inputs"], {"x": 10, "y": 20})
        self.assertEqual(log_data["outputs"], 30)
        self.assertIsNone(log_data["quality"])

    async def test_async_function(self):
        """Test instrumentation of an asynchronous function."""
        task_name = "async_test_task"

        @instrument_task(task_name, log_dir=self.log_dir)
        async def async_func(a: str, b: str) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return a + b

        result = await async_func("hello", " world")
        self.assertEqual(result, "hello world")

        log_file = os.path.join(self.log_dir, f"{task_name}.jsonl")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r") as f:
            log_data = json.loads(f.readline())

        self.assertEqual(log_data["task_name"], task_name)
        self.assertEqual(log_data["inputs"], {"a": "hello", "b": " world"})
        self.assertEqual(log_data["outputs"], "hello world")
        self.assertIsNone(log_data["quality"])

    @patch("builtins.input", return_value="yes")
    def test_sync_function_quality_labeling_yes(self, mock_input):
        """Test sync function with quality labeling enabled (input 'yes')."""
        task_name = "sync_quality_yes"

        @instrument_task(task_name, log_dir=self.log_dir, enable_quality_labeling=True)
        def sync_quality_func(val: bool) -> bool:
            return not val

        result = sync_quality_func(True)
        self.assertFalse(result)

        log_file = os.path.join(self.log_dir, f"{task_name}.jsonl")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r") as f:
            log_data = json.loads(f.readline())

        self.assertEqual(log_data["task_name"], task_name)
        self.assertEqual(log_data["inputs"], {"val": True})
        self.assertEqual(log_data["outputs"], False)

        # The instrument module accepts multiple forms of "yes" (case-insensitive)
        # Expect either "True", true, or 1 depending on how it's saved
        quality_value = log_data["quality"]
        self.assertTrue(
            quality_value is True or
            quality_value == "True" or
            quality_value == "true" or
            quality_value == 1
        )

        mock_input.assert_called_once()

    @patch("builtins.input", return_value="no")
    def test_sync_function_quality_labeling_no(self, mock_input):
        """Test sync function with quality labeling enabled (input 'no')."""
        task_name = "sync_quality_no"

        @instrument_task(task_name, log_dir=self.log_dir, enable_quality_labeling=True)
        def sync_quality_func(val: int) -> int:
            return val * 2

        result = sync_quality_func(5)
        self.assertEqual(result, 10)

        log_file = os.path.join(self.log_dir, f"{task_name}.jsonl")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r") as f:
            log_data = json.loads(f.readline())

        self.assertEqual(log_data["task_name"], task_name)
        self.assertEqual(log_data["inputs"], {"val": 5})
        self.assertEqual(log_data["outputs"], 10)
        self.assertFalse(log_data["quality"])
        mock_input.assert_called_once()

    @patch("aioconsole.ainput", return_value="YES")
    async def test_async_function_quality_labeling_yes(self, mock_ainput):
        """Test async function with quality labeling enabled (input 'yes')."""
        task_name = "async_quality_yes"

        @instrument_task(task_name, log_dir=self.log_dir, enable_quality_labeling=True)
        async def async_quality_func(data: list) -> int:
            await asyncio.sleep(0.01)
            return len(data)

        result = await async_quality_func([1, 2, 3])
        self.assertEqual(result, 3)

        log_file = os.path.join(self.log_dir, f"{task_name}.jsonl")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r") as f:
            log_data = json.loads(f.readline())

        self.assertEqual(log_data["task_name"], task_name)
        self.assertEqual(log_data["inputs"], {"data": [1, 2, 3]})
        self.assertEqual(log_data["outputs"], 3)
        self.assertTrue(log_data["quality"])
        mock_ainput.assert_awaited_once()

    @patch("aioconsole.ainput", return_value="No")
    async def test_async_function_quality_labeling_no(self, mock_ainput):
        """Test async function with quality labeling enabled (input 'no')."""
        task_name = "async_quality_no"

        @instrument_task(task_name, log_dir=self.log_dir, enable_quality_labeling=True)
        async def async_quality_func(text: str) -> str:
            await asyncio.sleep(0.01)
            return text.upper()

        result = await async_quality_func("test")
        self.assertEqual(result, "TEST")

        log_file = os.path.join(self.log_dir, f"{task_name}.jsonl")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r") as f:
            log_data = json.loads(f.readline())

        self.assertEqual(log_data["task_name"], task_name)
        self.assertEqual(log_data["inputs"], {"text": "test"})
        self.assertEqual(log_data["outputs"], "TEST")
        self.assertFalse(log_data["quality"])
        mock_ainput.assert_awaited_once()

    def test_default_log_dir(self):
        """Test that the default log directory 'llm_logs' is used."""
        task_name = "default_dir_task"
        default_log_dir = "llm_logs"

        # Ensure the default log dir doesn't exist before the test
        if os.path.exists(default_log_dir):
            shutil.rmtree(default_log_dir)  # Use shutil.rmtree for directories

        @instrument_task(task_name)  # No log_dir specified
        def sync_func_default(a: int) -> int:
            return a + 1

        sync_func_default(1)

        log_file = os.path.join(default_log_dir, f"{task_name}.jsonl")
        self.assertTrue(os.path.exists(log_file))

        # Clean up the default log directory after the test
        if os.path.exists(default_log_dir):
            shutil.rmtree(default_log_dir)


if __name__ == "__main__":
    unittest.main()
