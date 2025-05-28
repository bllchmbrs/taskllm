"""
Provides a PromptSpec implementation for tests.
"""
from typing import Any, List, Optional, Set
from enum import Enum


from taskllm.optimizer.prompt.meta import MetaPromptSpecBase, PromptMode


# Local test model enum to avoid import conflicts
class ModelsEnum(Enum):
    """Mock model enum for testing."""
    TEST_MODEL = "test_model"


class PromptSpec(MetaPromptSpecBase):
    """
    Basic implementation for testing purposes that extends MetaPromptSpecBase.
    """
    content: str
    model: ModelsEnum = ModelsEnum.TEST_MODEL
    prompt_execution_model: ModelsEnum = ModelsEnum.TEST_MODEL

    def get_system_message(self) -> str:
        """Returns a system message for the prompt."""
        return "You are a helpful assistant."

    def get_user_message(self) -> str:
        """Returns the user message content."""
        return self.content

    def get_content(self) -> str:
        """Returns the prompt content."""
        return self.content

    async def variation_types(self) -> Set[str]:
        """Returns possible variation types."""
        return {"content"}

    async def variation_weights(self) -> List[float]:
        """Returns weights for variation types."""
        return [1.0]

    async def vary(self, variation_type: str = "content") -> Optional["PromptSpec"]:
        """Creates a variation of this prompt spec."""
        return self.model_copy(update={"content": f"{self.content} (varied)"})

    @classmethod
    async def generate_spec(
        cls,
        guidance: str,
        keys: List[str],
        expected_output_type: Any,
    ) -> Optional["PromptSpec"]:
        """Generates a new prompt spec from guidance."""
        # Simple implementation for testing
        return PromptSpec(
            input_user_task_goal=guidance,
            input_variable_keys=keys,
            input_expected_output_type=expected_output_type,
            mode=PromptMode.SIMPLE,
            content=f"Test prompt for: {guidance}"
        )