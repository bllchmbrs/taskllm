"""
Simplified test models with proper serialization for testing.
"""
import json
from typing import Any, Dict, List, Optional, Set, Type, cast

from pydantic import BaseModel

from taskllm.ai import LLMConfig
from taskllm.optimizer.data import Row
from taskllm.optimizer.prompt.meta import MetaPrompt, PromptMode


class SimpleTestOutput(BaseModel):
    """Minimal test output model with proper serialization."""
    value: str = ""
    score: float = 0.0

    def model_dump(self) -> Dict[str, Any]:
        """Return a dict representation for serialization."""
        return {"value": self.model_dump_json("value"), "score": self.model_dump_json("score")}

    def model_dump_json(self, field_name=None) -> Any:
        """Return JSON-serializable value."""
        if field_name == "value":
            return self.value
        elif field_name == "score":
            return self.score
        else:
            return json.dumps({"value": self.value, "score": self.score})

    def __dict__(self) -> Dict[str, Any]:
        """Support direct dict conversion."""
        return {"value": self.value, "score": self.score}

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON serializable format."""
        return {"value": self.value, "score": self.score}


class SimplePromptSpec(BaseModel):
    """Minimal prompt spec for testing."""
    content: str
    input_user_task_goal: str = "Test task"
    input_variable_keys: List[str] = ["input"]
    input_expected_output_type: Type[Any] = SimpleTestOutput
    mode: PromptMode = PromptMode.SIMPLE
    
    def get_content(self) -> str:
        """Get the content of the prompt."""
        return self.content
    
    def get_system_message(self) -> str:
        """Get the system message."""
        return "You are a helpful assistant."
    
    def get_user_message(self) -> str:
        """Get the user message."""
        return self.content
        
    async def variation_types(self) -> Set[str]:
        """Get the types of variations supported."""
        return {"content"}
        
    async def variation_weights(self) -> List[float]:
        """Get the weights for variations."""
        return [1.0]
        
    async def vary(self, variation_type: str = "content") -> Optional["SimplePromptSpec"]:
        """Create a variation."""
        return SimplePromptSpec(
            content=f"{self.content} (varied)",
            input_user_task_goal=self.input_user_task_goal,
            input_variable_keys=self.input_variable_keys,
            input_expected_output_type=self.input_expected_output_type,
            mode=self.mode
        )


class PromptWithType:
    """Simplified PromptWithType for testing."""
    
    def __init__(self, meta_prompt: MetaPrompt, scores: Optional[List[float]] = None):
        self.meta_prompt = meta_prompt
        self._scores = scores or [0.8]
        
    async def calculate_scores(self, rows: List[Row], scoring_function: Any) -> float:
        """Calculate scores for the given rows."""
        return sum(self._scores) / len(self._scores) if self._scores else 0.0
        
    async def get_scores(self, rows: List[Row], scoring_function: Any) -> List[float]:
        """Get scores for the given rows."""
        return self._scores or [0.0]


# Helper functions for creating test objects

def create_simple_meta_prompt(content: str = "Test prompt content") -> MetaPrompt:
    """Create a simple MetaPrompt for testing."""
    spec = SimplePromptSpec(content=content)
    return MetaPrompt(
        spec=spec,
        expected_output_type=SimpleTestOutput,
        config=LLMConfig(
            model="test-model",
            temperature=0.1,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    )


def create_simple_prompt_with_type(
    content: str = "Test prompt content", 
    scores: Optional[List[float]] = None
) -> PromptWithType:
    """Create a simple PromptWithType for testing."""
    meta_prompt = create_simple_meta_prompt(content)
    return PromptWithType(meta_prompt=meta_prompt, scores=scores)


def create_minimal_test_dataset(size: int = 5) -> List[Row]:
    """Create a minimal test dataset with the given number of rows."""
    return [
        Row(
            input_variables={"input": f"test input {i}"},
            expected_output=SimpleTestOutput(value=f"output {i}", score=0.5 + i/10),
            is_labeled=True,
            task_name="test_task",
            timestamp=f"2023-01-{i+1:02d}T00:00:00Z"
        ) 
        for i in range(size)
    ]