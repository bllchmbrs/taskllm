import json
from typing import Any, Dict, Generic, List, TypeVar

from pydantic import BaseModel
from sklearn.model_selection import train_test_split

T = TypeVar("T", bound=BaseModel)


class Row(BaseModel, Generic[T]):
    input_variables: Dict[str, Any]
    expected_output: T | None = None
    is_labeled: bool = False
    task_name: str
    timestamp: str

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "Row":
        input_variables = data.get("inputs", {})
        expected_output = data.get("outputs")
        task_name = data.get("task_name", "")
        timestamp = data.get("timestamp", "")
        is_labeled = expected_output is not None  # infer from presence of output

        return cls(
            input_variables=input_variables,
            expected_output=expected_output,
            is_labeled=is_labeled,
            task_name=task_name,
            timestamp=timestamp,
        )

    @classmethod
    def create(
        cls,
        input_dictionary: Dict[str, Any],
        output: T | None,
    ) -> "Row":
        task_name = input_dictionary.get("task_name", "")
        timestamp = input_dictionary.get("timestamp", "")
        is_labeled = output is not None  # infer from presence of output

        return cls(
            input_variables=input_dictionary,
            expected_output=output,
            is_labeled=is_labeled,
            task_name=task_name,
            timestamp=timestamp,
        )

    def get_template_keys(self) -> List[str]:
        return list(self.input_variables.keys())

    def get_variables(self) -> Dict[str, Any]:
        return self.input_variables

    def to_dict(self):
        return {
            "inputs": self.input_variables,
            "outputs": self.expected_output,
            "task_name": self.task_name,
            "timestamp": self.timestamp,
            "quality": None,  # This was in the original jsonl
        }


class DataSet(BaseModel, Generic[T]):
    name: str
    rows: List[Row[T]]
    _training_rows: List[Row[T]] = []
    _test_rows: List[Row[T]] = []

    def to_file(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row.to_dict()) + "\n")

    def num_labelled_rows(self) -> int:
        return sum(1 for row in self.rows if row.is_labeled)

    def num_unlabelled_rows(self) -> int:
        return sum(1 for row in self.rows if not row.is_labeled)

    def train_test_split(self, test_size: float = 0.2) -> None:
        self._training_rows, self._test_rows = train_test_split(
            self.rows, test_size=test_size, random_state=56
        )

    @property
    def training_rows(self) -> List[Row[T]]:
        if not self._training_rows:
            self.train_test_split()
        return self._training_rows

    @property
    def test_rows(self) -> List[Row[T]]:
        if not self._test_rows:
            self.train_test_split()
        return self._test_rows
