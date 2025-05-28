import json
import os
import tempfile

import pytest
from pydantic import BaseModel

from taskllm.optimizer.data import DataSet, Row


class TestOutput(BaseModel):
    """Test output model for Row generic type."""
    value: str
    score: float

    def model_dump(self):
        """Return a dict representation for serialization."""
        return {"value": self.value, "score": self.score}


@pytest.mark.skip("Needs further implementation")
def test_row_creation():
    """Test Row.create method."""
    # Input dictionary for testing
    input_dict = {
        "task_name": "test_task",
        "timestamp": "2023-01-01T00:00:00Z",
        "key1": "value1",
        "key2": 42,
    }

    # Output value
    output = TestOutput(value="test_result", score=0.95)

    # Create a row
    row = Row.create(input_dict, output)

    # Verify the row properties
    assert row.task_name == "test_task"
    assert row.timestamp == "2023-01-01T00:00:00Z"
    assert row.input_variables == input_dict
    assert row.expected_output == output
    assert row.is_labeled is True


@pytest.mark.skip("Needs further implementation")
def test_row_creation_without_output():
    """Test Row.create method without an output value."""
    # Input dictionary for testing
    input_dict = {
        "task_name": "test_task",
        "timestamp": "2023-01-01T00:00:00Z",
        "key1": "value1",
    }

    # Create a row without output
    row = Row.create(input_dict, None)

    # Verify the row properties
    assert row.task_name == "test_task"
    assert row.timestamp == "2023-01-01T00:00:00Z"
    assert row.input_variables == input_dict
    assert row.expected_output is None
    assert row.is_labeled is False


@pytest.mark.skip("Needs further implementation")
def test_row_create_from_dict():
    """Test Row.create_from_dict method."""
    # Data dict representing a logged row
    data = {
        "inputs": {"prompt": "Test prompt", "parameter": 10},
        "outputs": TestOutput(value="Test output", score=0.8).model_dump(),
        "task_name": "test_task",
        "timestamp": "2023-01-01T00:00:00Z",
        "quality": True,
    }

    # Create a row from the dictionary
    row = Row.create_from_dict(data)

    # Verify the row properties
    assert row.input_variables == data["inputs"]
    assert row.expected_output == data["outputs"]
    assert row.task_name == data["task_name"]
    assert row.timestamp == data["timestamp"]
    assert row.is_labeled is True


@pytest.mark.skip("Needs further implementation")
def test_row_create_from_dict_without_outputs():
    """Test Row.create_from_dict method with missing outputs."""
    # Data dict without outputs
    data = {
        "inputs": {"prompt": "Test prompt", "parameter": 10},
        "task_name": "test_task",
        "timestamp": "2023-01-01T00:00:00Z",
    }

    # Create a row from the dictionary
    row = Row.create_from_dict(data)

    # Verify the row properties
    assert row.input_variables == data["inputs"]
    assert row.expected_output is None
    assert row.task_name == data["task_name"]
    assert row.timestamp == data["timestamp"]
    assert row.is_labeled is False


@pytest.mark.skip("Needs further implementation")
def test_row_get_template_keys():
    """Test Row.get_template_keys method."""
    # Create a row with several input variables
    row = Row[TestOutput](
        input_variables={"key1": "value1", "key2": 42, "key3": True},
        task_name="test_task",
        timestamp="2023-01-01T00:00:00Z",
        is_labeled=False,
        expected_output=None,
    )

    # Get template keys
    keys = row.get_template_keys()

    # Verify the keys match
    assert set(keys) == {"key1", "key2", "key3"}


@pytest.mark.skip("Needs further implementation")
def test_row_get_variables():
    """Test Row.get_variables method."""
    # Input variables dictionary
    variables = {"key1": "value1", "key2": 42, "key3": True}

    # Create a row
    row = Row[TestOutput](
        input_variables=variables,
        task_name="test_task",
        timestamp="2023-01-01T00:00:00Z",
        is_labeled=False,
        expected_output=None,
    )

    # Get variables
    result = row.get_variables()

    # Verify we get back the original dict
    assert result == variables


@pytest.mark.skip("Needs further implementation")
def test_row_to_dict():
    """Test Row.to_dict method."""
    # Create a row
    output_model = TestOutput(value="test_value", score=0.9)
    row = Row[TestOutput](
        input_variables={"key1": "value1", "key2": 42},
        expected_output=output_model,
        is_labeled=True,
        task_name="test_task",
        timestamp="2023-01-01T00:00:00Z",
    )

    # Convert to dict
    result = row.to_dict()

    # Verify the dict structure
    assert result["inputs"] == {"key1": "value1", "key2": 42}
    assert result["outputs"] == output_model.model_dump()
    assert result["task_name"] == "test_task"
    assert result["timestamp"] == "2023-01-01T00:00:00Z"
    assert result["quality"] is None  # Quality is always None in to_dict


@pytest.mark.skip("Needs further implementation")
def test_dataset_creation(sample_dataset):
    """Test DataSet creation."""
    # Verify dataset properties
    assert sample_dataset.name == "test_dataset"
    assert len(sample_dataset.rows) == 10

    # Verify all rows are labeled
    for row in sample_dataset.rows:
        assert row.is_labeled is True
        assert row.expected_output is not None


@pytest.mark.skip("Needs further implementation")
def test_dataset_to_file():
    """Test DataSet.to_file method."""
    # Create a minimal dataset
    rows = [
        Row[TestOutput](
            input_variables={"prompt": f"Test prompt {i}"},
            expected_output=TestOutput(value=f"Test output {i}", score=0.8),
            is_labeled=True,
            task_name="test_task",
            timestamp="2023-01-01T00:00:00Z",
        )
        for i in range(3)
    ]
    dataset = DataSet(name="test_dataset", rows=rows)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name

    try:
        # Write dataset to file
        dataset.to_file(file_path)

        # Read back the file and verify
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Verify we have the right number of lines
        assert len(lines) == 3

        # Parse each line as JSON and verify
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["inputs"]["prompt"] == f"Test prompt {i}"
            assert data["outputs"]["value"] == f"Test output {i}"
            assert data["task_name"] == "test_task"
    finally:
        # Clean up
        os.unlink(file_path)


@pytest.mark.skip("Needs further implementation")
def test_dataset_num_labelled_rows(sample_dataset):
    """Test DataSet.num_labelled_rows method."""
    # All rows in sample_dataset are labeled
    assert sample_dataset.num_labelled_rows() == 10

    # Add an unlabeled row
    sample_dataset.rows.append(
        Row[TestOutput](
            input_variables={"prompt": "Unlabeled prompt"},
            expected_output=None,
            is_labeled=False,
            task_name="test_task",
            timestamp="2023-01-01T00:00:00Z",
        )
    )

    # Verify the count
    assert sample_dataset.num_labelled_rows() == 10
    assert sample_dataset.num_unlabelled_rows() == 1


@pytest.mark.skip("Needs further implementation")
def test_dataset_train_test_split(sample_dataset):
    """Test DataSet.train_test_split method."""
    # Split dataset with default test_size=0.2
    sample_dataset.train_test_split()

    # Verify split proportions (8 train, 2 test)
    assert len(sample_dataset.training_rows) == 8
    assert len(sample_dataset.test_rows) == 2

    # Verify that all original rows are accounted for
    all_rows = set(sample_dataset.training_rows + sample_dataset.test_rows)
    assert len(all_rows) == 10

    # Create a new dataset with more rows for testing different split
    rows = [
        Row[TestOutput](
            input_variables={"prompt": f"Test prompt {i}"},
            expected_output=TestOutput(value=f"Test output {i}", score=0.8),
            is_labeled=True,
            task_name="test_task",
            timestamp="2023-01-01T00:00:00Z",
        )
        for i in range(100)
    ]
    dataset = DataSet(name="large_dataset", rows=rows)

    # Split with custom test_size=0.3
    dataset.train_test_split(test_size=0.3)

    # Verify split proportions (70 train, 30 test)
    assert len(dataset.training_rows) == 70
    assert len(dataset.test_rows) == 30


@pytest.mark.skip("Needs further implementation")
def test_dataset_training_and_test_rows_properties(sample_dataset):
    """Test that training_rows and test_rows properties trigger split if not done."""
    # Reset the internal split data
    sample_dataset._training_rows = []
    sample_dataset._test_rows = []

    # Access training_rows should trigger split
    train_rows = sample_dataset.training_rows
    assert len(train_rows) == 8

    # Reset again
    sample_dataset._training_rows = []
    sample_dataset._test_rows = []

    # Access test_rows should trigger split
    test_rows = sample_dataset.test_rows
    assert len(test_rows) == 2