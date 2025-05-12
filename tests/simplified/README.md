# Simplified Tests for TaskLLM

This directory contains the streamlined test suite for TaskLLM, following the principles from the test streamlining plan.

## Key Principles

1. **Simplicity**: Tests focus on critical functionality only
2. **Interface Testing**: Tests verify interface behavior, not implementation details
3. **Low Maintenance**: Tests avoid complex mocks and fixtures
4. **Reliability**: Tests are designed to be stable and not flaky

## Contents

- `test_models.py`: Simplified test models with proper serialization
- `test_optimizer_interface.py`: Tests for the core optimizer interfaces
- `test_instrument.py`: Minimal tests for instrumentation functionality
- `conftest.py`: Simple fixtures for tests

## Running Tests

Run only the simplified tests:

```bash
pytest tests/simplified/
```

## Design Decisions

1. **Test Model Simplification**:
   - Used minimal Pydantic models with complete serialization support
   - Avoided complex inheritance hierarchies

2. **Interface Focus**:
   - Tests verify that optimizers implement the expected interface
   - Implementation details are not tested directly

3. **Reduced Fixture Usage**:
   - Helper functions provide test data instead of complex fixtures
   - Initialization is done inline where possible

4. **Direct Instantiation**:
   - Objects are created directly with minimal parameters
   - Complex setup is avoided in favor of simpler tests