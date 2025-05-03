from ..data import Row


def bool_scoring_function(row: Row, output: bool | None) -> float:
    if output is None:
        return -10
    if row.expected_output == output:
        return 1
    return 0


def semi_supervised_bool_scoring(row: Row, output: bool | None) -> float:
    if output is None:
        return -10

    # For labeled data, use ground truth
    if row.is_labeled:
        return 1 if row.expected_output == output else 0

    # For unlabeled data, use model confidence or other heuristics
    return 0.5  # Neutral score for unlabeled data
