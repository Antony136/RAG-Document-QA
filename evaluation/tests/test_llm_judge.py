import pytest

from evaluation.generation.llm_judge import (
    validate_judgment,
)


def valid_judgment():
    return {
        "correctness": {
            "score": 5,
            "reason": "Correct answer.",
        },
        "relevance": {
            "score": 4,
            "reason": "Directly answers the question.",
        },
        "faithfulness": {
            "score": 5,
            "reason": "Supported by context.",
        },
        "context_relevance": {
            "score": 4,
            "reason": "Useful context.",
        },
    }


def test_validate_judgment_accepts_valid_result():
    result = validate_judgment(
        valid_judgment()
    )

    assert result["correctness"]["score"] == 5
    assert result["relevance"]["score"] == 4


def test_validate_judgment_rejects_missing_dimension():
    judgment = valid_judgment()

    del judgment["faithfulness"]

    with pytest.raises(ValueError):
        validate_judgment(judgment)


def test_validate_judgment_rejects_score_below_one():
    judgment = valid_judgment()

    judgment["correctness"]["score"] = 0

    with pytest.raises(ValueError):
        validate_judgment(judgment)


def test_validate_judgment_rejects_score_above_five():
    judgment = valid_judgment()

    judgment["correctness"]["score"] = 6

    with pytest.raises(ValueError):
        validate_judgment(judgment)


def test_validate_judgment_rejects_non_integer_score():
    judgment = valid_judgment()

    judgment["correctness"]["score"] = 4.5

    with pytest.raises(ValueError):
        validate_judgment(judgment)


def test_validate_judgment_rejects_missing_reason():
    judgment = valid_judgment()

    del judgment["relevance"]["reason"]

    with pytest.raises(ValueError):
        validate_judgment(judgment)