import pytest

from liquid_memory_agents.datasets.longmemeval import official_hypothesis_rows


def test_official_hypothesis_format() -> None:
    assert official_hypothesis_rows(["q1"], ["answer"]) == [
        {"question_id": "q1", "hypothesis": "answer"}
    ]
    with pytest.raises(ValueError):
        official_hypothesis_rows(["q1"], [])
