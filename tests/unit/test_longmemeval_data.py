import json

from liquid_memory_agents.datasets.longmemeval import (
    LongMemEvalExample,
    load_longmemeval,
    official_hypothesis_rows,
    stratified_sample,
)


def test_longmemeval_loader_preserves_ids_and_dates(tmp_path) -> None:
    source = tmp_path / "sample.json"
    source.write_text(json.dumps([{
        "question_id": "q-1",
        "question_type": "temporal-reasoning",
        "question": "Where did I go?",
        "question_date": "2025-02-01",
        "answer": "Orion",
        "haystack_session_ids": ["session-10"],
        "haystack_dates": ["2025-01-01"],
        "haystack_sessions": [[
            {"role": "user", "content": "I went to Orion."},
            {"role": "assistant", "content": "Noted."},
        ]],
        "answer_session_ids": ["session-10"],
    }]), encoding="utf-8")

    example = load_longmemeval(source)[0]

    assert example.question_id == "q-1"
    assert example.question_type == "temporal-reasoning"
    assert example.question_date == "2025-02-01"
    assert example.sessions[0]["session_id"] == "session-10"
    assert example.sessions[0]["date"] == "2025-01-01"
    assert example.sessions[0]["turns"][1]["role"] == "assistant"
    assert example.answer_session_ids == ("session-10",)
    assert official_hypothesis_rows(["q-1"], ["Orion"]) == [
        {"question_id": "q-1", "hypothesis": "Orion"}
    ]


def test_longmemeval_loader_normalizes_numeric_answers_to_text(tmp_path) -> None:
    source = tmp_path / "numeric.json"
    source.write_text(json.dumps([{
        "question_id": 17,
        "question_type": "single-session-user",
        "question": "How many?",
        "question_date": None,
        "answer": 42,
        "haystack_session_ids": [3],
        "haystack_dates": [None],
        "haystack_sessions": [[
            {"role": "user", "content": 42},
        ]],
        "answer_session_ids": [3],
    }]), encoding="utf-8")

    example = load_longmemeval(source)[0]

    assert example.question_id == "17"
    assert example.answer == "42"
    assert example.sessions[0]["session_id"] == "3"
    assert example.sessions[0]["turns"][0]["content"] == "42"
    assert example.answer_session_ids == ("3",)


def test_stratified_sample_is_proportional_and_deterministic() -> None:
    def example(index: int, question_type: str) -> LongMemEvalExample:
        return LongMemEvalExample(
            question_id=str(index),
            question_type=question_type,
            question="question",
            question_date=None,
            answer="answer",
            sessions=(),
            answer_session_ids=(),
        )

    examples = [
        *(example(index, "majority") for index in range(8)),
        *(example(index, "minority") for index in range(8, 10)),
    ]

    first = stratified_sample(examples, sample_size=5, seed=7)
    second = stratified_sample(examples, sample_size=5, seed=7)

    assert [row.question_id for row in first] == [
        row.question_id for row in second
    ]
    assert [row.question_type for row in first].count("majority") == 4
    assert [row.question_type for row in first].count("minority") == 1
