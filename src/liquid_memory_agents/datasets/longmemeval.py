from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LongMemEvalExample:
    question_id: str
    question_type: str
    question: str
    question_date: str | None
    answer: str
    sessions: tuple[dict[str, Any], ...]
    answer_session_ids: tuple[str, ...]


def load_longmemeval(path: str | Path) -> list[LongMemEvalExample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload["data"]
    examples = []
    for row in rows:
        ids = [str(value) for value in row.get("haystack_session_ids", [])]
        dates = row.get("haystack_dates", [None] * len(ids))
        sessions = tuple(
            {
                "session_id": sid,
                "date": None if date is None else str(date),
                "turns": [
                    {
                        **turn,
                        "role": str(turn.get("role", "user")),
                        "content": str(turn.get("content", "")),
                    }
                    for turn in turns
                ],
            }
            for sid, date, turns in zip(ids, dates, row["haystack_sessions"])
        )
        examples.append(
            LongMemEvalExample(
                question_id=str(row["question_id"]),
                question_type=str(row.get("question_type", "unknown")),
                question=str(row["question"]),
                question_date=(
                    None
                    if row.get("question_date") is None
                    else str(row["question_date"])
                ),
                answer=str(row["answer"]),
                sessions=sessions,
                answer_session_ids=tuple(
                    str(value) for value in row.get("answer_session_ids", [])
                ),
            )
        )
    return examples


def stratified_sample(
    examples: list[LongMemEvalExample],
    sample_size: int,
    seed: int,
) -> list[LongMemEvalExample]:
    """Select a deterministic proportional sample over question types."""
    if sample_size < 1 or sample_size > len(examples):
        raise ValueError("sample_size must be between 1 and the dataset size")

    groups: dict[str, list[tuple[int, LongMemEvalExample]]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.question_type, []).append((index, example))

    raw_quotas = {
        question_type: sample_size * len(group) / len(examples)
        for question_type, group in groups.items()
    }
    quotas = {
        question_type: math.floor(raw_quota)
        for question_type, raw_quota in raw_quotas.items()
    }
    remaining = sample_size - sum(quotas.values())
    allocation_order = sorted(
        groups,
        key=lambda question_type: (
            -(raw_quotas[question_type] - quotas[question_type]),
            question_type,
        ),
    )
    for question_type in allocation_order[:remaining]:
        quotas[question_type] += 1

    rng = random.Random(seed)
    selected: list[tuple[int, LongMemEvalExample]] = []
    for question_type in sorted(groups):
        selected.extend(rng.sample(groups[question_type], quotas[question_type]))
    return [example for _, example in sorted(selected)]


def official_hypothesis_rows(question_ids: list[str], predictions: list[str]) -> list[dict[str, str]]:
    if len(question_ids) != len(predictions):
        raise ValueError("question/prediction length mismatch")
    return [
        {"question_id": question_id, "hypothesis": prediction}
        for question_id, prediction in zip(question_ids, predictions)
    ]
