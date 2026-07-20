from liquid_memory_agents.datasets.synthetic import generate_examples, split_examples
import json


def test_boolean_subset_is_balanced_and_queries_are_not_turns() -> None:
    examples = generate_examples(40, horizon=100, seed=7)
    answers = [x.answer for x in examples if x.question_type == "boolean"]
    assert answers.count("yes") == answers.count("no")
    assert all(example.query not in example.turns for example in examples)
    assert all(len(example.roles) == len(example.turns) for example in examples)
    assert all(len(example.timestamps) == len(example.turns) for example in examples)
    assert all(len(example.session_ids) == len(example.turns) for example in examples)
    assert {example.question_type for example in examples} == {
        "stable_fact", "correction", "repeated_fact", "contradictory_update",
        "boolean", "goal", "response_style", "temporal", "multi_fact", "paraphrase",
    }


def test_generation_is_seeded_and_splits_do_not_overlap() -> None:
    first = generate_examples(20, 100, 9)
    second = generate_examples(20, 100, 9)
    assert first == second
    splits = split_examples(first)
    ids = [{x.example_id for x in rows} for rows in splits.values()]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]


def test_corrections_update_ground_truth_after_the_correction_turn() -> None:
    examples = generate_examples(20, 100, 7)
    example = next(row for row in examples if row.question_type == "correction")
    before, correction = example.evidence_indices
    assert json.loads(example.ground_truth_trace[before])["destination"] != example.answer
    assert json.loads(example.ground_truth_trace[correction])["destination"] == example.answer
