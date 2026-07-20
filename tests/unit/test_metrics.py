from liquid_memory_agents.evaluation.metrics import exact_match, token_f1


def test_metrics_accept_numeric_references_defensively() -> None:
    assert exact_match("42", 42)
    assert token_f1("42", 42) == 1.0
