from liquid_memory_agents.evaluation.metrics import locomo_qa_score


def test_locomo_category_five_abstention() -> None:
    assert locomo_qa_score("No information available.", "irrelevant", 5) == 1.0
    assert locomo_qa_score("I do not know.", "irrelevant", 5) == 0.0


def test_locomo_category_one_comma_separated_partial_answers() -> None:
    assert locomo_qa_score("camping, painting", "painting, camping", 1) == 1.0


def test_locomo_category_three_uses_pre_semicolon_reference() -> None:
    assert locomo_qa_score("psychology", "psychology; inferred from support", 3) == 1.0
