from __future__ import annotations

import re
from collections import Counter


def _tokens(text: object) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", str(text).casefold())


def exact_match(prediction: object, reference: object) -> bool:
    return " ".join(_tokens(prediction)) == " ".join(_tokens(reference))


def token_f1(prediction: object, reference: object) -> float:
    predicted, expected = Counter(_tokens(prediction)), Counter(_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected:
        return float(predicted == expected)
    precision, recall = overlap / sum(predicted.values()), overlap / sum(expected.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def locomo_qa_score(prediction: object, reference: object, category: int) -> float:
    """Reproduce the released LoCoMo QA evaluator's category-specific score."""
    from nltk.stem import PorterStemmer

    prediction_text = str(prediction)
    reference_text = str(reference)
    if category == 5:
        lowered = prediction_text.casefold()
        return float(
            "no information available" in lowered or "not mentioned" in lowered
        )
    if category == 3:
        reference_text = reference_text.split(";", 1)[0].strip()
    stemmer = PorterStemmer()

    def score(left: str, right: str) -> float:
        articles = {"a", "an", "the", "and"}
        predicted = Counter(
            stemmer.stem(token) for token in _tokens(left) if token not in articles
        )
        expected = Counter(
            stemmer.stem(token) for token in _tokens(right) if token not in articles
        )
        overlap = sum((predicted & expected).values())
        if not predicted or not expected or overlap == 0:
            return 0.0
        precision = overlap / sum(predicted.values())
        recall = overlap / sum(expected.values())
        return 2 * precision * recall / (precision + recall)

    if category == 1:
        predictions = [part.strip() for part in prediction_text.split(",")]
        references = [part.strip() for part in reference_text.split(",")]
        return sum(max(score(pred, ref) for pred in predictions) for ref in references) / len(references)
    if category in {2, 3, 4}:
        return score(prediction_text, reference_text)
    raise ValueError(f"unsupported LoCoMo QA category: {category}")


def paired_bootstrap(
    a: list[float], b: list[float], samples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    import numpy as np

    if len(a) != len(b) or not a:
        raise ValueError("paired non-empty arrays are required")
    rng, delta = np.random.default_rng(seed), np.asarray(a) - np.asarray(b)
    values = [float(rng.choice(delta, len(delta), replace=True).mean()) for _ in range(samples)]
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))
