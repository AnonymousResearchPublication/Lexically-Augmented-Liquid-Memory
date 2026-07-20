from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class SyntheticExample:
    example_id: str
    turns: tuple[str, ...]
    query: str
    answer: str
    question_type: str
    evidence_indices: tuple[int, ...]
    ground_truth_trace: tuple[str, ...]
    roles: tuple[str, ...]
    timestamps: tuple[str, ...]
    session_ids: tuple[str, ...]


def _entity(rng: random.Random) -> str:
    starts = ("Al", "Bel", "Cor", "Dra", "El", "Fen", "Gal", "Iri", "Jun", "Kel", "Lum", "Nor")
    ends = ("ara", "eton", "iri", "ova", "une", "yx", "ador", "elis", "orin", "essa")
    return rng.choice(starts) + rng.choice(ends) + f"-{rng.randrange(1000, 9999)}"


def generate_examples(count: int, horizon: int, seed: int) -> list[SyntheticExample]:
    """Generate deterministic open-vocabulary memory QA with per-turn truth traces."""
    rng, examples = random.Random(seed), []
    for index in range(count):
        person, old_place, new_place = _entity(rng), _entity(rng), _entity(rng)
        project, collaborator = _entity(rng), _entity(rng)
        boolean = bool((index // 8) % 2)
        correction_at = rng.randrange(max(2, horizon // 3), max(3, 2 * horizon // 3))
        question_type = (
            "stable_fact", "correction", "repeated_fact", "contradictory_update",
            "boolean", "goal", "response_style", "temporal", "multi_fact", "paraphrase",
        )[index % 10]
        state: dict[str, object] = {}
        scheduled: dict[int, tuple[str, dict[str, object]]] = {}
        if question_type == "stable_fact":
            scheduled[0] = (f"{person}'s hometown is {old_place}.", {"hometown": old_place})
            query, answer, evidence = f"What is {person}'s hometown?", old_place, (0,)
        elif question_type in {"correction", "contradictory_update"}:
            scheduled[0] = (f"{person} plans to visit {old_place}.", {"destination": old_place})
            scheduled[correction_at] = (
                f"Correction: {person} now plans to visit {new_place}, not {old_place}.",
                {"destination": new_place},
            )
            query, answer, qtype = (
                f"Where does {person} currently plan to visit?",
                new_place,
                question_type,
            )
            evidence = (0, correction_at)
            question_type = qtype
        elif question_type == "repeated_fact":
            scheduled[0] = (f"{person}'s preferred venue is {old_place}.", {"venue": old_place})
            scheduled[correction_at] = (
                f"To repeat, {person}'s preferred venue remains {old_place}.", {"venue": old_place}
            )
            query, answer, evidence = f"Which venue does {person} prefer?", old_place, (0, correction_at)
        elif question_type == "boolean":
            scheduled[0] = (
                f"{person} {'does' if boolean else 'does not'} want concise responses.",
                {"concise": boolean},
            )
            query, answer, evidence = (
                f"Does {person} want concise responses?", "yes" if boolean else "no", (0,)
            )
        elif question_type == "goal":
            scheduled[0] = (
                f"{person}'s current goal is to complete project {project}.", {"goal": project}
            )
            query, answer, evidence = f"What project does {person} aim to complete?", project, (0,)
        elif question_type == "response_style":
            old_style, new_style = "detailed", "bullet-only"
            scheduled[0] = (
                f"{person} prefers {old_style} responses.", {"response_style": old_style}
            )
            scheduled[correction_at] = (
                f"{person} now wants {new_style} responses instead.", {"response_style": new_style}
            )
            query, answer, evidence = f"How should responses to {person} be formatted?", new_style, (0, correction_at)
        elif question_type == "temporal":
            scheduled[0] = (
                f"{person} met {collaborator} before visiting {old_place}.",
                {"first_event": collaborator, "second_event": old_place},
            )
            query, answer, evidence = f"Who did {person} meet before visiting {old_place}?", collaborator, (0,)
        elif question_type == "multi_fact":
            scheduled[0] = (
                f"{person}'s hometown is {old_place}.", {"hometown": old_place}
            )
            scheduled[correction_at] = (
                f"{person}'s current project is {project}.", {"project": project}
            )
            query, answer, evidence = (
                f"Give {person}'s hometown and current project.",
                f"{old_place} and {project}",
                (0, correction_at),
            )
        else:
            scheduled[0] = (
                f"{person}'s preferred destination is {old_place}.", {"destination": old_place}
            )
            query, answer, evidence = (
                f"Where would {person} most enjoy going?", old_place, (0,)
            )
        turns, trace, roles, timestamps, session_ids = [], [], [], [], []
        for turn_index in range(horizon):
            if turn_index in scheduled:
                text, updates = scheduled[turn_index]
                state.update(updates)
                role = "user"
            else:
                text = f"Unrelated note {turn_index} concerns {_entity(rng)}."
                role = "assistant" if turn_index % 3 == 0 else "user"
            turns.append(text)
            trace.append(json.dumps(state, sort_keys=True))
            roles.append(role)
            timestamps.append(
                str(date(2024, 1, 1) + timedelta(days=turn_index // 10))
            )
            session_ids.append(f"session-{turn_index // 25:04d}")
        fingerprint = hashlib.sha256(
            f"{seed}:{horizon}:{index}:{query}:{answer}".encode()
        ).hexdigest()[:16]
        examples.append(
            SyntheticExample(
                f"syn-{fingerprint}", tuple(turns), query, answer,
                question_type, evidence, tuple(trace), tuple(roles),
                tuple(timestamps), tuple(session_ids),
            )
        )
    return examples


def split_examples(examples: list[SyntheticExample], train=0.7, validation=0.15):
    if train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError("invalid split fractions")
    first, second = int(len(examples) * train), int(len(examples) * (train + validation))
    splits = {"train": examples[:first], "validation": examples[first:second], "test": examples[second:]}
    id_sets = [{example.example_id for example in split} for split in splits.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("train/validation/test overlap")
    return splits
