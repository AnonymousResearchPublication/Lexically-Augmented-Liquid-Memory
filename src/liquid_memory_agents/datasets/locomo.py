from __future__ import annotations

import json
import re
from pathlib import Path

from .longmemeval import LongMemEvalExample


_SESSION_RE = re.compile(r"^session_(\d+)$")


def load_locomo(path: str | Path) -> list[LongMemEvalExample]:
    """Load the released LoCoMo QA task into the shared conversational schema.

    The release does not include image files. When present, the released BLIP
    caption is appended to the turn text, making this an explicitly text-only
    evaluation over dialogue plus provided captions.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("LoCoMo payload must be a non-empty list")
    examples: list[LongMemEvalExample] = []
    for sample in payload:
        sample_id = str(sample["sample_id"])
        conversation = sample["conversation"]
        speaker_a = str(conversation.get("speaker_a", "speaker_a"))
        speaker_b = str(conversation.get("speaker_b", "speaker_b"))
        sessions = []
        evidence_to_session: dict[str, str] = {}
        session_keys = sorted(
            (key for key in conversation if _SESSION_RE.match(key)),
            key=lambda key: int(_SESSION_RE.match(key).group(1)),  # type: ignore[union-attr]
        )
        for key in session_keys:
            number = _SESSION_RE.match(key).group(1)  # type: ignore[union-attr]
            date = conversation.get(f"session_{number}_date_time")
            turns = []
            for turn in conversation[key]:
                speaker = str(turn.get("speaker", "unknown"))
                role = "user" if speaker == speaker_a else (
                    "assistant" if speaker == speaker_b else "user"
                )
                text = str(turn.get("text", ""))
                caption = str(turn.get("blip_caption", "")).strip()
                if caption:
                    text = f"{text}\n[Released image caption: {caption}]"
                dialog_id = str(turn.get("dia_id", ""))
                if dialog_id:
                    evidence_to_session[dialog_id] = key
                turns.append({
                    "role": role,
                    "content": f"{speaker}: {text}",
                    "dialog_id": dialog_id,
                })
            sessions.append({
                "session_id": key,
                "date": None if date is None else str(date),
                "turns": turns,
            })
        for qa_index, qa in enumerate(sample.get("qa", [])):
            evidence = [str(value) for value in (qa.get("evidence") or [])]
            answer_sessions = tuple(dict.fromkeys(
                evidence_to_session[value]
                for value in evidence
                if value in evidence_to_session
            ))
            answer = qa.get("answer", "")
            if isinstance(answer, list):
                answer = "; ".join(str(value) for value in answer)
            examples.append(LongMemEvalExample(
                question_id=f"{sample_id}-qa-{qa_index}",
                question_type=f"category-{qa.get('category', 'unknown')}",
                question=str(qa["question"]),
                question_date=None,
                answer=str(answer),
                sessions=tuple(sessions),
                answer_session_ids=answer_sessions,
            ))
    return examples
