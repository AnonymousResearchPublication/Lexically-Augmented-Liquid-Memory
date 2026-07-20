import json

from liquid_memory_agents.datasets.locomo import load_locomo


def test_load_locomo_maps_sessions_evidence_and_captions(tmp_path) -> None:
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps([{
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 Jan 2024",
            "session_1": [
                {
                    "speaker": "Alice",
                    "dia_id": "d1",
                    "text": "I adopted Pepper.",
                    "blip_caption": "a black dog",
                },
                {"speaker": "Bob", "dia_id": "d2", "text": "Wonderful!"},
            ],
        },
        "qa": [{
            "question": "Who did Alice adopt?",
            "answer": "Pepper",
            "category": 1,
            "evidence": ["d1"],
        }],
    }]), encoding="utf-8")

    examples = load_locomo(path)
    assert len(examples) == 1
    example = examples[0]
    assert example.question_id == "conv-1-qa-0"
    assert example.question_type == "category-1"
    assert example.answer_session_ids == ("session_1",)
    assert example.sessions[0]["turns"][0]["role"] == "user"
    assert "Released image caption: a black dog" in example.sessions[0]["turns"][0]["content"]
