import torch

from liquid_memory_agents.liquid import BoundedLexicalMemory


def test_bounded_lexical_memory_retrieves_exact_text_and_has_fixed_budget() -> None:
    memory = BoundedLexicalMemory(
        embedding_dim=3, capacity=2, top_k=1, max_text_bytes=64
    )
    memory.update("The exact code is Zephyr-4821.", torch.tensor([1.0, 0.0, 0.0]))
    memory.update("An unrelated note.", torch.tensor([0.0, 1.0, 0.0]))
    context, metadata = memory.retrieve(torch.tensor([1.0, 0.0, 0.0]))
    assert "Zephyr-4821" in context
    assert metadata["lexical_entries"] == 2
    assert memory.reserved_bytes == 2 * (3 * 4 + 64)


def test_correction_replaces_redundant_entry_at_capacity() -> None:
    memory = BoundedLexicalMemory(
        embedding_dim=2, capacity=1, top_k=1, max_text_bytes=64
    )
    memory.update("Destination is Oldtown.", torch.tensor([1.0, 0.0]))
    memory.update(
        "Correction: destination is Newtown instead.",
        torch.tensor([1.0, 0.0]),
    )
    context, _ = memory.retrieve(torch.tensor([1.0, 0.0]))
    assert "Newtown" in context
    assert "Oldtown" not in context


def test_configured_correction_cues_control_replacement() -> None:
    memory = BoundedLexicalMemory(
        embedding_dim=2,
        capacity=1,
        top_k=1,
        max_text_bytes=64,
        correction_cues=["revision:"],
    )
    memory.update("old value", torch.tensor([1.0, 0.0]))
    memory.update("actually new value", torch.tensor([1.0, 0.0]))
    assert memory.texts[0] == "old value"
    memory.update("revision: new value", torch.tensor([1.0, 0.0]))
    assert memory.texts[0] == "revision: new value"


def test_retrieval_reports_session_ids() -> None:
    memory = BoundedLexicalMemory(
        embedding_dim=2, capacity=2, top_k=2, max_text_bytes=128
    )
    memory.update(
        "session=session-a; role=user; content=Alpha",
        torch.tensor([1.0, 0.0]),
    )
    memory.update(
        "session=session-b; role=user; content=Beta",
        torch.tensor([0.0, 1.0]),
    )

    _, metadata = memory.retrieve(torch.tensor([1.0, 0.0]))

    assert metadata["retrieved_session_ids"] == ["session-a", "session-b"]
