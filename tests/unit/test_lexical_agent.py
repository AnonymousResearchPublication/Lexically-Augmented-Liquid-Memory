import pytest

torch = pytest.importorskip("torch")

from liquid_memory_agents.agents import LexicalMemoryAgent


class Encoder:
    dimension = 3

    def encode(self, text):
        values = [text] if isinstance(text, str) else text
        rows = []
        for value in values:
            lowered = value.casefold()
            rows.append(torch.tensor([
                float("paris" in lowered),
                float("tokyo" in lowered),
                1.0,
            ]))
        result = torch.stack(rows)
        return result[0] if isinstance(text, str) else result


class Answerer:
    embedding_dimension = 8

    def __init__(self):
        self.prefix_called = False

    def answer(self, query, context, query_timestamp=None):
        return context

    def answer_with_prefix(self, *args, **kwargs):
        self.prefix_called = True
        raise AssertionError("lexical-only must not inject Liquid virtual tokens")


def test_lexical_only_uses_same_sidecar_without_liquid_prefix() -> None:
    answerer = Answerer()
    agent = LexicalMemoryAgent(
        Encoder(),
        answerer,
        capacity=2,
        top_k=1,
        max_text_bytes=128,
    )
    agent.observe(
        {"role": "user", "content": "The venue is Paris."},
        timestamp="2026-01-01",
        session_id="s1",
    )
    response = agent.answer("Which venue is Paris?")
    assert agent.name == "lexical_only"
    assert "The venue is Paris." in response.text
    assert not answerer.prefix_called
    assert response.metadata["latent_enabled"] is False
    assert agent.memory_size_bytes() == 2 * (3 * 4 + 128)


def test_lexical_only_query_does_not_mutate_memory() -> None:
    agent = LexicalMemoryAgent(Encoder(), Answerer(), capacity=2, top_k=1)
    agent.observe("Remember Tokyo.")
    before_embeddings = agent.memory.embeddings.clone()
    before_texts = list(agent.memory.texts)
    agent.answer("What should be remembered?")
    assert torch.equal(before_embeddings, agent.memory.embeddings)
    assert before_texts == agent.memory.texts
