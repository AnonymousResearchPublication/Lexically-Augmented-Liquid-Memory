import inspect

import pytest

torch = pytest.importorskip("torch")

from liquid_memory_agents.agents import (
    LexicalMemoryAgent,
    LALMMemoryAgent,
    RAGMemoryAgent,
    VanillaAgent,
    WindowMemoryAgent,
)


class Encoder:
    dimension = 6

    def encode(self, text):
        values = torch.tensor(
            [len(text) % 7, text.count("alpha"), text.count("beta"), 1, 2, 3],
            dtype=torch.float32,
        )
        return values + 0.01


class Answerer:
    embedding_dimension = 10

    def answer(self, query, context, query_timestamp=None):
        return f"answer:{query}:{len(context)}"

    def answer_with_prefix(
        self, query, prefix, query_timestamp=None, context=""
    ):
        assert prefix.shape == (4, self.embedding_dimension)
        return f"prefix-answer:{query}:{len(context)}"

def make_agents():
    encoder, answerer = Encoder(), Answerer()

    def liquid():
        return LALMMemoryAgent(
            encoder, answerer, state_dim=8, slots=4, prefix_tokens=4
        )

    def rag():
        return RAGMemoryAgent(encoder, answerer, top_k=2)

    return [
        VanillaAgent(answerer),
        WindowMemoryAgent(answerer, max_messages=2),
        rag(),
        liquid(),
        LexicalMemoryAgent(encoder, answerer, capacity=4, top_k=2),
    ]


def test_every_agent_implements_and_executes_uniform_interface() -> None:
    required = {
        "reset", "observe", "answer", "get_memory_context",
        "memory_size_bytes", "diagnostics",
    }
    for agent in make_agents():
        assert required <= set(dir(agent))
        agent.reset()
        agent.observe(
            {"role": "user", "content": "remember alpha"},
            timestamp="2026-01-01",
            session_id="session-1",
        )
        before = agent.memory_size_bytes()
        response = agent.answer("what should be remembered?", "2026-01-02")
        after = agent.memory_size_bytes()
        assert isinstance(response.text, str) and response.text
        assert isinstance(response.metadata, dict)
        assert isinstance(agent.diagnostics(), dict)
        assert before == after, f"query mutated {agent.name} memory"


def test_agent_method_signatures_include_temporal_metadata() -> None:
    for agent in make_agents():
        observe = inspect.signature(agent.observe)
        answer = inspect.signature(agent.answer)
        assert "timestamp" in observe.parameters
        assert "session_id" in observe.parameters
        assert "query_timestamp" in answer.parameters


def test_rag_logical_payload_counts_shared_text_once() -> None:
    agent = RAGMemoryAgent(Encoder(), Answerer(), top_k=1)
    agent.observe(
        {"role": "user", "content": "remember alpha"},
        timestamp="2026-01-01",
        session_id="session-1",
    )
    record = agent.turns[0]
    assert agent.sessions["session-1"][0] is record.text
    expected = record.embedding.numel() * record.embedding.element_size()
    expected += len(record.text.encode("utf-8")) + 8
    assert agent.memory_size_bytes() == expected
