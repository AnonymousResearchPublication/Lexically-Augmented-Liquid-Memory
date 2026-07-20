import pytest

torch = pytest.importorskip("torch")

from liquid_memory_agents.agents.liquid import LALMMemoryAgent


class Encoder:
    dimension = 6

    def encode(self, text):
        # Deterministic test encoder only; primary configuration uses SentenceTransformer.
        value = float(sum(text.encode("utf-8")) % 31) / 31
        return torch.tensor([value, 1, 2, 3, 4, 5], dtype=torch.float32)


class Answerer:
    embedding_dimension = 10

    def answer_with_prefix(self, query, prefix, query_timestamp=None, context=""):
        self.last_prefix = prefix.detach().clone()
        return f"free text for {query}"


def test_query_never_updates_memory_and_raw_turn_is_not_retained() -> None:
    agent = LALMMemoryAgent(Encoder(), Answerer(), state_dim=8, slots=4)
    secret = "Caledonia-9281"
    agent.observe(f"My destination is {secret}.", timestamp="2026-01-01", session_id="s1")
    before = agent.memory.state.clone()
    response = agent.answer("Where am I going?")
    assert torch.equal(before, agent.memory.state)
    assert response.memory_context == ""
    assert secret not in repr(agent.__dict__)


def test_reset_clears_state() -> None:
    agent = LALMMemoryAgent(Encoder(), Answerer(), state_dim=8, slots=4)
    agent.observe("remember this")
    agent.reset()
    assert torch.count_nonzero(agent.memory.state) == 0
    assert agent.memory_size_bytes() == 4 * 8 * 4


def test_pure_liquid_has_explicit_name_and_only_latent_storage() -> None:
    agent = LALMMemoryAgent(
        Encoder(),
        Answerer(),
        state_dim=8,
        slots=4,
        lexical_enabled=False,
        condition_name="pure_liquid",
    )
    assert agent.name == "pure_liquid"
    assert agent.memory_size_bytes() == 4 * 8 * 4


def test_lalm_zero_prefix_removes_prefix_signal() -> None:
    answerer = Answerer()
    agent = LALMMemoryAgent(
        Encoder(),
        answerer,
        state_dim=8,
        slots=4,
        condition_name="lalm_zero_prefix",
    )
    agent.observe("remember this")
    agent.answer("what should I remember?")
    assert agent.name == "lalm_zero_prefix"
    assert agent._last_read["prefix_control"] == "zero"
    assert torch.count_nonzero(answerer.last_prefix) == 0


def test_lalm_random_prefix_is_deterministic_and_norm_matched() -> None:
    agent = LALMMemoryAgent(
        Encoder(),
        Answerer(),
        state_dim=8,
        slots=4,
        condition_name="lalm_random_prefix",
    )
    prefix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    first = agent._controlled_prefix(prefix, "query")
    second = agent._controlled_prefix(prefix, "query")
    assert torch.equal(first, second)
    assert torch.isclose(first.norm(), prefix.norm(), rtol=1e-5, atol=1e-5)
    assert not torch.equal(first, prefix)


def test_lalm_permuted_prefix_preserves_values_but_breaks_positions() -> None:
    agent = LALMMemoryAgent(
        Encoder(),
        Answerer(),
        state_dim=8,
        slots=4,
        condition_name="lalm_permuted_prefix",
    )
    prefix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    permuted = agent._controlled_prefix(prefix, "query")
    assert torch.equal(torch.sort(permuted.reshape(-1)).values, prefix.reshape(-1))
    assert not torch.equal(permuted, prefix)
