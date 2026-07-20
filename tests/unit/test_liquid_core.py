import pytest

torch = pytest.importorskip("torch")

from liquid_memory_agents.liquid.cell import AdaptiveLiquidCell
from liquid_memory_agents.liquid.llm_adapter import StateToPrefixAdapter
from liquid_memory_agents.liquid.memory_bank import LiquidMemoryBank
from liquid_memory_agents.liquid.query_reader import QueryConditionedReader


def test_adaptive_tau_is_positive_and_bounded() -> None:
    cell = AdaptiveLiquidCell(12, 8, 4, tau_min=0.2, tau_max=3.0)
    output, diagnostics = cell(torch.zeros(4, 8), torch.randn(12))
    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()
    assert diagnostics["tau"].min() >= 0.2
    assert diagnostics["tau"].max() <= 3.0


def test_fixed_memory_bytes_do_not_depend_on_updates() -> None:
    bank = LiquidMemoryBank(AdaptiveLiquidCell(12, 8, 4), dtype=torch.float32)
    expected = 4 * 8 * 4
    for _ in range(500):
        bank.update(torch.randn(12))
        assert bank.persistent_bytes == expected


def test_query_reader_uses_every_slot() -> None:
    reader = QueryConditionedReader(12, 8)
    _, weights = reader(torch.randn(12), torch.randn(4, 8))
    assert weights.shape == (4,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_prefix_adapter_is_open_vocabulary_not_candidate_lookup() -> None:
    adapter = StateToPrefixAdapter(8, 32, prefix_tokens=6)
    assert adapter(torch.randn(8)).shape == (6, 32)
    assert not hasattr(adapter, "candidates")
    assert not hasattr(adapter, "fact_schema")


def test_prefix_can_cross_a_mixed_precision_llm_boundary() -> None:
    adapter = StateToPrefixAdapter(8, 32, prefix_tokens=6)
    prefix = adapter(torch.randn(8))
    token_embeddings = torch.randn(4, 32, dtype=torch.bfloat16)
    cast_prefix = prefix.to(token_embeddings.dtype)
    combined = torch.cat((cast_prefix, token_embeddings), dim=0)
    assert combined.dtype == torch.bfloat16
    assert combined.shape == (10, 32)
