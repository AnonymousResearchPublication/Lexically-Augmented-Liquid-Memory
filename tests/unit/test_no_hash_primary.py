from pathlib import Path


def test_primary_source_has_no_hash_encoder() -> None:
    root = Path(__file__).parents[2] / "src" / "liquid_memory_agents"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "HashEmbedding" not in source
    assert "hash_encoder" not in source.casefold()
