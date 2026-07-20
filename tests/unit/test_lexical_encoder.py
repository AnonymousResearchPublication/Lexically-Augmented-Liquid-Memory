import torch

from liquid_memory_agents.embeddings import LexicalAugmentedEncoder


class SemanticStub:
    dimension = 3

    def encode(self, text):
        count = 1 if isinstance(text, str) else len(text)
        values = torch.ones((count, self.dimension), dtype=torch.float32)
        return values[0] if isinstance(text, str) else values


def test_lexical_encoder_preserves_position_and_shape() -> None:
    encoder = LexicalAugmentedEncoder(SemanticStub(), max_bytes=4)
    first = encoder.encode("ab")
    second = encoder.encode("ba")
    assert first.shape == (8,)
    assert not torch.equal(first[3:7], second[3:7])
    batch = encoder.encode(["ab", "ba"])
    assert batch.shape == (2, 8)
    assert batch[0, -1].item() == 0.5
