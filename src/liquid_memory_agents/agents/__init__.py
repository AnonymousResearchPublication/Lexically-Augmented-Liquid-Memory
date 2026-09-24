"""Agent exports are loaded lazily so dataset tooling does not require Torch."""

from .base import AgentResponse, BaseAgent

__all__ = [
    "AgentResponse", "BaseAgent", "VanillaAgent", "WindowMemoryAgent",
    "RAGMemoryAgent", "BoundedRAGMemoryAgent", "LALMMemoryAgent", "LexicalMemoryAgent",
]

_MODULES = {
    "VanillaAgent": ".vanilla",
    "WindowMemoryAgent": ".window",
    "RAGMemoryAgent": ".rag",
    "BoundedRAGMemoryAgent": ".rag",
    "LALMMemoryAgent": ".liquid",
    "LexicalMemoryAgent": ".lexical",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
