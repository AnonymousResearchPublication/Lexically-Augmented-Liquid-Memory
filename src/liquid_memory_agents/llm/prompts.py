def answer_system_prompt(abstention_response: str = "I do not know.") -> str:
    return (
        "Answer the question using only memory made available by the configured agent. "
        "Return only the shortest answer span, without explanation or a complete sentence. "
        f"If there is insufficient evidence, say exactly '{abstention_response}'"
    )


SYSTEM_PROMPT = answer_system_prompt()

def render_prompt(query: str, context: str = "", query_timestamp: str | None = None) -> str:
    date = query_timestamp or "unknown"
    return f"Memory context:\n{context}\n\nQuestion date: {date}\nQuestion: {query}"
