def complexity_table() -> list[dict[str, str]]:
    return [
        {"agent": "vanilla", "storage": "O(1)", "update": "O(1)", "read": "O(1)"},
        {"agent": "window", "storage": "O(wL)", "update": "O(L)", "read": "O(wL)"},
        {"agent": "rag", "storage": "O(nd+nL)", "update": "O(d)", "read": "O(nd) flat IP"},
        {
            "agent": "lalm",
            "storage": "O(Sd + C(e+B))",
            "update": "O(Sd^2+Sdp+C^2e)",
            "read": "O(Sd+Ce)+decoder",
        },
    ]
