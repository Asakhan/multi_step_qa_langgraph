"""S1 (상한): S0과 동일 그래프, generate 노드 모델만 gemini-3.1-pro 로 교체."""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from ..state import AgentState
from .common import make_retrieve_node, make_generate_node


def build():
    g = StateGraph(AgentState)
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("generate", make_generate_node("s1", role="generate", node_name="generate"))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
