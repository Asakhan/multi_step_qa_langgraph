"""S4 (분해): decompose -> solve -> aggregate -> END.
Decomposer+Solver+Aggregator 3역할 모두 gemini-2.5-flash. sub-question 최대 5개.
solve 는 sub-question 순회하며 이전 부분답을 다음 컨텍스트에 누적.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from ..state import AgentState
from .common import make_decompose_node, make_solve_node, make_aggregate_s4_node


def build():
    g = StateGraph(AgentState)
    g.add_node("decompose", make_decompose_node("s4"))
    g.add_node("solve", make_solve_node("s4"))
    g.add_node("aggregate", make_aggregate_s4_node("s4"))

    g.add_edge(START, "decompose")
    g.add_edge("decompose", "solve")
    g.add_edge("solve", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()
