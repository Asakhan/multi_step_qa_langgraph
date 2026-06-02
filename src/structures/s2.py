"""S2 (검증): retrieve -> generate -> verify 조건부.
verify=PASS면 END, FAIL이면 critique 기록 후 generate로 1회만 재생성(retry_count 통제),
재생성 후엔 검증 결과와 무관하게 END.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .. import config
from ..state import AgentState
from .common import (
    make_retrieve_node,
    make_generate_node,
    make_verify_node,
    make_prep_retry_node,
)

_MAX_RETRY = int(config.experiment_cfg()["structure_rules"]["s2_max_retry"])


def _route_after_generate(state):
    # 첫 생성(retry_count==0)이면 검증으로, 재생성(>0)이면 종료.
    return "verify" if int(state.get("retry_count", 0)) == 0 else END


def _route_after_verify(state):
    if state.get("verify_verdict") == "PASS":
        return END
    if int(state.get("retry_count", 0)) < _MAX_RETRY:
        return "prep_retry"
    return END


def build():
    g = StateGraph(AgentState)
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("generate", make_generate_node("s2", role="generate", node_name="generate"))
    g.add_node("verify", make_verify_node("s2"))
    g.add_node("prep_retry", make_prep_retry_node())

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", _route_after_generate, {"verify": "verify", END: END})
    g.add_conditional_edges("verify", _route_after_verify, {"prep_retry": "prep_retry", END: END})
    g.add_edge("prep_retry", "generate")
    return g.compile()
