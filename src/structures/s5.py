"""S5 (실험 3 — 동적 신뢰도 캐스케이드, 논문 3.4.3).

retrieve -> generate(s5, confidence 산출) -> gate ->
    gate_decision=="stop"(확신)     -> END                     (저비용 단일 호출로 종료)
    gate_decision=="escalate"(불확실)-> verify -> (PASS->END /
                                         FAIL & retry<max -> prep_retry -> generate_s5)

핵심 통제:
- 신뢰도 신호는 모델 자기보고(generate_s5.txt). self-consistency 는 temperature=0 에서
  반복 불일치가 발생하지 않으므로 사용하지 않는다(논문 3.1).
- 재생성 경로(retry_count>0)에서는 gate 를 건너뛰고 곧바로 verify 로 가며, 검증 결과와
  무관하게 END(s2 패턴과 동일) → 무한루프 방지.
- verify / prep_retry 는 s2 와 동일한 make_verify_node('s5')·make_prep_retry_node() 재사용.
"""
from __future__ import annotations

import os

from langgraph.graph import StateGraph, START, END

from .. import config
from ..state import AgentState
from .common import (
    make_retrieve_node,
    make_generate_s5_node,
    make_gate_node,
    make_verify_node,
    make_prep_retry_node,
)

_RULES = config.experiment_cfg()["structure_rules"]
_MAX_RETRY = int(_RULES["s5_max_retry"])


def _threshold() -> float:
    """게이트 임계값: .env S5_THRESHOLD 우선, 없으면 experiment.yaml 기본값(스윕 편의)."""
    raw = os.environ.get("S5_THRESHOLD", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(_RULES["s5_confidence_threshold"])


def _route_after_generate(state):
    # 첫 생성(retry_count==0) → gate 로. 재생성(>0) → gate 건너뛰고 곧장 verify(무한루프 방지).
    return "gate" if int(state.get("retry_count", 0)) == 0 else "verify"


def _route_after_gate(state):
    # 확신(stop) → END, 불확실(escalate) → verify(실험2와 동일 검증).
    return END if state.get("gate_decision") == "stop" else "verify"


def _route_after_verify(state):
    if state.get("verify_verdict") == "PASS":
        return END
    if int(state.get("retry_count", 0)) < _MAX_RETRY:
        return "prep_retry"
    return END


def build():
    g = StateGraph(AgentState)
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("generate", make_generate_s5_node("s5", role="generate", node_name="generate"))
    g.add_node("gate", make_gate_node(_threshold()))
    g.add_node("verify", make_verify_node("s5"))
    g.add_node("prep_retry", make_prep_retry_node())

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", _route_after_generate,
                            {"gate": "gate", "verify": "verify"})
    g.add_conditional_edges("gate", _route_after_gate, {"verify": "verify", END: END})
    g.add_conditional_edges("verify", _route_after_verify, {"prep_retry": "prep_retry", END: END})
    g.add_edge("prep_retry", "generate")
    return g.compile()
