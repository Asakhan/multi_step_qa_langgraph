"""S6 (실험 3 개선안 — self-probe 신뢰도 캐스케이드).

지도교수 피드백(m1, 문헌D: Kadavath et al. 2022) 및 S5 로그 재보정 분석(작성관리
§3-1)에서 드러난 문제 — S5 의 모델 자기보고 확신도(generate_s5.txt)가 몇 개 값
(약 0.7 / 약 0.95~1.0)에만 몰려 있어(clustering) 임계값을 0.75~0.95 사이에서 바꿔도
게이트 판정이 전혀 안 바뀌는 문제 — 를 해결하기 위한 구조.

retrieve -> generate(s5 와 동일, expressions+code_conflict 계산용으로 재사용) ->
    self_probe(독립 재확인 질문을 temperature>0 에서 n_samples 회 반복, True 비율을
               유효 신뢰도로 사용 — generate 의 verbalized confidence 를 대체) ->
    gate -> stop(확신)-> END / escalate(불확실) -> verify -> (PASS->END /
                                                     FAIL & retry<max -> prep_retry -> generate)

s5 와의 유일한 차이는 "신뢰도를 어떻게 산출하는가"이며, 그래프 구조·verify·retry 로직은
s5 와 동일하다(비교 가능성 유지).
"""
from __future__ import annotations

import os

from langgraph.graph import StateGraph, START, END

from .. import config
from ..state import AgentState
from .common import (
    make_retrieve_node,
    make_generate_s5_node,
    make_self_probe_node,
    make_gate_node,
    make_verify_node,
    make_prep_retry_node,
)

_RULES = config.experiment_cfg()["structure_rules"]
_MAX_RETRY = int(_RULES["s6_max_retry"])
_N_SAMPLES = int(_RULES["s6_probe_samples"])
_PROBE_TEMPERATURE = float(_RULES["s6_probe_temperature"])


def _threshold() -> float:
    """게이트 임계값: .env S6_THRESHOLD 우선, 없으면 experiment.yaml 기본값."""
    raw = os.environ.get("S6_THRESHOLD", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(_RULES["s6_confidence_threshold"])


def _route_after_generate(state):
    # 첫 생성(retry_count==0) -> self_probe 로. 재생성(>0) -> 곧장 verify(무한루프 방지, s5 와 동일).
    return "self_probe" if int(state.get("retry_count", 0)) == 0 else "verify"


def _route_after_gate(state):
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
    g.add_node("generate", make_generate_s5_node("s6", role="generate", node_name="generate"))
    g.add_node("self_probe", make_self_probe_node("s6", _N_SAMPLES, _PROBE_TEMPERATURE))
    g.add_node("gate", make_gate_node(_threshold()))
    g.add_node("verify", make_verify_node("s6"))
    g.add_node("prep_retry", make_prep_retry_node())

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", _route_after_generate,
                            {"self_probe": "self_probe", "verify": "verify"})
    g.add_edge("self_probe", "gate")
    g.add_conditional_edges("gate", _route_after_gate, {"verify": "verify", END: END})
    g.add_conditional_edges("verify", _route_after_verify, {"prep_retry": "prep_retry", END: END})
    g.add_edge("prep_retry", "generate")
    return g.compile()
