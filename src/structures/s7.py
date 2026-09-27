"""S7 (실험 C — 반사실 게이트 재현 기반 구조, counterfactual gate replay base).

그래프: retrieve -> generate(s5 프롬프트, 자기보고 confidence) -> verify(전수) ->
        PASS -> END / FAIL & retry<max -> prep_retry -> generate -> END

S5(동적 캐스케이드)와의 차이는 **게이트를 두지 않는다**는 한 가지뿐이다. 즉 모든 run 에
검증을 적용하되(=S2 의 검증 정책), 생성은 S5 와 동일한 프롬프트로 자기보고 confidence 를
받고, **1차 생성의 답·신호를 스냅샷으로 보존**한다.

왜 이렇게 설계하는가 (실험 A 결과에 대한 대응):
  실험 A 파일럿에서 "검증기를 더 엄격하게 만드는" 경로(L2·L3)는 실패했고, 병목은
  검증 엄격도가 아니라 **무엇을 검증할지 고르는 게이트의 선별 정확도**로 확인되었다
  (실험 B: 신호 결합 시 정밀도 0.239→0.343, 재현율 0.700→0.900).
  게이트 정책을 바꿔 실측하려면 정책마다 180 run 을 다시 돌려야 하는데, 그 정책들은
  전부 "1차 생성 결과에 검증을 적용할지 말지"의 이진 선택일 뿐이다. 따라서
  **모든 run 에 검증을 적용해 두 갈래(검증 안 함 = 1차 답 / 검증함 = 최종 답)를 한 번에
  관측**하면, 임의의 게이트 정책 성능을 추가 호출 없이 정확히 재구성할 수 있다.

  이 재구성의 타당성은 내부 검증으로 확인한다: 자기보고 임계값 0.8 정책을 재현했을 때
  실제 S5 실측(발동률·TSR)이 복원되는지 대조한다(src/replay_gates.py).

재현성: 기존 S0~S6 그래프·프롬프트·모델은 일절 변경하지 않는다. S7 은 신규 구조다.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .. import config
from ..state import AgentState
from .common import (
    make_retrieve_node,
    make_generate_s7_node,
    make_verify_node,
    make_prep_retry_node,
)

_RULES = config.experiment_cfg()["structure_rules"]
_MAX_RETRY = int(_RULES["s7_max_retry"])


def _route_after_generate(state):
    # 첫 생성(retry_count==0) → verify. 재생성(>0) → END (S2·S5 와 동일한 무한루프 방지).
    return "verify" if int(state.get("retry_count", 0)) == 0 else END


def _route_after_verify(state):
    if state.get("verify_verdict") == "PASS":
        return END
    if int(state.get("retry_count", 0)) < _MAX_RETRY:
        return "prep_retry"
    return END


def build(structure: str = "s7"):
    """S7 그래프. `structure` 는 models.yaml 의 노드->모델 배정을 고르는 키다.

    기본값 "s7" 은 기존 실험 C(gemini-2.5-flash 단일 모델)와 완전히 동일하다.
    기반 모델 일반화(H6) 검정을 위한 변형(s7_gpt / s7_claude)과 상위 모델 검증기
    실험(s7_vpro)은 그래프/프롬프트/채점기를 그대로 두고 모델 배정만 바꾼다.
    """
    g = StateGraph(AgentState)
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("generate", make_generate_s7_node(structure, role="generate", node_name="generate"))
    g.add_node("verify", make_verify_node(structure))
    g.add_node("prep_retry", make_prep_retry_node())

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", _route_after_generate, {"verify": "verify", END: END})
    g.add_conditional_edges("verify", _route_after_verify, {"prep_retry": "prep_retry", END: END})
    g.add_edge("prep_retry", "generate")
    return g.compile()
