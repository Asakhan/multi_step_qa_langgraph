"""오케스트레이션 구조 S0~S4 (LangGraph StateGraph). 동일 AgentState/도구/프롬프트 공유."""
from __future__ import annotations


def build_graph(structure: str):
    """구조 이름 -> 컴파일된 LangGraph 그래프."""
    structure = structure.lower()
    if structure == "s0":
        from .s0 import build
    elif structure == "s1":
        from .s1 import build
    elif structure == "s2":
        from .s2 import build
    elif structure == "s3":
        from .s3 import build
    elif structure == "s4":
        from .s4 import build
    elif structure == "s5":
        from .s5 import build
    elif structure == "s6":
        from .s6 import build
    elif structure == "s7":
        from .s7 import build
    elif structure in ("s7_gpt", "s7_claude", "s7_vpro"):
        # S7 과 동일 그래프. models.yaml 의 모델 배정만 다르다.
        from .s7 import build as _build_s7
        return _build_s7(structure)
    else:
        raise ValueError(f"알 수 없는 구조: {structure}")
    return build()
