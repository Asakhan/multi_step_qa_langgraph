"""모든 구조(S0~S4)가 공유하는 단일 AgentState 스키마.

구조 간 차이는 노드 구성·엣지 연결로만 발생해야 하므로, 스키마는 하나로 고정한다.
병렬 fan-out(S3)에서 여러 노드가 동시에 쓰는 필드(token_usage, candidate_answers,
node_latencies)는 reducer(operator.add) 를 붙여 누락 없이 누적되게 한다.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class TokenRecord(TypedDict):
    node: str
    model: str          # 정규 모델명(pricing key)
    model_string: str   # 실제 API 모델 문자열
    input_tokens: int
    output_tokens: int


class AgentState(TypedDict, total=False):
    # --- 입력/식별 ---
    query: str
    dataset: str
    question_id: str
    task_type: str
    lang: str

    # --- 검색 ---
    retrieved_evidence: list[dict[str, Any]]   # [{text, source{...}, chunk_id}]
    context_chunks: list[dict[str, Any]]       # FinQA 문항별 self-contained 컨텍스트(text_N/table_N)

    # --- 생성/후보 ---
    candidate_answers: Annotated[list[dict[str, Any]], operator.add]  # S3 병렬 누적
    final_answer: str
    reasoning_trace: str
    used_expression: list[str]                 # 모델이 사용한 산술식 문자열들

    # --- 검증(S2) ---
    critique: str
    retry_count: int
    verify_verdict: str                        # PASS|FAIL (마지막 검증 결과)

    # --- 분해/해결(S4) ---
    sub_questions: list[str]
    sub_answers: list[dict[str, Any]]          # [{sub_question, partial_answer, reasoning, expressions, evidence}]

    # --- 계측 (reducer 로 노드별 누락 없이 누적) ---
    token_usage: Annotated[list[TokenRecord], operator.add]
    node_latencies: Annotated[list[dict[str, Any]], operator.add]
    latency: float                             # 그래프 전체 wall-clock (runner 가 기록)

    # --- 집계 메타 ---
    aggregation_info: dict[str, Any]           # S3: 채택 후보/군집 정보
    run_error: str                             # run 단위 치명 오류 메시지


def new_state(query: str, *, dataset: str, question_id: str, task_type: str, lang: str,
              context_chunks: list | None = None) -> AgentState:
    return AgentState(
        query=query,
        dataset=dataset,
        question_id=question_id,
        task_type=task_type,
        lang=lang,
        retrieved_evidence=[],
        context_chunks=context_chunks or [],
        candidate_answers=[],
        final_answer="",
        reasoning_trace="",
        used_expression=[],
        critique="",
        retry_count=0,
        verify_verdict="",
        sub_questions=[],
        sub_answers=[],
        token_usage=[],
        node_latencies=[],
        latency=0.0,
        aggregation_info={},
        run_error="",
    )
