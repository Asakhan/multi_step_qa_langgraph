"""구조 공통 노드 빌더 + JSON 호출 헬퍼.

모든 노드는 저자 확정 프롬프트를 system(역할/규칙/출력형식) + user(INPUTS 슬롯) 로 분리해
사용한다. 프롬프트 파일은 'INPUTS:' 기준으로 분할되며, 변수 입력(query/evidence/...)은
user 메시지에만 주입된다(저자 명세: "{...} 표기는 user 메시지로 주입되는 슬롯").
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

from .. import config, llm_clients, metrics, tools
from ..prompt_loader import system_prompt

_MODELS = config.models_cfg()
_EXP = config.experiment_cfg()


class JSONParseError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _split_prompt(name: str) -> tuple[str, str]:
    """프롬프트를 (system_part, inputs_template) 로 분할. INPUTS 이후가 user 템플릿."""
    full = system_prompt(name)
    marker = "\nINPUTS:"
    idx = full.find(marker)
    if idx < 0:
        return full, ""
    system_part = full[:idx].rstrip()
    inputs_template = full[idx + 1:]  # 'INPUTS:' 로 시작
    return system_part, inputs_template


def _extract_json(text: str):
    """모델 응답에서 JSON 객체 추출. 실패 시 None."""
    if not text:
        return None
    t = text.strip()
    # 코드펜스 제거
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        frag = t[start:end + 1]
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None


# 저자 확정 프롬프트(common_rules)는 근거 부족 시 정확히 'INSUFFICIENT_EVIDENCE' 를
# (JSON 이 아닌 토큰으로) 출력하도록 지시한다. 이는 모델의 정당한 기권 응답이므로
# 파싱 실패가 아니라 '오답(기권)' 으로 집계한다.
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
_ABSTENTION = {
    "final_answer": INSUFFICIENT,
    "partial_answer": INSUFFICIENT,
    "verdict": "FAIL",
    "critique": INSUFFICIENT,
    "sub_questions": [],
    "reasoning": INSUFFICIENT,
    "expressions": [],
}


def _is_abstention(text: str) -> bool:
    t = (text or "").strip().strip("`").strip().strip('"').strip("'").strip()
    if not t:
        return False
    return t.upper() == INSUFFICIENT or (len(t) <= 80 and INSUFFICIENT in t.upper())


def call_json(logical_key: str, prompt_name: str, slots: dict, node_name: str):
    """JSON 출력 노드 호출. 파싱 실패 시 동일 프롬프트로 json_parse_retries 회 재호출.

    모델이 'INSUFFICIENT_EVIDENCE' 로 기권하면 기권 dict 를 반환(오답 집계).
    반환: (parsed_dict, token_records). 최종 실패 시 JSONParseError.
    """
    system_part, inputs_template = _split_prompt(prompt_name)
    user_message = inputs_template.format(**slots)
    json_retries = int(_EXP["robustness"]["json_parse_retries"])

    records = []
    last_text = None
    for _ in range(json_retries + 1):
        res = llm_clients.call(logical_key, system_part, user_message, node_name)
        records.append(res.token_record())
        parsed = _extract_json(res.text)
        if parsed is not None:
            return parsed, records
        if _is_abstention(res.text):
            return dict(_ABSTENTION), records
        last_text = res.text
    raise JSONParseError(
        f"JSON 파싱 실패 (node={node_name}, model={logical_key}): {last_text!r:.200}")


def _logical(structure: str, role: str) -> str:
    return _MODELS["structures"][structure][role]


def _evidence_str(state) -> str:
    return tools.format_evidence(state.get("retrieved_evidence", []))


# ---------------------------------------------------------------------------
# 노드 빌더
# ---------------------------------------------------------------------------

def make_retrieve_node():
    """retrieve: query -> retrieved_evidence (top_k=50)."""
    def retrieve_node(state):
        t0 = time.perf_counter()
        chunks = tools.retrieve(state["query"], state["dataset"],
                                context_chunks=state.get("context_chunks"))
        return {
            "retrieved_evidence": chunks,
            "node_latencies": [{"node": "retrieve", "seconds": time.perf_counter() - t0}],
        }
    return retrieve_node


def make_generate_node(structure: str, role: str = "generate", node_name: str = "generate"):
    """generate (S0/S1/S2): query+evidence(+critique) -> final_answer/reasoning/expressions."""
    logical = _logical(structure, role)

    def generate_node(state):
        t0 = time.perf_counter()
        slots = {
            "query": state["query"],
            "evidence": _evidence_str(state),
            "critique": state.get("critique", "") or "",
        }
        parsed, records = call_json(logical, "generate.txt", slots, node_name)
        return {
            "final_answer": str(parsed.get("final_answer", "")),
            "reasoning_trace": str(parsed.get("reasoning", "")),
            "used_expression": list(parsed.get("expressions", []) or []),
            "token_usage": records,
            "node_latencies": [{"node": node_name, "seconds": time.perf_counter() - t0}],
        }
    return generate_node


def _reasoning_code_conflict(final_answer: str, expressions: list[str]):
    """보조 신뢰도 신호(논문 3.4.3): '추론 답(final_answer) vs 코드 실행 답(expressions)' 불일치.

    used_expression 의 각 산술식을 tools.safe_calc 로 재실행한 값들 중 **어느 하나라도**
    final_answer 와 채점기(metrics.is_correct, 동일 허용오차·퍼센트 정합)로 정합하면 일관(False).
    하나도 정합하지 않으면 모델이 제시한 산술이 자신의 답을 재현하지 못한 것이므로 충돌(True).

    반환: True(충돌) | False(일관) | None(미적용 — 비수치 답이거나 실행 가능한 식 없음).
    self-consistency 는 temperature=0 에서 불가하므로(논문 3.1) 이 결정적 보조신호를 쓴다.
    """
    fn = metrics.normalize_number(final_answer)
    if fn is None:
        return None  # 비수치 답(항목명/텍스트 등) → 보조신호 미적용, 자기보고 유지
    computed = []
    for e in (expressions or []):
        v, err = tools.safe_calc(e)
        if err is None and v is not None:
            computed.append(v)
    if not computed:
        return None  # 코드 실행 가능한 산술식 없음 → 미적용
    for cv in computed:
        if metrics.is_correct(cv, final_answer).correct:
            return False  # 산술이 답을 재현 → 일관
    return True  # 어떤 식도 답을 재현 못함 → 충돌


def make_generate_s5_node(structure: str, role: str = "generate", node_name: str = "generate"):
    """generate (S5, 실험3): generate_node 와 동일하되 generate_s5.txt 를 써서
    모델 자기보고 confidence(0~1)를 산출하고, 보조신호(추론답 vs 코드실행답 불일치)로
    보정한 '유효 confidence' 를 state['confidence'] 로 기록(gate 가 이를 임계값과 비교).

    논문 3.4.3 — 1차 신호=자기보고. 단 temperature=0 에서 모델이 과신해 1.0 을 남발하면
    캐스케이드가 퇴화하므로, 모델이 제시한 산술식이 자신의 답을 재현하지 못하는(결정적)
    충돌이 감지되면 유효 confidence 를 0.0 으로 강제 하향해 검증(verify)으로 에스컬레이트한다.
    일관(충돌 아님)·비수치 답이면 자기보고 값을 그대로 사용한다(보수적: 오탐 escalate 최소화).
    """
    logical = _logical(structure, role)

    def generate_s5_node(state):
        t0 = time.perf_counter()
        slots = {
            "query": state["query"],
            "evidence": _evidence_str(state),
            "critique": state.get("critique", "") or "",
        }
        parsed, records = call_json(logical, "generate_s5.txt", slots, node_name)
        final_answer = str(parsed.get("final_answer", ""))
        expressions = list(parsed.get("expressions", []) or [])
        try:
            self_conf = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            self_conf = 0.0
        self_conf = max(0.0, min(1.0, self_conf))  # 0~1 클램프

        conflict = _reasoning_code_conflict(final_answer, expressions)
        # 보조신호 보정: 결정적 충돌이면 0.0 으로 강제 하향(escalate 유도), 아니면 자기보고 유지.
        eff_conf = 0.0 if conflict is True else self_conf
        return {
            "final_answer": final_answer,
            "reasoning_trace": str(parsed.get("reasoning", "")),
            "used_expression": expressions,
            "confidence": eff_conf,
            "confidence_self_report": self_conf,   # 모델 원자기보고(분석/H3용)
            "code_conflict": bool(conflict) if conflict is not None else False,
            "token_usage": records,
            "node_latencies": [{"node": node_name, "seconds": time.perf_counter() - t0}],
        }
    return generate_s5_node


def make_gate_node(threshold: float):
    """gate (S5, 실험3): generate 자기보고 confidence 를 임계값과 비교해 분기 결정.
    confidence >= threshold → gate_decision='stop'(확신, END),
    아니면 → 'escalate'(불확실, verify 로). LLM 호출 없음(토큰 0). 지연만 기록한다.
    """
    def gate_node(state):
        t0 = time.perf_counter()
        conf = float(state.get("confidence", 0.0) or 0.0)
        decision = "stop" if conf >= threshold else "escalate"
        return {
            "gate_decision": decision,
            "node_latencies": [{"node": "gate", "seconds": time.perf_counter() - t0}],
        }
    return gate_node


def make_verify_node(structure: str):
    """verify (S2, o3): -> verdict/critique. FAIL 이면 critique 기록."""
    logical = _logical(structure, "verify")

    def verify_node(state):
        t0 = time.perf_counter()
        slots = {
            "query": state["query"],
            "evidence": _evidence_str(state),
            "candidate_answer": state.get("final_answer", ""),
            "candidate_reasoning": state.get("reasoning_trace", ""),
        }
        parsed, records = call_json(logical, "verify.txt", slots, "verify")
        verdict = str(parsed.get("verdict", "FAIL")).upper().strip()
        critique = str(parsed.get("critique", "") or "")
        update = {
            "verify_verdict": "PASS" if verdict == "PASS" else "FAIL",
            "token_usage": records,
            "node_latencies": [{"node": "verify", "seconds": time.perf_counter() - t0}],
        }
        if update["verify_verdict"] == "FAIL":
            update["critique"] = critique
        return update
    return verify_node


def make_prep_retry_node():
    """S2 재생성 진입 전 retry_count 증가(상한 통제)."""
    def prep_retry_node(state):
        return {"retry_count": int(state.get("retry_count", 0)) + 1}
    return prep_retry_node


def make_vote_gen_node(structure: str, role: str, node_name: str):
    """gen_* (S3): query+evidence -> candidate_answers 에 append (병렬)."""
    logical = _logical(structure, role)

    def vote_gen_node(state):
        t0 = time.perf_counter()
        slots = {"query": state["query"], "evidence": _evidence_str(state)}
        parsed, records = call_json(logical, "vote_generate.txt", slots, node_name)
        candidate = {
            "node": node_name,
            "logical_key": logical,
            "model": config.model_pricing_key(logical),
            "final_answer": str(parsed.get("final_answer", "")),
            "reasoning": str(parsed.get("reasoning", "")),
            "expressions": list(parsed.get("expressions", []) or []),
        }
        return {
            "candidate_answers": [candidate],
            "token_usage": records,
            "node_latencies": [{"node": node_name, "seconds": time.perf_counter() - t0}],
        }
    return vote_gen_node


def make_decompose_node(structure: str):
    """decompose (S4): query -> sub_questions (최대 5개)."""
    logical = _logical(structure, "decompose")
    max_sub = int(_EXP["structure_rules"]["s4_max_sub_questions"])

    def decompose_node(state):
        t0 = time.perf_counter()
        slots = {"query": state["query"]}
        parsed, records = call_json(logical, "decompose.txt", slots, "decompose")
        subs = [str(s) for s in (parsed.get("sub_questions") or []) if str(s).strip()]
        if not subs:
            subs = [state["query"]]
        subs = subs[:max_sub]  # 무한 분해 방지
        return {
            "sub_questions": subs,
            "token_usage": records,
            "node_latencies": [{"node": "decompose", "seconds": time.perf_counter() - t0}],
        }
    return decompose_node


def make_solve_node(structure: str):
    """solve (S4): sub_questions 순차 해결. 각 단계 retrieve + 이전 부분답 누적."""
    logical = _logical(structure, "solve")

    def solve_node(state):
        t0 = time.perf_counter()
        records = []
        sub_answers = []
        all_chunks = []
        seen_chunk = set()
        prior_lines = []
        for i, sub_q in enumerate(state.get("sub_questions", []), 1):
            chunks = tools.retrieve(sub_q, state["dataset"],
                                    context_chunks=state.get("context_chunks"))
            for c in chunks:
                cid = c.get("chunk_id") or c["text"][:50]
                if cid not in seen_chunk:
                    seen_chunk.add(cid)
                    all_chunks.append(c)
            prior = "\n".join(prior_lines) if prior_lines else "(none yet)"
            slots = {
                "sub_question": sub_q,
                "evidence": tools.format_evidence(chunks),
                "prior_partial_answers": prior,
            }
            parsed, recs = call_json(logical, "solve.txt", slots, "solve")
            records.extend(recs)
            pa = str(parsed.get("partial_answer", ""))
            entry = {
                "sub_question": sub_q,
                "partial_answer": pa,
                "reasoning": str(parsed.get("reasoning", "")),
                "expressions": list(parsed.get("expressions", []) or []),
            }
            sub_answers.append(entry)
            prior_lines.append(f"{i}. Q: {sub_q} -> A: {pa}")
        return {
            "sub_answers": sub_answers,
            "retrieved_evidence": all_chunks,
            "token_usage": records,
            "node_latencies": [{"node": "solve", "seconds": time.perf_counter() - t0}],
        }
    return solve_node


def make_aggregate_s4_node(structure: str):
    """aggregate (S4): sub_answers 종합 -> final_answer."""
    logical = _logical(structure, "aggregate")

    def aggregate_node(state):
        t0 = time.perf_counter()
        subs = state.get("sub_answers", [])
        sub_lines = [f"{i}. Q: {s['sub_question']} -> A: {s['partial_answer']}"
                     for i, s in enumerate(subs, 1)]
        slots = {
            "query": state["query"],
            "sub_answers": "\n".join(sub_lines) if sub_lines else "(none)",
            "evidence": _evidence_str(state),
        }
        parsed, records = call_json(logical, "aggregate_s4.txt", slots, "aggregate")
        # 누적 산술식: solve 단계 + aggregate
        exprs = []
        for s in subs:
            exprs.extend(s.get("expressions", []))
        exprs.extend(parsed.get("expressions", []) or [])
        return {
            "final_answer": str(parsed.get("final_answer", "")),
            "reasoning_trace": str(parsed.get("reasoning", "")),
            "used_expression": exprs,
            "token_usage": records,
            "node_latencies": [{"node": "aggregate", "seconds": time.perf_counter() - t0}],
        }
    return aggregate_node
