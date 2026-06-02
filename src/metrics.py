"""정답 판정(is_correct) + 정규화 규칙 + 비용/TSR/TPS 산출.

수치형: 단위 표기 차이 인정(억원/백만원/천원/원/%/%pt/배/개, 천단위 구분기호, 통화기호
정규화) 후 상대오차 < 1%(절대오차·정수·극소값 특례 포함, FinQA evaluation_schema 와 정합).
텍스트형(항목명/사업부문명): 공백·괄호주석·대소문자 정규화 후 일치/포함.

정규화 규칙은 함수로 분리하고 tests/test_metrics.py 로 단위 테스트한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config

TEXT_UNITS = {"항목명", "사업부문명"}

# 자릿수 단위(스케일 적용) — 긴 접미사 우선
_MAGNITUDE_UNITS = [
    ("조원", 1e12),
    ("억원", 1e8),
    ("백만원", 1e6),
    ("만원", 1e4),
    ("천원", 1e3),
    ("원", 1.0),
]
# 스케일 없이 제거하는 단위
_PLAIN_UNITS = ["%pt", "%p", "%", "배", "개", "명", "건", "회", "점", "pt", "x", "X"]


def normalize_number(s) -> float | None:
    """문자열을 단위 정규화하여 float 로 변환. 변환 불가 시 None.

    예: '44.9%'->44.9, '1,234백만원'->1.234e9, '4,273천원'->4273000, '-35.0'->-35.0
    """
    if s is None:
        return None
    t = str(s).strip()
    if t == "":
        return None
    t = t.replace(",", "").replace(" ", "")
    t = t.replace("₩", "").replace("$", "")
    t = re.sub(r"(?i)usd", "", t)

    scale = 1.0
    for unit, mult in _MAGNITUDE_UNITS:
        if t.endswith(unit):
            t = t[: -len(unit)]
            scale = mult
            break
    else:
        for unit in _PLAIN_UNITS:
            if t.endswith(unit):
                t = t[: -len(unit)]
                break

    m = re.search(r"[-+]?\d*\.?\d+", t)
    if not m:
        return None
    try:
        return float(m.group()) * scale
    except ValueError:
        return None


def normalize_text(s) -> str:
    """공백·괄호주석·대소문자·구분점 정규화."""
    t = str(s or "").strip().lower()
    t = re.sub(r"[\(\（][^\)\）]*[\)\）]", "", t)   # 괄호 주석 제거
    t = re.sub(r"\s+", "", t)
    for ch in ["·", ",", ".", "'", '"', "’", "“", "”"]:
        t = t.replace(ch, "")
    return t


def _jcfg() -> dict:
    return config.experiment_cfg()["judging"]


def numeric_close(gold: float, pred: float) -> bool:
    j = _jcfg()
    rel_tol = float(j["relative_tolerance"])
    abs_tol = float(j["absolute_tolerance"])
    int_abs_tol = float(j["integer_absolute_tolerance"])
    small_thr = float(j["small_value_threshold"])
    small_abs = float(j["small_value_absolute_tolerance"])

    diff = abs(gold - pred)
    if abs(gold) < small_thr:
        return diff <= small_abs
    if float(gold).is_integer() and diff <= int_abs_tol:
        return True
    if diff <= abs_tol:
        return True
    if abs(gold) > 0 and diff / abs(gold) <= rel_tol:
        return True
    return False


@dataclass
class Judgement:
    correct: bool
    judge_flag: bool   # 경계 케이스 — 저자 검토용
    mode: str          # 'numeric' | 'text'


def is_correct(gold, pred, unit: str | None = None) -> Judgement:
    """gold 와 pred 의 정답 일치 판정. unit 으로 텍스트/수치 분기."""
    # 단위 메타로 텍스트형 강제
    force_text = unit in TEXT_UNITS if unit else False

    if not force_text:
        gn = normalize_number(gold)
        pn = normalize_number(pred)
        if gn is not None and pn is not None:
            ok = numeric_close(gn, pn)
            # 경계 플래그: 정답은 아니지만 5배 허용오차 내(근접)면 검토 대상
            flag = False
            if not ok and abs(gn) > 0:
                if abs(gn - pn) / abs(gn) <= 5 * float(_jcfg()["relative_tolerance"]):
                    flag = True
            return Judgement(ok, flag, "numeric")
        if gn is not None and pn is None:
            # gold 는 수치인데 pred 파싱 불가 → 검토 대상
            # 텍스트 경로로 한 번 더 시도 후 flag
            tj = _text_judge(gold, pred)
            return Judgement(tj.correct, True, "numeric")

    return _text_judge(gold, pred)


def _text_judge(gold, pred) -> Judgement:
    g = normalize_text(gold)
    p = normalize_text(pred)
    if not g:
        return Judgement(False, True, "text")
    if g == p:
        return Judgement(True, False, "text")
    if g in p:
        # 포함은 정답으로 인정하되 부분 일치는 검토 플래그
        return Judgement(True, True, "text")
    return Judgement(False, False, "text")


# ---------------------------------------------------------------------------
# 비용 / 토큰 / 지표
# ---------------------------------------------------------------------------

def record_cost(record: dict) -> float:
    """단일 토큰 레코드의 USD 비용."""
    pricing = config.pricing()[record["model"]]
    return (
        record["input_tokens"] / 1_000_000.0 * pricing["input"]
        + record["output_tokens"] / 1_000_000.0 * pricing["output"]
    )


def run_cost(token_usage: list[dict]) -> float:
    return sum(record_cost(r) for r in token_usage)


def run_tokens(token_usage: list[dict]) -> tuple[int, int]:
    it = sum(r["input_tokens"] for r in token_usage)
    ot = sum(r["output_tokens"] for r in token_usage)
    return it, ot


def aggregate_metrics(runs: list[dict]) -> dict:
    """raw run 레코드 리스트 -> 구조 집계 지표.

    각 run 레코드 필수 키: success(0/1), input_tokens, output_tokens, cost, latency,
    dataset, task_type.
    """
    n = len(runs)
    if n == 0:
        return {"n_runs": 0}
    succ = sum(r["success"] for r in runs)
    tot_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in runs)
    tot_cost = sum(r["cost"] for r in runs)

    def grp(key):
        out = {}
        keys = sorted({r[key] for r in runs})
        for k in keys:
            sub = [r for r in runs if r[key] == k]
            s = sum(x["success"] for x in sub)
            out[k] = {
                "n_runs": len(sub),
                "n_success": s,
                "TSR": s / len(sub),
                "avg_latency": sum(x["latency"] for x in sub) / len(sub),
                "avg_tokens": sum(x["input_tokens"] + x["output_tokens"] for x in sub) / len(sub),
                "avg_cost": sum(x["cost"] for x in sub) / len(sub),
            }
        return out

    return {
        "n_runs": n,
        "n_success": succ,
        "TSR": succ / n,
        "avg_input_tokens": sum(r["input_tokens"] for r in runs) / n,
        "avg_output_tokens": sum(r["output_tokens"] for r in runs) / n,
        "avg_total_tokens": tot_tokens / n,
        "avg_latency": sum(r["latency"] for r in runs) / n,
        "total_cost_usd": tot_cost,
        "avg_cost_usd": tot_cost / n,
        # TPS (Tokens per Success): 성공 1건당 평균 토큰
        "TPS": (tot_tokens / succ) if succ > 0 else None,
        "cost_per_success_usd": (tot_cost / succ) if succ > 0 else None,
        "by_dataset": grp("dataset"),
        "by_task_type": grp("task_type"),
    }
