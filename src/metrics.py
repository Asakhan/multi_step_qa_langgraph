"""정답 판정(is_correct) + 정규화 규칙 + 비용/TSR/TPS 산출.

수치형: 단위 표기 차이 인정(억원/백만원/천원/원/%/%pt/배/개, 천단위 구분기호, 통화기호
정규화) 후 상대오차 < 1%(절대오차·정수·극소값 특례 포함, FinQA evaluation_schema 와 정합).
텍스트형(항목명/사업부문명): 공백·괄호주석·대소문자 정규화 후 일치/포함.

정규화 규칙은 함수로 분리하고 tests/test_metrics.py 로 단위 테스트한다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from . import config

TEXT_UNITS = {"항목명", "사업부문명"}

# 유니코드 기호 정규화(U+2212 등) — 표기 차이로 인한 오판정 방지.
# NFKC 가 U+2212(−)/대시류를 ASCII '-' 로 바꾸지 않으므로 명시 치환을 NFKC 앞에 둔다.
_UNICODE_SYMBOL_MAP = {
    "−": "-", "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "％": "%", "，": ",", " ": " ",
}


def _ascii_symbols(t: str) -> str:
    """숫자 파싱 전 유니코드 기호를 ASCII 로 정규화(값·부호 불변, 과교정 아님)."""
    for k, v in _UNICODE_SYMBOL_MAP.items():
        if k in t:
            t = t.replace(k, v)
    return unicodedata.normalize("NFKC", t)

# gold_answer_unit 이 이 집합이면 gold 값이 이미 "퍼센트 크기"(비율 소수가 아님)임을 뜻한다.
# (예: kdart gold "44.9"(unit=%)는 44.9% 를 의미하므로 비율 0.449 로 오인해 ×100 하면 안 됨)
PERCENT_UNITS = {"%", "%p", "%pt"}

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


def _normalize_number_ex(s) -> tuple[float | None, bool]:
    """문자열을 단위 정규화하여 (float|None, is_percent) 로 반환.

    is_percent 는 값이 **명시적 퍼센트 표기**('...%')였는지 여부. 단 '%pt'/'%p'(퍼센트
    포인트)는 차이값이라 비율 환산 대상이 아니므로 is_percent=False 로 둔다. 이 플래그로
    비교 단계에서 비율 소수↔퍼센트 100배 스케일 정합을 안전하게 판단한다.

    예: '44.9%'->(44.9,True), '15.65%'->(15.65,True), '0.15653'->(0.15653,False),
        '5%pt'->(5.0,False), '1,234백만원'->(1.234e9,False)
    """
    if s is None:
        return None, False
    t = str(s).strip()
    if t == "":
        return None, False
    t = _ascii_symbols(t)   # 유니코드 기호 정규화(U+2212 등)
    t = t.replace(",", "").replace(" ", "")
    t = t.replace("₩", "").replace("$", "")
    t = re.sub(r"(?i)usd", "", t)

    scale = 1.0
    is_percent = False
    for unit, mult in _MAGNITUDE_UNITS:
        if t.endswith(unit):
            t = t[: -len(unit)]
            scale = mult
            break
    else:
        for unit in _PLAIN_UNITS:
            if t.endswith(unit):
                t = t[: -len(unit)]
                if unit == "%":           # 순수 퍼센트만(%pt/%p 제외)
                    is_percent = True
                break

    m = re.search(r"[-+]?\d*\.?\d+", t)
    if not m:
        return None, False
    try:
        return float(m.group()) * scale, is_percent
    except ValueError:
        return None, False


def normalize_number(s) -> float | None:
    """문자열을 단위 정규화하여 float 로 변환. 변환 불가 시 None.

    예: '44.9%'->44.9, '1,234백만원'->1.234e9, '4,273천원'->4273000, '-35.0'->-35.0
    """
    return _normalize_number_ex(s)[0]


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


def _percent_reconcile(gn: float, g_pct: bool, pn: float, p_pct: bool) -> bool:
    """비율 소수 ↔ 퍼센트 표기의 100배 스케일 정합 여부(같은 부호 한정).

    **정확히 한쪽만** 퍼센트('%')이고 다른 쪽이 비율 소수(|x|<=1)일 때에만 ×100 하여
    비교한다. 양쪽 다 퍼센트이거나 양쪽 다 비표기면(둘 다 절대수치/비율) 스케일하지 않아
    무분별 보정을 막는다. 부호가 다르면 numeric_close 가 실패하므로 여기서 정답 처리되지
    않는다(부호 불일치는 _sign_only_mismatch 가 flag 로만 표시).
    """
    if g_pct == p_pct:
        return False
    if p_pct and not g_pct and abs(gn) <= 1.0 + 1e-9:
        return numeric_close(gn * 100.0, pn)
    if g_pct and not p_pct and abs(pn) <= 1.0 + 1e-9:
        return numeric_close(gn, pn * 100.0)
    return False


def _sign_only_mismatch(gn: float, g_pct: bool, pn: float, p_pct: bool) -> bool:
    """부호만 다르고 크기는 정합(필요시 퍼센트 100배 보정 후)하는가.

    True 면 '감소/증가' 부호 누락 의심 → 자동 정답 처리하지 않고 judge_flag 로만 남긴다.
    """
    if (gn < 0) == (pn < 0):
        return False
    ag, ap = abs(gn), abs(pn)
    if numeric_close(ag, ap):
        return True
    if g_pct != p_pct:
        if p_pct and not g_pct and ag <= 1.0 + 1e-9 and numeric_close(ag * 100.0, ap):
            return True
        if g_pct and not p_pct and ap <= 1.0 + 1e-9 and numeric_close(ag, ap * 100.0):
            return True
    return False


def is_correct(gold, pred, unit: str | None = None) -> Judgement:
    """gold 와 pred 의 정답 일치 판정. unit 으로 텍스트/수치 분기."""
    # 단위 메타로 텍스트형 강제
    force_text = unit in TEXT_UNITS if unit else False

    if not force_text:
        gn, g_pct = _normalize_number_ex(gold)
        pn, p_pct = _normalize_number_ex(pred)
        # gold 단위가 퍼센트면 gold 는 이미 퍼센트 크기(비율 소수 아님) → ×100 오인 방지.
        if unit in PERCENT_UNITS:
            g_pct = True
        if gn is not None and pn is not None:
            ok = numeric_close(gn, pn)
            # 비율↔퍼센트 100배 스케일 정합(같은 부호, 명시적 % 한쪽 한정)
            if not ok and _percent_reconcile(gn, g_pct, pn, p_pct):
                ok = True
            flag = False
            if not ok:
                # (b) 부호만 다르고 크기 정합 → 자동정답 금지, 검토 플래그만.
                if _sign_only_mismatch(gn, g_pct, pn, p_pct):
                    flag = True
                # 경계 플래그: 정답은 아니지만 5배 허용오차 내(근접)면 검토 대상
                elif abs(gn) > 0 and abs(gn - pn) / abs(gn) <= 5 * float(_jcfg()["relative_tolerance"]):
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
