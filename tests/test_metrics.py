"""metrics 정규화·판정 단위 테스트 + calc 안전성 + S3 집계."""
from __future__ import annotations

import pytest

from src.metrics import normalize_number, normalize_text, is_correct, numeric_close
from src.tools import calc, CalcError
from src.structures.aggregation import aggregate_votes


# ---- 수치 정규화 ----
@pytest.mark.parametrize("s,expected", [
    ("44.9", 44.9),
    ("44.9%", 44.9),
    ("4,273", 4273.0),
    ("4,273천원", 4273000.0),
    ("1,234백만원", 1_234_000_000.0),
    ("1억원", 100_000_000.0),
    ("-35.0", -35.0),
    ("3배", 3.0),
    ("5%pt", 5.0),
    ("$1,000", 1000.0),
])
def test_normalize_number(s, expected):
    assert normalize_number(s) == pytest.approx(expected)


def test_normalize_number_text():
    assert normalize_number("철강 부문") is None
    assert normalize_number("") is None


# ---- 단위 표기 차이 인정 ----
def test_unit_equivalence():
    assert is_correct("44.9", "44.9%").correct
    assert is_correct("1234000000", "1,234백만원").correct
    assert is_correct("94.0", "94").correct


def test_numeric_tolerance():
    assert numeric_close(0.14464, 0.1446)
    assert not numeric_close(0.14464, 14.464)
    assert numeric_close(-35.0, -35.001)
    assert numeric_close(705.25, 705.3)


def test_numeric_wrong():
    # % 표기가 전혀 없으면 100배 차이는 스케일 보정하지 않고 오답 유지(무분별 보정 방지)
    assert not is_correct("0.14464", "14.464").correct


# ---- (a) 퍼센트 ↔ 비율 소수 정합 (실패사례 기반 회귀) ----
def test_percent_ratio_match():
    # finqa: gold 비율소수 vs pred 퍼센트 → ×100 정합 → 정답
    assert is_correct("0.15653", "15.65%").correct
    assert is_correct("0.0585", "5.85%").correct
    assert is_correct("0.2895", "28.95%").correct
    assert is_correct("0.19859", "19.86%").correct


def test_percent_ratio_genuine_error_not_passed():
    # 사용자 예시: 0.1954(=19.54%) vs 16.35% → 스케일 후에도 불일치 → 진짜 오답(자동보정 금지)
    j = is_correct("0.1954", "16.35%")
    assert not j.correct
    # 19.54 vs 16.35 는 5배 허용오차(5%) 밖 → flag 도 아님
    assert not j.judge_flag


def test_percent_guard_requires_explicit_sign():
    # 한쪽에 % 표기가 없으면(둘 다 bare) 100배 정합이어도 보정하지 않음
    assert not is_correct("0.0585", "5.85").correct      # pred 에 % 없음
    # 양쪽 다 퍼센트인데 100배 차이 → 오답 유지(같은 단위)
    assert not is_correct("50%", "5000%").correct
    # 양쪽 다 절대수치인데 100배 차이 → 오답 유지
    assert not is_correct("747", "74700").correct


def test_percent_unit_blocks_false_scale():
    # gold_answer_unit='%' 이면 gold("0.5")는 0.5% 의미(비율 아님) → 50% 와 스케일 정합 금지
    assert not is_correct("0.5", "50%", unit="%").correct
    # 반면 unit 미지정(finqa)인 0.5 는 비율(=50%)로 보아 50% 와 정합
    assert is_correct("0.5", "50%").correct
    # kdart % gold 는 동일 크기 pred 와 정상 정답
    assert is_correct("44.9", "44.9%", unit="%").correct


# ---- (a-2) 선언 자릿수 단위 정합 (백만원 이중 적용 회귀) ----
def test_declared_magnitude_unit_reconcile():
    """gold_answer_unit 이 자릿수 단위면 접미사 없는 쪽을 그 단위로 해석한다.

    회귀 근거: K-DART gold 는 공시 원표의 단위로 표기된 맨숫자다. 이 처리가 없으면
    모델이 같은 값을 단위까지 붙여 답할 때 배수가 두 번 적용되어, 숫자를 맞히고도
    오답이 된다(DART_T1_002·DART_T1_003 — 네 구조 전부가 이 이유로 오답 처리되었다).
    """
    # 같은 값을 단위까지 붙여 답한 경우 → 정답 (수정 전에는 10^6 배 어긋나 오답이었다)
    assert is_correct("739,764,256", "739,764,256 백만원", unit="백만원").correct
    assert is_correct("2,326,663", "2,326,663 백만원", unit="백만원").correct
    # 단위를 붙이지 않은 답의 기존 판정은 바뀌지 않는다
    assert is_correct("739,764,256", "739764256", unit="백만원").correct
    # 원 단위로 환산해 답해도 절대 크기가 같으면 정답
    assert is_correct("739,764,256", "739,764,256,000,000원", unit="백만원").correct
    # 억원도 동일하게 동작
    assert is_correct("1,234", "1,234 억원", unit="억원").correct
    # 선언 단위와 다른 자릿수 단위로 **올바르게 환산해** 답한 경우도 정답
    # (DART_T3_006 회귀: gold 283,180.52억원 = pred 28,318,052백만원 = 28.318조원)
    assert is_correct("283,180.52", "28,318,052 백만원", unit="억원").correct


def test_declared_magnitude_unit_does_not_loosen_scoring():
    """선언 단위 정합은 진짜 단위·배율 오류를 정답으로 만들지 않는다."""
    # 환산 배수가 어긋나면(100배 초과) 오답 유지
    assert not is_correct("283,180.52", "28,318,052 억원", unit="억원").correct
    assert not is_correct("2,326,663", "23,266,630 백만원", unit="백만원").correct
    # 선언 단위와 다른 자릿수 단위를 붙이면 배수가 어긋나 오답
    assert not is_correct("1,234", "1,234 백만원", unit="억원").correct
    # 자릿수 단위가 아닌 선언 단위(개·원)는 기존 동작 그대로
    assert is_correct("100", "100개", unit="개").correct
    assert not is_correct("(801)", "6,690원", unit="원").correct
    # unit 미지정이면 종전과 동일 (gold 가 절대 크기인 기존 케이스 보존)
    assert is_correct("1234000000", "1,234백만원").correct


# ---- (b) 부호 불일치: 자동 정답 금지, judge_flag 로만 표시 ----
def test_sign_mismatch_flags_not_correct():
    # finqa: gold 음수(감소) vs pred 절댓값 퍼센트 → 오답이되 검토 플래그
    j = is_correct("-0.12381", "12.38%")
    assert not j.correct and j.judge_flag
    # kdart: gold 음수 % vs pred 양수 % (크기 동일) → 오답 + 플래그
    j2 = is_correct("-91.05", "91.05%", unit="%")
    assert not j2.correct and j2.judge_flag
    # 부호 반대 방향(gold 양수, pred 음수)도 동일
    j3 = is_correct("1.16", "-1.16%", unit="%")
    assert not j3.correct and j3.judge_flag


# ---- 텍스트 판정 ----
def test_text_match():
    assert is_correct("서울대공원", "서울대공원", unit="항목명").correct
    assert is_correct("철강 부문", "철강부문", unit="사업부문명").correct
    assert not is_correct("홍보전산과", "홍보팀", unit="항목명").correct


def test_text_unit_forces_text_even_with_digit():
    # 단위가 항목명이면 숫자 포함 텍스트도 텍스트로 비교
    j = is_correct("제1사업부문", "제1사업부문", unit="사업부문명")
    assert j.correct and j.mode == "text"


def test_normalize_text_paren():
    assert normalize_text("연결 (당기)") == normalize_text("연결")


# ---- calc 안전성 ----
def test_calc_basic():
    assert calc("(123-100)/100*100") == pytest.approx(23.0)
    assert calc("153.7 - 139.9") == pytest.approx(13.8)
    assert calc("13.8 / 139.9 = 0.09864") == pytest.approx(13.8 / 139.9)


def test_calc_rejects_code():
    with pytest.raises(CalcError):
        calc("__import__('os').system('ls')")
    with pytest.raises(CalcError):
        calc("a + b")


# ---- S3 집계 ----
def _cand(model, logical, ans):
    return {"node": model, "logical_key": logical, "model": model, "final_answer": ans}


def test_vote_majority_numeric():
    cands = [_cand("gemini-2.5-flash", "gemini_flash", "94.0"),
             _cand("gpt-4o-mini", "gpt4o_mini", "94"),
             _cand("claude-haiku-4.5", "claude_haiku", "100")]
    chosen, info = aggregate_votes(cands, "gemini_flash")
    assert info["method"] == "majority"
    assert normalize_number(chosen["final_answer"]) == pytest.approx(94.0)


def test_vote_three_way_tie_uses_priority():
    cands = [_cand("gemini-2.5-flash", "gemini_flash", "1"),
             _cand("gpt-4o-mini", "gpt4o_mini", "2"),
             _cand("claude-haiku-4.5", "claude_haiku", "3")]
    chosen, info = aggregate_votes(cands, "gemini_flash")
    assert info["method"] == "tiebreak"
    assert chosen["logical_key"] == "gemini_flash"


def test_vote_text_majority():
    cands = [_cand("g", "gemini_flash", "철강 부문"),
             _cand("p", "gpt4o_mini", "철강부문"),
             _cand("c", "claude_haiku", "화학 부문")]
    chosen, info = aggregate_votes(cands, "gemini_flash")
    assert info["method"] == "majority"
    assert "철강" in chosen["final_answer"]
