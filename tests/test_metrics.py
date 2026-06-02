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
    assert not is_correct("0.14464", "14.464").correct


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
