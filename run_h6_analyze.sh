#!/usr/bin/env bash
# H6(기반 모델 간 일반화) · 실험 D(상위 모델 검증기) 분석. LLM 호출 없음.
#
# 각 구조의 3개 데이터셋 샤드를 풀링(140문항)하여 반사실 게이트 재현을 돌린다.
# 기준선인 gemini-2.5-flash(s7)는 results_expC / results_expC_ext 에 이미 있으므로 함께 낸다.
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

mkdir -p results_h6/analysis

run_one () {  # $1=structure  $2=label  $3... = dirs
  local s="$1"; shift
  local label="$1"; shift
  local out="results_h6/analysis/${s}"
  mkdir -p "$out"
  echo "### ${s} — ${label}"
  RESULTS_DIR=results_h6 "$PY" -m src.replay_gates \
      --structure "$s" --dirs "$@" --out "$out" --label "$label" \
      > "$out/console.log" 2>&1
  "$PY" -m src.report_expC "$out/replay_gates.json" --out "$out/결과표.md" \
      > /dev/null 2>&1 || echo "  (report_expC 실패 — console.log 확인)"
  echo "  -> $out/결과표.md"
}

# 기준선 (기실행분 재사용)
run_one s7 "gemini-2.5-flash · 140문항" results_expC results_expC_ext

# 신규
run_one s7_gpt    "gpt-4o-mini · 140문항" \
    results_h6/s7_gpt__finqa results_h6/s7_gpt__finqa_ext results_h6/s7_gpt__kdart
run_one s7_claude "claude-haiku-4.5 · 140문항" \
    results_h6/s7_claude__finqa results_h6/s7_claude__finqa_ext results_h6/s7_claude__kdart
run_one s7_vpro   "실험 D 상위 검증기(claude-sonnet-4-5) · 140문항" \
    results_h6/s7_vpro__finqa results_h6/s7_vpro__finqa_ext results_h6/s7_vpro__kdart

echo "[h6-analyze] 완료 $(date +%H:%M:%S)"
