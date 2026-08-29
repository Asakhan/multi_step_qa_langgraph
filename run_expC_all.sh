#!/usr/bin/env bash
# 실험 C 전체 재현 — 데이터 준비 → 실행 → 분석 → 표 생성.
#
# 재현성(실험계획서 §3-E): 시드 고정(20260821), 모델 문자열은 .env, 단가는 config/pricing.yaml.
# 모든 단계가 idempotent 하므로 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 진행한다.
#
#   ./run_expC_all.sh              # 전체
#   ./run_expC_all.sh analyze      # 분석·표만 (LLM 호출 없음)
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

MODE="${1:-all}"

if [ "$MODE" = "all" ]; then
  echo "### 0) 확대 표본 준비 (FinQA 난이도 층화 80문항, 시드 20260821)"
  "$PY" -m src.prepare_finqa_ext

  echo "### 1) 기존 60문항 표본에서 S7 실행 (180 run)"
  RESULTS_DIR=results_expC "$PY" -m src.runner --structure s7 --runs 3 \
      --datasets finqa kdart --no-email

  echo "### 2) 확대 80문항 표본에서 S7 실행 (240 run)"
  RESULTS_DIR=results_expC_ext "$PY" -m src.runner --structure s7 --runs 3 \
      --datasets finqa_ext --no-email
fi

mkdir -p results_expC_pooled
echo "### 3) 반사실 게이트 재현 분석 — 기존 60문항"
RESULTS_DIR=results_expC "$PY" -m src.replay_gates \
    --out results_expC/analysis_expC --label "기존 60문항(FinQA 20 + K-DART 40)" \
    > results_expC/analysis_expC_console.log 2>&1

echo "### 4) 반사실 게이트 재현 분석 — 확대 80문항(독립 표본)"
RESULTS_DIR=results_expC_ext "$PY" -m src.replay_gates \
    --out results_expC_ext/analysis_expC --label "확대 80문항(FinQA 층화 독립표본)" \
    > results_expC_ext/analysis_expC_console.log 2>&1

echo "### 5) 반사실 게이트 재현 분석 — 풀링 140문항 (주분석)"
RESULTS_DIR=results_expC "$PY" -m src.replay_gates \
    --dirs results_expC results_expC_ext \
    --out results_expC_pooled/analysis_expC --label "풀링 140문항" \
    > results_expC_pooled/analysis_expC_console.log 2>&1

echo "### 6) 마크다운 결과표 생성"
for d in results_expC/analysis_expC results_expC_ext/analysis_expC results_expC_pooled/analysis_expC; do
  "$PY" -m src.report_expC "$d/replay_gates.json" --out "$d/결과표.md"
done

echo "### 7) 실험 B 재실행 (게이트 신호 오프라인 평가) — 참조"
RESULTS_DIR=results_main "$PY" -m src.eval_gate_signals s0 s5 \
    > results_expC_pooled/expB_재실행.log 2>&1 || true

echo "[expC-all] 완료 $(date +%H:%M:%S)"
