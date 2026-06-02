#!/usr/bin/env bash
# S1(gemini-3.1-pro) 재개 — 일일 quota 리셋 후 1회 실행 예약용(at).
# 멱등: 이미 성공한 런은 건너뛰고 오류/미실행분만 채운다. 결과는 results_finqa_format/s1.
set -uo pipefail
cd "$(dirname "$0")"
export RESULTS_DIR="/root/multi_step_qa_langgraph/results_finqa_format"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
mkdir -p logs "$RESULTS_DIR"
LOG="logs/s1_resume.log"

echo "[s1_resume] $(date -u +%FT%TZ) S1 재개 시작 (RESULTS_DIR=$RESULTS_DIR)" >> "$LOG"
"$PY" -m src.runner --structure s1 --datasets finqa kdart >> "$LOG" 2>&1
rc=$?
echo "[s1_resume] $(date -u +%FT%TZ) S1 재개 종료 (exit=$rc)" >> "$LOG"

# S1까지 끝났으면 최종 통계분석/리포트 생성(파레토). 실패해도 무시.
if [ "$rc" -eq 0 ]; then
  echo "[s1_resume] $(date -u +%FT%TZ) 최종 분석 실행" >> "$LOG"
  "$PY" -m src.analysis >> "$LOG" 2>&1 || echo "[s1_resume] 분석 실패(무시)" >> "$LOG"
fi
