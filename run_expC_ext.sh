#!/usr/bin/env bash
# 실험 C 확대 표본 — FinQA 난이도 층화 확대분(finqa_ext 80문항) × 3반복 = 240 run.
#
# 목적(실험계획서 §1.3·§3-C): 60문항에서는 McNemar 불일치쌍이 4개뿐이어서 어떤 효과도
# 검출할 수 없다. 난이도를 균형 배치한 독립 표본 80문항을 더해 검정력을 확보하고,
# 동시에 실험 C(반사실 게이트 재현) 결과가 표본을 바꿔도 유지되는지 확인한다.
#
# 결과 디렉토리를 분리하므로 기존 60문항 결과(results_expC)와 섞이지 않는다.
# 중단 시 같은 명령 재실행 → 완료분 자동 건너뜀(idempotent resume).
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export RESULTS_DIR="results_expC_ext"

echo "=========================================================="
echo "[expC-ext] S7 확대표본 실행 시작  $(date +%H:%M:%S)  (결과: $RESULTS_DIR/s7)"
echo "=========================================================="
"$PY" -m src.runner --structure s7 --runs 3 --datasets finqa_ext --no-email
echo "[expC-ext] 종료코드=$? $(date +%H:%M:%S)"
