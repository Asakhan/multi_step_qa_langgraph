#!/usr/bin/env bash
# 실험 C — 반사실 게이트 재현(counterfactual gate replay) 본 실행.
#
# 설계: 구조 S7(게이트 없음 + verify 전수 + 1차 생성 스냅샷)로 60문항 × 3반복 = 180 run.
#       이 로그 하나에서 임의의 게이트 정책(무검증/항상검증/자기보고/학습라우터/무작위/oracle)의
#       정확도·비용을 추가 LLM 호출 없이 재구성한다(src/replay_gates.py).
#
# 반복 3회는 기존 실험 0·2·3·4(각 180 run)와 동일 규모로 맞춘 것 — 문항 단위 McNemar 비교가
# 같은 분석 단위에서 성립해야 하기 때문이다.
#
# 사용법:  ./run_expC.sh          # 중단 시 같은 명령 재실행 → 완료분 자동 건너뜀(idempotent resume)
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export RESULTS_DIR="results_expC"

echo "=========================================================="
echo "[expC] S7 실행 시작  $(date +%H:%M:%S)   (결과: $RESULTS_DIR/s7)"
echo "=========================================================="
"$PY" -m src.runner --structure s7 --runs 3 --datasets finqa kdart --no-email
echo "[expC] 종료코드=$? $(date +%H:%M:%S)"
