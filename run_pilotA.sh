#!/usr/bin/env bash
# 실험 A 파일럿 — 검증기 엄격도 3종의 FAIL 발동률·정확도 비교.
#
# 설계: 구조는 s2(항상 검증)로 고정하여 게이트 교란을 제거하고, 검증 프롬프트만 교체한다.
#       60문항 × 1반복 = 레벨당 60 run (temperature=0 이라 파일럿에는 1반복으로 충분).
#       레벨별로 RESULTS_DIR 을 분리하므로 중간에 끊겨도 완료분은 보존되며,
#       같은 명령을 다시 실행하면 완료분을 건너뛰고 이어서 진행한다(idempotent resume).
#
# 사용법:  ./run_pilotA.sh            # L1(기본)→L2→L3 순차
#          ./run_pilotA.sh L3         # 특정 레벨만
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

run_level () {
  local level="$1" prompt="$2"
  local outdir="results_pilotA_${level}"
  echo ""
  echo "=========================================================="
  echo "[pilotA] ${level} 시작  (프롬프트: ${prompt})  $(date +%H:%M:%S)"
  echo "=========================================================="
  VERIFY_PROMPT="$prompt" RESULTS_DIR="$outdir" \
    "$PY" -m src.orchestrator --structures s2 --runs 1 \
      --datasets finqa kdart --no-final-analysis
  echo "[pilotA] ${level} 종료코드=$? $(date +%H:%M:%S)"
}

LEVELS="${*:-L1 L2 L3}"
for L in $LEVELS; do
  case "$L" in
    L1) run_level L1 "verify.txt" ;;       # 기본(저자 확정본) — 대조군
    L2) run_level L2 "verify_L2.txt" ;;    # 항목별 판정 + OR 규칙
    L3) run_level L3 "verify_L3.txt" ;;    # 적대적 검증 + 입증책임 전환
    *)  echo "[pilotA] 알 수 없는 레벨: $L" ;;
  esac
done

echo ""
echo "[pilotA] 전체 완료 $(date +%H:%M:%S)"
