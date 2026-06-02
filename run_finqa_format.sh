#!/usr/bin/env bash
# FinQA-포맷 K-DART 실험 전용 런처. 결과를 results_finqa_format/ 에 저장(기존 results/ 와 분리).
# kdart 는 self-contained 컨텍스트(임베딩·검색 없음), FinQA 와 동일 파이프라인.
#
# 사용법:
#   ./run_finqa_format.sh prepare              # kdart_qa_finqa_format.json -> kdart_finqa.jsonl 재생성
#   ./run_finqa_format.sh smoke                # 표본 회귀검증 (데이터셋당 1문항, S1~S4, 전경, 메일X)
#   ./run_finqa_format.sh all --run-s0         # S0~S4 전체 백그라운드 실행
#   ./run_finqa_format.sh status | stop        # 진행 확인 / 중단
set -euo pipefail
cd "$(dirname "$0")"

export RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results_finqa_format}"
PY="${PYTHON:-.venv/bin/python}"; [ -x "$PY" ] || PY="python3"
PID_FILE="logs/run_finqa_format.pid"
mkdir -p logs "$RESULTS_DIR"

cmd="${1:-}"; shift || true

case "$cmd" in
  prepare)
    exec "$PY" -m src.prepare_kdart "$@"
    ;;
  smoke)
    echo "[run_ff] 표본 회귀검증: S1~S4 × 데이터셋당 1문항 × 3반복 (RESULTS_DIR=$RESULTS_DIR)"
    exec "$PY" -m src.orchestrator --structures s1 s2 s3 s4 \
         --limit-per-dataset 1 --no-final-analysis "$@"
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "[run_ff] 실행 중 (PID $(cat "$PID_FILE"))"
    else
      echo "[run_ff] 실행 중인 프로세스 없음"
    fi
    latest="$(ls -t logs/run_ff_*.log 2>/dev/null | head -1 || true)"
    [[ -n "$latest" ]] && { echo "--- ${latest} (tail) ---"; tail -n 30 "$latest"; }
    echo "--- 구조별 완료 run 수 ($RESULTS_DIR) ---"
    for s in s0 s1 s2 s3 s4; do
      f="$RESULTS_DIR/$s/raw_runs.jsonl"
      [[ -f "$f" ]] && echo "  $s: $(wc -l < "$f") runs"
    done
    ;;
  stop)
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      echo "[run_ff] 중단 신호 전송 (PID ${pid})"; rm -f "$PID_FILE"
    fi
    pkill -TERM -f "src.orchestrator" 2>/dev/null || true
    pkill -TERM -f "src.runner" 2>/dev/null || true
    echo "[run_ff] 종료 신호 전송 완료"
    ;;
  s0|s1|s2|s3|s4|all)
    structures=("$cmd")
    while [[ $# -gt 0 && "$1" != --* ]]; do structures+=("$1"); shift; done
    ts="$(date +%Y%m%d_%H%M%S)"; log="logs/run_ff_${ts}.log"
    args="--structures ${structures[*]} $*"
    echo "[run_ff] 백그라운드 실행: ${args}"
    echo "[run_ff] RESULTS_DIR=$RESULTS_DIR  로그: ${log}"
    setsid bash -c "echo \$\$ > '$PID_FILE'; export RESULTS_DIR='$RESULTS_DIR'; \
      exec '$PY' -m src.orchestrator $args > '$log' 2>&1" </dev/null &
    sleep 0.5
    echo "[run_ff] PID $(cat "$PID_FILE" 2>/dev/null). './run_finqa_format.sh status|stop' 로 제어."
    ;;
  ""|help|-h|--help)
    sed -n '2,12p' "$0"
    ;;
  *)
    echo "[run_ff] 알 수 없는 명령: $cmd"; sed -n '2,12p' "$0"; exit 1
    ;;
esac
