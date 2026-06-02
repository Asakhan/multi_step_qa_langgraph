#!/usr/bin/env bash
# 진입점. S0~S4 개별/전체 실행, 백그라운드(nohup+setsid) 구동, 상태/중단, 인덱스/회귀검증/워처.
#
# 사용법:
#   ./run.sh build-index                # FinQA finqa.jsonl 생성 + K-DART 인덱스 검증
#   ./run.sh smoke                      # 표본 회귀검증 (구조별 데이터셋당 2문항, S1~S4)
#   ./run.sh s2                         # S2만 백그라운드 실행
#   ./run.sh s1 s3 s4                   # 일부만
#   ./run.sh all                        # S1~S4 전체 (S0 제외)
#   ./run.sh all --run-s0               # S0 포함
#   ./run.sh status                     # 진행 확인 (pid + 로그 tail)
#   ./run.sh stop                       # 중단
#   ./run.sh watch                      # 10분 진행 워처 (이 세션이 살아있는 동안만)
#   ./run.sh analysis                   # 통계 분석 + 최종 리포트
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
PID_FILE="logs/run.pid"
mkdir -p logs results

cmd="${1:-}"; shift || true

start_bg() {
  # 남은 인자에서 --run-s0 추출
  local extra=()
  local structures=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-s0|--limit-per-dataset|--runs|--datasets|--no-final-analysis)
        extra+=("$1"); shift
        # 값이 필요한 옵션은 다음 토큰도 가져감
        if [[ "${extra[-1]}" == "--limit-per-dataset" || "${extra[-1]}" == "--runs" ]]; then
          extra+=("$1"); shift
        elif [[ "${extra[-1]}" == "--datasets" ]]; then
          while [[ $# -gt 0 && "$1" != --* ]]; do extra+=("$1"); shift; done
        fi
        ;;
      *) structures+=("$1"); shift ;;
    esac
  done
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local log="logs/run_${ts}.log"
  local args="--structures ${structures[*]}"
  [[ ${#extra[@]} -gt 0 ]] && args="$args ${extra[*]}"
  echo "[run.sh] 백그라운드 실행: ${args}"
  echo "[run.sh] 로그: ${log}"
  # setsid 로 새 세션(세션 종료에 영향 없음). bash 의 $$ 가 exec 로 python 으로 대체되므로
  # PID_FILE 에는 실제 orchestrator(python) PID 가 기록되고, 그것이 세션/그룹 리더가 된다.
  setsid bash -c "echo \$\$ > '$PID_FILE'; exec '$PY' -m src.orchestrator $args > '$log' 2>&1" </dev/null &
  sleep 0.5
  echo "[run.sh] PID $(cat "$PID_FILE" 2>/dev/null) (logs/run.pid). './run.sh status' 로 확인, './run.sh stop' 으로 중단."
}

case "$cmd" in
  build-index)
    exec "$PY" -m src.build_index "$@"
    ;;
  smoke)
    echo "[run.sh] 표본 회귀검증: S1~S4 × 데이터셋당 2문항 × 3반복 (전경 실행)"
    exec "$PY" -m src.orchestrator --structures s1 s2 s3 s4 --limit-per-dataset 2 "$@"
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "[run.sh] 실행 중 (PID $(cat "$PID_FILE"))"
    else
      echo "[run.sh] 실행 중인 프로세스 없음"
    fi
    latest="$(ls -t logs/run_*.log 2>/dev/null | head -1 || true)"
    [[ -n "$latest" ]] && { echo "--- ${latest} (tail) ---"; tail -n 30 "$latest"; }
    echo "--- 구조별 완료 run 수 ---"
    for s in s0 s1 s2 s3 s4; do
      f="results/$s/raw_runs.jsonl"
      [[ -f "$f" ]] && echo "  $s: $(wc -l < "$f") runs"
    done
    ;;
  stop)
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      # 프로세스 그룹 전체 종료 (setsid 로 그룹장) + 자식 runner 백업 종료
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      echo "[run.sh] 중단 신호 전송 (PID ${pid}, 프로세스 그룹)"
      rm -f "$PID_FILE"
    else
      echo "[run.sh] PID 파일 없음 — 잔여 프로세스 탐색"
    fi
    pkill -TERM -f "src.orchestrator" 2>/dev/null || true
    pkill -TERM -f "src.runner" 2>/dev/null || true
    echo "[run.sh] orchestrator/runner 종료 신호 전송 완료"
    ;;
  watch)
    exec "$PY" -m src.reporter --watch "$@"
    ;;
  analysis)
    exec "$PY" -m src.analysis "$@"
    ;;
  s0|s1|s2|s3|s4|all)
    start_bg "$cmd" "$@"
    ;;
  ""|help|-h|--help)
    sed -n '2,20p' "$0"
    ;;
  *)
    echo "[run.sh] 알 수 없는 명령: $cmd"; sed -n '2,20p' "$0"; exit 1
    ;;
esac
