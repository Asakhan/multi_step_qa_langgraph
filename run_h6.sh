#!/usr/bin/env bash
# H6(기반 모델 간 일반화) + 실험 D(상위 모델 검증기) 병렬 실행.
#
# 설계 근거:
#   S7 은 게이트 없이 verify 를 전수 실행하고 1차 생성 스냅샷을 남기므로, 로그 하나에서
#   무검증·항상검증·자기보고 θ 스윕·무작위 게이트·oracle 상하한을 추가 호출 없이 재구성한다
#   (src/replay_gates.py). 따라서 "300문항 × 4구조 × 3모델"을 전부 실행할 필요가 없고,
#   모델당 S7 한 번(140문항 × 3반복 = 420 run)이면 4구조 전부의 게이트 정책이 복원된다.
#
#   기존 s7(gemini-2.5-flash)은 results_expC / results_expC_ext 에 이미 완주해 있으므로
#   여기서는 신규 2모델 + 상위 검증기만 실행한다.
#
# 실행 단위:
#   구조 3종 × 데이터셋 3종(finqa 20 / finqa_ext 80 / kdart 40) = 9 프로세스 병렬.
#   샤드마다 RESULTS_DIR 을 분리해 raw_runs.jsonl append 경합을 없앤다.
#   중단 후 같은 명령을 다시 실행하면 완료분은 run_id 로 건너뛴다(idempotent resume).
#
# 사용법:
#   ./run_h6.sh start     # 9개 샤드 병렬 착수
#   ./run_h6.sh status    # 샤드별 진행 run 수
#   ./run_h6.sh stop      # 전체 중단
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# .env 에 없거나 잘못된 모델 문자열을 여기서 확정한다(.env 는 수정하지 않는다).
export MODEL_GPT4O_MINI="${MODEL_GPT4O_MINI:-gpt-4o-mini}"
export MODEL_CLAUDE_HAIKU="claude-haiku-4-5-20251001"
export MODEL_CLAUDE_SONNET="${MODEL_CLAUDE_SONNET:-claude-sonnet-4-5}"

STRUCTS="s7_gpt s7_claude s7_vpro"
DATASETS="finqa finqa_ext kdart"
BASE="results_h6"

cmd="${1:-status}"

case "$cmd" in
  start)
    mkdir -p logs "$BASE"
    for s in $STRUCTS; do
      for d in $DATASETS; do
        out="$BASE/${s}__${d}"
        log="logs/h6_${s}__${d}.log"
        mkdir -p "$out"
        echo "[h6] 착수: $s / $d  -> $out  (로그 $log)"
        RESULTS_DIR="$out" "$PY" -m src.runner \
          --structure "$s" --runs 3 --datasets "$d" --no-email \
          > "$log" 2>&1 &
        sleep 1
      done
    done
    echo "[h6] 9개 샤드 착수 완료. 전체 종료까지 대기한다."
    wait
    echo "[h6] 전체 샤드 종료 $(date +%H:%M:%S)"
    ;;
  status)
    total=0
    for s in $STRUCTS; do
      for d in $DATASETS; do
        f="$BASE/${s}__${d}/${s}/raw_runs.jsonl"
        n=0; [[ -f "$f" ]] && n=$(wc -l < "$f")
        total=$((total+n))
        alive="종료"
        pgrep -f "structure $s --runs 3 --datasets $d" >/dev/null 2>&1 && alive="실행중"
        printf "  %-12s %-10s %5d run  [%s]\n" "$s" "$d" "$n" "$alive"
      done
    done
    echo "  ---- 합계 ${total} / 1260 run"
    ;;
  stop)
    pkill -f "src.runner --structure s7_" 2>/dev/null || true
    echo "[h6] 중단 신호 전송"
    ;;
  *)
    sed -n '2,25p' "$0"
    ;;
esac
