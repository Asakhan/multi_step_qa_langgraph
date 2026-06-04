"""저장된 raw_runs.jsonl 을 **모델 재호출 없이** 수정된 채점기로 재채점한다.

배경: 원시 실패의 상당수가 추론 오류가 아니라 채점 결함(특히 비율 소수↔퍼센트 100배
불일치)에 의한 오탐이었다(src/metrics.py 수정). 이 스크립트는 이미 저장된
(gold_answer, final_answer, gold_answer_unit) 만으로 metrics.is_correct 를 다시 호출해
success / judge_flag / judge_mode 만 갱신한다. 토큰·비용·latency 등은 그대로 둔다.

  python -m src.rejudge

- 입력(읽기전용): results_finqa_format/{s0,s2,s3,s4}/raw_runs.jsonl
- 출력(원본 보존): results_finqa_format_rejudged/{s0,s2,s3,s4}/{raw_runs.jsonl, metrics.json, failures.csv}
- 멱등: 항상 원본을 입력으로 읽고 출력 디렉토리를 덮어쓰므로 다시 돌려도 동일 결과.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config, metrics, runner

SRC_DIR = config.ROOT / "results_finqa_format"
OUT_DIR = config.ROOT / "results_finqa_format_rejudged"
STRUCTURES = ["s0", "s2", "s3", "s4"]  # S1 은 quota 미완(있으면 자동 포함)


def _rejudge_record(r: dict) -> dict:
    """단일 run 레코드 재채점. run_error 레코드는 그대로 둔다."""
    if r.get("run_error"):
        return r
    j = metrics.is_correct(r.get("gold_answer"), r.get("final_answer"),
                           r.get("gold_answer_unit"))
    out = dict(r)
    out["success"] = int(j.correct)
    out["judge_flag"] = bool(j.judge_flag)
    out["judge_mode"] = j.mode
    return out


def _present_structures() -> list[str]:
    structs = list(STRUCTURES)
    s1 = SRC_DIR / "s1" / "raw_runs.jsonl"
    if s1.exists() and "s1" not in structs:
        structs.append("s1")
    return [s for s in structs if (SRC_DIR / s / "raw_runs.jsonl").exists()]


def rejudge() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig_results_dir = config.RESULTS_DIR
    summary: dict[str, dict] = {}
    deltas: dict[str, dict] = {}
    try:
        for st in _present_structures():
            src = SRC_DIR / st / "raw_runs.jsonl"
            rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
            old_succ = sum(1 for r in rows if not r.get("run_error") and r.get("success"))
            new_rows = [_rejudge_record(r) for r in rows]
            new_succ = sum(1 for r in new_rows if not r.get("run_error") and r.get("success"))
            n_valid = sum(1 for r in new_rows if not r.get("run_error"))

            out_st = OUT_DIR / st
            out_st.mkdir(parents=True, exist_ok=True)
            with open(out_st / "raw_runs.jsonl", "w", encoding="utf-8") as fh:
                for r in new_rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 검증된 집계·failures.csv 로직 재사용: RESULTS_DIR 을 rejudged 로 임시 전환.
            config.RESULTS_DIR = OUT_DIR
            summary[st] = runner.finalize_structure(st)
            deltas[st] = {
                "n_valid": n_valid,
                "old_success": old_succ, "new_success": new_succ,
                "old_TSR": (old_succ / n_valid) if n_valid else None,
                "new_TSR": (new_succ / n_valid) if n_valid else None,
                "delta_success": new_succ - old_succ,
            }
    finally:
        config.RESULTS_DIR = orig_results_dir

    # 콘솔 요약
    print(f"[rejudge] 입력 {SRC_DIR} → 출력 {OUT_DIR}")
    print(f"{'구조':<6}{'유효':>6}{'원success':>10}{'보정success':>12}{'원TSR':>9}{'보정TSR':>10}{'Δ':>5}")
    for st in _present_structures():
        d = deltas[st]
        print(f"{st:<6}{d['n_valid']:>6}{d['old_success']:>10}{d['new_success']:>12}"
              f"{(d['old_TSR'] or 0):>9.3f}{(d['new_TSR'] or 0):>10.3f}{d['delta_success']:>+5}")
    return {"summary": summary, "deltas": deltas}


if __name__ == "__main__":
    rejudge()
