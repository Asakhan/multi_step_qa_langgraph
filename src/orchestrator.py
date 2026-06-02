"""백그라운드 실험 본체. 구조들을 독립 프로세스로 순차 실행 후 최종 분석·메일.

  python -m src.orchestrator --structures s1 s2 s3 s4 [--run-s0]
                             [--runs 3] [--datasets finqa kdart]
                             [--limit-per-dataset N] [--no-final-analysis]

- 각 구조를 별도 subprocess(python -m src.runner)로 실행 → 한 구조 실패가 다른 구조에
  영향을 주지 않음. 각 구조 완료 시 runner 가 결과 저장 + 구조별 메일 발송.
- 전 구조 완료 후 통계 분석(analysis) 실행 + 최종 종합 리포트 메일(파레토 첨부).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone

from . import config


def _expand(structures: list[str], run_s0: bool) -> list[str]:
    out = []
    for s in structures:
        if s == "all":
            out.extend(["s1", "s2", "s3", "s4"])
        else:
            out.append(s)
    if run_s0 and "s0" not in out:
        out = ["s0"] + out
    # 중복 제거(순서 보존)
    seen = set()
    ordered = []
    for s in out:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def main(argv=None):
    ap = argparse.ArgumentParser(description="실험 오케스트레이터 (백그라운드 본체)")
    ap.add_argument("--structures", nargs="+", required=True)
    ap.add_argument("--run-s0", action="store_true")
    ap.add_argument("--runs", type=int, default=config.experiment_cfg()["repetition"]["runs_per_question"])
    ap.add_argument("--datasets", nargs="+", default=["finqa", "kdart"], choices=["finqa", "kdart"])
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--no-final-analysis", action="store_true")
    args = ap.parse_args(argv)

    structures = _expand(args.structures, args.run_s0)
    print(f"[orchestrator] {datetime.now(timezone.utc).isoformat()} 구조: {structures}", flush=True)

    # 전체 필요한 env 사전 검증(빠른 실패). self-contained 컨텍스트라 임베딩 불요.
    config.validate_env(structures, need_email=True, need_embedding=False)

    results = {}
    for s in structures:
        cmd = [sys.executable, "-m", "src.runner", "--structure", s,
               "--runs", str(args.runs), "--datasets", *args.datasets]
        if args.limit_per_dataset:
            cmd += ["--limit-per-dataset", str(args.limit_per_dataset)]
        print(f"\n[orchestrator] === {s.upper()} 시작: {' '.join(cmd)} ===", flush=True)
        try:
            proc = subprocess.run(cmd, cwd=str(config.ROOT))
            results[s] = proc.returncode
            print(f"[orchestrator] {s.upper()} 종료 코드 {proc.returncode}", flush=True)
        except Exception as exc:  # noqa: BLE001 — 한 구조 실패가 다른 구조를 막지 않음
            results[s] = -1
            print(f"[orchestrator] {s.upper()} 예외: {exc}", flush=True)

    print(f"\n[orchestrator] 구조별 종료코드: {results}", flush=True)

    if not args.no_final_analysis:
        try:
            from . import analysis, notify
            print("[orchestrator] 최종 통계 분석 실행", flush=True)
            res = analysis.run_full_analysis()
            body = analysis.final_report_text(res)
            print(body, flush=True)
            attachments = [res["pareto"]["plot"]] if res.get("pareto", {}).get("plot") else []
            notify.send_final_report(body, attachments)
        except Exception as exc:  # noqa: BLE001
            print(f"[orchestrator] 최종 분석/메일 실패: {exc}", flush=True)

    print("[orchestrator] 완료", flush=True)
    return results


if __name__ == "__main__":
    main()
