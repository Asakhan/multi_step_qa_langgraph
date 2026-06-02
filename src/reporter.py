"""진행상황 워처 — 부모(채팅) 세션이 살아있는 동안에만 10분마다 진행 메일.

  python -m src.reporter --watch --interval 600

세션이 종료되면 이 워처도 함께 종료된다(진행 보고만 멈추고, 실험 본체는 nohup 으로 계속).
한 번만 요약을 보고 종료하려면 --once.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, notify

EXP = config.experiment_cfg()


def _count_runs(structure: str) -> dict:
    path = config.RESULTS_DIR / structure / "raw_runs.jsonl"
    if not path.exists():
        return {"total": 0, "success": 0, "error": 0, "cost": 0.0}
    total = success = error = 0
    cost = 0.0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        total += 1
        if r.get("run_error"):
            error += 1
        else:
            success += int(r.get("success", 0))
            cost += float(r.get("cost", 0.0))
    return {"total": total, "success": success, "error": error, "cost": cost}


def _planned_per_structure(datasets, limit_per_dataset, runs_per_q) -> int:
    from .datasets import questions_by_dataset
    by = questions_by_dataset()
    n_q = 0
    for ds in datasets:
        qs = by.get(ds, [])
        n_q += min(len(qs), limit_per_dataset) if limit_per_dataset else len(qs)
    return n_q * runs_per_q


def build_progress(structures, datasets, limit_per_dataset, runs_per_q, started_at) -> tuple[str, dict]:
    planned = _planned_per_structure(datasets, limit_per_dataset, runs_per_q)
    lines = [f"[진행 보고] {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}", ""]
    total_done = total_planned = 0
    total_cost = 0.0
    finished = []
    in_progress = []
    for s in structures:
        c = _count_runs(s)
        total_done += c["total"]
        total_planned += planned
        total_cost += c["cost"]
        frac = c["total"] / planned if planned else 0
        state = "완료" if c["total"] >= planned else ("진행중" if c["total"] > 0 else "대기")
        if state == "완료":
            finished.append(s)
        elif state == "진행중":
            in_progress.append(s)
        lines.append(f"  {s.upper()}: {c['total']}/{planned} ({frac:.0%}) "
                     f"성공 {c['success']} 오류 {c['error']} 비용 ${c['cost']:.3f} [{state}]")
    elapsed = max(1e-6, time.time() - started_at)
    rate = total_done / elapsed  # runs/sec (이번 세션 관측)
    remaining = max(0, total_planned - total_done)
    eta_sec = remaining / rate if rate > 0 else None
    lines += ["",
              f"완료 구조: {', '.join(s.upper() for s in finished) or '없음'}",
              f"진행 중: {', '.join(s.upper() for s in in_progress) or '없음'}",
              f"전체 진행: {total_done}/{total_planned} ({(total_done/total_planned if total_planned else 0):.0%})",
              f"누적 비용(관측): ${total_cost:.3f}",
              f"예상 잔여시간: {(_fmt_eta(eta_sec))}"]
    meta = {"total_done": total_done, "total_planned": total_planned,
            "finished": finished, "in_progress": in_progress, "cost": total_cost}
    return "\n".join(lines), meta


def _fmt_eta(sec):
    if sec is None:
        return "추정 불가 (아직 데이터 부족)"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m}분"


def main(argv=None):
    ap = argparse.ArgumentParser(description="진행상황 워처")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=EXP["reporting"]["progress_interval_seconds"])
    ap.add_argument("--structures", nargs="+", default=["s1", "s2", "s3", "s4"])
    ap.add_argument("--datasets", nargs="+", default=["finqa", "kdart"])
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--runs", type=int, default=EXP["repetition"]["runs_per_question"])
    args = ap.parse_args(argv)

    started = time.time()
    structures = []
    for s in args.structures:
        structures.extend(["s1", "s2", "s3", "s4"] if s == "all" else [s])

    def tick():
        body, _ = build_progress(structures, args.datasets, args.limit_per_dataset, args.runs, started)
        print(body, flush=True)
        notify.send_progress(body)

    tick()
    if args.once:
        return
    if args.watch:
        try:
            while True:
                time.sleep(args.interval)
                tick()
        except KeyboardInterrupt:
            print("[reporter] 종료 (세션 종료/인터럽트)", flush=True)


if __name__ == "__main__":
    main()
