"""전 실험 로그 재채점 — 현행 채점기로 success 를 다시 매긴다 (결함 A·B 반영).

## 왜 필요한가

로그의 `success` 는 **실행 당시의 채점기**로 확정되어 파일에 박히므로, 나중에 채점기를
고쳐도 자동으로 갱신되지 않는다. 그 결과 두 문제가 남아 있었다.

- **결함 A — 자릿수 단위 이중 적용**: `is_correct` 가 퍼센트 단위에는 "gold 가 이미 그
  단위로 표현되어 있다"는 처리를 두면서 백만원·억원 등 자릿수 단위에는 두지 않았다.
  모델이 같은 값을 단위까지 붙여 답하면 배수가 한 번 더 곱해져 오답이 되었다.
  `metrics._apply_declared_magnitude` 로 수정했다.
- **결함 B — 로그가 현행 코드와 불일치**: 유니코드 기호 정규화(U+2212 등)가
  2026-06-04(커밋 376410b)에 추가되었는데, 그 이전에 실행된 s2·s5(2026-06-02)는
  재채점되지 않았다. 같은 gold·같은 답인데 s0·s6(2026-08-21 실행)과 판정이 갈렸다.

## 무엇을 하는가

`raw_runs.jsonl` 의 각 run 을 현행 `metrics.is_correct` 로 다시 판정해 `success`,
`judge_flag`, `judge_mode` 를 갱신하고, `metrics.json` 을 재생성한다. S7 로그는
1차 생성 스냅샷(`first_answer`)의 `first_success`·`first_judge_flag` 도 함께 갱신한다
(반사실 재현의 stop 갈래 성능이 여기에 달려 있다).

**LLM 을 호출하지 않는다** — 저장된 답 문자열을 다시 채점할 뿐이다.

원본은 `raw_runs.jsonl.prefix_YYYYMMDD.bak` 로 보존한다.

사용:
    .venv/Scripts/python.exe -m src.rescore_all --dry-run     # 영향만 확인
    .venv/Scripts/python.exe -m src.rescore_all               # 실제 적용
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import metrics

# (결과 디렉토리, 구조 하위폴더) — 본실험·실험 C·실험 A 전량
TARGETS = [
    ("results_main", "s0"),
    ("results_main", "s2"),
    ("results_main", "s5"),
    ("results_main", "s6"),
    ("results_expC", "s7"),
    ("results_expC_ext", "s7"),
    ("results_pilotA_L1", "s2"),
    ("results_pilotA_L2", "s2"),
    ("results_pilotA_L3", "s2"),
]


def _rescore_run(r: dict) -> tuple[dict, bool]:
    """한 run 을 현행 채점기로 재판정. (갱신된 run, 변경여부) 반환."""
    changed = False
    unit = r.get("gold_answer_unit")
    gold = r.get("gold_answer")

    jud = metrics.is_correct(gold, r.get("final_answer"), unit)
    if int(r.get("success", 0)) != int(jud.correct):
        changed = True
    r["success"] = int(jud.correct)
    r["judge_flag"] = bool(jud.judge_flag)
    r["judge_mode"] = jud.mode

    # S7 의 1차 생성 스냅샷 — 반사실 재현의 stop 갈래가 이 값을 쓴다.
    if "first_answer" in r:
        fj = metrics.is_correct(gold, r.get("first_answer"), unit)
        if int(r.get("first_success", 0)) != int(fj.correct):
            changed = True
        r["first_success"] = int(fj.correct)
        r["first_judge_flag"] = bool(fj.judge_flag)
        r["first_judge_mode"] = fj.mode
    return r, changed


def _recompute_metrics(runs: list[dict], structure: str) -> dict:
    """metrics.json 재생성 — 기존 runner 산출 스키마와 같은 키를 채운다."""
    n = len(runs)
    ok = [r for r in runs if r["success"]]
    tot_in = sum(float(r.get("input_tokens") or 0) for r in runs)
    tot_out = sum(float(r.get("output_tokens") or 0) for r in runs)
    tot_tok = sum(float(r.get("total_tokens") or 0) for r in runs)
    tot_cost = sum(float(r.get("cost") or 0) for r in runs)

    def _group(key: str) -> dict:
        g = defaultdict(list)
        for r in runs:
            g[r.get(key, "")].append(r)
        out = {}
        for k, v in sorted(g.items()):
            ns = sum(x["success"] for x in v)
            out[k] = {
                "n_runs": len(v),
                "n_success": ns,
                "TSR": ns / len(v),
                "avg_latency": sum(float(x.get("latency") or 0) for x in v) / len(v),
                "avg_tokens": sum(float(x.get("total_tokens") or 0) for x in v) / len(v),
                "avg_cost": sum(float(x.get("cost") or 0) for x in v) / len(v),
            }
        return out

    return {
        "n_runs": n,
        "n_success": len(ok),
        "TSR": len(ok) / n if n else 0.0,
        "avg_input_tokens": tot_in / n if n else 0.0,
        "avg_output_tokens": tot_out / n if n else 0.0,
        "avg_total_tokens": tot_tok / n if n else 0.0,
        "avg_latency": sum(float(r.get("latency") or 0) for r in runs) / n if n else 0.0,
        "total_cost_usd": tot_cost,
        "avg_cost_usd": tot_cost / n if n else 0.0,
        "TPS": tot_tok / len(ok) if ok else None,
        "cost_per_success_usd": tot_cost / len(ok) if ok else None,
        "by_dataset": _group("dataset"),
        "by_task_type": _group("task_type"),
        "structure": structure,
        "n_error_runs": sum(1 for r in runs if r.get("run_error")),
        "n_judge_flagged": sum(1 for r in runs if r.get("judge_flag")),
        "rescored_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rescore_note": ("현행 채점기로 재채점됨 — 선언 자릿수 단위 정합(결함 A) + "
                         "유니코드 기호 정규화 반영(결함 B). LLM 재호출 없음."),
    }


def process(target: tuple[str, str], dry_run: bool, stamp: str) -> dict | None:
    results_dir, structure = target
    path = Path(results_dir) / structure / "raw_runs.jsonl"
    if not path.exists():
        return None
    runs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    before = sum(int(r.get("success", 0)) for r in runs)
    before_first = sum(int(r.get("first_success", 0)) for r in runs if "first_answer" in r)

    changed_runs = []
    for r in runs:
        old_s = int(r.get("success", 0))
        old_f = int(r.get("first_success", 0)) if "first_answer" in r else None
        r, ch = _rescore_run(r)
        if ch:
            changed_runs.append({
                "question_id": r.get("question_id"), "run_index": r.get("run_index"),
                "task_type": r.get("task_type"), "dataset": r.get("dataset"),
                "gold": r.get("gold_answer"), "unit": r.get("gold_answer_unit"),
                "pred": r.get("final_answer"),
                "success": f"{old_s}->{r['success']}",
                "first_success": (f"{old_f}->{r['first_success']}"
                                  if old_f is not None and old_f != r["first_success"] else None),
            })

    after = sum(int(r.get("success", 0)) for r in runs)
    after_first = sum(int(r.get("first_success", 0)) for r in runs if "first_answer" in r)
    n = len(runs)
    summary = {
        "dir": results_dir, "structure": structure, "n_runs": n,
        "TSR_before": before / n, "TSR_after": after / n,
        "delta_runs": after - before,
        "n_changed": len(changed_runs),
        "changed": changed_runs,
    }
    if before_first or after_first:
        summary["first_TSR_before"] = before_first / n
        summary["first_TSR_after"] = after_first / n
        summary["first_delta_runs"] = after_first - before_first

    if not dry_run:
        bak = path.with_suffix(f".jsonl.pre_rescore_{stamp}.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        with open(path, "w", encoding="utf-8") as fh:
            for r in runs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        mpath = path.parent / "metrics.json"
        if mpath.exists():
            mbak = mpath.with_suffix(f".json.pre_rescore_{stamp}.bak")
            if not mbak.exists():
                shutil.copy2(mpath, mbak)
        mpath.write_text(
            json.dumps(_recompute_metrics(runs, structure), ensure_ascii=False, indent=1),
            encoding="utf-8")
        summary["backup"] = str(bak)
    return summary


def process_exp1(csv_path: Path, dry_run: bool, stamp: str) -> dict | None:
    """실험 1(CrewAI) 재채점본 CSV — LangGraph 와 **같은 채점기**로 맞춘다.

    비교 대상 구조가 서로 다른 채점 기준으로 매겨져 있으면 Table 3 의 구조 간 비교가
    성립하지 않는다. 결함 A 가 드러난 계기가 바로 이 비대칭이었으므로(CrewAI 는 단위를
    붙이지 않는 출력 양식이라 통과, LangGraph 는 붙여서 탈락) 반드시 함께 재채점한다.

    채점 대상은 `final_answer` 가 아니라 **`final_answer_normalized`** 다. CrewAI 는
    퍼센트 답에 '%' 를 붙이지 않는 출력 양식이라(예: gold 0.15653 에 '15.65265' 로 답)
    기존 재채점 파이프라인이 추론의 ×100 연산을 근거로 '%' 를 복원해 두었고, `success`
    는 그 복원본 기준이다. 원문으로 채점하면 이 21건이 전부 오답이 되어 CrewAI 점수만
    부당하게 깎인다.

    이 '% 복원'이 LangGraph 에 없는 특혜인지 확인했다 — LangGraph 로그에는 같은 상황
    (gold 가 비율 소수, pred 가 '%' 없는 100배 값)이 **0건**이다. LangGraph 모델은
    퍼센트 답에 항상 '%' 를 붙이므로 `_percent_reconcile` 이 그대로 처리한다. 즉 두
    프레임워크가 서로 다른 경로로 같은 정합을 받은 것이지 기준이 다른 것이 아니다.
    """
    import csv as _csv
    if not csv_path.exists():
        return None
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
        fields = list(rows[0].keys()) if rows else []
    if not rows:
        return None

    before = sum(int(float(r.get("success") or 0)) for r in rows)
    changed = []
    for r in rows:
        answer = r.get("final_answer_normalized") or r.get("final_answer")
        jud = metrics.is_correct(r.get("gold_answer"), answer,
                                 r.get("gold_answer_unit"))
        old = int(float(r.get("success") or 0))
        if old != int(jud.correct):
            changed.append({
                "question_id": r.get("id"), "run_index": r.get("run_index"),
                "task_type": r.get("task_type"), "dataset": r.get("dataset"),
                "gold": r.get("gold_answer"), "unit": r.get("gold_answer_unit"),
                "pred": answer, "success": f"{old}->{int(jud.correct)}",
                "first_success": None,
            })
        r["success"] = str(int(jud.correct))
        r["judge_flag"] = str(bool(jud.judge_flag))
    after = sum(int(float(r.get("success") or 0)) for r in rows)
    n = len(rows)
    summary = {"dir": str(csv_path.parent), "structure": "exp1(CrewAI)", "n_runs": n,
               "TSR_before": before / n, "TSR_after": after / n,
               "delta_runs": after - before, "n_changed": len(changed), "changed": changed}
    if not dry_run:
        bak = csv_path.with_suffix(f".csv.pre_rescore_{stamp}.bak")
        if not bak.exists():
            shutil.copy2(csv_path, bak)
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        summary["backup"] = str(bak)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="전 실험 로그 재채점 (LLM 호출 없음)")
    ap.add_argument("--dry-run", action="store_true", help="적용하지 않고 영향만 출력")
    ap.add_argument("--out", default="results_phase1", help="요약 JSON 출력 폴더")
    ap.add_argument("--exp1-csv",
                    default=r"C:/Users/asakh/Documents/개인/대학원/2026/졸업논문/"
                            r"실험결과/results_exp1_rescored.csv")
    args = ap.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d")
    mode = "DRY RUN — 파일을 쓰지 않는다" if args.dry_run else "적용"
    print(f"=== 전 실험 재채점 ({mode}) ===\n")

    results, total_delta, total_changed = [], 0, 0
    jobs = [(t, None) for t in TARGETS] + [(None, Path(args.exp1_csv))]
    for t, csvp in jobs:
        s = (process(t, args.dry_run, stamp) if t
             else process_exp1(csvp, args.dry_run, stamp))
        if s is None:
            print(f"  {t[0] + '/' + t[1] if t else csvp}: 없음 — 건너뜀")
            continue
        results.append(s)
        total_delta += s["delta_runs"]
        total_changed += s["n_changed"]
        line = (f"  {s['dir']}/{s['structure']}: TSR {s['TSR_before']:.4f} -> "
                f"{s['TSR_after']:.4f} ({s['delta_runs']:+d} run, 변경 {s['n_changed']}건)")
        if "first_TSR_before" in s:
            line += (f"  | 1차생성 {s['first_TSR_before']:.4f} -> "
                     f"{s['first_TSR_after']:.4f} ({s['first_delta_runs']:+d})")
        print(line)

    print(f"\n합계: {total_delta:+d} run, 판정 변경 {total_changed}건")
    if not args.dry_run:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "rescore_all_summary.json"
        p.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"요약: {p}")
        print(f"원본 백업: *.pre_rescore_{stamp}.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
