"""단일 구조 실행 엔진.

  python -m src.runner --structure s2 [--runs 3] [--datasets finqa kdart] [--limit-per-dataset N]

- 각 문항 × 3반복 전수 로깅 -> results/sX/raw_runs.jsonl (append, idempotent resume).
- 구조 완료 시 metrics.json + failures.csv 저장 후 해당 구조 결과 메일 발송.
- 토큰/지연 노드별 누락 없는 합산을 검산(assert).
- 누적 비용 상한(COST_CAP_USD) 초과 시 경고 메일 후 일시정지.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import config, metrics, tools
from .datasets import Question, load_questions
from .state import new_state
from .structures import build_graph

EXP = config.experiment_cfg()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _results_dir(structure: str) -> Path:
    d = config.RESULTS_DIR / structure
    d.mkdir(parents=True, exist_ok=True)
    return d


def _raw_path(structure: str) -> Path:
    return _results_dir(structure) / "raw_runs.jsonl"


def _done_run_ids(structure: str) -> set[str]:
    path = _raw_path(structure)
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not rec.get("run_error"):  # 실패 run 은 재시도 허용
                    done.add(rec["run_id"])
            except Exception:
                continue
    return done


def _node_token_breakdown(token_usage: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in token_usage:
        node = r["node"]
        d = out.setdefault(node, {"input_tokens": 0, "output_tokens": 0, "n_calls": 0})
        d["input_tokens"] += r["input_tokens"]
        d["output_tokens"] += r["output_tokens"]
        d["n_calls"] += 1
    return out


def _assert_token_accounting(token_usage: list[dict], breakdown: dict) -> None:
    """노드별 합 == 전체 합 검산 (누락 없는 합산)."""
    it_total = sum(r["input_tokens"] for r in token_usage)
    ot_total = sum(r["output_tokens"] for r in token_usage)
    it_bd = sum(d["input_tokens"] for d in breakdown.values())
    ot_bd = sum(d["output_tokens"] for d in breakdown.values())
    assert it_total == it_bd and ot_total == ot_bd, (
        f"토큰 합산 불일치: total=({it_total},{ot_total}) breakdown=({it_bd},{ot_bd})")


def _gold_chunk_retrieved(q: Question, retrieved_ids: list[str]) -> bool:
    """gold evidence chunk 가 검색 결과에 포함됐는지 (E4 단서).
    FinQA: gold evidence_chunks(table_6 등)가 retrieved(text_N/table_N)에 포함.
    K-DART: gold evidence_chunks(전역 chunk_id)가 retrieved 에 포함.
    """
    rid = set(retrieved_ids)
    gold = q.evidence_chunks or []
    if not gold:
        return False
    return any(gc in rid for gc in gold)


def run_one(graph, structure: str, q: Question, run_index: int) -> dict:
    run_id = f"{q.qid}_{structure}_{run_index:02d}"
    init = new_state(q.query, dataset=q.dataset, question_id=q.qid,
                     task_type=q.task_type, lang=q.lang,
                     context_chunks=q.context_chunks)
    t0 = time.perf_counter()
    error = ""
    try:
        # recursion_limit 넉넉히 (S2 재진입/ S4 순회 안전)
        final = graph.invoke(init, config={"recursion_limit": 50})
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - t0
        return {
            "run_id": run_id, "structure": structure, "dataset": q.dataset,
            "question_id": q.qid, "task_type": q.task_type, "orig_task_type": q.orig_task_type,
            "run_index": run_index, "query": q.query, "gold_answer": q.gold_answer,
            "gold_answer_unit": q.gold_answer_unit, "final_answer": "",
            "success": 0, "judge_flag": True, "judge_mode": "error",
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0,
            "latency": latency, "timestamp": _now_iso(),
            "run_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
    latency = time.perf_counter() - t0

    token_usage = final.get("token_usage", [])
    breakdown = _node_token_breakdown(token_usage)
    _assert_token_accounting(token_usage, breakdown)
    it, ot = metrics.run_tokens(token_usage)
    cost = metrics.run_cost(token_usage)
    pred = final.get("final_answer", "")
    jud = metrics.is_correct(q.gold_answer, pred, q.gold_answer_unit)

    retrieved = final.get("retrieved_evidence", [])
    retrieved_ids = [c.get("chunk_id", "") for c in retrieved]

    return {
        "run_id": run_id,
        "structure": structure,
        "dataset": q.dataset,
        "question_id": q.qid,
        "task_type": q.task_type,
        "orig_task_type": q.orig_task_type,
        "lang": q.lang,
        "run_index": run_index,
        "query": q.query,
        "gold_answer": q.gold_answer,
        "gold_answer_unit": q.gold_answer_unit,
        "final_answer": pred,
        "reasoning_trace": final.get("reasoning_trace", ""),
        "used_expression": final.get("used_expression", []),
        "retrieved_chunk_ids": retrieved_ids,
        "retrieved_evidence": retrieved,
        "gold_chunk_retrieved": _gold_chunk_retrieved(q, retrieved_ids),
        "evidence_quotes": q.evidence_quotes,
        "evidence_chunks": q.evidence_chunks,
        "reasoning_hops": q.reasoning_hops,
        "success": int(jud.correct),
        "judge_flag": jud.judge_flag,
        "judge_mode": jud.mode,
        "input_tokens": it,
        "output_tokens": ot,
        "total_tokens": it + ot,
        "cost": cost,
        "latency": latency,
        "token_breakdown": breakdown,
        "retry_count": final.get("retry_count", 0),
        "verify_verdict": final.get("verify_verdict", ""),
        "critique": final.get("critique", ""),
        # 실험3(S5) 게이트 신호 — H3 보조가설(게이트 발동 집합 vs 실제 오답 집합) 비교용
        "confidence": final.get("confidence", 0.0),                       # 유효(보정 후) — gate 입력
        "confidence_self_report": final.get("confidence_self_report", 0.0),  # 모델 원자기보고
        "code_conflict": final.get("code_conflict", False),              # 보조신호: 추론답 vs 코드실행답 충돌
        "gate_decision": final.get("gate_decision", ""),
        "escalated": final.get("gate_decision", "") == "escalate",
        "sub_questions": final.get("sub_questions", []),
        "sub_answers": final.get("sub_answers", []),
        "aggregation_info": final.get("aggregation_info", {}),
        "timestamp": _now_iso(),
        "run_error": "",
    }


def _cumulative_cost(structure: str) -> float:
    path = _raw_path(structure)
    if not path.exists():
        return 0.0
    total = 0.0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            total += float(json.loads(line).get("cost", 0.0))
        except Exception:
            continue
    return total


def run_structure(structure: str, runs_per_q: int, datasets: list[str],
                  limit_per_dataset: int | None) -> dict:
    print(f"\n===== 구조 {structure.upper()} 실행 시작 =====", flush=True)
    graph = build_graph(structure)
    questions = [q for q in load_questions() if q.dataset in datasets]
    if limit_per_dataset:
        by: dict[str, list[Question]] = {}
        for q in questions:
            by.setdefault(q.dataset, []).append(q)
        questions = []
        for ds in datasets:
            questions.extend(by.get(ds, [])[:limit_per_dataset])

    done = _done_run_ids(structure)
    raw_path = _raw_path(structure)
    cost_cap = config.cost_cap_usd()
    cum_cost = _cumulative_cost(structure)

    total_planned = len(questions) * runs_per_q
    n_done_existing = sum(1 for q in questions for r in range(1, runs_per_q + 1)
                          if f"{q.qid}_{structure}_{r:02d}" in done)
    print(f"  계획 {total_planned}회 중 기존 완료 {n_done_existing}회 → 이어서 진행", flush=True)

    paused = False
    with open(raw_path, "a", encoding="utf-8") as out:
        for q in questions:
            for r in range(1, runs_per_q + 1):
                run_id = f"{q.qid}_{structure}_{r:02d}"
                if run_id in done:
                    continue
                if cost_cap is not None and cum_cost >= cost_cap:
                    paused = True
                    _emit_cost_warning(structure, cum_cost, cost_cap)
                    break
                rec = run_one(graph, structure, q, r)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                cum_cost += rec.get("cost", 0.0)
                status = "OK" if rec["success"] else ("ERR" if rec.get("run_error") else "x")
                print(f"  [{structure}] {run_id} {status} "
                      f"tok={rec.get('total_tokens',0)} ${rec.get('cost',0):.4f} "
                      f"{rec.get('latency',0):.1f}s cum=${cum_cost:.3f}", flush=True)
            if paused:
                break

    summary = finalize_structure(structure)
    summary["paused_by_cost_cap"] = paused
    if not paused:
        _emit_structure_email(structure, summary)
    return summary


def finalize_structure(structure: str) -> dict:
    """raw_runs.jsonl -> metrics.json + failures.csv 산출."""
    raw_path = _raw_path(structure)
    runs = []
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    valid = [r for r in runs if not r.get("run_error")]
    agg = metrics.aggregate_metrics(valid) if valid else {"n_runs": 0}
    agg["structure"] = structure
    agg["n_error_runs"] = sum(1 for r in runs if r.get("run_error"))
    agg["n_judge_flagged"] = sum(1 for r in valid if r.get("judge_flag"))

    with open(_results_dir(structure) / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2)

    _write_failures_csv(structure, runs)
    return agg


def _write_failures_csv(structure: str, runs: list[dict]) -> None:
    """실패(success=0) 케이스를 오류분류용 CSV 로 추출 (저자 수기 분류)."""
    path = _results_dir(structure) / "failures.csv"
    cols = [
        "run_id", "structure", "dataset", "question_id", "task_type", "run_index",
        "query", "gold_answer", "pred_answer", "reasoning_trace", "used_expression",
        "retrieved_chunks", "gold_chunk", "evidence_quotes",
        "error_type",            # 저자 기입 (비움)
        "gold_chunk_retrieved",  # E4 단서
        "expr_recompute",        # E2 단서 (used_expression Python 재실행)
        "judge_flag", "run_error",
    ]
    failures = [r for r in runs if (not r.get("success")) ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        # E1~E5 정의를 헤더 주석으로
        fh.write("# E1 Evidence Selection: 잘못된 청크/같은 청크 내 잘못된 수치 선택\n")
        fh.write("# E2 Numerical Computation: 수치는 맞으나 계산 단계 오류\n")
        fh.write("# E3 Multi-step Accumulation: 중간 단계는 맞으나 누적 후 최종 답 오류\n")
        fh.write("# E4 Retrieval Failure: RAG가 gold chunk 미검색\n")
        fh.write("# E5 Other\n")
        fh.write("# error_type 열을 저자가 직접 기입. gold_chunk_retrieved=False면 E4 후보.\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in failures:
            exprs = r.get("used_expression", []) or []
            recompute = []
            for e in exprs:
                v, err = tools.safe_calc(e)
                recompute.append(f"{e} => {v if err is None else 'ERR:'+err}")
            gold_chunk = ";".join(r.get("evidence_chunks", []) or [])
            w.writerow({
                "run_id": r.get("run_id", ""),
                "structure": r.get("structure", structure),
                "dataset": r.get("dataset", ""),
                "question_id": r.get("question_id", ""),
                "task_type": r.get("task_type", ""),
                "run_index": r.get("run_index", ""),
                "query": r.get("query", ""),
                "gold_answer": r.get("gold_answer", ""),
                "pred_answer": r.get("final_answer", ""),
                "reasoning_trace": r.get("reasoning_trace", ""),
                "used_expression": " | ".join(exprs),
                "retrieved_chunks": ";".join(r.get("retrieved_chunk_ids", []) or [])[:1000],
                "gold_chunk": gold_chunk,
                "evidence_quotes": " | ".join(r.get("evidence_quotes", []) or []),
                "error_type": "",
                "gold_chunk_retrieved": r.get("gold_chunk_retrieved", ""),
                "expr_recompute": " | ".join(recompute),
                "judge_flag": r.get("judge_flag", ""),
                "run_error": r.get("run_error", ""),
            })


def _emit_structure_email(structure: str, summary: dict) -> None:
    try:
        from . import notify
        notify.send_structure_report(structure, summary)
    except Exception as exc:  # noqa: BLE001 — 메일 실패가 실험을 막지 않음
        print(f"  [warn] 구조 완료 메일 발송 실패: {exc}", flush=True)


def _emit_cost_warning(structure: str, cum: float, cap: float) -> None:
    print(f"  [COST CAP] 누적 ${cum:.2f} >= 상한 ${cap:.2f} → 일시정지", flush=True)
    try:
        from . import notify
        notify.send_cost_warning(structure, cum, cap)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 비용 경고 메일 실패: {exc}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="단일 구조 실행")
    ap.add_argument("--structure", required=True, choices=config.ALL_STRUCTURES)
    ap.add_argument("--runs", type=int, default=EXP["repetition"]["runs_per_question"])
    ap.add_argument("--datasets", nargs="+", default=["finqa", "kdart"], choices=["finqa", "kdart"])
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="회귀 검증용: 데이터셋별 N문항만")
    ap.add_argument("--no-email", action="store_true")
    args = ap.parse_args(argv)

    need_email = not args.no_email
    # 두 데이터셋 모두 self-contained 컨텍스트라 임베딩 검색이 없음 → 임베딩 키 불요.
    config.validate_env([args.structure], need_email=need_email, need_embedding=False)

    summary = run_structure(args.structure, args.runs, args.datasets, args.limit_per_dataset)
    print(f"\n[{args.structure}] 완료 — TSR={summary.get('TSR')}, "
          f"TPS={summary.get('TPS')}, runs={summary.get('n_runs')}", flush=True)
    return summary


if __name__ == "__main__":
    main()
    sys.exit(0)
