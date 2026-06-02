"""통계 분석: McNemar, bootstrap CI, within-run 순효과, 효과크기, 파레토.

  python -m src.analysis

- TSR 차이: McNemar 정확검정(쌍체, 문항 단위 다수결 정답)
- 비용지표(log Token/Latency/TPS): 부트스트랩 CI n=1,000
- within-run 순효과: 동일 문항·구조 3회 반복 내 정답 전환 집계
- H1: S2·S3 TPS 를 S0 와 비교 (적어도 하나가 유의하게 낮으면 지지)
- H2: S4 vs S0 TSR McNemar, 과업유형(T1~T5, 특히 T4) 분해
- 효과크기: Cohen's d / Cramer's V
- 파레토: 정확도-비용 평면 시각화, S0-S1 보간선, 프론티어 식별

오류유형(E1~E5) 분류는 저자 수기 작업이므로 자동화하지 않는다(failures.csv 만 준비).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import config

N_BOOT = 1000
SEED = 20260601  # 재현성 고정 시드


def _load_runs(structure: str) -> list[dict]:
    path = config.RESULTS_DIR / structure / "raw_runs.jsonl"
    if not path.exists():
        return []
    runs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if not r.get("run_error"):
                runs.append(r)
    return runs


def _available_structures() -> list[str]:
    return [s for s in config.ALL_STRUCTURES if _load_runs(s)]


# ---------------------------------------------------------------------------
# 문항 단위 집계
# ---------------------------------------------------------------------------

def _item_success_counts(runs: list[dict]) -> dict[str, dict]:
    """question_id -> {'count': 성공수(0~3), 'n': 반복수, 'majority': 0/1, 'task_type':..}"""
    by: dict[str, dict] = {}
    for r in runs:
        qid = r["question_id"]
        d = by.setdefault(qid, {"count": 0, "n": 0, "task_type": r.get("task_type")})
        d["count"] += int(r["success"])
        d["n"] += 1
    for d in by.values():
        d["majority"] = 1 if d["count"] >= math.ceil(d["n"] / 2) else 0
    return by


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

def mcnemar_test(a_runs: list[dict], b_runs: list[dict]) -> dict:
    """문항 단위 다수결 정답으로 McNemar 정확검정 (a vs b)."""
    from statsmodels.stats.contingency_tables import mcnemar
    A = _item_success_counts(a_runs)
    B = _item_success_counts(b_runs)
    common = sorted(set(A) & set(B))
    b01 = b10 = n00 = n11 = 0
    for qid in common:
        a = A[qid]["majority"]
        b = B[qid]["majority"]
        if a == 1 and b == 1:
            n11 += 1
        elif a == 0 and b == 0:
            n00 += 1
        elif a == 1 and b == 0:
            b10 += 1   # a 성공, b 실패
        else:
            b01 += 1   # a 실패, b 성공
    table = [[n11, b10], [b01, n00]]
    res = mcnemar(table, exact=True)
    return {
        "n_items": len(common),
        "table_2x2": {"both_correct": n11, "a_only": b10, "b_only": b01, "both_wrong": n00},
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "discordant": b10 + b01,
        "a_tsr_items": sum(A[q]["majority"] for q in common) / max(1, len(common)),
        "b_tsr_items": sum(B[q]["majority"] for q in common) / max(1, len(common)),
    }


# ---------------------------------------------------------------------------
# 부트스트랩 CI
# ---------------------------------------------------------------------------

def _rng():
    import numpy as np
    return np.random.default_rng(SEED)


def bootstrap_log_metric(runs: list[dict], field: str, n_boot: int = N_BOOT) -> dict:
    """log(field) 평균의 부트스트랩 CI (field: total_tokens|latency)."""
    import numpy as np
    vals = np.array([max(1e-9, float(r[field])) for r in runs if r.get(field) is not None])
    if len(vals) == 0:
        return {}
    logv = np.log(vals)
    rng = _rng()
    boots = np.array([np.mean(rng.choice(logv, size=len(logv), replace=True))
                      for _ in range(n_boot)])
    return {
        "mean_log": float(np.mean(logv)),
        "ci95_log": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "geo_mean": float(np.exp(np.mean(logv))),
        "geo_ci95": [float(np.exp(np.percentile(boots, 2.5))),
                     float(np.exp(np.percentile(boots, 97.5)))],
    }


def bootstrap_tps(runs: list[dict], n_boot: int = N_BOOT) -> dict:
    """TPS = sum(total_tokens)/sum(success) 의 부트스트랩 CI (log 스케일 보고)."""
    import numpy as np
    toks = np.array([float(r["total_tokens"]) for r in runs])
    succ = np.array([float(r["success"]) for r in runs])
    n = len(runs)
    if n == 0:
        return {}
    rng = _rng()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = succ[idx].sum()
        if s > 0:
            boots.append(toks[idx].sum() / s)
    boots = np.array(boots)
    point = float(toks.sum() / succ.sum()) if succ.sum() > 0 else None
    out = {"tps": point, "n_boot_valid": len(boots)}
    if len(boots):
        out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out["log_ci95"] = [float(np.percentile(np.log(boots), 2.5)),
                           float(np.percentile(np.log(boots), 97.5))]
    return out


def bootstrap_tps_diff(x_runs: list[dict], base_runs: list[dict], n_boot: int = N_BOOT) -> dict:
    """TPS(x) - TPS(base) 의 부트스트랩 CI. 상한 < 0 이면 x 가 유의하게 낮음."""
    import numpy as np
    def arrs(runs):
        return (np.array([float(r["total_tokens"]) for r in runs]),
                np.array([float(r["success"]) for r in runs]))
    xt, xs = arrs(x_runs)
    bt, bs = arrs(base_runs)
    rng = _rng()
    diffs = []
    for _ in range(n_boot):
        xi = rng.integers(0, len(xt), size=len(xt))
        bi = rng.integers(0, len(bt), size=len(bt))
        if xs[xi].sum() > 0 and bs[bi].sum() > 0:
            diffs.append(xt[xi].sum() / xs[xi].sum() - bt[bi].sum() / bs[bi].sum())
    diffs = np.array(diffs)
    if not len(diffs):
        return {}
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "tps_diff_point": float(np.median(diffs)),
        "ci95": [lo, hi],
        "x_lower_than_base_sig": hi < 0,   # 상한이 0 미만이면 x TPS 가 유의하게 낮음
    }


# ---------------------------------------------------------------------------
# within-run 순효과 (net flip)
# ---------------------------------------------------------------------------

def within_run_net_flip(x_runs: list[dict], base_runs: list[dict]) -> dict:
    X = _item_success_counts(x_runs)
    B = _item_success_counts(base_runs)
    common = sorted(set(X) & set(B))
    inc = dec = same = 0
    for qid in common:
        if X[qid]["count"] > B[qid]["count"]:
            inc += 1
        elif X[qid]["count"] < B[qid]["count"]:
            dec += 1
        else:
            same += 1
    return {"n_items": len(common), "increased": inc, "decreased": dec,
            "same": same, "net_flip": inc - dec}


# ---------------------------------------------------------------------------
# 효과 크기
# ---------------------------------------------------------------------------

def cohens_d_log(x_runs: list[dict], y_runs: list[dict], field: str) -> float | None:
    import numpy as np
    x = np.log([max(1e-9, float(r[field])) for r in x_runs])
    y = np.log([max(1e-9, float(r[field])) for r in y_runs])
    if len(x) < 2 or len(y) < 2:
        return None
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return float((x.mean() - y.mean()) / sp)


def cramers_v_success(structures_runs: dict[str, list[dict]]) -> float | None:
    """구조 × 성공 분할표의 Cramér's V."""
    import numpy as np
    from scipy.stats import chi2_contingency
    structs = list(structures_runs)
    table = []
    for s in structs:
        runs = structures_runs[s]
        ones = sum(r["success"] for r in runs)
        table.append([ones, len(runs) - ones])
    table = np.array(table)
    if table.sum() == 0 or table.shape[0] < 2:
        return None
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    k = min(table.shape) - 1
    if k <= 0:
        return None
    return float(math.sqrt(chi2 / (n * k)))


# ---------------------------------------------------------------------------
# 파레토
# ---------------------------------------------------------------------------

def _structure_point(runs: list[dict]) -> dict:
    n = len(runs)
    succ = sum(r["success"] for r in runs)
    cost = sum(r["cost"] for r in runs)
    return {"TSR": succ / n if n else 0.0,
            "avg_cost": cost / n if n else 0.0,
            "n": n}


def pareto_plot(points: dict[str, dict], out_path: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        print(f"  [analysis] 시각화 라이브러리 없음: {exc}")
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, p in points.items():
        ax.scatter(p["avg_cost"], p["TSR"], s=120)
        ax.annotate(s.upper(), (p["avg_cost"], p["TSR"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=11)
    # S0-S1 단일모델 보간선
    if "s0" in points and "s1" in points:
        xs = [points["s0"]["avg_cost"], points["s1"]["avg_cost"]]
        ys = [points["s0"]["TSR"], points["s1"]["TSR"]]
        ax.plot(xs, ys, "--", color="gray", label="단일모델 기준선 (S0–S1)")
    # 파레토 프론티어 (비용 최소·정확도 최대)
    front = _pareto_frontier(points)
    if len(front) >= 2:
        fx = [points[s]["avg_cost"] for s in front]
        fy = [points[s]["TSR"] for s in front]
        order = sorted(range(len(fx)), key=lambda i: fx[i])
        ax.plot([fx[i] for i in order], [fy[i] for i in order],
                "-", color="crimson", alpha=0.6, label="Pareto frontier")
    ax.set_xlabel("문항당 평균 비용 (USD)")
    ax.set_ylabel("정확도 (TSR)")
    ax.set_title("정확도–비용 파레토 (S0~S4)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _pareto_frontier(points: dict[str, dict]) -> list[str]:
    """비용 낮을수록·TSR 높을수록 우월. 비지배 집합 반환."""
    items = list(points.items())
    front = []
    for s, p in items:
        dominated = False
        for s2, p2 in items:
            if s2 == s:
                continue
            if p2["avg_cost"] <= p["avg_cost"] and p2["TSR"] >= p["TSR"] and \
               (p2["avg_cost"] < p["avg_cost"] or p2["TSR"] > p["TSR"]):
                dominated = True
                break
        if not dominated:
            front.append(s)
    return front


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run_full_analysis() -> dict:
    structures = _available_structures()
    runs = {s: _load_runs(s) for s in structures}
    out: dict = {"structures_present": structures, "n_boot": N_BOOT, "seed": SEED}

    # 구조별 포인트/지표
    out["points"] = {s: _structure_point(runs[s]) for s in structures}
    out["bootstrap"] = {}
    for s in structures:
        out["bootstrap"][s] = {
            "log_token": bootstrap_log_metric(runs[s], "total_tokens"),
            "log_latency": bootstrap_log_metric(runs[s], "latency"),
            "tps": bootstrap_tps(runs[s]),
        }

    base = "s0" if "s0" in structures else None
    out["baseline"] = base

    # H1: S2·S3 TPS vs S0
    out["H1"] = {"baseline": base, "tested": {}, "supported": None}
    if base:
        any_lower = False
        for s in ("s2", "s3"):
            if s in structures:
                diff = bootstrap_tps_diff(runs[s], runs[base])
                out["H1"]["tested"][s] = diff
                if diff.get("x_lower_than_base_sig"):
                    any_lower = True
        out["H1"]["supported"] = any_lower
    else:
        out["H1"]["note"] = "S0 baseline 미실행 — --run-s0 로 baseline 실행 후 H1 검정 가능"

    # H2: S4 vs S0 TSR McNemar + 과업유형 분해
    out["H2"] = {"baseline": base}
    if base and "s4" in structures:
        out["H2"]["mcnemar_overall"] = mcnemar_test(runs["s4"], runs[base])
        out["H2"]["within_run_net_flip"] = within_run_net_flip(runs["s4"], runs[base])
        bytt = {}
        task_types = sorted({r.get("task_type") for r in runs["s4"]})
        for tt in task_types:
            s4_tt = [r for r in runs["s4"] if r.get("task_type") == tt]
            s0_tt = [r for r in runs[base] if r.get("task_type") == tt]
            if s4_tt and s0_tt:
                bytt[tt] = {
                    "mcnemar": mcnemar_test(s4_tt, s0_tt),
                    "net_flip": within_run_net_flip(s4_tt, s0_tt),
                }
        out["H2"]["by_task_type"] = bytt
        out["H2"]["note_T4"] = "T4(다단계 조건추론) 분해 결과는 by_task_type['T4'] 참조"
        out["H2"]["error_type_note"] = (
            "E2(계산오류)·E3(누적오류) 감소로 메커니즘 확인은 저자 수기 분류(failures.csv) 후 수행")
    else:
        out["H2"]["note"] = "S0 baseline 또는 S4 미실행 — 두 구조 모두 필요"

    # 쌍별 McNemar (참고)
    out["pairwise_mcnemar"] = {}
    for i, a in enumerate(structures):
        for b in structures[i + 1:]:
            out["pairwise_mcnemar"][f"{a}_vs_{b}"] = mcnemar_test(runs[a], runs[b])

    # 효과 크기
    out["effect_size"] = {"cramers_v_success": cramers_v_success(runs)}
    if base:
        out["effect_size"]["cohens_d_log_token_vs_s0"] = {
            s: cohens_d_log(runs[s], runs[base], "total_tokens")
            for s in structures if s != base}

    # 파레토
    analysis_dir = config.RESULTS_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    png = pareto_plot(out["points"], analysis_dir / "pareto.png")
    out["pareto"] = {
        "frontier": _pareto_frontier(out["points"]),
        "plot": str(png) if png else None,
    }

    with open(analysis_dir / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return out


def final_report_text(analysis: dict) -> str:
    L = ["=== 최종 종합 리포트 ===", ""]
    L.append(f"실행된 구조: {', '.join(s.upper() for s in analysis['structures_present'])}")
    L.append(f"부트스트랩 n={analysis['n_boot']}, seed={analysis['seed']}")
    L.append("")
    L.append("[구조별 정확도·비용]")
    for s, p in analysis["points"].items():
        tps = analysis["bootstrap"][s]["tps"].get("tps")
        L.append(f"  {s.upper()}: TSR {p['TSR']:.3f}, 평균비용 ${p['avg_cost']:.5f}, "
                 f"TPS {tps:.0f}" if tps else
                 f"  {s.upper()}: TSR {p['TSR']:.3f}, 평균비용 ${p['avg_cost']:.5f}, TPS N/A")
    L.append("")
    h1 = analysis["H1"]
    L.append(f"[H1] baseline={h1.get('baseline')} supported={h1.get('supported')}")
    for s, d in (h1.get("tested") or {}).items():
        L.append(f"  {s.upper()} TPS-S0 diff CI95={d.get('ci95')} "
                 f"유의저하={d.get('x_lower_than_base_sig')}")
    if h1.get("note"):
        L.append(f"  note: {h1['note']}")
    L.append("")
    h2 = analysis["H2"]
    L.append(f"[H2] baseline={h2.get('baseline')}")
    if "mcnemar_overall" in h2:
        m = h2["mcnemar_overall"]
        L.append(f"  S4 vs S0 McNemar p={m['pvalue']:.4g} "
                 f"(S4 항목TSR {m['a_tsr_items']:.3f} vs S0 {m['b_tsr_items']:.3f})")
        nf = h2["within_run_net_flip"]
        L.append(f"  within-run net flip: +{nf['increased']}/-{nf['decreased']} "
                 f"(net {nf['net_flip']}, same {nf['same']})")
        if "T4" in (h2.get("by_task_type") or {}):
            t4 = h2["by_task_type"]["T4"]["mcnemar"]
            L.append(f"  [T4] McNemar p={t4['pvalue']:.4g} "
                     f"(S4 {t4['a_tsr_items']:.3f} vs S0 {t4['b_tsr_items']:.3f})")
    if h2.get("note"):
        L.append(f"  note: {h2['note']}")
    L.append("")
    L.append(f"[파레토 프론티어] {', '.join(s.upper() for s in analysis['pareto']['frontier'])}")
    L.append(f"효과크기 Cramér's V(성공): {analysis['effect_size'].get('cramers_v_success')}")
    L.append("")
    L.append("오류유형(E1~E5) 분류는 저자 수기 작업: results/sX/failures.csv 참조")
    L.append("상세: results/analysis/analysis.json, results/analysis/pareto.png")
    return "\n".join(L)


if __name__ == "__main__":
    res = run_full_analysis()
    print(final_report_text(res))
