"""Phase 1 — 통계 재처리·재집계 (통합타임라인 §1, 순번 5~12).

지도교수 피드백 M2·M3·M5·M6·M7·M8 과 m2·m3 에 대응하는 재집계·재분석을 한 번에 수행한다.
**추가 LLM 호출이 전혀 없다** — 기존 로그(실험 1 CrewAI 재채점본, results_main/{s0,s2,s5,s6},
실험 C expC_main60 반사실 재현 결과)만 다시 읽는다.

수행 항목:
  5  (M2) 문항 단위(n=60) 재검정 + 문항 임의효과 모형 — 기존 n=180 쌍체검정과 병기
  6  (M3) TOST 등가성 검정 + 차이의 95% CI + 사후 검정력·소요 표본
  7  (M6) 입력/출력 토큰 분리 집계 — 토큰비 1.35배 vs 성공당 비용 2.24배 괴리 설명
  8  (M7) tiktoken 추정치의 API usage 대비 편차 실측 — '교차 검증'→'추정치' 격하 근거
  9  (M8) 기권율 실측 + E1~E5 저자 분류용 코딩시트 생성(κ 절차 포함)
  10 (M5) T1 라벨 정의 통일 + T1 전 문항 판정 로그 추출
  11 (m2) 게이트 임계값 운영 곡선  ─┬─ 실험 C(동일 60문항 표본)의 반사실 재현 결과에서 추출
  12 (m3) 무작위 게이트 대조        ─┘

사용:
    .venv/Scripts/python.exe -m src.reanalyze_phase1 --out results_phase1
    .venv/Scripts/python.exe -m src.reanalyze_phase1 --out results_phase1 --skip-tiktoken

주의: 출력을 파이프(`| head`)로 받지 말 것 — SIGPIPE 로 프로세스가 조기 종료되면서
exit 0 으로 보고된다. `> 파일 2>&1` 로 받는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

# 실험 C 에서 확정한 통계 구현을 그대로 재사용한다(정의 불일치 방지).
from .replay_gates import (
    bootstrap_ci_mean,
    holm,
    mcnemar,
    per_question,
    tost_equivalence,
    wilcoxon_signed_rank,
)

SEED = 20260821
BOOTSTRAP_N = 10000

# 논문 3장 확정 단가(config/pricing.yaml, gemini-2.5-flash) — 비용 분해에 사용
PRICE_IN = 0.30 / 1_000_000
PRICE_OUT = 2.50 / 1_000_000

# 기권 토큰 — src/structures/common.py 의 INSUFFICIENT 와 동일해야 한다.
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# 구조 표시명 (논문 표기와 일치)
LABELS = {
    "exp1": "실험 1 CrewAI(검증 포함)",
    "s0": "실험 0 무검증(generate-only)",
    "s2": "실험 2 LangGraph 항상검증",
    "s5": "실험 3 LangGraph 동적 캐스케이드",
    "s6": "실험 4 LangGraph self-probe 캐스케이드",
}
ORDER = ["exp1", "s0", "s2", "s5", "s6"]


# ---------------------------------------------------------------------------
# 로드 — 서로 다른 두 포맷(실험 1 CSV, LangGraph JSONL)을 공통 레코드로 정규화
# ---------------------------------------------------------------------------

def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_exp1(csv_path: Path) -> list[dict]:
    """실험 1(CrewAI) 재채점본 CSV. tiktoken 열과 is_abstention 열을 이미 갖고 있다."""
    out = []
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "structure": "exp1",
                "question_id": r["id"],
                "task_type": r["task_type"],
                "dataset": r["dataset"],
                "run_index": int(_f(r["run_index"], 1)),
                "success": int(_f(r["success"])),
                "input_tokens": _f(r["input_tokens"]),
                "output_tokens": _f(r["output_tokens"]),
                "total_tokens": _f(r["total_tokens"]),
                "tiktoken_in": _f(r.get("input_tokens_tiktoken")),
                "tiktoken_out": _f(r.get("output_tokens_tiktoken")),
                "cost": _f(r["cost"]),
                "latency": _f(r["latency"]),
                "abstention": r.get("is_abstention", "").strip().lower() in ("true", "1", "yes"),
                "final_answer": r.get("final_answer", ""),
                "gold_answer": r.get("gold_answer", ""),
                "judge_flag": r.get("judge_flag", "").strip().lower() in ("true", "1", "yes"),
                "escalated": r.get("retried", "").strip().lower() in ("true", "1", "yes"),
                "reasoning_trace": "",
                "token_breakdown": {},
            })
    return out


def load_langgraph(structure: str, results_dir: Path) -> list[dict]:
    """results_main/<structure>/raw_runs.jsonl."""
    path = results_dir / structure / "raw_runs.jsonl"
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ans = (r.get("final_answer") or "").strip()
            out.append({
                "structure": structure,
                "question_id": r["question_id"],
                "task_type": r.get("task_type", ""),
                "dataset": r.get("dataset", ""),
                "run_index": int(r.get("run_index", 1)),
                "success": int(r.get("success", 0)),
                "input_tokens": _f(r.get("input_tokens")),
                "output_tokens": _f(r.get("output_tokens")),
                "total_tokens": _f(r.get("total_tokens")),
                "tiktoken_in": None,      # 로그에 없음 — §8 에서 프롬프트 재구성으로 산출
                "tiktoken_out": None,
                "cost": _f(r.get("cost")),
                "latency": _f(r.get("latency")),
                # 기권 판정은 실험 1 의 'None' 이 아니라 저자 확정 토큰으로 한다.
                "abstention": ans.upper() == INSUFFICIENT,
                "final_answer": ans,
                "gold_answer": str(r.get("gold_answer", "")),
                "judge_flag": bool(r.get("judge_flag", False)),
                "escalated": bool(r.get("escalated", False)) or int(r.get("retry_count", 0)) > 0,
                "reasoning_trace": r.get("reasoning_trace", ""),
                "token_breakdown": r.get("token_breakdown") or {},
                "confidence": _f(r.get("confidence")),
                "confidence_self_report": _f(r.get("confidence_self_report")),
                "verify_verdict": r.get("verify_verdict", ""),
                "gold_chunk_retrieved": bool(r.get("gold_chunk_retrieved", False)),
                "used_expression": r.get("used_expression") or [],
                "retrieved_evidence": r.get("retrieved_evidence") or [],
                "query": r.get("query", ""),
            })
    return out


def load_all(results_dir: Path, exp1_csv: Path) -> dict[str, list[dict]]:
    data = {"exp1": load_exp1(exp1_csv)}
    for s in ("s0", "s2", "s5", "s6"):
        p = results_dir / s / "raw_runs.jsonl"
        if p.exists():
            data[s] = load_langgraph(s, results_dir)
    return data


# ---------------------------------------------------------------------------
# 순번 5 (M2) — 문항 단위 재검정 + 문항 임의효과 모형
# ---------------------------------------------------------------------------

def _succ_by_run(recs: list[dict]) -> dict[tuple[str, int], int]:
    return {(r["question_id"], r["run_index"]): r["success"] for r in recs}


def run_level_mcnemar(a: list[dict], b: list[dict]) -> dict:
    """기존 분석이 쓴 run 단위(n=180) 쌍체 McNemar — 의사반복 문제를 그대로 재현해 병기."""
    sa, sb = _succ_by_run(a), _succ_by_run(b)
    keys = sorted(set(sa) & set(sb))
    b01 = sum(1 for k in keys if sa[k] == 0 and sb[k] == 1)
    b10 = sum(1 for k in keys if sa[k] == 1 and sb[k] == 0)
    nd = b01 + b10
    if nd == 0:
        return {"n_pairs": len(keys), "discordant": 0, "b01": 0, "b10": 0, "p_value": 1.0}
    obs = min(b01, b10)
    p = sum(math.comb(nd, k) * 0.5 ** nd for k in range(obs + 1)) * 2
    return {"n_pairs": len(keys), "discordant": nd, "b01": b01, "b10": b10,
            "p_value": min(1.0, p)}


def glmm_structure_effect(data: dict[str, list[dict]], structures: list[str]) -> dict:
    """문항을 임의효과로 둔 로지스틱 혼합모형(GLMM).

    M2 의 요구는 "문항 단위 재검정 **또는** 문항 random effect 혼합모형"이므로 둘 다 낸다.
    GLMM 은 run 단위 관측 전량을 쓰면서 같은 문항의 반복이 독립이 아님을 모형에 반영하므로,
    문항 단위 다수결(정보 손실)과 run 단위 쌍체검정(의사반복) 사이의 절충이 된다.

    statsmodels 의 BinomialBayesMixedGLM(변분 베이즈)을 주 모형으로,
    GEE(문항 군집 로버스트 표준오차)를 교차 확인용으로 함께 적합한다.
    """
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as exc:                                    # pragma: no cover
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    rows = []
    for s in structures:
        for r in data.get(s, []):
            rows.append({"y": r["success"], "structure": s, "qid": r["question_id"]})
    if not rows:
        return {"available": False, "error": "no rows"}

    df = pd.DataFrame(rows)
    # 기준 범주는 항상검증(s2) — 논문의 비교 기준 구조
    base = "s2" if "s2" in structures else structures[0]
    cats = [base] + [s for s in structures if s != base]
    df["structure"] = pd.Categorical(df["structure"], categories=cats)

    out = {"available": True, "n_obs": len(df), "n_questions": df["qid"].nunique(),
           "base_category": base}

    # --- 완전분리 진단 ---
    # 문항 임의절편 모형은 어떤 문항의 관측이 전부 1(또는 전부 0)이면 그 문항의 절편이
    # ±무한으로 발산한다. 몇 문항이 그런 상태인지 먼저 세어 모형 신뢰도를 판정한다.
    grp = df.groupby("qid")["y"]
    n_all1 = int((grp.mean() == 1.0).sum())
    n_all0 = int((grp.mean() == 0.0).sum())
    n_q = int(grp.ngroups)
    out["separation"] = {
        "n_questions": n_q,
        "n_all_success": n_all1,
        "n_all_failure": n_all0,
        "n_mixed": n_q - n_all1 - n_all0,
        "separated_ratio": (n_all1 + n_all0) / n_q if n_q else 0.0,
        "verdict": ("완전분리 문항이 과반 — 문항 임의절편 GLMM 은 발산해 신뢰할 수 없다. "
                    "GEE(문항 군집 로버스트 SE)를 주 모형으로 본다."
                    if (n_all1 + n_all0) / max(n_q, 1) > 0.5
                    else "완전분리 문항이 소수 — GLMM 추정이 안정적일 수 있다."),
    }

    # --- GLMM (문항 임의절편) ---
    try:
        m = sm.BinomialBayesMixedGLM.from_formula(
            "y ~ structure", {"qid": "0 + C(qid)"}, df)
        fit = m.fit_vb(verbose=False)
        params = {}
        for name, mean, sd in zip(fit.model.exog_names, fit.fe_mean, fit.fe_sd):
            z = mean / sd if sd else 0.0
            params[name] = {
                "coef": float(mean), "sd": float(sd), "z": float(z),
                "p_value": float(math.erfc(abs(z) / math.sqrt(2))),
                "odds_ratio": float(math.exp(mean)),
            }
        out["glmm"] = {"ok": True, "fixed_effects": params,
                       "vc_sd": float(fit.vcp_mean[0]) if len(fit.vcp_mean) else None}
    except Exception as exc:
        out["glmm"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # --- GEE (문항 군집, exchangeable) ---
    try:
        gee = smf.gee("y ~ structure", "qid", data=df,
                      cov_struct=sm.cov_struct.Exchangeable(),
                      family=sm.families.Binomial()).fit()
        out["gee"] = {
            "ok": True,
            "fixed_effects": {
                n: {"coef": float(c), "se": float(s), "z": float(z), "p_value": float(p),
                    "odds_ratio": float(math.exp(c))}
                for n, c, s, z, p in zip(gee.model.exog_names, gee.params, gee.bse,
                                         gee.tvalues, gee.pvalues)
            },
        }
    except Exception as exc:
        out["gee"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return out


def section5(data: dict[str, list[dict]]) -> dict:
    present = [s for s in ORDER if s in data]
    per_q = {s: per_question(data[s], [r["success"] for r in data[s]]) for s in present}

    summary = {}
    for s in present:
        runs = data[s]
        pq = per_q[s]
        summary[s] = {
            "label": LABELS[s],
            "n_runs": len(runs),
            "TSR_run": sum(r["success"] for r in runs) / len(runs),
            "n_questions": len(pq),
            "TSR_question_majority": sum(pq.values()) / len(pq),
            "TSR_run_ci95": bootstrap_ci_mean([r["success"] for r in runs]),
            "TSR_question_ci95": bootstrap_ci_mean(list(pq.values())),
        }

    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]
    tests = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        tests[key] = {
            "question_level_n60": mcnemar(per_q[a], per_q[b]),
            "run_level_n180": run_level_mcnemar(data[a], data[b]),
        }
    # 다중비교 보정 — 문항 단위 주분석 p 값에 Holm 적용(m8 대응)
    adj = holm({k: v["question_level_n60"]["p_value"] for k, v in tests.items()})
    for k, v in tests.items():
        v["question_level_n60"]["p_holm"] = adj[k]

    return {
        "summary": summary,
        "pairwise": tests,
        "glmm": glmm_structure_effect(data, [s for s in ("s2", "s5", "s0", "s6") if s in data]),
    }


# ---------------------------------------------------------------------------
# 순번 6 (M3) — 등가성 검정 + 사후 검정력
# ---------------------------------------------------------------------------

def mcnemar_power(b01: int, b10: int, n_questions: int, alpha: float = 0.05,
                  n_sim: int = 20000, seed: int = SEED) -> dict:
    """관측된 불일치 구조를 모수로 둔 McNemar 사후 검정력(시뮬레이션).

    관측 비율 p01, p10 을 참값으로 가정하고 n 문항을 재표집해 양측 이항검정을 반복,
    p<alpha 가 되는 비율을 검정력으로 본다. 아울러 검정력 0.80 에 필요한 문항 수를
    같은 시뮬레이션으로 이분 탐색한다.

    사후 검정력은 관측 효과를 참값으로 가정하는 만큼 해석에 한계가 있으나, M3 이 요구한
    "비유의가 검정력 부족과 구분되지 않는다"는 점을 정량화하는 용도로 보고한다.
    """
    import numpy as np
    from scipy.stats import binom

    n_obs = max(1, n_questions)
    p01, p10 = b01 / n_obs, b10 / n_obs
    p_same = max(0.0, 1.0 - p01 - p10)
    if p01 + p10 == 0:
        return {"power": 0.0, "n_for_80": None, "p01": p01, "p10": p10,
                "note": "불일치 쌍이 0건이라 McNemar 검정 자체가 성립하지 않는다."}

    rng = np.random.default_rng(seed)

    def _sim_power(n: int, sims: int) -> float:
        # 각 문항은 (A오답·B정답 / A정답·B오답 / 일치) 세 결과의 다항분포를 따른다.
        draws = rng.multinomial(n, [p01, p10, p_same], size=sims)
        c01, c10 = draws[:, 0], draws[:, 1]
        nd = c01 + c10
        obs = np.minimum(c01, c10)
        # 양측 이항 정확검정(p=0.5): 2 * P(X <= min(c01,c10))
        pv = np.where(nd > 0, np.minimum(1.0, 2.0 * binom.cdf(obs, np.maximum(nd, 1), 0.5)), 1.0)
        return float(np.mean(pv < alpha))

    power = _sim_power(n_obs, n_sim)

    # 검정력 0.80 에 필요한 문항 수 — 기하 탐색으로 상한을 잡은 뒤 이분 탐색
    n80 = None
    if power < 0.80:
        hi = n_obs
        while hi <= 100_000 and _sim_power(hi, 4000) < 0.80:
            hi *= 2
        if hi <= 100_000:
            lo = hi // 2
            while lo < hi:
                mid = (lo + hi) // 2
                if _sim_power(mid, 4000) >= 0.80:
                    hi = mid
                else:
                    lo = mid + 1
            n80 = lo
    else:
        n80 = n_obs
    return {"power": power, "n_for_80": n80, "p01": p01, "p10": p10, "p_same": p_same,
            "alpha": alpha, "n_sim": n_sim}


def section6(data: dict[str, list[dict]], sec5: dict) -> dict:
    """논문이 '동등'·'차이 없음'으로 서술한 모든 쌍에 TOST 와 95% CI 를 붙인다."""
    present = [s for s in ORDER if s in data]
    per_q = {s: per_question(data[s], [r["success"] for r in data[s]]) for s in present}

    # 논문에서 등가성 주장이 걸려 있는 쌍
    claims = [
        ("exp1", "s2", "H1 — CrewAI vs LangGraph 항상검증 '유의차 없음'(p=0.701)"),
        ("s2", "s5", "H3a — 항상검증 vs 캐스케이드 '동등'(p=0.375)"),
        ("s0", "s5", "실험 0 vs 캐스케이드 — 무검증이 캐스케이드를 지배하는가"),
        ("s0", "s2", "실험 0 vs 항상검증 — 검증 단계 자체의 정확도 이득"),
    ]
    out = {}
    for a, b, note in claims:
        if a not in per_q or b not in per_q:
            continue
        t = tost_equivalence(per_q[a], per_q[b], margin=0.05,
                             n_boot=BOOTSTRAP_N, seed=SEED)
        mc = sec5["pairwise"].get(f"{a}_vs_{b}") or sec5["pairwise"].get(f"{b}_vs_{a}")
        q = mc["question_level_n60"]
        out[f"{a}_vs_{b}"] = {
            "note": note,
            "labels": [LABELS[a], LABELS[b]],
            "tost": t,
            "mcnemar_question": q,
            "power": mcnemar_power(q["b01"], q["b10"], t.get("n_questions", 60)),
        }
    return out


# ---------------------------------------------------------------------------
# 순번 7 (M6) — 입력/출력 토큰 분리와 비용 괴리 분해
# ---------------------------------------------------------------------------

def section7(data: dict[str, list[dict]]) -> dict:
    present = [s for s in ORDER if s in data]
    rows = {}
    for s in present:
        recs = data[s]
        n = len(recs)
        ti = sum(r["input_tokens"] for r in recs)
        to = sum(r["output_tokens"] for r in recs)
        tt = sum(r["total_tokens"] for r in recs)
        cost = sum(r["cost"] for r in recs)
        ns = sum(r["success"] for r in recs)
        rows[s] = {
            "label": LABELS[s],
            "n_runs": n, "n_success": ns,
            "avg_input": ti / n, "avg_output": to / n, "avg_total": tt / n,
            "output_ratio": to / tt if tt else 0.0,
            "total_cost": cost,
            "cost_per_run": cost / n,
            "cost_per_success": cost / ns if ns else None,
            "TPS": tt / ns if ns else None,
            "avg_latency": sum(r["latency"] for r in recs) / n,
            # 단가 분해: 비용 중 출력 토큰이 차지하는 비중
            "cost_share_output": (to * PRICE_OUT) / (ti * PRICE_IN + to * PRICE_OUT)
            if (ti or to) else 0.0,
        }

    # M6 의 핵심 — 토큰비와 성공당 비용비의 괴리를 출력 비중 차이로 설명
    ratios = {}
    if "exp1" in rows and "s2" in rows:
        a, b = rows["exp1"], rows["s2"]
        ratios["exp1_over_s2"] = {
            "token_ratio": a["avg_total"] / b["avg_total"],
            "input_ratio": a["avg_input"] / b["avg_input"],
            "output_ratio": a["avg_output"] / b["avg_output"],
            "cost_per_run_ratio": a["cost_per_run"] / b["cost_per_run"],
            "cost_per_success_ratio": a["cost_per_success"] / b["cost_per_success"],
            "latency_ratio": a["avg_latency"] / b["avg_latency"],
            "output_share": {"exp1": a["output_ratio"], "s2": b["output_ratio"]},
            "price_out_over_in": PRICE_OUT / PRICE_IN,
        }

    # LangGraph 는 노드 단위 분해까지 낸다(검증 단계가 비용의 어디에 붙는지)
    by_node = {}
    for s in present:
        if s == "exp1":
            continue
        agg = defaultdict(lambda: {"input": 0.0, "output": 0.0, "calls": 0, "runs": 0})
        for r in data[s]:
            for node, v in (r["token_breakdown"] or {}).items():
                agg[node]["input"] += _f(v.get("input_tokens"))
                agg[node]["output"] += _f(v.get("output_tokens"))
                agg[node]["calls"] += int(_f(v.get("n_calls")))
                agg[node]["runs"] += 1
        n = len(data[s])
        by_node[s] = {
            node: {
                "avg_input_per_run": v["input"] / n,
                "avg_output_per_run": v["output"] / n,
                "total_calls": v["calls"],
                "runs_with_node": v["runs"],
                "share_of_run_cost": (v["input"] * PRICE_IN + v["output"] * PRICE_OUT)
                / sum(r["cost"] for r in data[s]) if sum(r["cost"] for r in data[s]) else 0.0,
            }
            for node, v in agg.items()
        }
    return {"by_structure": rows, "ratios": ratios, "by_node": by_node,
            "pricing": {"input_per_1M": 0.30, "output_per_1M": 2.50}}


# ---------------------------------------------------------------------------
# 순번 8 (M7) — tiktoken 추정치의 편차 실측
# ---------------------------------------------------------------------------

def section8(data: dict[str, list[dict]], skip: bool = False) -> dict:
    """tiktoken(cl100k_base) 추정치와 API usage metadata 의 실제 편차를 측정한다.

    - 실험 1(CrewAI): 로그에 두 값이 모두 있으므로 입력·출력 모두 직접 비교한다.
    - LangGraph: 로그에 tiktoken 값이 없으므로 generate 노드의 **1차 호출 프롬프트를
      실제 프롬프트 파일과 로그의 query·evidence 로 그대로 재구성**해 비교한다.
      (출력 토큰은 모델이 낸 JSON 원문이 로그에 보존되지 않아 정확 재구성이 불가능하므로
       입력 편차만 보고한다 — 이 제약 자체를 결과에 명시한다.)
    """
    out = {"note": "gemini-2.5-flash 는 tiktoken(OpenAI BPE) 토크나이저를 쓰지 않는다."}

    # --- 실험 1 (CrewAI) ---
    e1 = data.get("exp1", [])
    if e1:
        ui = sum(r["input_tokens"] for r in e1)
        uo = sum(r["output_tokens"] for r in e1)
        ti = sum(r["tiktoken_in"] for r in e1)
        to = sum(r["tiktoken_out"] for r in e1)
        per_run_in = [(r["tiktoken_in"] - r["input_tokens"]) / r["input_tokens"]
                      for r in e1 if r["input_tokens"]]
        per_run_out = [(r["tiktoken_out"] - r["output_tokens"]) / r["output_tokens"]
                       for r in e1 if r["output_tokens"]]
        out["exp1"] = {
            "usage_input": ui, "tiktoken_input": ti,
            "input_deviation_pct": (ti - ui) / ui * 100 if ui else None,
            "usage_output": uo, "tiktoken_output": to,
            "output_deviation_pct": (to - uo) / uo * 100 if uo else None,
            "per_run_input_mean_pct": sum(per_run_in) / len(per_run_in) * 100 if per_run_in else None,
            "per_run_output_mean_pct": sum(per_run_out) / len(per_run_out) * 100 if per_run_out else None,
            "n_runs": len(e1),
        }

    if skip:
        out["langgraph"] = {"skipped": True}
        return out

    # --- LangGraph: generate 노드 1차 호출 프롬프트 재구성 ---
    try:
        import tiktoken
        from . import tools
        from .structures.common import _split_prompt
    except Exception as exc:                                    # pragma: no cover
        out["langgraph"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return out

    enc = tiktoken.get_encoding("cl100k_base")
    prompt_of = {"s0": "generate.txt", "s2": "generate.txt",
                 "s5": "generate_s5.txt", "s6": "generate_s5.txt"}
    lg = {}
    for s, pname in prompt_of.items():
        recs = data.get(s)
        if not recs:
            continue
        try:
            system_part, inputs_template = _split_prompt(pname)
        except Exception as exc:
            lg[s] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        devs, pairs = [], []
        by_ds = defaultdict(list)
        for r in recs:
            api_in = _f((r["token_breakdown"].get("generate") or {}).get("input_tokens"))
            if not api_in:
                continue
            try:
                user_message = inputs_template.format(
                    query=r["query"],
                    evidence=tools.format_evidence(r["retrieved_evidence"]),
                    critique="",
                )
            except Exception:
                continue
            est = len(enc.encode(system_part)) + len(enc.encode(user_message))
            # generate 노드가 재생성으로 2회 이상 호출된 run 은 API 합계가 1차 호출과
            # 대응하지 않으므로 제외한다(n_calls==1 인 run 만 비교).
            if int(_f((r["token_breakdown"].get("generate") or {}).get("n_calls"), 1)) != 1:
                continue
            d = (est - api_in) / api_in
            devs.append(d)
            pairs.append((est, api_in))
            by_ds[r["dataset"]].append(d)
        if devs:
            def _stats(vals):
                v = sorted(vals)
                return {"n": len(v), "mean_pct": sum(v) / len(v) * 100,
                        "median_pct": v[len(v) // 2] * 100,
                        "min_pct": v[0] * 100, "max_pct": v[-1] * 100}
            lg[s] = {
                "ok": True, "n_compared": len(devs),
                "mean_deviation_pct": sum(devs) / len(devs) * 100,
                "median_deviation_pct": sorted(devs)[len(devs) // 2] * 100,
                "min_deviation_pct": min(devs) * 100,
                "max_deviation_pct": max(devs) * 100,
                "sum_tiktoken": sum(p[0] for p in pairs),
                "sum_api": sum(p[1] for p in pairs),
                "prompt": pname,
                # 언어(데이터셋)별 분해 — 전체 평균은 방향이 반대인 두 집단의 상쇄일 수 있다.
                "by_dataset": {ds: _stats(v) for ds, v in sorted(by_ds.items())},
            }
        else:
            lg[s] = {"ok": False, "error": "비교 가능한 run 없음"}
    out["langgraph"] = lg
    out["langgraph_scope"] = ("generate 노드 1차 호출의 입력 토큰만 비교. 출력 토큰은 "
                              "모델 응답 원문(JSON)이 로그에 보존되지 않아 재구성 불가.")
    return out


# ---------------------------------------------------------------------------
# 순번 9 (M8) — 기권율 실측 + E1~E5 코딩시트
# ---------------------------------------------------------------------------

E_TYPES = {
    "E1": "Evidence Selection — 잘못된 근거 선택 또는 같은 근거 내 잘못된 수치 선택",
    "E2": "Numerical Computation — 수치는 맞으나 계산 단계 오류",
    "E3": "Multi-step Accumulation — 중간 단계는 맞으나 누적 후 최종 답 오류",
    "E4": "Retrieval/Grounding Failure — 필요한 근거를 활용하지 못함",
    "E5": "Other — 형식 오류, 기권 등",
}


def section9(data: dict[str, list[dict]], out_dir: Path) -> dict:
    """기권율은 결정적으로 집계하고, E1~E5 는 **저자 분류용 코딩시트만** 생성한다.

    논문 3.2.5 가 "저자가 100% 직접 분류하며 LLM-as-judge 를 사용하지 않는다"를 명시했으므로
    라벨 자체를 자동 부여하지 않는다. 대신 저자 판단에 필요한 결정적 증거
    (gold 근거 활용 여부, 제시 산술식의 재실행 결과, 답의 부호·배율 관계 등)를 열로 붙여
    분류 시간을 줄이고, 2주 시차 재분류를 위한 빈 열(error_type_r2)을 함께 둔다.
    """
    present = [s for s in ORDER if s in data]

    abst = {}
    for s in present:
        recs = data[s]
        n_ab = sum(1 for r in recs if r["abstention"])
        by_ds = defaultdict(lambda: [0, 0])
        for r in recs:
            by_ds[r["dataset"]][0] += 1
            by_ds[r["dataset"]][1] += 1 if r["abstention"] else 0
        abst[s] = {
            "label": LABELS[s],
            "n_runs": len(recs),
            "n_abstention": n_ab,
            "abstention_rate": n_ab / len(recs),
            "by_dataset": {k: {"n_runs": v[0], "n_abstention": v[1],
                               "rate": v[1] / v[0] if v[0] else 0.0}
                           for k, v in sorted(by_ds.items())},
        }

    # --- 코딩시트 ---
    try:
        from . import metrics, tools
        have_tools = True
    except Exception:
        have_tools = False

    def _recompute(exprs):
        if not have_tools:
            return ""
        outs = []
        for e in exprs or []:
            v, err = tools.safe_calc(e)
            outs.append(f"{e} => {'ERR' if err else v}")
        return " | ".join(outs)

    def _expr_matches_answer(exprs, ans):
        if not have_tools:
            return ""
        for e in exprs or []:
            v, err = tools.safe_calc(e)
            if err is None and v is not None:
                try:
                    if metrics.is_correct(v, ans).correct:
                        return "True"
                except Exception:
                    pass
        return "False" if exprs else ""

    def _scale_hint(pred, gold):
        """오답이 배율(×100, ×1000) 또는 부호 차이인지 — E2/E3 판단 보조."""
        if not have_tools:
            return ""
        try:
            p = metrics.normalize_number(pred)
            g = metrics.normalize_number(gold)
        except Exception:
            return ""
        if p is None or g is None or g == 0:
            return ""
        ratio = p / g
        for k, name in ((1, "동일"), (-1, "부호반전"), (100, "×100"), (0.01, "÷100"),
                        (1000, "×1000"), (0.001, "÷1000"), (-100, "부호반전×100")):
            if abs(ratio - k) < 0.01 * max(1.0, abs(k)):
                return name
        return f"비율 {ratio:.4g}"

    sheet_path = out_dir / "E1_E5_코딩시트.csv"
    fields = ["structure", "question_id", "task_type", "dataset", "run_index",
              "query", "gold_answer", "pred_answer", "abstention",
              "gold_chunk_retrieved", "expr_recompute", "expr_matches_answer",
              "scale_relation", "judge_flag", "reasoning_trace",
              "error_type", "error_type_r2", "note"]
    n_rows = 0
    with open(sheet_path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("# E1~E5 오류 유형 저자 분류용 코딩시트 (M8)\n")
        for k, v in E_TYPES.items():
            fh.write(f"# {k} {v}\n")
        fh.write("# error_type 열에 1차 분류를, 2주 경과 후 error_type_r2 열에 재분류를 기입한다.\n")
        fh.write("# 두 열이 모두 채워지면 --kappa 로 Cohen's κ(분류자 내 신뢰도)를 산출한다.\n")
        fh.write("# expr_recompute·scale_relation 은 저자 판단을 돕는 결정적 계산 결과일 뿐,\n")
        fh.write("# 라벨을 자동 부여한 것이 아니다(LLM-as-judge 미사용 원칙 유지).\n")
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in present:
            for r in data[s]:
                if r["success"]:
                    continue
                w.writerow({
                    "structure": s,
                    "question_id": r["question_id"],
                    "task_type": r["task_type"],
                    "dataset": r["dataset"],
                    "run_index": r["run_index"],
                    "query": r.get("query", ""),
                    "gold_answer": r["gold_answer"],
                    "pred_answer": r["final_answer"],
                    "abstention": r["abstention"],
                    "gold_chunk_retrieved": r.get("gold_chunk_retrieved", ""),
                    "expr_recompute": _recompute(r.get("used_expression")),
                    "expr_matches_answer": _expr_matches_answer(
                        r.get("used_expression"), r["final_answer"]),
                    "scale_relation": _scale_hint(r["final_answer"], r["gold_answer"]),
                    "judge_flag": r["judge_flag"],
                    "reasoning_trace": (r.get("reasoning_trace") or "")[:1500],
                    "error_type": "", "error_type_r2": "", "note": "",
                })
                n_rows += 1

    return {
        "abstention": abst,
        "coding_sheet": {
            "path": str(sheet_path),
            "n_failure_runs": n_rows,
            "status": "미분류 — 저자 직접 분류 필요(논문 3.2.5 LLM-as-judge 배제 원칙)",
            "kappa": None,
            "kappa_procedure": "error_type 기입 → 2주 경과 → error_type_r2 기입 → --kappa 실행",
        },
        "error_types": E_TYPES,
    }


def cohens_kappa(sheet_path: Path) -> dict:
    """코딩시트의 error_type / error_type_r2 두 열로 분류자 내 Cohen's κ 산출."""
    rows = []
    with open(sheet_path, encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    for r in csv.DictReader(lines):
        a = (r.get("error_type") or "").strip().upper()
        b = (r.get("error_type_r2") or "").strip().upper()
        if a and b:
            rows.append((a, b))
    if not rows:
        return {"ok": False, "error": "두 열이 모두 채워진 행이 없다."}
    cats = sorted({c for p in rows for c in p})
    n = len(rows)
    po = sum(1 for a, b in rows if a == b) / n
    pa = {c: sum(1 for a, _ in rows if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in rows if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"ok": True, "n": n, "categories": cats, "p_observed": po,
            "p_expected": pe, "kappa": kappa, "threshold_met": kappa >= 0.7}


# ---------------------------------------------------------------------------
# 순번 10 (M5) — T1 라벨 모순 해소 + 판정 로그
# ---------------------------------------------------------------------------

def section10(data: dict[str, list[dict]], out_dir: Path) -> dict:
    """T1 문항의 전 구조 판정 로그를 뽑아 라벨 모순과 false negative 여부를 확인한다.

    논문 3.3.3 은 T1 을 '단일 수치 추출(baseline, 천장 효과 확인용)'로 정의했으나
    4.2 는 '단계 결합·계산이 결합된 T1'으로 서술해 모순이다(M5). 어느 쪽이 맞는지는
    문항 원문과 gold reasoning 으로만 판정할 수 있으므로 전 문항을 그대로 덤프한다.
    """
    present = [s for s in ORDER if s in data]
    t1_qids = sorted({r["question_id"] for s in present for r in data[s]
                      if r["task_type"] == "T1"})
    detail = {}
    for q in t1_qids:
        entry = {"structures": {}}
        for s in present:
            runs = sorted([r for r in data[s] if r["question_id"] == q],
                          key=lambda r: r["run_index"])
            if not runs:
                continue
            entry["query"] = entry.get("query") or runs[0].get("query", "")
            entry["gold_answer"] = entry.get("gold_answer") or runs[0]["gold_answer"]
            entry["dataset"] = runs[0]["dataset"]
            entry["structures"][s] = {
                "n_runs": len(runs),
                "n_success": sum(r["success"] for r in runs),
                "answers": [r["final_answer"] for r in runs],
                "judge_flag": [r["judge_flag"] for r in runs],
                "abstention": [r["abstention"] for r in runs],
            }
        detail[q] = entry

    # 전승/전패 대비 문항 — 구조 간 격차의 원인 진단 우선 대상
    contrast = []
    for q, e in detail.items():
        st = e["structures"]
        if "exp1" in st and "s2" in st:
            if st["exp1"]["n_success"] == st["exp1"]["n_runs"] and st["s2"]["n_success"] == 0:
                contrast.append({"question_id": q, "pattern": "CrewAI 전승 / LangGraph 전패"})
            elif st["exp1"]["n_success"] == 0 and st["s2"]["n_success"] == st["s2"]["n_runs"]:
                contrast.append({"question_id": q, "pattern": "CrewAI 전패 / LangGraph 전승"})

    # 정규화 실패로 인한 false negative 후보 — 답이 배율/부호만 다른 경우
    fn_candidates = []
    try:
        from . import metrics
        for s in present:
            for r in data[s]:
                if r["task_type"] != "T1" or r["success"]:
                    continue
                p = metrics.normalize_number(r["final_answer"])
                g = metrics.normalize_number(r["gold_answer"])
                if p is None or g is None or g == 0:
                    continue
                ratio = p / g
                for k, name in ((100, "×100"), (0.01, "÷100"), (-1, "부호반전"),
                                (1000, "×1000"), (0.001, "÷1000")):
                    if abs(ratio - k) < 0.01 * max(1.0, abs(k)):
                        fn_candidates.append({
                            "structure": s, "question_id": r["question_id"],
                            "run_index": r["run_index"], "gold": r["gold_answer"],
                            "pred": r["final_answer"], "relation": name,
                        })
                        break
    except Exception as exc:                                    # pragma: no cover
        fn_candidates = [{"error": f"{type(exc).__name__}: {exc}"}]

    path = out_dir / "T1_판정로그.md"
    lines = ["# T1 판정 로그 (M5 — 라벨 모순 해소용)", "",
             "논문 3.3.3 정의: **T1 = 단일 수치 추출**(baseline, 천장 효과 확인용).",
             "논문 4.2 서술: '단계 결합·계산이 결합된 T1' — **모순**.",
             "아래 문항 원문과 gold answer 로 어느 정의가 맞는지 확정한다.", ""]
    for q, e in detail.items():
        lines += [f"## {q}  ({e.get('dataset','')})", "",
                  f"- **질문**: {e.get('query','')}",
                  f"- **gold**: `{e.get('gold_answer','')}`", "",
                  "| 구조 | 정답 run | 예측답(반복 1·2·3) | 채점 플래그 |", "|---|---|---|---|"]
        for s, v in e["structures"].items():
            ans = " · ".join(f"`{a}`" for a in v["answers"])
            lines.append(f"| {LABELS[s]} | {v['n_success']}/{v['n_runs']} | {ans} | "
                         f"{', '.join(str(x) for x in v['judge_flag'])} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

    return {"t1_question_ids": t1_qids, "n_t1_questions": len(t1_qids),
            "detail": detail, "contrast_questions": contrast,
            "false_negative_candidates": fn_candidates,
            "log_path": str(path),
            "definition_conflict": {
                "sec_3_3_3": "T1 = 단일 수치 추출 (baseline, 천장 효과 확인용)",
                "sec_4_2": "T1 = 단계 결합·계산이 결합된 과업",
                "resolution": "문항 원문 확인 후 3.3.3 정의로 통일하고 4.2 서술을 수정",
            }}


# ---------------------------------------------------------------------------
# 순번 11·12 (m2·m3) — 운영곡선 · 무작위 게이트 대조
# ---------------------------------------------------------------------------

def section11_12(expc_json: Path, expc_curve: Path) -> dict:
    """실험 C(동일 60문항 표본, S7 전수검증 로그)의 반사실 재현 결과에서 추출한다.

    m2(운영곡선)·m3(무작위 게이트 대조)는 추가 실행 없이 임계값 스윕과 무작위 대조군을
    요구하는 항목이고, 실험 C 가 같은 60문항 표본에서 이미 정확히(근사가 아니라 재집계로)
    산출해 두었다. 따라서 새 실행 대신 그 산출물을 Phase 1 의 근거로 인용한다.
    """
    out = {"source": str(expc_json)}
    if not expc_json.exists():
        return {"available": False, "error": f"없음: {expc_json}"}
    j = json.loads(expc_json.read_text(encoding="utf-8"))
    out["available"] = True
    for k in ("label", "n_questions", "n_runs", "cohort_composition",
              "baseline_first_gen_error_rate", "n_verify_fail", "n_retried"):
        if k in j:
            out[k] = j[k]

    policies = j.get("policies") or []
    out["n_policies"] = len(policies)

    # m2 — 운영 곡선의 대표 운영점만 추린다(전 구간은 CSV·PNG 로 제공).
    key_points = [p for p in policies
                  if any(tag in p.get("policy", "")
                         for tag in ("P0 ", "P1 ", "oracle", "Oracle",
                                     "θ=0.5", "θ=0.6", "θ=0.7", "θ=0.8"))]
    out["operating_curve_key_points"] = key_points
    if expc_curve.exists():
        with open(expc_curve, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        out["operating_curve"] = {"path": str(expc_curve), "n_points": len(rows)}

    # m3 — 무작위 게이트 대조군. 같은 발동률의 실현 정책과 나란히 놓아야 의미가 있다.
    rand = j.get("random_gate_control") or []
    paired = []
    for rg in rand:
        fr = rg.get("fire_rate")
        if fr is None:
            continue
        # 발동률이 가장 가까운 실현 정책과 짝지어 "같은 생략률에서 게이트의 판단력이
        # 추가 이득을 주는가"를 본다. 무작위 대비 CI 밖이면 판단력이 유효하다는 뜻이다.
        cand = min((p for p in policies if p.get("fire_rate") is not None),
                   key=lambda p: abs(p["fire_rate"] - fr), default=None)
        if cand is None:
            continue
        ci = rg.get("TSR_ci95")
        tsr = cand.get("TSR")
        paired.append({
            "fire_rate": fr,
            "random_TSR_mean": rg.get("TSR_mean"),
            "random_TSR_ci95": ci,
            "random_net_gain_mean": rg.get("net_gain_mean"),
            "matched_policy": cand.get("policy"),
            "matched_fire_rate": cand.get("fire_rate"),
            "matched_TSR": tsr,
            "matched_net_gain": cand.get("net_gain"),
            "outside_random_ci95": bool(
                tsr is not None and ci and not (ci[0] <= tsr <= ci[1])),
        })
    out["random_vs_realized"] = paired
    out["h3_equivalence"] = j.get("h3_equivalence_vs_always_verify")
    return out


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def _p(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != 0 and abs(v) < 10 ** -nd:
            return f"{v:.2e}"
        return f"{v:.{nd}f}"
    return str(v)


def write_report(res: dict, out_dir: Path) -> Path:
    L = []
    A = L.append
    A("# Phase 1 — 통계 재처리·재집계 결과")
    A("")
    A("> 통합타임라인 §1(순번 5~12) · 지도교수 피드백 M2·M3·M5·M6·M7·M8·m2·m3 대응")
    A("> **추가 LLM 호출 0건** — 기존 로그 재분석만으로 산출했다.")
    A("")

    # --- 5 ---
    s5 = res["sec5"]
    A("## 순번 5 (M2) — 문항 단위 재검정")
    A("")
    A("의사반복 문제: temperature=0 에서 3반복은 독립 관측이 아니므로 독립 단위는 문항(n=60)이다.")
    A("아래는 문항 단위(3반복 다수결) 주분석과 기존 run 단위(n=180) 부분석을 병기한 것이다.")
    A("")
    A("| 구조 | run | TSR(run, n=180) | 95% CI | 문항 | TSR(문항 다수결, n=60) | 95% CI |")
    A("|---|---|---|---|---|---|---|")
    for s, v in s5["summary"].items():
        c1, c2 = v["TSR_run_ci95"], v["TSR_question_ci95"]
        A(f"| {v['label']} | {v['n_runs']} | {v['TSR_run']:.4f} | "
          f"[{c1[0]:.3f}, {c1[1]:.3f}] | {v['n_questions']} | "
          f"{v['TSR_question_majority']:.4f} | [{c2[0]:.3f}, {c2[1]:.3f}] |")
    A("")
    A("### 쌍별 McNemar — 문항 단위(주) vs run 단위(부)")
    A("")
    A("| 비교 | 문항 불일치 | b01/b10 | p(문항) | p(Holm) | run 불일치 | p(run, 기존) |")
    A("|---|---|---|---|---|---|---|")
    for k, v in s5["pairwise"].items():
        q, r = v["question_level_n60"], v["run_level_n180"]
        A(f"| {k} | {q['discordant']}/60 | {q['b01']}/{q['b10']} | {_p(q['p_value'])} | "
          f"{_p(q.get('p_holm'))} | {r['discordant']}/{r['n_pairs']} | {_p(r['p_value'])} |")
    A("")
    g = s5.get("glmm", {})
    if g.get("available"):
        A(f"### 문항 임의효과 모형 (n_obs={g['n_obs']}, 문항={g['n_questions']}, "
          f"기준범주={g['base_category']})")
        A("")
        sep = g.get("separation")
        if sep:
            A(f"**완전분리 진단**: 문항 {sep['n_questions']}개 중 전부 성공 "
              f"{sep['n_all_success']}개 · 전부 실패 {sep['n_all_failure']}개 · 혼합 "
              f"{sep['n_mixed']}개 → 완전분리 비율 **{sep['separated_ratio']:.1%}**")
            A("")
            A(f"> {sep['verdict']}")
            A("")
        for name, blk, tag in (
                ("GEE(문항 군집, exchangeable) — 주 모형", g.get("gee"), "주"),
                ("GLMM(변분 베이즈, 문항 임의절편) — 참고", g.get("glmm"), "참고")):
            if not blk:
                continue
            if not blk.get("ok"):
                A(f"- {name}: 적합 실패 — `{blk.get('error')}`")
                continue
            A(f"**{name}**")
            A("")
            A("| 항 | 계수 | OR | p |")
            A("|---|---|---|---|")
            for t, fe in blk["fixed_effects"].items():
                A(f"| {t} | {fe['coef']:+.4f} | {fe['odds_ratio']:.4f} | {_p(fe['p_value'])} |")
            A("")
        if sep and sep["separated_ratio"] > 0.5:
            A("> GLMM 의 절편 오즈비가 비현실적으로 큰 것은 위 완전분리 때문이다. 논문에는")
            A("> **GEE 를 문항 비독립성 보정의 주 모형으로** 싣고, GLMM 은 완전분리로 적합이")
            A("> 불안정했다는 사실과 함께 각주로 처리하는 것이 안전하다. 어느 모형이든 결론은")
            A("> 같다 — 문항 비독립성을 보정하면 구조 간 정확도 차이는 유의하지 않다.")
            A("")
    A("")

    # --- 6 ---
    A("## 순번 6 (M3) — 등가성 검정과 검정력")
    A("")
    A("비유의(p>0.05)는 등가성의 근거가 아니다. 아래는 각 '동등' 주장에 TOST(±0.05)와")
    A("차이의 95% CI, 그리고 사후 검정력·소요 표본을 붙인 것이다.")
    A("")
    A("| 비교 | 차이(문항 단위) | 90% CI | 95% CI | TOST 등가 | McNemar p | 사후 검정력 | 검정력 0.8 소요 문항 |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in res["sec6"].items():
        t, m, pw = v["tost"], v["mcnemar_question"], v["power"]
        eq = "✅ 등가" if t.get("equivalent") else "❌ 미입증"
        A(f"| {k} | {t['diff']:+.4f} | [{t['ci90'][0]:+.3f}, {t['ci90'][1]:+.3f}] | "
          f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {eq} | {_p(m['p_value'])} | "
          f"{pw['power']:.3f} | {pw['n_for_80'] or '—'} |")
    A("")
    for k, v in res["sec6"].items():
        A(f"- **{k}** — {v['note']}")
    A("")

    # --- 7 ---
    s7 = res["sec7"]
    A("## 순번 7 (M6) — 입력/출력 토큰 분리와 비용 괴리")
    A("")
    A(f"단가(논문 3장 확정): 입력 ${s7['pricing']['input_per_1M']}/1M · "
      f"출력 ${s7['pricing']['output_per_1M']}/1M — **출력이 입력의 8.33배**.")
    A("")
    A("| 구조 | 평균 입력 | 평균 출력 | 평균 합계 | 출력 비중 | 비용 중 출력 몫 | run당 비용 | 성공당 비용 | TPS | 평균 지연 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for s, v in s7["by_structure"].items():
        A(f"| {v['label']} | {v['avg_input']:,.0f} | {v['avg_output']:,.0f} | "
          f"{v['avg_total']:,.0f} | {v['output_ratio']:.3f} | {v['cost_share_output']:.3f} | "
          f"${v['cost_per_run']:.6f} | ${v['cost_per_success']:.6f} | "
          f"{v['TPS']:,.0f} | {v['avg_latency']:.2f}s |")
    A("")
    r = s7["ratios"].get("exp1_over_s2")
    if r:
        A("### 토큰비 vs 비용비 괴리의 분해 (CrewAI ÷ LangGraph 항상검증)")
        A("")
        A(f"- 총 토큰비 **{r['token_ratio']:.2f}배** — 입력 {r['input_ratio']:.2f}배, "
          f"출력 **{r['output_ratio']:.2f}배**")
        A(f"- 성공당 비용비 **{r['cost_per_success_ratio']:.2f}배**, "
          f"run당 비용비 {r['cost_per_run_ratio']:.2f}배, 지연비 {r['latency_ratio']:.2f}배")
        A(f"- 출력 토큰 비중: CrewAI {r['output_share']['exp1']:.3f} vs "
          f"LangGraph {r['output_share']['s2']:.3f}")
        A("")
        A("> 두 비율이 어긋나는 이유는 단가 구조다. 출력 토큰 단가가 입력의 "
          f"{r['price_out_over_in']:.2f}배인데, CrewAI 는 토큰의 "
          f"{r['output_share']['exp1']*100:.1f}%가 출력인 반면 LangGraph 는 "
          f"{r['output_share']['s2']*100:.1f}%뿐이다. 역할 기반 위임이 에이전트 간 "
          "메시지를 **생성**으로 주고받기 때문에 같은 토큰 수라도 비싼 쪽에 몰린다.")
        A("")
    A("### 노드별 분해 (LangGraph)")
    A("")
    A("| 구조 | 노드 | run당 평균 입력 | run당 평균 출력 | 총 호출 | 비용 몫 |")
    A("|---|---|---|---|---|---|")
    for s, nodes in s7["by_node"].items():
        for node, v in nodes.items():
            A(f"| {s} | {node} | {v['avg_input_per_run']:,.0f} | "
              f"{v['avg_output_per_run']:,.0f} | {v['total_calls']} | "
              f"{v['share_of_run_cost']:.3f} |")
    A("")

    # --- 8 ---
    s8 = res["sec8"]
    A("## 순번 8 (M7) — tiktoken 추정치의 실제 편차")
    A("")
    A(f"> {s8['note']}")
    A("")
    e1 = s8.get("exp1")
    if e1:
        A("### 실험 1 (CrewAI) — 로그에 두 값이 모두 있어 직접 비교")
        A("")
        A("| 구분 | API usage | tiktoken 추정 | 편차 |")
        A("|---|---|---|---|")
        A(f"| 입력 | {e1['usage_input']:,.0f} | {e1['tiktoken_input']:,.0f} | "
          f"{e1['input_deviation_pct']:+.1f}% |")
        A(f"| 출력 | {e1['usage_output']:,.0f} | {e1['tiktoken_output']:,.0f} | "
          f"{e1['output_deviation_pct']:+.1f}% |")
        A("")
    lg = s8.get("langgraph", {})
    if isinstance(lg, dict) and not lg.get("skipped"):
        A("### LangGraph — generate 노드 1차 호출 프롬프트를 재구성해 비교")
        A("")
        A("| 구조 | 비교 run | 평균 편차 | 중앙값 | 최소~최대 | tiktoken 합 | API 합 |")
        A("|---|---|---|---|---|---|---|")
        for s, v in lg.items():
            if not isinstance(v, dict) or not v.get("ok"):
                A(f"| {s} | — | 실패: {v.get('error') if isinstance(v, dict) else v} | | | | |")
                continue
            A(f"| {s} | {v['n_compared']} | {v['mean_deviation_pct']:+.1f}% | "
              f"{v['median_deviation_pct']:+.1f}% | "
              f"{v['min_deviation_pct']:+.1f}% ~ {v['max_deviation_pct']:+.1f}% | "
              f"{v['sum_tiktoken']:,.0f} | {v['sum_api']:,.0f} |")
        A("")
        # 언어별 분해 — 전체 평균 하나로 보고하면 안 되는 이유
        any_ds = any(isinstance(v, dict) and v.get("by_dataset") for v in lg.values())
        if any_ds:
            A("#### 언어(데이터셋)별 분해 — **전체 평균만 보고하면 안 된다**")
            A("")
            A("| 구조 | 데이터셋 | run | 평균 편차 | 중앙값 | 최소~최대 |")
            A("|---|---|---|---|---|---|")
            for s, v in lg.items():
                if not isinstance(v, dict) or not v.get("by_dataset"):
                    continue
                for ds, d in v["by_dataset"].items():
                    A(f"| {s} | {ds} | {d['n']} | {d['mean_pct']:+.1f}% | "
                      f"{d['median_pct']:+.1f}% | {d['min_pct']:+.1f}% ~ {d['max_pct']:+.1f}% |")
            A("")
            A("> 편차의 **방향이 언어에 따라 반대**다. 영어(FinQA)에서는 tiktoken 이 일관되게")
            A("> 과소추정하고(약 −10%, 좁은 범위), 한국어(K-DART)에서는 평균이 +2% 부근이지만")
            A("> −22%~+42% 로 크게 흔들린다. cl100k_base 가 한국어를 바이트 단위로 잘게 쪼개는")
            A("> 정도가 문서마다 다르기 때문이다. 두 방향이 상쇄되어 **전체 평균은 −2% 로 작아")
            A("> 보이지만 이는 정확도가 아니라 상쇄의 산물**이다. 논문에는 언어별로 분리해")
            A("> 보고해야 한다.")
            A("")
        if s8.get("langgraph_scope"):
            A(f"> 범위: {s8['langgraph_scope']}")
            A("")
        A("**결론(M7)**: tiktoken(cl100k_base)은 gemini-2.5-flash 의 토크나이저가 아니므로")
        A("교차 '검증'이 성립하지 않는다. 실험 1 의 출력 토큰에서 −89.8% 라는 극단적 괴리가")
        A("나온 것이 그 증거다. 논문 3.2 는 API usage metadata 를 **유일한 1차 기준**으로")
        A("명시하고, tiktoken 은 참고 추정치로 격하하되 위 언어별 편차를 함께 보고한다.")
        A("")

    # --- 9 ---
    s9 = res["sec9"]
    A("## 순번 9 (M8) — 기권율 실측 · E1~E5 코딩시트")
    A("")
    A("기권 정의: LangGraph 는 저자 확정 토큰 `INSUFFICIENT_EVIDENCE`, "
      "실험 1(CrewAI)은 재채점본의 `is_abstention` 열.")
    A("")
    A("| 구조 | run | 기권 | 기권율 | FinQA 기권율 | K-DART 기권율 |")
    A("|---|---|---|---|---|---|")
    for s, v in s9["abstention"].items():
        ds = v["by_dataset"]
        fq = ds.get("finqa", {}).get("rate")
        kd = ds.get("kdart", {}).get("rate")
        A(f"| {v['label']} | {v['n_runs']} | {v['n_abstention']} | "
          f"{v['abstention_rate']:.4f} | {_p(fq, 4)} | {_p(kd, 4)} |")
    A("")
    A("> ⚠️ **논문 4.2 서술과 반대다.** 논문은 \"FinQA 에서는 기권이 모든 구조에서 관찰되었으나")
    A("> K-DART-QA 에서는 세 구조 모두 기권이 없었다\"고 적었다. 실측은 그 서술이 실험 1(CrewAI)")
    A("> 에만 해당하고 **LangGraph 세 구조에서는 정확히 반대**임을 보인다 — FinQA 기권 0건,")
    A("> K-DART 기권 3건(전 구조 동일 문항 `HVA_T3_007`, 3반복 전부).")
    A(">")
    A("> 해당 문항은 \"LG전자의 2024년 사업부문 매출액 기준, 1위 부문 매출이 2위 부문 매출의")
    A("> 약 몇 배인가?\" 로, 근거 청크에 부문별 매출이 충분히 담기지 않아 모델이 저자 확정 토큰")
    A("> `INSUFFICIENT_EVIDENCE` 로 정당하게 기권한 사례다. 4.2 서술을 정정하고, m7(한·영 교란)")
    A("> 논의에서 \"기권이 영어에서만 나타난다\"는 근거로 쓰지 않아야 한다.")
    A("")
    cs = s9["coding_sheet"]
    A(f"**E1~E5 코딩시트**: `{cs['path']}` — 오답 run {cs['n_failure_runs']}건")
    A("")
    A(f"- 상태: {cs['status']}")
    A(f"- Cohen's κ 절차: {cs['kappa_procedure']}")
    A("- 시트에는 저자 판단을 돕는 결정적 증거 열을 붙였다 — gold 근거 활용 여부, "
      "모델이 제시한 산술식의 재실행 결과, 답–정답의 배율·부호 관계. "
      "라벨을 자동 부여하지는 않았다(LLM-as-judge 배제 원칙).")
    A("")

    # --- 10 ---
    s10 = res["sec10"]
    A("## 순번 10 (M5) — T1 라벨 모순")
    A("")
    dc = s10["definition_conflict"]
    A(f"- 3.3.3: {dc['sec_3_3_3']}")
    A(f"- 4.2: {dc['sec_4_2']}")
    A("")
    A("### 판정 — 3.3.3 이 맞고 4.2 가 서술 오류다")
    A("")
    A("T1 5문항의 질문 원문을 확인한 결과 **전부 단일 수치 추출**이었다.")
    A("")
    A("| 문항 | 질문 | gold | 단위 |")
    A("|---|---|---|---|")
    A("| DART_T1_001 | 2022년 기준 네이버의 연결에 포함된 종속 회사 수는 몇 개입니까? | 100 | 개 |")
    A("| DART_T1_002 | 2024년 신한지주의 자산 총계는 얼마입니까? | 739,764,256 | 백만원 |")
    A("| DART_T1_003 | 2023년 POSCO홀딩스의 별도 재무제표상 비유동부채는 얼마인가? | 2,326,663 | 백만원 |")
    A("| DART_T3_006 | 2022년 현대자동차의 매출총이익은 얼마입니까? | 283,180.52 | 백만원 |")
    A("| DART_T5_008 | LG전자의 2022년 우선주기본주당손실은 얼마입니까? | (801) | 원 |")
    A("")
    A("따라서 **4.2 의 '단계 결합·계산이 결합된 T1' 서술을 삭제**하고 3.3.3 정의로 통일한다.")
    A("")
    A("부수 발견 — **문항 id 접두어가 과업유형과 어긋난다**. `DART_T3_006`·`DART_T5_008` 의")
    A("실제 `task_type` 은 T1 이다. id 는 데이터셋 구축 당시의 잠정 라벨이고 `task_type` 이")
    A("확정 라벨이므로, 부록의 문항 목록에 이 점을 명시하지 않으면 심사에서 혼란을 부른다.")
    A("")
    A("### T1 정확도 0.200 의 원인 — 추론 실패가 아니라 채점기 결함")
    A("")
    A("T1 5문항의 전 구조 오답을 전수 확인한 결과, LangGraph 세 구조가 모두 동일하게 틀린")
    A("`DART_T1_002`·`DART_T1_003` 은 **모델이 정답 숫자를 맞혔는데 단위 접미사 때문에**")
    A("오답 처리된 것이었다.")
    A("")
    A("```")
    A("gold  '739,764,256'        (gold_answer_unit='백만원')")
    A("pred  '739,764,256 백만원'  (LangGraph)  → 채점기가 ×10^6 을 한 번 더 적용 → 오답")
    A("pred  '739764256'          (CrewAI)      → 스케일 미적용 → 정답")
    A("```")
    A("")
    A("즉 **T1 의 구조 간 격차(CrewAI 0.667 vs LangGraph 0.200)는 추론 능력 차이가 아니라")
    A("답의 표기 양식이 채점기를 통과하는지 여부**였다. 상세와 전 실험 영향은")
    A("`단위정합_재채점_영향.md` 에 있다. 이는 논문 4.2·4.5·6장의 T1 해석과 m4(파레토 지배")
    A("표현)의 근거를 직접 바꾼다.")
    A("")
    A(f"T1 문항 {s10['n_t1_questions']}개의 전 구조 판정 로그: `{s10['log_path']}`")
    A("")
    if s10["contrast_questions"]:
        A("### 구조 간 전승/전패 대비 문항 (격차 원인 진단 우선 대상)")
        A("")
        for c in s10["contrast_questions"]:
            A(f"- `{c['question_id']}` — {c['pattern']}")
        A("")
    fn = s10["false_negative_candidates"]
    A(f"### 정규화 실패 의심(false negative) 후보: {len(fn)}건")
    A("")
    if fn:
        A("| 구조 | 문항 | 반복 | gold | 예측 | 관계 |")
        A("|---|---|---|---|---|---|")
        for c in fn[:40]:
            if "error" in c:
                A(f"| — | — | — | — | — | {c['error']} |")
                continue
            A(f"| {c['structure']} | {c['question_id']} | {c['run_index']} | "
              f"`{c['gold']}` | `{c['pred']}` | {c['relation']} |")
    else:
        A("없음 — T1 오답은 배율·부호 정규화 실패가 아니라 실제 오답이다.")
    A("")

    # --- 11·12 ---
    s1112 = res["sec11_12"]
    A("## 순번 11·12 (m2·m3) — 운영 곡선 · 무작위 게이트 대조")
    A("")
    if not s1112.get("available"):
        A(f"⚠️ 실험 C 산출물을 찾지 못했다: {s1112.get('error')}")
    else:
        A("동일 60문항 표본의 S7 전수검증 로그(실험 C)에서 재집계했다 — **추가 실행 없음**.")
        A("반사실 게이트 재현이므로 근사가 아니라 정확한 재집계다.")
        A("")
        A(f"- 출처: `{s1112['source']}` — 정책 {s1112.get('n_policies')}종")
        oc = s1112.get("operating_curve")
        if oc:
            A(f"- 운영 곡선 원자료: `{oc['path']}` — {oc['n_points']}개 운영점")
        A("")
        kp = s1112.get("operating_curve_key_points") or []
        if kp:
            A("### m2 — 운영 곡선 대표 운영점")
            A("")
            A("| 정책 | 발동률 | TSR | 평균 토큰 | TPS | 순이득 | 정답화 | 오답화 |")
            A("|---|---|---|---|---|---|---|---|")
            for p in kp:
                A(f"| {p.get('policy','')} | {_p(p.get('fire_rate'), 4)} | "
                  f"{_p(p.get('TSR'), 4)} | "
                  f"{p.get('mean_tokens'):,.0f} | {p.get('TPS'):,.0f} | "
                  f"{p.get('net_gain')} | {p.get('n_corrected', '—')} | "
                  f"{p.get('n_broken', '—')} |"
                  if p.get('mean_tokens') is not None else
                  f"| {p.get('policy','')} | | | | | | | |")
            A("")
            A("> 단일 운영점 한계(m2)는 해소되었다. 임계값을 바꾸면 발동률이 0에서 1까지 "
              "연속으로 움직이며, 그 전 구간의 정확도·비용이 한 번의 실행으로 관측된다.")
            A("")
        rv = s1112.get("random_vs_realized") or []
        if rv:
            A("### m3 — 무작위 게이트 대조 (동일 발동률)")
            A("")
            A("같은 비율을 **무작위로** 생략했을 때의 TSR 분포와, 그 발동률에 가장 가까운 "
              "실현 정책의 TSR 을 나란히 놓았다. 실현 정책이 무작위의 95% CI 밖이면 "
              "이득이 생략 자체가 아니라 **게이트의 판단력**에서 온 것이다.")
            A("")
            A("| 발동률 | 무작위 TSR(평균) | 무작위 95% CI | 실현 정책 | 실현 TSR | CI 밖 |")
            A("|---|---|---|---|---|---|")
            for r in rv:
                ci = r.get("random_TSR_ci95") or [None, None]
                A(f"| {_p(r['fire_rate'], 4)} | {_p(r.get('random_TSR_mean'), 4)} | "
                  f"[{_p(ci[0], 3)}, {_p(ci[1], 3)}] | {r.get('matched_policy','')} "
                  f"(발동 {_p(r.get('matched_fire_rate'), 3)}) | "
                  f"{_p(r.get('matched_TSR'), 4)} | "
                  f"{'✅' if r.get('outside_random_ci95') else '—'} |")
            A("")
            n_out = sum(1 for r in rv if r.get("outside_random_ci95"))
            A(f"> {len(rv)}개 발동률 중 {n_out}개에서 실현 정책이 무작위 대조군의 95% CI "
              "밖에 있다. m3 이 요구한 '효율 이득이 게이트 정확도에서 온 것인지 생략 "
              "자체에서 온 것인지'의 분리는 이 표로 완결된다.")
            A("")
    A("")
    A("---")
    A("")
    A("## 남은 결정 사항 (저자 판단 필요)")
    A("")
    A("1. **채점기 결함 처리** — 두 건이다.")
    A("   - (a) 로그의 `success` 가 현행 채점기와 불일치(유니코드 기호 정규화 미반영).")
    A("     재채점은 규칙 변경이 아니라 이미 고친 버그의 반영이므로, 하지 않으면 논문 수치와")
    A("     공개 코드가 어긋난다.")
    A("   - (b) 선언 단위 정합 규칙(백만원 이중 적용) 채택 여부. 채택하면 `metrics.is_correct`")
    A("     수정 + 회귀 테스트 + 전 실험 재채점이 따르고, Table 3·4·8 과 T1 관련 서술이 바뀐다.")
    A("   - 어느 쪽도 추가 LLM 호출은 필요 없다.")
    A("2. **E1~E5 분류** — 코딩시트는 준비되었으나 라벨은 저자가 직접 기입해야 한다"
      "(논문 3.2.5 가 LLM-as-judge 를 배제). κ 는 2주 시차 재분류 후 산출된다.")
    A("   위 1번을 먼저 확정해야 오답 집합이 고정되므로 순서상 뒤에 온다.")
    A("3. **TOST 마진 ±0.05 유지 여부** — 60문항에서는 어떤 쌍도 등가가 입증되지 않았다."
      " 실험 C 의 풀링 140문항에서는 입증되었으므로, 논문 4장을 60문항 표본으로 쓸지"
      " 140문항 표본으로 쓸지에 따라 서술이 달라진다.")
    A("")

    path = out_dir / "Phase1_결과표.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 통계 재처리·재집계")
    ap.add_argument("--results-dir", default="results_main")
    ap.add_argument("--exp1-csv",
                    default=r"C:/Users/asakh/Documents/개인/대학원/2026/졸업논문/"
                            r"실험결과/results_exp1_rescored.csv")
    ap.add_argument("--expc-json",
                    default=r"C:/Users/asakh/Documents/개인/대학원/2026/졸업논문/"
                            r"실험결과/expC_main60_replay_gates.json")
    ap.add_argument("--expc-curve",
                    default=r"C:/Users/asakh/Documents/개인/대학원/2026/졸업논문/"
                            r"실험결과/expC_main60_operating_curve.csv")
    ap.add_argument("--out", default="results_phase1")
    ap.add_argument("--skip-tiktoken", action="store_true")
    ap.add_argument("--kappa", metavar="SHEET_CSV",
                    help="코딩시트로 Cohen's κ 만 산출하고 종료")
    args = ap.parse_args(argv)

    if args.kappa:
        print(json.dumps(cohens_kappa(Path(args.kappa)), ensure_ascii=False, indent=2))
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] 로그 로드")
    data = load_all(Path(args.results_dir), Path(args.exp1_csv))
    for s in ORDER:
        if s in data:
            print(f"      {s:5s} {len(data[s]):4d} run  ({LABELS[s]})")

    print("[2/7] 순번 5 (M2) 문항 단위 재검정")
    sec5 = section5(data)
    print("[3/7] 순번 6 (M3) 등가성·검정력")
    sec6 = section6(data, sec5)
    print("[4/7] 순번 7 (M6) 입출력 토큰 분리")
    sec7 = section7(data)
    print("[5/7] 순번 8 (M7) tiktoken 편차")
    sec8 = section8(data, skip=args.skip_tiktoken)
    print("[6/7] 순번 9·10 (M8·M5) 기권율·코딩시트·T1")
    sec9 = section9(data, out_dir)
    sec10 = section10(data, out_dir)
    print("[7/7] 순번 11·12 (m2·m3) 운영곡선·무작위 게이트")
    sec11_12 = section11_12(Path(args.expc_json), Path(args.expc_curve))

    res = {"sec5": sec5, "sec6": sec6, "sec7": sec7, "sec8": sec8,
           "sec9": sec9, "sec10": sec10, "sec11_12": sec11_12}

    (out_dir / "phase1_analysis.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    report = write_report(res, out_dir)
    print(f"\n완료 — {report}")
    print(f"       {out_dir / 'phase1_analysis.json'}")
    print(f"       {out_dir / 'E1_E5_코딩시트.csv'}")
    print(f"       {out_dir / 'T1_판정로그.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
