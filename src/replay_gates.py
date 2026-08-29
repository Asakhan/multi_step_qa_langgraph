"""실험 C — 반사실 게이트 재현(counterfactual gate replay).

S7 로그(게이트 없음 + verify 전수 + 1차 생성 스냅샷) 하나에서 **임의의 게이트 정책**의
정확도·비용·지연을 추가 LLM 호출 없이 재구성한다.

원리
----
캐스케이드의 게이트는 run 단위 이진 선택 e(r) ∈ {stop, escalate} 이며, 두 갈래의 결과는
S7 로그에 모두 기록되어 있다.

    e=stop      → 답 = first_answer,  비용 = 1차 generate 토큰
    e=escalate  → 답 = final_answer,  비용 = 1차 generate + verify (+ FAIL 시 재생성)

따라서 정책 π 의 성능은 로그를 정책에 따라 재집계하기만 하면 된다. 게이트 자체는 LLM
호출이 없으므로(토큰 0) 이 재구성은 근사가 아니라 **정확**하다. 유일한 가정은
"검증 여부가 1차 생성 결과에 영향을 주지 않는다"는 것인데, 검증은 생성 이후에 실행되므로
설계상 성립한다(temperature=0 통제 하 동일 프롬프트).

재구성 타당성 검증: 자기보고 임계값 0.8 정책을 재현해 실제 S5 실측(발동률·TSR)과 대조한다.

평가 정책
--------
  P0 무검증(s0 등가)        : 전부 stop
  P1 항상검증(s2 등가)      : 전부 escalate
  P2 자기보고 게이트(s5 등가): eff_conf < θ 이면 escalate — θ 스윕
  P3 학습 라우터(실험 B B-5): 외부신호+자기보고 로지스틱 회귀 점수 ≥ θ 이면 escalate — θ 스윕
                              **문항 단위 GroupKFold 교차적합**으로 out-of-fold 점수만 사용
                              (같은 문항이 학습·평가에 동시에 들어가는 누출을 차단)
  P4 무작위 게이트(대조군)   : 동일 발동률에서 무작위 선택 — 부트스트랩 평균·95% CI
                              "이득이 게이트의 판단력에서 온 것인지 검증 자체에서 온 것인지" 분리
  P5 oracle 게이트(상한)     : 1차 답이 오답인 run 만 escalate — 게이트 개선의 이론적 상한

  python -m src.replay_gates [--structure s7] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import config, metrics

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")

SEED = 20260821
BOOTSTRAP_N = 10000


# ---------------------------------------------------------------------------
# 로그 로딩 + run 단위 반사실 레코드 구성
# ---------------------------------------------------------------------------

def load_runs(structure: str, results_dir: Path | None = None) -> list[dict]:
    base = results_dir or config.RESULTS_DIR
    path = base / structure / "raw_runs.jsonl"
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("run_error"):
            out.append(r)
    return out


def _seq_split(r: dict) -> tuple[int, int, int]:
    """호출 시퀀스를 (1차 generate, verify, 재생성 generate) 토큰으로 분할."""
    seq = r.get("token_usage_seq") or []
    gen1 = ver = gen2 = 0
    seen_verify = False
    for c in seq:
        t = int(c["input_tokens"]) + int(c["output_tokens"])
        if c["node"] == "verify":
            seen_verify = True
            ver += t
        elif seen_verify:
            gen2 += t
        else:
            gen1 += t
    return gen1, ver, gen2


def _seq_split_cost(r: dict) -> tuple[float, float, float]:
    """비용도 동일하게 3분할. pricing 은 metrics.run_cost 와 같은 경로를 쓴다."""
    seq = r.get("token_usage_seq") or []
    g1 = v = g2 = 0.0
    seen = False
    for c in seq:
        rec = {"model": c.get("model", ""), "input_tokens": c["input_tokens"],
               "output_tokens": c["output_tokens"], "node": c["node"]}
        cost = metrics.run_cost([rec])
        if c["node"] == "verify":
            seen = True
            v += cost
        elif seen:
            g2 += cost
        else:
            g1 += cost
    return g1, v, g2


def _seq_split_latency(r: dict) -> tuple[float, float, float]:
    lat = r.get("node_latency_seq") or []
    g1 = v = g2 = 0.0
    seen = False
    for c in lat:
        s = float(c.get("seconds", 0.0))
        node = c.get("node", "")
        if node == "verify":
            seen = True
            v += s
        elif node in ("retrieve", "generate"):
            if seen:
                g2 += s
            else:
                g1 += s
    return g1, v, g2


def build_records(runs: list[dict]) -> list[dict]:
    """정책 재구성에 필요한 최소 정보로 run 을 압축."""
    recs = []
    for r in runs:
        if not r.get("first_answer") and r.get("first_success") is None:
            raise SystemExit(
                "1차 생성 스냅샷이 없는 로그입니다 — 실험 C 는 S7 로그(first_* 필드)가 필요합니다.")
        g1t, vt, g2t = _seq_split(r)
        g1c, vc, g2c = _seq_split_cost(r)
        g1l, vl, g2l = _seq_split_latency(r)
        recs.append({
            "run_id": r["run_id"],
            "question_id": r["question_id"],
            "run_index": r["run_index"],
            "dataset": r["dataset"],
            "task_type": r["task_type"],
            "reasoning_hops": r.get("reasoning_hops"),
            # 두 갈래 결과
            "stop_success": int(r["first_success"]),
            "escalate_success": int(r["success"]),
            "stop_answer": r.get("first_answer", ""),
            "escalate_answer": r.get("final_answer", ""),
            "verify_verdict": r.get("verify_verdict", ""),
            "retried": int(r.get("retry_count", 0)) > 0,
            # 비용
            "tok_stop": g1t, "tok_escalate": g1t + vt + g2t,
            # S5 등가 비용 — S5(s5.py)는 재생성 후 verify 를 한 번 더 호출한다. 그 2차 판정은
            # 라우팅에 영향을 주지 않으므로(어느 판정이든 END) 순수 낭비 비용이다. S7 은 이를
            # 제거했으므로(S2 와 동일 정책) 실측 S5 와 비용을 대조할 때만 추정 보정치를 쓴다.
            # 추정치는 같은 run 의 1차 verify 토큰(입력 근거·답 길이가 사실상 동일)을 재사용한다.
            "tok_escalate_s5equiv": g1t + vt + g2t + (vt if g2t > 0 else 0),
            "cost_stop": g1c, "cost_escalate": g1c + vc + g2c,
            "lat_stop": g1l, "lat_escalate": g1l + vl + g2l,
            # 게이트 신호(1차 생성 시점에 관측 가능한 것만)
            "conf_self": float(r.get("first_confidence_self", 0.0) or 0.0),
            "code_conflict": bool(r.get("first_code_conflict", False)),
            "first_expressions": r.get("first_expressions", []) or [],
            "first_reasoning": r.get("first_reasoning", "") or "",
            "retrieved_evidence": r.get("retrieved_evidence", []) or [],
            "first_output_tokens": int(r.get("first_output_tokens", 0) or 0),
        })
    return recs


# ---------------------------------------------------------------------------
# 게이트 신호 (실험 B 에서 판별력이 확인된 것만 — B1·B4b 는 제외 권고 반영)
# ---------------------------------------------------------------------------

def _evidence_text(rec: dict) -> str:
    parts = []
    for c in rec["retrieved_evidence"]:
        parts.append(str(c.get("text", "")) if isinstance(c, dict) else str(c))
    return " ".join(parts)


def sig_no_expression(rec) -> float:
    """B-1b 수치 답인데 산술식 없음."""
    if metrics.normalize_number(rec["stop_answer"]) is None:
        return 0.0
    return 0.0 if rec["first_expressions"] else 1.0


def sig_evidence_coverage(rec) -> float:
    """B-2 추론에 인용된 수치 중 근거에서 찾을 수 없는 비율."""
    ev = _evidence_text(rec)
    if not ev:
        return 0.0
    ev_nums = {x.replace(",", "") for x in _NUM.findall(ev)}
    if not ev_nums:
        return 0.0
    cited = [x.replace(",", "") for x in _NUM.findall(rec["first_reasoning"])]
    cited = [c for c in cited if len(c.replace(".", "").replace("-", "")) >= 3]
    if not cited:
        return 0.0
    return sum(1 for c in cited if c not in ev_nums) / len(cited)


def sig_hops(rec) -> float:
    """B-3 문항이 요구하는 추론 홉 수(사전 라벨)."""
    try:
        return float(rec.get("reasoning_hops") or 0)
    except (TypeError, ValueError):
        return 0.0


def sig_n_expr(rec) -> float:
    """B-3b 모델이 쓴 산술식 개수."""
    return float(len(rec["first_expressions"]))


def sig_format_risk(rec) -> float:
    """B-4 답 형식 위험도(퍼센트·단위·부호)."""
    fa = str(rec["stop_answer"])
    risk = 0.0
    if "%" in fa:
        risk += 1
    if re.search(r"(억|백만|천|원|배|명|개|pt|USD|\$)", fa):
        risk += 1
    if fa.strip().startswith("-") or "감소" in fa:
        risk += 1
    return risk


def sig_code_conflict(rec) -> float:
    """B-1 추론답 vs 코드실행답 불일치. (실험 B 에서 AUC 0.489 — 참조용으로만 산출)"""
    return 1.0 if rec["code_conflict"] else 0.0


def sig_self_report(rec) -> float:
    """(대조) 자기보고 확신도를 위험도로 반전."""
    return 1.0 - rec["conf_self"]


# 라우터 특징 — 실험 B 판정("B-1 계산불일치·B-4b 출력토큰은 판별력 없어 제외")을 반영한
# 축소 세트. 자기보고는 실험 B 에서 보완적 정보가 확인되었으므로 포함한다.
ROUTER_FEATURES = {
    "B1b 산술식없음": sig_no_expression,
    "B2 근거미발견율": sig_evidence_coverage,
    "B3 추론홉수": sig_hops,
    "B3b 산술식개수": sig_n_expr,
    "B4 답형식위험": sig_format_risk,
    "자기보고(반전)": sig_self_report,
}
# 자기보고를 뺀 외부신호 전용 세트(자기평가 의존 탈피 가능성 확인용)
ROUTER_FEATURES_EXT = {k: v for k, v in ROUTER_FEATURES.items() if not k.startswith("자기보고")}

# 참조용 전체 신호(AUC 재현성 확인)
ALL_SIGNALS = dict(ROUTER_FEATURES)
ALL_SIGNALS["B1 계산불일치"] = sig_code_conflict


# ---------------------------------------------------------------------------
# 학습 라우터 — 문항 단위 GroupKFold 교차적합 out-of-fold 점수
# ---------------------------------------------------------------------------

def router_oof_scores(recs: list[dict], features: dict, folds: int = 5):
    """문항을 그룹으로 묶어 교차적합한 로지스틱 회귀의 out-of-fold 오답확률.

    같은 문항의 3반복 run 이 학습·평가에 갈라 들어가면 문항 누출로 성능이 과대추정되므로,
    문항 전체를 한 fold 에 묶는다(GroupKFold). 반환되는 점수는 모두 해당 문항을 보지 않은
    모델의 예측이다.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return None, None

    X = np.array([[fn(r) for fn in features.values()] for r in recs], dtype=float)
    y = np.array([1 - r["stop_success"] for r in recs])   # 1 = 1차 답이 오답
    groups = np.array([r["question_id"] for r in recs])

    n_groups = len(set(groups))
    folds = min(folds, n_groups)
    if folds < 2 or y.sum() < 2 or (len(y) - y.sum()) < 2:
        return None, None

    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = y[tr].mean() if len(tr) else 0.5
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=SEED))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    # 전체 데이터로 적합한 최종 모델의 계수(해석·재현용, 평가에는 쓰지 않음)
    final = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=SEED))
    final.fit(X, y)
    lr = final.named_steps["logisticregression"]
    coefs = dict(zip(features.keys(), [round(float(c), 4) for c in lr.coef_[0]]))
    coefs["_intercept"] = round(float(lr.intercept_[0]), 4)
    return list(oof), coefs


def auc(scores, labels) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# 정책 평가
# ---------------------------------------------------------------------------

def evaluate(recs: list[dict], escalate: list[bool], label: str) -> dict:
    """게이트 결정 벡터로부터 정책 성능을 재구성."""
    n = len(recs)
    succ = [r["escalate_success"] if e else r["stop_success"] for r, e in zip(recs, escalate)]
    tok = [r["tok_escalate"] if e else r["tok_stop"] for r, e in zip(recs, escalate)]
    cost = [r["cost_escalate"] if e else r["cost_stop"] for r, e in zip(recs, escalate)]
    lat = [r["lat_escalate"] if e else r["lat_stop"] for r, e in zip(recs, escalate)]

    fired = [i for i, e in enumerate(escalate) if e]
    # 순이득: 발동 run 중 (오답→정답) − (정답→오답)
    gained = sum(1 for i in fired
                 if recs[i]["stop_success"] == 0 and recs[i]["escalate_success"] == 1)
    broken = sum(1 for i in fired
                 if recs[i]["stop_success"] == 1 and recs[i]["escalate_success"] == 0)
    # 게이트 선별 품질: 발동 집합이 실제 오답(1차)을 얼마나 맞췄는가
    tp = sum(1 for i in fired if recs[i]["stop_success"] == 0)
    fp = len(fired) - tp
    total_wrong = sum(1 for r in recs if r["stop_success"] == 0)
    prec = tp / len(fired) if fired else float("nan")
    rec_ = tp / total_wrong if total_wrong else float("nan")

    # 조건부 이득 — 발동 run 에 한정한 (검증 후 정답률 − 무검증 정답률).
    # 실험 A 파일럿의 핵심 지표(+28.6%p)와 같은 정의이므로 직접 비교할 수 있다.
    if fired:
        cond_esc = sum(recs[i]["escalate_success"] for i in fired) / len(fired)
        cond_stop = sum(recs[i]["stop_success"] for i in fired) / len(fired)
    else:
        cond_esc = cond_stop = float("nan")

    tsr = sum(succ) / n
    return {
        "policy": label,
        "cond_escalate_tsr": cond_esc,
        "cond_stop_tsr": cond_stop,
        "cond_gain": cond_esc - cond_stop,
        "n_runs": n,
        "fire_rate": len(fired) / n,
        "n_fired": len(fired),
        "TSR": tsr,
        "TSR_ci95": bootstrap_ci_mean(succ),
        "mean_tokens": sum(tok) / n,
        "TPS": (sum(tok) / n) / tsr if tsr > 0 else float("nan"),
        "mean_cost": sum(cost) / n,
        "cost_per_success": (sum(cost) / n) / tsr if tsr > 0 else float("nan"),
        "mean_latency": sum(lat) / n,
        "net_gain": gained - broken,
        "n_corrected": gained,
        "n_broken": broken,
        "gate_precision": prec,
        "gate_recall": rec_,
        "_success_vec": succ,
        "_token_vec": tok,
    }


def bootstrap_ci_mean(values, n_boot: int = BOOTSTRAP_N, seed: int = SEED):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return (lo, hi)


def random_gate_distribution(recs: list[dict], k: int, n_boot: int = 2000,
                             seed: int = SEED) -> dict:
    """무작위 게이트 대조군 — 발동률을 k/n 로 고정하고 대상만 무작위 선택.

    실험계획서 §3-E 필수 대조군. 이 분포와 학습 라우터를 비교하면 "이득이 게이트의
    선별력에서 온 것인가, 아니면 그저 검증 횟수에서 온 것인가"를 분리할 수 있다.
    """
    rng = random.Random(seed)
    n = len(recs)
    idx = list(range(n))
    tsrs, toks, nets = [], [], []
    for _ in range(n_boot):
        pick = set(rng.sample(idx, k))
        esc = [i in pick for i in idx]
        s = sum(recs[i]["escalate_success"] if esc[i] else recs[i]["stop_success"]
                for i in idx) / n
        t = sum(recs[i]["tok_escalate"] if esc[i] else recs[i]["tok_stop"] for i in idx) / n
        g = sum(1 for i in pick if recs[i]["stop_success"] == 0 and recs[i]["escalate_success"] == 1)
        b = sum(1 for i in pick if recs[i]["stop_success"] == 1 and recs[i]["escalate_success"] == 0)
        tsrs.append(s); toks.append(t); nets.append(g - b)
    tsrs.sort()
    return {
        "policy": f"P4 무작위게이트(발동 {k}/{n})",
        "fire_rate": k / n,
        "TSR_mean": sum(tsrs) / len(tsrs),
        "TSR_ci95": (tsrs[int(0.025 * len(tsrs))], tsrs[int(0.975 * len(tsrs)) - 1]),
        "mean_tokens": sum(toks) / len(toks),
        "net_gain_mean": sum(nets) / len(nets),
    }


# ---------------------------------------------------------------------------
# 통계 검정 (문항 단위 주분석)
# ---------------------------------------------------------------------------

def per_question(recs: list[dict], succ_vec: list[int]) -> dict[str, int]:
    """문항 단위 성공 = 3반복 다수결(기존 분석과 동일 규약)."""
    by = defaultdict(list)
    for r, s in zip(recs, succ_vec):
        by[r["question_id"]].append(s)
    return {q: int(sum(v) * 2 > len(v)) for q, v in by.items()}


def mcnemar(a: dict[str, int], b: dict[str, int]) -> dict:
    """문항 단위 McNemar 정확검정(이항). a=정책A, b=정책B."""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)   # A오답 B정답
    b10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)   # A정답 B오답
    nd = b01 + b10
    if nd == 0:
        return {"discordant": 0, "b01": 0, "b10": 0, "p_value": 1.0}
    # 양측 이항검정 (p=0.5)
    def _pmf(k):
        return math.comb(nd, k) * 0.5 ** nd
    obs = min(b01, b10)
    p = sum(_pmf(k) for k in range(0, obs + 1)) * 2
    return {"discordant": nd, "b01": b01, "b10": b10, "p_value": min(1.0, p)}


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict:
    """대응 Wilcoxon 부호순위검정(정규근사, 동순위 보정)."""
    d = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    n = len(d)
    if n < 1:
        return {"n_nonzero": 0, "p_value": 1.0, "z": 0.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_pos = sum(ranks[i] for i in range(n) if d[i] > 0)
    w_neg = sum(ranks[i] for i in range(n) if d[i] < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return {"n_nonzero": n, "p_value": 1.0, "z": 0.0}
    z = (w - mu) / sigma
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return {"n_nonzero": n, "p_value": min(1.0, p), "z": z, "W": w}


def tost_equivalence(a: dict[str, int], b: dict[str, int], margin: float = 0.05,
                     n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> dict:
    """대응 표본 TOST 등가성 검정 (부트스트랩 신뢰구간 방식).

    실험계획서 §3-E: "차이 없음"을 주장하는 모든 곳에 TOST 와 차이의 95% CI 를 병기한다.
    McNemar 의 비유의(p>0.05)는 "차이가 없다"의 근거가 될 수 없다(검정력 부족과 구분되지
    않음). TOST 는 반대로 "차이가 ±margin 안에 있다"를 적극적으로 입증한다.

    문항 단위 대응 차이 d = b - a 의 90% CI(양측 5% 두 개)가 (-margin, +margin) 안에
    완전히 들어가면 등가 판정. 이항 대응 자료이므로 부트스트랩(문항 재표집)을 쓴다.
    margin=0.05 는 문항 60개에서 3문항에 해당하는 실무적 무의미 차이 기준이다.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"equivalent": None, "margin": margin}
    diffs = [b[k] - a[k] for k in keys]
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo90 = boots[int(0.05 * n_boot)]
    hi90 = boots[int(0.95 * n_boot) - 1]
    lo95 = boots[int(0.025 * n_boot)]
    hi95 = boots[int(0.975 * n_boot) - 1]
    return {
        "n_questions": n,
        "margin": margin,
        "diff": obs,
        "ci90": (lo90, hi90),
        "ci95": (lo95, hi95),
        "equivalent": bool(lo90 > -margin and hi90 < margin),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni 보정."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


# ---------------------------------------------------------------------------
# H5 — 난이도 조절효과(moderation): 검증 이득이 난이도에 따라 달라지는가
# ---------------------------------------------------------------------------

def stratified_gain(recs: list[dict]) -> dict:
    """층별로 '검증 전수 적용 시의 이득'을 분해한다.

    반사실 로그에서는 모든 run 이 검증을 받았으므로, 층별 (stop 정답률, escalate 정답률,
    정답화/오답화 건수)를 그대로 계산할 수 있다. 이것이 H5(난이도 조절효과) 검정의 기술
    통계이며, 층 간 이득 차이가 크면 '난이도 층화가 불일치율을 높인다'는 실험계획서
    §3-C 의 설계 근거가 된다.
    """
    def _bucket_hops(h):
        try:
            h = int(h or 0)
        except (TypeError, ValueError):
            h = 0
        if h <= 1:
            return "1홉 이하(단일조회)"
        if h == 2:
            return "2홉"
        return "3홉 이상"

    out = {}
    for key_name, keyfn in [("dataset", lambda r: r["dataset"]),
                            ("task_type", lambda r: r["task_type"]),
                            ("hops", lambda r: _bucket_hops(r.get("reasoning_hops")))]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            groups[keyfn(r)].append(r)
        rows = []
        for g, rs in sorted(groups.items()):
            nn = len(rs)
            stop = sum(r["stop_success"] for r in rs) / nn
            esc = sum(r["escalate_success"] for r in rs) / nn
            fail = sum(1 for r in rs if r["verify_verdict"] == "FAIL")
            corrected = sum(1 for r in rs if r["stop_success"] == 0 and r["escalate_success"] == 1)
            broken = sum(1 for r in rs if r["stop_success"] == 1 and r["escalate_success"] == 0)
            rows.append({
                "stratum": g, "n_runs": nn,
                "stop_tsr": stop, "escalate_tsr": esc, "gain": esc - stop,
                "verify_fail_rate": fail / nn,
                "n_corrected": corrected, "n_broken": broken,
                "net_gain": corrected - broken,
                "oracle_headroom": 1.0 - stop,   # 완벽한 게이트+검증이 회복할 수 있는 최대치
            })
        out[key_name] = rows
    return out


def operating_curve_plot(results: list[dict], rand_rows: list[dict], out_path: Path):
    """발동률–정확도 운영곡선. 실험계획서 §3-A 가 요구한 핵심 그림.

    정책군(자기보고/학습라우터/무작위/oracle)을 같은 축에 올려, 같은 발동률에서 어느
    게이트가 더 높은 정확도를 내는지 — 즉 '이득이 게이트의 선별력에서 오는지'를 보인다.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:  # noqa: BLE001
        print(f"  [plot] 시각화 라이브러리 없음: {exc}")
        return None
    # 한글 폰트(윈도우 맑은고딕) — 없으면 기본 폰트로 진행
    for cand in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if any(f.name == cand for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    def pick(pred):
        rs = [r for r in results if pred(r)]
        rs.sort(key=lambda r: r["fire_rate"])
        return [r["fire_rate"] for r in rs], [r["TSR"] for r in rs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    x, y = pick(lambda r: r["policy"].startswith("P2"))
    ax.plot(x, y, "o-", label="자기보고 게이트 (S5 계열)", color="#c44e52")
    x, y = pick(lambda r: r["policy"].startswith("P3") and "전체신호" in r["policy"])
    ax.plot(x, y, "s-", label="학습 라우터 (외부신호+자기보고)", color="#4c72b0")
    x, y = pick(lambda r: r["policy"].startswith("P3") and "외부신호만" in r["policy"])
    ax.plot(x, y, "^-", label="학습 라우터 (외부신호만)", color="#55a868")
    if rand_rows:
        rr = sorted(rand_rows, key=lambda r: r["fire_rate"])
        ax.plot([r["fire_rate"] for r in rr], [r["TSR_mean"] for r in rr],
                "--", color="gray", label="무작위 게이트 (대조군)")
        ax.fill_between([r["fire_rate"] for r in rr],
                        [r["TSR_ci95"][0] for r in rr], [r["TSR_ci95"][1] for r in rr],
                        color="gray", alpha=0.18)
    p0 = [r for r in results if r["policy"].startswith("P0")][0]
    p1 = [r for r in results if r["policy"].startswith("P1")][0]
    p5 = [r for r in results if r["policy"].startswith("P5")][0]
    p6l = [r for r in results if r["policy"].startswith("P6")]
    p6 = p6l[0] if p6l else None
    ax.axhline(p0["TSR"], color="black", lw=1, ls=":", label=f"무검증 {p0['TSR']:.3f}")
    ax.scatter([p1["fire_rate"]], [p1["TSR"]], marker="*", s=220, color="#8172b2",
               zorder=5, label=f"항상검증 {p1['TSR']:.3f}")
    ax.scatter([p5["fire_rate"]], [p5["TSR"]], marker="D", s=90, color="#ccb974",
               zorder=5, label=f"oracle(정확도상한) {p5['TSR']:.3f}")
    if p6:
        ax.scatter([p6["fire_rate"]], [p6["TSR"]], marker="P", s=130, color="#937860",
                   zorder=6, label=f"oracle+(비용하한) {p6['TSR']:.3f}")
    ax.set_xlabel("게이트 발동률 (검증에 회부된 run 비율)")
    ax.set_ylabel("정확도 (TSR)")
    ax.set_title("발동률–정확도 운영곡선")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for pred, sty, lab, col in [
        (lambda r: r["policy"].startswith("P2"), "o-", "자기보고 게이트", "#c44e52"),
        (lambda r: r["policy"].startswith("P3") and "전체신호" in r["policy"], "s-",
         "학습 라우터(전체신호)", "#4c72b0"),
        (lambda r: r["policy"].startswith("P3") and "외부신호만" in r["policy"], "^-",
         "학습 라우터(외부신호만)", "#55a868"),
    ]:
        rs = sorted([r for r in results if pred(r)], key=lambda r: r["mean_tokens"])
        ax.plot([r["mean_tokens"] for r in rs], [r["TSR"] for r in rs], sty,
                label=lab, color=col)
    marks = [(p0, "o", "black"), (p1, "*", "#8172b2"), (p5, "D", "#ccb974")]
    if p6:
        marks.append((p6, "P", "#937860"))
    for r, m, c in marks:
        ax.scatter([r["mean_tokens"]], [r["TSR"]], marker=m, s=150, color=c, zorder=5)
        ax.annotate(r["policy"].split(" ")[0], (r["mean_tokens"], r["TSR"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("run 당 평균 총 토큰")
    ax.set_ylabel("정확도 (TSR)")
    ax.set_title("비용–정확도 트레이드오프")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("실험 C — 게이트 정책 운영곡선 (반사실 재현, S7 로그 기반)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="실험 C — 반사실 게이트 재현")
    ap.add_argument("--structure", default="s7")
    ap.add_argument("--out", default=None, help="산출물 디렉토리(기본 RESULTS_DIR/analysis_expC)")
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="여러 결과 디렉토리를 풀링해 분석(예: results_expC results_expC_ext). "
                         "미지정 시 RESULTS_DIR 하나만 쓴다.")
    ap.add_argument("--label", default="", help="리포트 제목에 붙일 표본 라벨")
    args = ap.parse_args(argv)

    if args.dirs:
        runs = []
        for d in args.dirs:
            rs = load_runs(args.structure, Path(d).resolve())
            print(f"  [로드] {d}: {len(rs)} run")
            runs.extend(rs)
    else:
        runs = load_runs(args.structure)
    recs = build_records(runs)
    n = len(recs)
    out_dir = Path(args.out) if args.out else (config.RESULTS_DIR / "analysis_expC")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    lab = f"  표본={args.label}" if args.label else ""
    n_q = len({r["question_id"] for r in recs})
    print(f"실험 C — 반사실 게이트 재현   구조={args.structure.upper()}  "
          f"run={n}  문항={n_q}{lab}")
    print("=" * 84)
    base_wrong = sum(1 for r in recs if r["stop_success"] == 0)
    print(f"1차 생성(무검증 등가) 오답 {base_wrong}/{n} = {base_wrong/n:.3f}")
    print(f"검증 전수 적용 시 FAIL 판정 {sum(1 for r in recs if r['verify_verdict']=='FAIL')}건, "
          f"재생성 발생 {sum(1 for r in recs if r['retried'])}건")
    print()

    results: list[dict] = []

    # --- P0 / P1 ---
    p0 = evaluate(recs, [False] * n, "P0 무검증(전부 stop)")
    p1 = evaluate(recs, [True] * n, "P1 항상검증(전부 escalate)")
    results += [p0, p1]

    # --- P2 자기보고 게이트 스윕 (S5 재현) ---
    def eff_conf(r):
        return 0.0 if r["code_conflict"] else r["conf_self"]

    p2_rows = []
    for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]:
        esc = [eff_conf(r) < th for r in recs]
        res = evaluate(recs, esc, f"P2 자기보고 θ={th}")
        res["threshold"] = th
        p2_rows.append(res)
    results += p2_rows

    # --- P3 학습 라우터 (문항 그룹 교차적합) ---
    p3_rows = []
    router_meta = {}
    for tag, feats in [("전체신호", ROUTER_FEATURES), ("외부신호만", ROUTER_FEATURES_EXT)]:
        oof, coefs = router_oof_scores(recs, feats)
        if oof is None:
            print("[경고] scikit-learn 미설치 또는 표본 부족 — 학습 라우터 생략")
            continue
        labels = [1 - r["stop_success"] for r in recs]
        a = auc(oof, labels)
        router_meta[tag] = {"oof_auc": a, "coefficients": coefs,
                            "features": list(feats.keys())}
        # 점수 임계값을 발동률 격자로 변환(발동률을 직접 통제해 비교를 쉽게)
        srt = sorted(oof, reverse=True)
        for target in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]:
            k = max(1, int(round(target * n)))
            thr = srt[k - 1]
            esc = [s >= thr for s in oof]
            res = evaluate(recs, esc, f"P3 라우터({tag}) 목표발동 {target:.0%}")
            res["threshold"] = thr
            res["router_variant"] = tag
            res["oof_auc"] = a
            p3_rows.append(res)
    results += p3_rows

    # --- P5 oracle (1차 답이 오답인 run 만 발동) ---
    p5 = evaluate(recs, [r["stop_success"] == 0 for r in recs], "P5 oracle 게이트(정확도 상한)")
    results.append(p5)

    # --- P6 oracle+ : 검증이 실제로 결과를 바꾸는 run 만 발동 (비용 하한) ---
    # 검증기가 고칠 수 있는 오답은 전체 오답의 일부뿐이다(재현율 한계). 게이트가 아무리
    # 완벽해도 '검증해도 안 바뀌는 오답'에 발동하면 비용만 쓴다. P6 는 결과가 바뀌는 run
    # (정답화 또는 오답화가 일어나는 run)에만 발동하는 가상의 게이트로, **현 검증기 성능
    # 하에서 도달 가능한 최소 비용**을 준다. P5 와 P6 의 차이가 "게이트 정밀도 개선의
    # 남은 여지", P0 와 P6 의 정확도 차이가 "검증기 재현율이 허용하는 최대 이득"이다.
    p6 = evaluate(recs,
                  [r["stop_success"] != r["escalate_success"] for r in recs],
                  "P6 oracle+ 게이트(비용 하한)")
    results.append(p6)

    # --- 출력: 정책 표 ---
    hdr = (f"{'정책':<34}{'발동률':>8}{'TSR':>8}{'TSR 95%CI':>18}"
           f"{'평균토큰':>10}{'TPS':>9}{'순이득':>7}{'조건부이득':>10}"
           f"{'정밀도':>8}{'재현율':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ci = r["TSR_ci95"]
        print(f"{r['policy']:<34}{r['fire_rate']:>8.3f}{r['TSR']:>8.3f}"
              f"{'['+_fmt(ci[0])+', '+_fmt(ci[1])+']':>18}"
              f"{r['mean_tokens']:>10.0f}{r['TPS']:>9.0f}"
              f"{r['net_gain']:>7d}{_fmt(r['cond_gain']):>10}"
              f"{_fmt(r['gate_precision']):>8}{_fmt(r['gate_recall']):>8}")
    print()

    # --- P4 무작위 게이트 대조군: 라우터 최선 발동률과 동일 발동률에서 비교 ---
    print("-" * 84)
    print("무작위 게이트 대조군 (동일 발동률, 부트스트랩 2,000회)")
    print("-" * 84)
    rand_rows = []
    compare_targets = sorted({r["n_fired"] for r in p3_rows} |
                             {r["n_fired"] for r in p2_rows if 0 < r["n_fired"] < n})
    for k in compare_targets:
        if k <= 0 or k >= n:
            continue
        rr = random_gate_distribution(recs, k)
        rand_rows.append(rr)
        print(f"  발동 {k:>3}/{n} ({k/n:>5.1%})  TSR {rr['TSR_mean']:.3f} "
              f"[{rr['TSR_ci95'][0]:.3f}, {rr['TSR_ci95'][1]:.3f}]  "
              f"평균토큰 {rr['mean_tokens']:>6.0f}  순이득 {rr['net_gain_mean']:+.2f}")
    print()

    # --- 라우터 진단 ---
    if router_meta:
        print("-" * 84)
        print("학습 라우터 진단 (문항 단위 GroupKFold out-of-fold)")
        print("-" * 84)
        for tag, m in router_meta.items():
            print(f"  {tag}: OOF AUC = {m['oof_auc']:.3f}")
            print(f"    계수(전체적합, 표준화 후): {m['coefficients']}")
        print()

    # --- 참조: 신호별 단독 AUC 재현성 (실험 B 를 독립 표본에서 재검증) ---
    print("-" * 84)
    print("신호별 단독 AUC — 실험 B(S5 로그) 결과의 독립 표본 재현성 확인")
    print("-" * 84)
    labels = [1 - r["stop_success"] for r in recs]
    sig_auc = {}
    for name, fn in ALL_SIGNALS.items():
        a = auc([fn(r) for r in recs], labels)
        sig_auc[name] = a
        print(f"  {name:<18} AUC {_fmt(a)}")
    print()

    # --- 통계 검정 (문항 단위 주분석) ---
    print("-" * 84)
    print("통계 검정 — 문항 단위(3반복 다수결) McNemar / run 단위 토큰 Wilcoxon")
    print("-" * 84)
    q0 = per_question(recs, p0["_success_vec"])
    tests = {}
    key_policies = [p1] + [r for r in p2_rows if abs(r.get("threshold", 0) - 0.8) < 1e-9]
    # 라우터는 각 변형에서 순이득 최대 지점을 대표로 검정
    for tag in {r["router_variant"] for r in p3_rows}:
        cand = [r for r in p3_rows if r["router_variant"] == tag]
        best = max(cand, key=lambda r: (r["net_gain"], r["TSR"], -r["mean_tokens"]))
        key_policies.append(best)
    key_policies.append(p5)
    key_policies.append(p6)

    raw_p = {}
    for pol in key_policies:
        qa = per_question(recs, pol["_success_vec"])
        mc = mcnemar(q0, qa)
        wl = wilcoxon_signed_rank(pol["_token_vec"], p0["_token_vec"])
        tests[pol["policy"]] = {"mcnemar_vs_P0": mc, "wilcoxon_tokens_vs_P0": wl}
        tt = tost_equivalence(q0, qa, margin=0.05)
        tests[pol["policy"]]["tost_vs_P0"] = tt
        raw_p[pol["policy"]] = mc["p_value"]
        print(f"  {pol['policy']:<40} vs P0 무검증")
        print(f"      정확도 McNemar: 불일치 {mc['discordant']} "
              f"(P0오답→정답 {mc['b01']}, P0정답→오답 {mc['b10']}) p={mc['p_value']:.4f}")
        print(f"      토큰 Wilcoxon : n={wl['n_nonzero']} p={wl['p_value']:.3e}")
        print(f"      TOST 등가성(±0.05): 차이 {tt['diff']:+.4f} "
              f"90%CI [{tt['ci90'][0]:+.4f}, {tt['ci90'][1]:+.4f}] "
              f"95%CI [{tt['ci95'][0]:+.4f}, {tt['ci95'][1]:+.4f}] "
              f"→ {'등가 입증' if tt['equivalent'] else '등가 미입증'}")
    adj = holm(raw_p)
    print()
    print("  Holm 보정 후 McNemar p:")
    for k, v in adj.items():
        print(f"      {k:<44} p_adj={v:.4f}")
    for k in tests:
        tests[k]["mcnemar_p_holm"] = adj[k]
    print()

    # --- S5 실측 대조: 재구성 타당성 검증 ---
    print("-" * 84)
    print("재구성 타당성 검증 — 자기보고 θ=0.8 정책 재현 vs 실제 S5 실측")
    print("-" * 84)
    p2_08 = [r for r in p2_rows if abs(r["threshold"] - 0.8) < 1e-9][0]
    s5_path = config.RESULTS_DIR.parent / "results_main" / "s5" / "raw_runs.jsonl"
    s5_line = None
    if s5_path.exists():
        s5 = [json.loads(l) for l in open(s5_path, encoding="utf-8") if l.strip()]
        s5 = [r for r in s5 if not r.get("run_error")]
        s5_fire = sum(1 for r in s5 if r.get("gate_decision") == "escalate") / len(s5)
        s5_tsr = sum(r["success"] for r in s5) / len(s5)
        s5_tok = sum(r["total_tokens"] for r in s5) / len(s5)
        s5_line = {"fire_rate": s5_fire, "TSR": s5_tsr, "mean_tokens": s5_tok, "n": len(s5)}
        esc08 = [eff_conf(r) < 0.8 for r in recs]
        tok_s5eq = sum(r["tok_escalate_s5equiv"] if e else r["tok_stop"]
                       for r, e in zip(recs, esc08)) / n
        s5_line["replay_fire_rate"] = p2_08["fire_rate"]
        s5_line["replay_TSR"] = p2_08["TSR"]
        s5_line["replay_mean_tokens"] = p2_08["mean_tokens"]
        s5_line["replay_mean_tokens_s5equiv"] = tok_s5eq
        print(f"  재현(P2 θ=0.8) : 발동률 {p2_08['fire_rate']:.3f}  TSR {p2_08['TSR']:.3f}  "
              f"평균토큰 {p2_08['mean_tokens']:.0f}")
        print(f"    └ S5 등가 비용 보정(재생성 후 2차 verify 포함): 평균토큰 {tok_s5eq:.0f}")
        print(f"  실측(S5 180run): 발동률 {s5_fire:.3f}  TSR {s5_tsr:.3f}  평균토큰 {s5_tok:.0f}")
        print("  → 발동률·TSR 이 근접하면 반사실 재구성이 실제 캐스케이드 실행을 잘 대리한다는 근거다.")
        print("  → 비용은 S7 이 S5 의 무의미한 2차 검증 호출을 제거했으므로 구조적으로 더 낮다.")
    else:
        print("  (results_main/s5 없음 — 대조 생략)")
    print()

    # --- H3 직접 검정: 항상검증과 '등가 정확도'이면서 가장 저렴한 정책 ---
    # H3(정밀화): "동적 캐스케이드는 항상 검증과 동등한 TSR 을 유지하며 비용을 유의하게
    # 낮춘다". 등가성은 McNemar 비유의가 아니라 TOST 로 판정해야 한다(§3-E).
    print("-" * 84)
    print("H3 직접 검정 — 항상검증(P1)과 등가 정확도이면서 비용이 낮은 정책")
    print("-" * 84)
    q1 = per_question(recs, p1["_success_vec"])
    h3_rows = []
    for pol in results:
        if pol is p1:
            continue
        qa = per_question(recs, pol["_success_vec"])
        tt = tost_equivalence(q1, qa, margin=0.05)
        mc = mcnemar(q1, qa)
        wl = wilcoxon_signed_rank(pol["_token_vec"], p1["_token_vec"])
        saving = 1.0 - pol["mean_tokens"] / p1["mean_tokens"]
        h3_rows.append({
            "policy": pol["policy"], "fire_rate": pol["fire_rate"], "TSR": pol["TSR"],
            "tost_equivalent_to_P1": tt["equivalent"], "tost_diff": tt["diff"],
            "tost_ci90": tt["ci90"], "mcnemar_p_vs_P1": mc["p_value"],
            "mean_tokens": pol["mean_tokens"], "token_saving_vs_P1": saving,
            "wilcoxon_p_vs_P1": wl["p_value"],
        })
    eq_rows = [r for r in h3_rows if r["tost_equivalent_to_P1"]]
    eq_rows.sort(key=lambda r: r["mean_tokens"])
    if eq_rows:
        print(f"  {'정책':<38}{'발동률':>8}{'TSR':>8}{'평균토큰':>10}"
              f"{'절감':>8}{'Wilcoxon p':>13}")
        for r in eq_rows[:8]:
            print(f"  {r['policy']:<38}{r['fire_rate']:>8.3f}{r['TSR']:>8.3f}"
                  f"{r['mean_tokens']:>10.0f}{r['token_saving_vs_P1']:>7.1%}"
                  f"{r['wilcoxon_p_vs_P1']:>13.2e}")
        best = eq_rows[0]
        print()
        print(f"  → 최저비용 등가정책: {best['policy']}")
        print(f"     TSR {best['TSR']:.3f} vs 항상검증 {p1['TSR']:.3f} "
              f"(차이 {best['tost_diff']:+.4f}, 90%CI "
              f"[{best['tost_ci90'][0]:+.4f}, {best['tost_ci90'][1]:+.4f}] ⊂ ±0.05 → 등가)")
        print(f"     토큰 {best['mean_tokens']:.0f} vs {p1['mean_tokens']:.0f} "
              f"= {best['token_saving_vs_P1']:.1%} 절감 (Wilcoxon p={best['wilcoxon_p_vs_P1']:.2e})")
    else:
        print("  ±0.05 마진에서 항상검증과 등가로 입증된 정책이 없다 "
              "(표본이 작아 CI 가 넓은 경우 흔하다 — 검정력 문제와 구분해 서술할 것).")
    print()

    # --- H5 난이도 조절효과 (검증 전수 적용 기준 층별 이득) ---
    print("-" * 84)
    print("H5 난이도 조절효과 — 층별 검증 이득 분해 (검증 전수 적용 기준)")
    print("-" * 84)
    strat = stratified_gain(recs)
    for key, rows in strat.items():
        print(f"  [{key}]")
        print(f"    {'층':<20}{'n':>5}{'무검증TSR':>10}{'검증TSR':>9}{'이득':>8}"
              f"{'FAIL률':>8}{'정답화':>7}{'오답화':>7}{'상한여지':>9}")
        for r in rows:
            print(f"    {r['stratum']:<20}{r['n_runs']:>5}{r['stop_tsr']:>10.3f}"
                  f"{r['escalate_tsr']:>9.3f}{r['gain']:>+8.3f}"
                  f"{r['verify_fail_rate']:>8.3f}{r['n_corrected']:>7}{r['n_broken']:>7}"
                  f"{r['oracle_headroom']:>9.3f}")
    print()

    # --- 운영곡선 그림 ---
    png = operating_curve_plot(results, rand_rows, out_dir / "operating_curve.png")
    if png:
        print(f"저장: {png}")

    # --- 저장 ---
    payload = {
        "structure": args.structure,
        "label": args.label,
        "source_dirs": args.dirs or [str(config.RESULTS_DIR)],
        "n_runs": n,
        "n_questions": n_q,
        "cohort_composition": {k: v for k, v in sorted(
            Counter(r["dataset"] for r in recs).items())},
        "seed": SEED,
        "baseline_first_gen_error_rate": base_wrong / n,
        "n_verify_fail": sum(1 for r in recs if r["verify_verdict"] == "FAIL"),
        "n_retried": sum(1 for r in recs if r["retried"]),
        "policies": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
        "random_gate_control": rand_rows,
        "router": router_meta,
        "signal_auc": sig_auc,
        "h5_stratified_gain": strat,
        "h3_equivalence_vs_always_verify": h3_rows,
        "tests": tests,
        "replay_validation_vs_s5": s5_line,
    }
    with open(out_dir / "replay_gates.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    # 운영곡선 CSV (발동률–정확도–비용)
    import csv
    with open(out_dir / "operating_curve.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "fire_rate", "TSR", "TSR_lo95", "TSR_hi95", "mean_tokens",
                    "TPS", "mean_cost", "cost_per_success", "mean_latency",
                    "net_gain", "n_corrected", "n_broken",
                    "cond_stop_tsr", "cond_escalate_tsr", "cond_gain",
                    "gate_precision", "gate_recall"])
        for r in results:
            w.writerow([r["policy"], f"{r['fire_rate']:.4f}", f"{r['TSR']:.4f}",
                        f"{r['TSR_ci95'][0]:.4f}", f"{r['TSR_ci95'][1]:.4f}",
                        f"{r['mean_tokens']:.1f}", f"{r['TPS']:.1f}",
                        f"{r['mean_cost']:.6f}", f"{r['cost_per_success']:.6f}",
                        f"{r['mean_latency']:.2f}",
                        r["net_gain"], r["n_corrected"], r["n_broken"],
                        _fmt(r["cond_stop_tsr"], 4), _fmt(r["cond_escalate_tsr"], 4),
                        _fmt(r["cond_gain"], 4),
                        f"{r['gate_precision']:.4f}" if not math.isnan(r["gate_precision"]) else "",
                        f"{r['gate_recall']:.4f}" if not math.isnan(r["gate_recall"]) else ""])

    # run 단위 반사실 원자료 (재현·재분석용)
    with open(out_dir / "replay_records.csv", "w", encoding="utf-8-sig", newline="") as fh:
        cols = ["run_id", "question_id", "run_index", "dataset", "task_type", "reasoning_hops",
                "stop_success", "escalate_success", "verify_verdict", "retried",
                "tok_stop", "tok_escalate", "cost_stop", "cost_escalate",
                "conf_self", "code_conflict"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)

    with open(out_dir / "h5_stratified_gain.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grouping", "stratum", "n_runs", "stop_tsr", "escalate_tsr", "gain",
                    "verify_fail_rate", "n_corrected", "n_broken", "net_gain",
                    "oracle_headroom"])
        for key, rows in strat.items():
            for r in rows:
                w.writerow([key, r["stratum"], r["n_runs"], f"{r['stop_tsr']:.4f}",
                            f"{r['escalate_tsr']:.4f}", f"{r['gain']:.4f}",
                            f"{r['verify_fail_rate']:.4f}", r["n_corrected"], r["n_broken"],
                            r["net_gain"], f"{r['oracle_headroom']:.4f}"])

    print(f"저장: {out_dir/'replay_gates.json'}")
    print(f"저장: {out_dir/'operating_curve.csv'}")
    print(f"저장: {out_dir/'replay_records.csv'}")
    return payload


if __name__ == "__main__":
    main()
