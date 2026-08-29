"""실험 B (오프라인) — 게이트 신호 후보의 오답 식별력 평가.

추가 LLM 호출 없이 기존 raw_runs.jsonl 만으로, "생성 시점에 얻을 수 있는 신호"들이
오답(success==0)을 얼마나 잘 식별하는지 AUC/정밀도/재현율로 평가한다.

배경: S5 의 자기보고 확신도는 소수 값에 군집(정밀도 0.239)하고, S6 의 self-probe 는
더 악화(0.111)되었다. 따라서 모델 자기평가에 의존하지 않는 신호를 찾아야 한다.

  python -m src.eval_gate_signals [구조명 ...]     (기본 s0)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import config, metrics, tools

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _load(structure: str) -> list[dict]:
    path = config.RESULTS_DIR / structure / "raw_runs.jsonl"
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if not r.get("run_error"):
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# 신호 산출 (모두 생성 시점에 계산 가능 — 정답을 참조하지 않는다)
# ---------------------------------------------------------------------------

def sig_code_conflict(r: dict) -> float:
    """B-1 추론답 vs 코드실행답 불일치. 1=충돌(위험)."""
    fa = r.get("final_answer", "")
    fn = metrics.normalize_number(fa)
    if fn is None:
        return 0.0
    computed = []
    for e in (r.get("used_expression") or []):
        v, err = tools.safe_calc(e)
        if err is None and v is not None:
            computed.append(v)
    if not computed:
        return 0.0
    for cv in computed:
        if metrics.is_correct(cv, fa).correct:
            return 0.0
    return 1.0


def sig_no_expression(r: dict) -> float:
    """B-1b 수치 답인데 산술식을 제시하지 않음. 1=위험."""
    if metrics.normalize_number(r.get("final_answer", "")) is None:
        return 0.0
    return 0.0 if (r.get("used_expression") or []) else 1.0


def _evidence_text(r: dict) -> str:
    parts = []
    for c in (r.get("retrieved_evidence") or []):
        if isinstance(c, dict):
            parts.append(str(c.get("text", "")))
        else:
            parts.append(str(c))
    return " ".join(parts)


def sig_evidence_coverage(r: dict) -> float:
    """B-2 추론에 등장한 수치가 근거에 실제로 있는지. 반환=미발견 비율(높을수록 위험)."""
    ev = _evidence_text(r)
    if not ev:
        return 0.0
    ev_nums = set(x.replace(",", "") for x in _NUM.findall(ev))
    if not ev_nums:
        return 0.0
    cited = [x.replace(",", "") for x in _NUM.findall(r.get("reasoning_trace", "") or "")]
    cited = [c for c in cited if len(c.replace(".", "").replace("-", "")) >= 3]  # 잡음 제거
    if not cited:
        return 0.0
    miss = sum(1 for c in cited if c not in ev_nums)
    return miss / len(cited)


def sig_hops(r: dict) -> float:
    """B-3 문항이 요구하는 추론 홉 수(사전 라벨). 클수록 위험."""
    try:
        return float(r.get("reasoning_hops") or 0)
    except (TypeError, ValueError):
        return 0.0


def sig_n_expr(r: dict) -> float:
    """B-3b 모델이 쓴 산술식 개수. 많을수록 다단계 → 위험."""
    return float(len(r.get("used_expression") or []))


def sig_format_risk(r: dict) -> float:
    """B-4 답 형식 위험도 — 퍼센트/단위/부호가 섞이면 오류율이 높다."""
    fa = str(r.get("final_answer", ""))
    risk = 0.0
    if "%" in fa:
        risk += 1
    if re.search(r"(억|백만|천|원|배|명|개|pt|USD|\$)", fa):
        risk += 1
    if fa.strip().startswith("-") or "감소" in fa:
        risk += 1
    return risk


def sig_output_tokens(r: dict) -> float:
    """B-4b 출력 토큰 수 — 장황한 추론은 어려운 문항의 대리지표."""
    return float(r.get("output_tokens") or 0)


def sig_self_report(r: dict) -> float:
    """(대조) 모델 자기보고 확신도. 낮을수록 위험 → 부호 반전해 위험도로 변환."""
    try:
        c = float(r.get("confidence_self_report") or 0.0)
    except (TypeError, ValueError):
        c = 0.0
    return 1.0 - c


SIGNALS = {
    "B1 계산불일치": sig_code_conflict,
    "B1b 산술식없음": sig_no_expression,
    "B2 근거미발견율": sig_evidence_coverage,
    "B3 추론홉수": sig_hops,
    "B3b 산술식개수": sig_n_expr,
    "B4 답형식위험": sig_format_risk,
    "B4b 출력토큰": sig_output_tokens,
    "(대조) 자기보고": sig_self_report,
}


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def auc(scores: list[float], labels: list[int]) -> float:
    """labels: 1=오답(양성). Mann-Whitney U 기반 AUC."""
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


def best_operating_point(scores: list[float], labels: list[int]):
    """임계값을 훑어 F1 최대 지점의 (임계값, 발동율, 정밀도, 재현율, F1)."""
    best = None
    for thr in sorted(set(scores)):
        fired = [(s >= thr) for s in scores]
        tp = sum(1 for f, y in zip(fired, labels) if f and y == 1)
        fp = sum(1 for f, y in zip(fired, labels) if f and y == 0)
        fn = sum(1 for f, y in zip(fired, labels) if not f and y == 1)
        if tp + fp == 0 or tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        rate = (tp + fp) / len(scores)
        if best is None or f1 > best[4]:
            best = (thr, rate, prec, rec, f1)
    return best


def logistic_cv(X: list[list[float]], y: list[int], folds: int = 5):
    """B-5 학습된 라우터 — 표준화 + 로지스틱 회귀, 층화 교차검증 AUC."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return None
    import numpy as np
    Xa, ya = np.array(X, dtype=float), np.array(y)
    if ya.sum() < folds or (len(ya) - ya.sum()) < folds:
        return None
    oof = np.zeros(len(ya))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260601)
    for tr, te in skf.split(Xa, ya):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(Xa[tr], ya[tr])
        oof[te] = m.predict_proba(Xa[te])[:, 1]
    return auc(list(oof), list(ya)), oof


def report(structure: str) -> None:
    runs = _load(structure)
    y = [1 - int(r["success"]) for r in runs]          # 1 = 오답(양성)
    print("=" * 78)
    print(f"구조 {structure.upper()} — run {len(runs)}개, 오답 {sum(y)}개 (오답률 {sum(y)/len(y):.3f})")
    print("=" * 78)
    print(f"{'신호':<18}{'AUC':>7}{'발동율':>9}{'정밀도':>9}{'재현율':>9}{'F1':>8}")
    print("-" * 78)

    cols = []
    for name, fn in SIGNALS.items():
        s = [fn(r) for r in runs]
        cols.append(s)
        a = auc(s, y)
        bop = best_operating_point(s, y)
        if bop:
            _, rate, prec, rec, f1 = bop
            print(f"{name:<18}{a:>7.3f}{rate:>9.2f}{prec:>9.3f}{rec:>9.3f}{f1:>8.3f}")
        else:
            print(f"{name:<18}{a:>7.3f}{'—':>9}{'—':>9}{'—':>9}{'—':>8}")

    # B-5 조합 모델 (자기보고 제외한 외부 신호만 / 전체)
    ext_idx = [i for i, n in enumerate(SIGNALS) if not n.startswith("(대조)")]
    X_ext = [[cols[i][j] for i in ext_idx] for j in range(len(runs))]
    X_all = [[c[j] for c in cols] for j in range(len(runs))]
    print("-" * 78)
    for label, X in [("B5 조합(외부신호만)", X_ext), ("B5 조합(자기보고 포함)", X_all)]:
        res = logistic_cv(X, y)
        if res is None:
            print(f"{label:<18}   scikit-learn 미설치 또는 표본 부족으로 생략")
            continue
        a, oof = res
        bop = best_operating_point(list(oof), y)
        if bop:
            _, rate, prec, rec, f1 = bop
            print(f"{label:<18}{a:>7.3f}{rate:>9.2f}{prec:>9.3f}{rec:>9.3f}{f1:>8.3f}")
        else:
            print(f"{label:<18}{a:>7.3f}")
    print()
    print("  참고 · 기존 게이트 실측: S5 정밀도 0.239 / 재현율 0.700, S6 정밀도 0.111 / 재현율 0.026")
    print("  AUC 0.5 = 무작위, 1.0 = 완벽. 양성=오답.")
    print()


def main(argv=None):
    args = (argv or sys.argv[1:]) or ["s0"]
    for st in args:
        p = config.RESULTS_DIR / st / "raw_runs.jsonl"
        if not p.exists():
            print(f"[건너뜀] {p} 없음")
            continue
        report(st)


if __name__ == "__main__":
    main()
