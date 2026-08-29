"""실험 C 결과를 논문·보고용 마크다운 표로 변환.

replay_gates.py 가 저장한 replay_gates.json 을 읽어, 사람이 다시 계산하지 않아도 되도록
표를 그대로 뽑아 준다. 수치를 손으로 옮겨 적는 과정에서 생기는 전사 오류를 막기 위한
장치다(부분 표본 외삽 사고 이후 도입한 절차 — 실험 A 15:27 사례 참조).

  python -m src.report_expC results_expC/analysis_expC/replay_gates.json [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _f(v, nd=3):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return "—"
    return f"{v:.{nd}f}"


def _pct(v, nd=1):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return "—"
    return f"{v*100:.{nd}f}%"


def policy_table(pols: list[dict], keep=None) -> str:
    rows = [p for p in pols if keep is None or keep(p)]
    out = ["| 정책 | 발동률 | TSR | TSR 95% CI | 평균 토큰 | TPS | 성공당 $ | 순이득 | 조건부 이득 | 게이트 정밀도 | 게이트 재현율 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        ci = p["TSR_ci95"]
        out.append(
            f"| {p['policy']} | {_pct(p['fire_rate'])} | **{_f(p['TSR'])}** | "
            f"[{_f(ci[0])}, {_f(ci[1])}] | {p['mean_tokens']:,.0f} | {p['TPS']:,.0f} | "
            f"{_f(p['mean_cost']/p['TSR'] if p['TSR'] else float('nan'), 4)} | "
            f"{p['net_gain']:+d} | {_f(p.get('cond_gain'))} | "
            f"{_f(p.get('gate_precision'))} | {_f(p.get('gate_recall'))} |")
    return "\n".join(out)


def random_table(rows: list[dict]) -> str:
    out = ["| 발동률 | 무작위 게이트 TSR (평균) | 95% CI | 평균 토큰 | 순이득 (평균) |",
           "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["fire_rate"]):
        out.append(f"| {_pct(r['fire_rate'])} | {_f(r['TSR_mean'])} | "
                   f"[{_f(r['TSR_ci95'][0])}, {_f(r['TSR_ci95'][1])}] | "
                   f"{r['mean_tokens']:,.0f} | {r['net_gain_mean']:+.2f} |")
    return "\n".join(out)


def strat_table(rows: list[dict]) -> str:
    out = ["| 층 | run | 무검증 TSR | 검증 TSR | 이득 | 검증 FAIL률 | 정답화 | 오답화 | 순이득 | oracle 여지 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['stratum']} | {r['n_runs']} | {_f(r['stop_tsr'])} | "
                   f"{_f(r['escalate_tsr'])} | {r['gain']:+.3f} | "
                   f"{_pct(r['verify_fail_rate'])} | {r['n_corrected']} | {r['n_broken']} | "
                   f"{r['net_gain']:+d} | {_f(r['oracle_headroom'])} |")
    return "\n".join(out)


def signal_table(sig: dict) -> str:
    out = ["| 신호 | AUC |", "|---|---|"]
    for k, v in sorted(sig.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0)):
        out.append(f"| {k} | {_f(v)} |")
    return "\n".join(out)


def test_table(tests: dict) -> str:
    out = ["| 정책 (vs P0 무검증) | McNemar 불일치쌍 | 개선/악화 | p | Holm p | TOST ±0.05 | 차이 95% CI | 토큰 Wilcoxon p |",
           "|---|---|---|---|---|---|---|---|"]
    for k, v in tests.items():
        mc = v["mcnemar_vs_P0"]
        wl = v["wilcoxon_tokens_vs_P0"]
        tt = v.get("tost_vs_P0", {})
        eq = tt.get("equivalent")
        eqs = "등가 입증" if eq else ("등가 미입증" if eq is not None else "—")
        ci = tt.get("ci95", (None, None))
        out.append(f"| {k} | {mc['discordant']} | {mc['b01']}/{mc['b10']} | "
                   f"{mc['p_value']:.4f} | {v.get('mcnemar_p_holm', float('nan')):.4f} | "
                   f"{eqs} | [{_f(ci[0], 4)}, {_f(ci[1], 4)}] | {wl['p_value']:.2e} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="실험 C 결과 -> 마크다운 표")
    ap.add_argument("json_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    d = json.load(open(args.json_path, encoding="utf-8"))
    pols = d["policies"]

    parts = []
    parts.append(f"# 실험 C 결과표 — {d.get('label') or d['structure'].upper()}\n")
    parts.append(f"- 표본: run {d['n_runs']}개 / 문항 {d.get('n_questions','?')}개 "
                 f"· 구성 {d.get('cohort_composition', {})}")
    parts.append(f"- 1차 생성(무검증 등가) 오답률: **{_f(d['baseline_first_gen_error_rate'])}**")
    parts.append(f"- 검증 전수 적용 시 FAIL 판정 {d['n_verify_fail']}건 "
                 f"({_pct(d['n_verify_fail']/d['n_runs'])}), 재생성 {d['n_retried']}건")
    parts.append(f"- 부트스트랩 시드 {d['seed']}, 반복 10,000회\n")

    parts.append("## 1. 기준 정책 · oracle 상한(정확도)·하한(비용)\n")
    parts.append(policy_table(pols, lambda p: p["policy"][:2] in ("P0", "P1", "P5", "P6")) + "\n")

    parts.append("## 2. 자기보고 게이트 (S5 계열) 임계값 스윕\n")
    parts.append(policy_table(pols, lambda p: p["policy"].startswith("P2")) + "\n")

    parts.append("## 3. 학습 라우터 게이트 (실험 B B-5 조합) — 문항 단위 교차적합\n")
    for tag in ("전체신호", "외부신호만"):
        rows = [p for p in pols if p["policy"].startswith("P3") and tag in p["policy"]]
        if not rows:
            continue
        auc = rows[0].get("oof_auc")
        parts.append(f"### 3.{1 if tag=='전체신호' else 2} {tag} (out-of-fold AUC = {_f(auc)})\n")
        parts.append(policy_table(rows) + "\n")

    if d.get("router"):
        parts.append("### 라우터 계수 (표준화 후, 전체 적합 — 해석용)\n")
        parts.append("| 변형 | OOF AUC | 계수 |")
        parts.append("|---|---|---|")
        for tag, m in d["router"].items():
            parts.append(f"| {tag} | {_f(m['oof_auc'])} | `{m['coefficients']}` |")
        parts.append("")

    parts.append("## 4. 무작위 게이트 대조군 (동일 발동률)\n")
    parts.append(random_table(d["random_gate_control"]) + "\n")

    parts.append("## 5. 통계 검정 (문항 단위 주분석)\n")
    parts.append(test_table(d["tests"]) + "\n")

    parts.append("## 6. 신호별 단독 판별력 (독립 표본 재현성)\n")
    parts.append(signal_table(d["signal_auc"]) + "\n")

    parts.append("## 7. H5 난이도 조절효과 — 층별 검증 이득 분해\n")
    for key, rows in d.get("h5_stratified_gain", {}).items():
        parts.append(f"### {key}\n")
        parts.append(strat_table(rows) + "\n")

    if d.get("replay_validation_vs_s5"):
        v = d["replay_validation_vs_s5"]
        p2 = [p for p in pols if p["policy"] == "P2 자기보고 θ=0.8"]
        parts.append("## 8. 반사실 재구성 타당성 검증 (vs 실제 S5 실측)\n")
        parts.append("| 구분 | 발동률 | TSR | 평균 토큰 |")
        parts.append("|---|---|---|---|")
        if p2:
            parts.append(f"| 재현 (P2 θ=0.8) | {_pct(p2[0]['fire_rate'])} | "
                         f"{_f(p2[0]['TSR'])} | {p2[0]['mean_tokens']:,.0f} |")
        parts.append(f"| 실측 (S5 {v['n']} run) | {_pct(v['fire_rate'])} | "
                     f"{_f(v['TSR'])} | {v['mean_tokens']:,.0f} |")
        parts.append("")

    text = "\n".join(parts)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"저장: {args.out}")
    else:
        print(text)
    return text


if __name__ == "__main__":
    main()
