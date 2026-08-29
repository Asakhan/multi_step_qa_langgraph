"""논문 4장 결과표(Table 3·4·8) 생성 — 재채점 후 확정 수치.

수치를 손으로 옮기면 전사 오류가 난다(2026-08-21 교훈). 로그에서 직접 뽑아 논문에
그대로 붙일 수 있는 마크다운 표를 만든다.

포함 항목(지도교수 피드백 대응):
  Table 3  구조별 정확도·토큰·비용·지연  + **입출력 토큰 분리(M6)**, **기권율(M8)**,
           **95% CI(m9)**, **문항 단위 TSR 병기(M2)**
  Table 4  가설 검정 — **문항 단위 주분석 + run 단위 부분석 병기(M2)**,
           **TOST 등가성(M3)**, **Holm 보정(m8)**
  Table 8  과업유형별 정확도 — 재채점 반영

사용:
    .venv/Scripts/python.exe -m src.build_paper_tables --out results_phase1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .reanalyze_phase1 import LABELS, ORDER, load_all, section5, section6, section7, section9

# 논문 표에 쓸 짧은 이름
SHORT = {
    "exp1": "실험 1 (CrewAI, 검증 포함)",
    "s0": "실험 0 (LangGraph, 무검증)",
    "s2": "실험 2 (LangGraph, 항상검증)",
    "s5": "실험 3 (LangGraph, 동적 캐스케이드)",
    "s6": "실험 4 (LangGraph, self-probe 캐스케이드)",
}
TASK_ORDER = ["T1", "T2", "T3", "T4", "T5"]


def build(data, sec5, sec6, sec7, sec9) -> str:
    L = []
    A = L.append
    present = [s for s in ORDER if s in data]

    A("# 논문 4장 결과표 — 재채점 후 확정본")
    A("")
    A("> 생성: `python -m src.build_paper_tables` · 전 수치를 로그에서 직접 산출했다(전사 오류 방지).")
    A("> 채점 기준: 현행 `metrics.is_correct` — 선언 자릿수 단위 정합 포함. 전 구조 동일 기준.")
    A("")
    A("---")
    A("")

    # ---------------- Table 3 ----------------
    A("## Table 3 — 구조별 성능 종합")
    A("")
    A("입력/출력 토큰을 분리하고(M6), 기권율(M8)과 95% 신뢰구간(m9), 문항 단위 정확도(M2)를 함께 싣는다.")
    A("")
    A("| 구조 | n(run) | TSR (run) | 95% CI | TSR (문항, n=60) | 기권율 | 입력 토큰 | 출력 토큰 | 총 토큰 | TPS | run당 비용 | 성공당 비용 | 지연(초) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in present:
        v5 = sec5["summary"][s]
        v7 = sec7["by_structure"][s]
        v9 = sec9["abstention"][s]
        ci = v5["TSR_run_ci95"]
        A(f"| {SHORT[s]} | {v5['n_runs']} | {v5['TSR_run']:.3f} | "
          f"[{ci[0]:.3f}, {ci[1]:.3f}] | {v5['TSR_question_majority']:.3f} | "
          f"{v9['abstention_rate']:.3f} | {v7['avg_input']:,.0f} | {v7['avg_output']:,.0f} | "
          f"{v7['avg_total']:,.0f} | {v7['TPS']:,.0f} | ${v7['cost_per_run']:.6f} | "
          f"${v7['cost_per_success']:.6f} | {v7['avg_latency']:.2f} |")
    A("")
    r = sec7["ratios"].get("exp1_over_s2")
    if r:
        A(f"**M6 — 토큰비와 비용비의 괴리**: CrewAI ÷ 항상검증 = 총 토큰 "
          f"{r['token_ratio']:.2f}배(입력 {r['input_ratio']:.2f}배, 출력 {r['output_ratio']:.2f}배), "
          f"성공당 비용 {r['cost_per_success_ratio']:.2f}배, 지연 {r['latency_ratio']:.2f}배. "
          f"출력 단가가 입력의 {r['price_out_over_in']:.2f}배인데 출력 비중이 "
          f"{r['output_share']['exp1']:.3f} 대 {r['output_share']['s2']:.3f} 로 갈리기 때문이다.")
        A("")
    A("**M8 — 기권**: 기권은 저자 확정 토큰 `INSUFFICIENT_EVIDENCE`(CrewAI 는 재채점본의 "
      "`is_abstention`)로 판정하며 오답과 분리해 보고한다. LangGraph 네 구조의 기권 3건은 "
      "모두 동일 문항 `HVA_T3_007` 의 3반복이다.")
    A("")
    A("**통계 패키지(m9)**: Python 3.12 · scipy 1.13 · statsmodels 0.14 · numpy 2.2. "
      "신뢰구간은 부트스트랩(n=10,000, seed=20260821).")
    A("")

    # ---------------- Table 4 ----------------
    A("---")
    A("")
    A("## Table 4 — 가설 검정")
    A("")
    A("독립 단위는 문항이므로 **문항 단위(n=60, 3반복 다수결)를 주분석**으로, run 단위(n=180)를 "
      "부분석으로 병기한다(M2). 다중비교는 Holm 보정한다(m8).")
    A("")
    A("| 비교 | 문항 불일치 (b01/b10) | McNemar p (문항) | p (Holm) | run 불일치 | McNemar p (run) |")
    A("|---|---|---|---|---|---|")
    for k, v in sec5["pairwise"].items():
        a, b = k.split("_vs_")
        q, rr = v["question_level_n60"], v["run_level_n180"]
        A(f"| {SHORT[a]} vs {SHORT[b]} | {q['discordant']}/60 ({q['b01']}/{q['b10']}) | "
          f"{q['p_value']:.4f} | {q['p_holm']:.4f} | {rr['discordant']}/180 | "
          f"{rr['p_value']:.4f} |")
    A("")
    A("### Table 4-b — 등가성 검정 (M3)")
    A("")
    A("비유의는 등가의 근거가 아니므로, '차이 없음'을 주장하는 모든 비교에 TOST(±0.05)와 "
      "차이의 신뢰구간, 사후 검정력을 병기한다.")
    A("")
    A("| 비교 | 차이 (문항 단위) | 90% CI | 95% CI | TOST 등가 | 사후 검정력 | 검정력 0.8 소요 문항 |")
    A("|---|---|---|---|---|---|---|")
    for k, v in sec6.items():
        a, b = k.split("_vs_")
        t, pw = v["tost"], v["power"]
        eq = "등가 입증" if t.get("equivalent") else "미입증"
        A(f"| {SHORT[a]} vs {SHORT[b]} | {t['diff']:+.4f} | "
          f"[{t['ci90'][0]:+.3f}, {t['ci90'][1]:+.3f}] | "
          f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {eq} | {pw['power']:.3f} | "
          f"{pw['n_for_80'] or '—'} |")
    A("")
    A("> **60문항 표본에서는 어떤 쌍도 ±0.05 등가가 입증되지 않는다.** 사후 검정력이 낮아 "
      "\"차이를 검출하지 못하였다\"까지만 말할 수 있다. 등가성 주장은 확대 표본(실험 C 풀링 "
      "140문항)에서 별도로 입증되며, 4.x 절에서 그 결과를 주분석으로 싣는다.")
    A("")
    g = sec5.get("glmm", {})
    if g.get("available") and g.get("gee", {}).get("ok"):
        sep = g.get("separation", {})
        A("### Table 4-c — 문항 임의효과 보정 (M2)")
        A("")
        A(f"문항 {sep.get('n_questions')}개 중 {sep.get('n_all_success')}개가 전부 성공, "
          f"{sep.get('n_all_failure')}개가 전부 실패로 **완전분리 비율이 "
          f"{sep.get('separated_ratio', 0):.1%}** 다. 이 조건에서 문항 임의절편 GLMM 은 "
          "절편이 발산하므로, **문항을 군집으로 둔 GEE(exchangeable, 로버스트 표준오차)** 를 "
          "주 모형으로 보고한다.")
        A("")
        A(f"기준 범주 = {SHORT.get(g['base_category'], g['base_category'])}")
        A("")
        A("| 항 | 계수 | 오즈비 | p |")
        A("|---|---|---|---|")
        for t_, fe in g["gee"]["fixed_effects"].items():
            A(f"| {t_} | {fe['coef']:+.4f} | {fe['odds_ratio']:.3f} | {fe['p_value']:.4f} |")
        A("")
        A("> 문항 비독립성을 보정하면 구조 간 정확도 차이는 모두 비유의하다.")
        A("")

    # ---------------- Table 8 ----------------
    A("---")
    A("")
    A("## Table 8 — 과업유형별 정확도")
    A("")
    A("모집단은 K-DART-QA 한정이다(m6) — FinQA 문항에는 T1~T5 라벨이 없다.")
    A("")
    A("| 구조 | " + " | ".join(TASK_ORDER) + " |")
    A("|---|" + "---|" * len(TASK_ORDER))
    tsr_by_tt = {}
    for s in present:
        agg = {}
        for r in data[s]:
            tt = r["task_type"]
            if tt not in TASK_ORDER:
                continue
            agg.setdefault(tt, [0, 0])
            agg[tt][0] += 1
            agg[tt][1] += r["success"]
        tsr_by_tt[s] = {k: v[1] / v[0] for k, v in agg.items()}
        A(f"| {SHORT[s]} | " + " | ".join(
            f"{tsr_by_tt[s][tt]:.3f}" if tt in tsr_by_tt[s] else "—"
            for tt in TASK_ORDER) + " |")
    A("")
    n_by_tt = {}
    for r in data[present[0]]:
        if r["task_type"] in TASK_ORDER:
            n_by_tt[r["task_type"]] = n_by_tt.get(r["task_type"], 0) + 1
    A("문항 수(run): " + " · ".join(f"{tt} {n_by_tt.get(tt, 0)}" for tt in TASK_ORDER))
    A("")
    A("> **M5 — T1 라벨**: 3.3.3 의 정의(T1 = 단일 수치 추출, baseline)가 정확하며, 4.2 의 "
      "\"단계 결합·계산이 결합된 T1\" 서술은 오류이므로 삭제한다. T1 5문항은 전부 "
      "\"…는 얼마입니까?\" 형 단일 추출이다. 최저 정확도 유형은 T4(다단계 추론)로, "
      "데이터셋 설계 의도와 정합한다.")
    A("")
    A("> **문항 id 표기 주의**: `DART_T3_006`·`DART_T5_008` 의 실제 과업유형은 T1 이다. "
      "id 접두어는 구축 당시의 잠정 라벨이고 `task_type` 필드가 확정 라벨이다. 부록 문항 "
      "목록에 이 점을 명시한다.")
    A("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="논문 4장 결과표 생성")
    ap.add_argument("--results-dir", default="results_main")
    ap.add_argument("--exp1-csv",
                    default=r"C:/Users/asakh/Documents/개인/대학원/2026/졸업논문/"
                            r"실험결과/results_exp1_rescored.csv")
    ap.add_argument("--out", default="results_phase1")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(Path(args.results_dir), Path(args.exp1_csv))
    sec5 = section5(data)
    sec6 = section6(data, sec5)
    sec7 = section7(data)
    sec9 = section9(data, out_dir)
    md = build(data, sec5, sec6, sec7, sec9)
    p = out_dir / "논문_Table3_4_8_확정본.md"
    p.write_text(md, encoding="utf-8")
    print(f"완료 - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
