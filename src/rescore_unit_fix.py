"""선언 단위(gold_answer_unit) 정합 재채점 — M5 false negative 검증용.

**기존 채점 결과를 덮어쓰지 않는다.** 기존 로그를 읽어 대안 채점 규칙을 적용했을 때
판정이 어떻게 달라지는지만 산출한다. 채택 여부는 저자가 결정한다.

## 발견한 결함

`metrics.is_correct(gold, pred, unit)` 은 퍼센트 단위에 대해서는 gold 가 이미 퍼센트
크기임을 `PERCENT_UNITS` 로 알려 ×100 오인을 막는다. 그런데 **자릿수 단위**(백만원·억원·
천원·조원)에는 대응 규칙이 없다.

K-DART-QA 의 gold 는 원표(DART 공시)의 단위를 그대로 쓰므로 `gold_answer_unit='백만원'`
문항의 gold 는 "백만원 단위로 표시된 맨숫자"다. 모델이 같은 값을 단위까지 붙여
`739,764,256 백만원` 으로 답하면 `_normalize_number_ex` 가 접미사를 보고 10^6 을 곱해
7.4×10^14 로 만들고, gold 7.4×10^8 과 비교해 **오답으로 판정**한다. 숫자는 완전히 같다.

이는 모델의 오류가 아니라 채점기의 단위 이중 적용이다. 같은 문항에 CrewAI 는 단위 없이
`739764256` 로 답해 정답 처리되었으므로, 구조 간 T1 격차의 일부는 **답 표기 양식 차이가
채점기를 통과하는지 여부**에서 발생한 인공물일 수 있다.

## 대안 규칙 (declared-unit reconciliation)

`_percent_reconcile` 과 대칭이 되도록 한 줄 규칙을 추가한다.

> gold_answer_unit 이 자릿수 단위이고, pred 의 접미사가 **그 단위와 같으면**,
> gold 와 pred 는 같은 단위로 표현된 것이므로 스케일을 적용하지 않고 맨숫자끼리 비교한다.

이 규칙은 단위를 붙이지 않은 답(CrewAI 형식)의 기존 판정을 바꾸지 않는다.
pred 가 **다른** 자릿수 단위를 붙였을 때는 스케일을 그대로 적용하므로 진짜 단위 오류는
계속 오답으로 남는다.

사용:
    .venv/Scripts/python.exe -m src.rescore_unit_fix --out results_phase1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from . import metrics
from .metrics import _MAGNITUDE_UNITS, _ascii_symbols

MAGNITUDE_NAMES = [u for u, _ in _MAGNITUDE_UNITS]


def _strip_declared_unit(pred, unit: str | None) -> tuple[str, bool]:
    """pred 의 접미사가 선언 단위와 같으면 그 접미사를 떼고 (문자열, True) 를 돌려준다.

    '원' 은 스케일 1.0 이라 판정에 영향이 없으므로 대상에서 제외한다.
    """
    if not unit or unit not in MAGNITUDE_NAMES or unit == "원":
        return str(pred), False
    t = _ascii_symbols(str(pred or "").strip()).replace(",", "").replace(" ", "")
    if t.endswith(unit):
        return t[: -len(unit)], True
    return str(pred), False


def is_correct_unitfix(gold, pred, unit: str | None = None):
    """선언 단위 정합 규칙을 얹은 판정. 그 외 동작은 metrics.is_correct 와 동일."""
    stripped, applied = _strip_declared_unit(pred, unit)
    if not applied:
        return metrics.is_correct(gold, pred, unit), False
    return metrics.is_correct(gold, stripped, unit), True


# ---------------------------------------------------------------------------

STRUCTS = ["s0", "s2", "s5", "s6"]
LABELS = {"s0": "실험 0 무검증", "s2": "실험 2 항상검증",
          "s5": "실험 3 동적 캐스케이드", "s6": "실험 4 self-probe 캐스케이드"}


def rescore(results_dir: Path) -> dict:
    out = {}
    changes = []
    for s in STRUCTS:
        path = results_dir / s / "raw_runs.jsonl"
        if not path.exists():
            continue
        runs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        # 세 갈래를 분리해 집계한다.
        #   log  = 로그에 기록된 success (실행 당시 채점기)
        #   cur  = 현행 metrics.is_correct 로 재채점 (코드가 이후 갱신되었을 수 있다)
        #   fix  = 현행 + 선언 단위 정합 규칙
        tot = {"log": 0, "cur": 0, "fix": 0}
        by_tt = defaultdict(lambda: {"n": 0, "log": 0, "cur": 0, "fix": 0})
        by_ds = defaultdict(lambda: {"n": 0, "log": 0, "cur": 0, "fix": 0})
        per_q = defaultdict(lambda: {"log": [], "cur": [], "fix": []})
        for r in runs:
            gold, pred = r.get("gold_answer"), r.get("final_answer")
            unit = r.get("gold_answer_unit")
            v = {
                "log": int(r.get("success", 0)),
                "cur": int(metrics.is_correct(gold, pred, unit).correct),
            }
            jud, applied = is_correct_unitfix(gold, pred, unit)
            v["fix"] = int(jud.correct)
            tt, ds = r.get("task_type", ""), r.get("dataset", "")
            for k in ("log", "cur", "fix"):
                tot[k] += v[k]
                by_tt[tt][k] += v[k]
                by_ds[ds][k] += v[k]
                per_q[r["question_id"]][k].append(v[k])
            by_tt[tt]["n"] += 1
            by_ds[ds]["n"] += 1
            if not (v["log"] == v["cur"] == v["fix"]):
                changes.append({
                    "structure": s, "question_id": r["question_id"],
                    "run_index": r.get("run_index"), "task_type": tt, "dataset": ds,
                    "gold": gold, "unit": unit, "pred": pred,
                    "log": v["log"], "cur": v["cur"], "fix": v["fix"],
                    "cause": ("채점기 코드 갱신 미반영" if v["log"] != v["cur"]
                              else "선언 단위 정합 규칙"),
                    "unit_rule_applied": applied,
                })
        n = len(runs)
        pq = {k: {q: int(sum(d[k]) * 2 > len(d[k])) for q, d in per_q.items()}
              for k in ("log", "cur", "fix")}
        out[s] = {
            "label": LABELS[s],
            "n_runs": n,
            "TSR_log": tot["log"] / n, "TSR_cur": tot["cur"] / n, "TSR_fix": tot["fix"] / n,
            "delta_log_to_cur": tot["cur"] - tot["log"],
            "delta_cur_to_fix": tot["fix"] - tot["cur"],
            "TSR_question_log": sum(pq["log"].values()) / len(pq["log"]),
            "TSR_question_cur": sum(pq["cur"].values()) / len(pq["cur"]),
            "TSR_question_fix": sum(pq["fix"].values()) / len(pq["fix"]),
            "by_task_type": {
                tt: {"n": d["n"], "TSR_log": d["log"] / d["n"],
                     "TSR_cur": d["cur"] / d["n"], "TSR_fix": d["fix"] / d["n"]}
                for tt, d in sorted(by_tt.items())
            },
            "by_dataset": {
                ds: {"n": d["n"], "TSR_log": d["log"] / d["n"],
                     "TSR_cur": d["cur"] / d["n"], "TSR_fix": d["fix"] / d["n"]}
                for ds, d in sorted(by_ds.items())
            },
        }
    return {"by_structure": out, "changed_runs": changes}


def write_report(res: dict, out_dir: Path) -> Path:
    L = []
    A = L.append
    A("# 선언 단위 정합 재채점 — M5 false negative 검증")
    A("")
    A("> **기존 채점 결과를 덮어쓰지 않았다.** 대안 규칙을 적용했을 때의 차이만 산출한 것이다.")
    A("")
    A("## 결함")
    A("")
    A("`metrics.is_correct` 는 퍼센트 단위에는 `PERCENT_UNITS` 로 gold 가 이미 퍼센트")
    A("크기임을 알려 ×100 오인을 막지만, **자릿수 단위(백만원·억원·천원·조원)에는 대응")
    A("규칙이 없다.** K-DART-QA 의 gold 는 공시 원표의 단위로 표기된 맨숫자이므로,")
    A("모델이 같은 값에 단위를 붙여 답하면 채점기가 스케일을 **한 번 더** 적용해 오답이 된다.")
    A("")
    A("```")
    A("gold  '739,764,256'  (gold_answer_unit='백만원')  → 7.39764e+08")
    A("pred  '739,764,256 백만원'                        → 7.39764e+14   ✗ 오답 판정")
    A("pred  '739764256'    (단위 미표기, CrewAI 형식)    → 7.39764e+08   ✓ 정답 판정")
    A("```")
    A("")
    A("숫자는 동일하다. 판정을 가른 것은 추론이 아니라 **답의 표기 양식**이다.")
    A("")
    A("## 대안 규칙")
    A("")
    A("> gold_answer_unit 이 자릿수 단위이고 pred 의 접미사가 **그 단위와 같으면**,")
    A("> 둘은 같은 단위로 표현된 것이므로 스케일 없이 맨숫자끼리 비교한다.")
    A("")
    A("단위를 붙이지 않은 답의 기존 판정은 바뀌지 않으며, pred 가 **다른** 자릿수 단위를")
    A("붙인 진짜 단위 오류는 계속 오답으로 남는다.")
    A("")
    A("## 두 번째 결함 — 로그의 판정이 현행 채점기와 이미 불일치한다")
    A("")
    A("재집계 중 확인된 별개 문제다. 로그의 `success` 값은 **실행 당시의 채점기**로 매겨졌는데,")
    A("그 뒤 채점기에 유니코드 기호 정규화(U+2212 등)가 추가되었고 **LangGraph 로그는 재채점되지")
    A("않았다.** 실험 1(CrewAI)만 재채점본이 있다. 따라서 아래 표는 세 갈래를 분리해 보고한다.")
    A("")
    A("- **로그**: 로그에 기록된 `success` — 논문 Table 3·4 의 현재 수치")
    A("- **현행코드**: 지금의 `metrics.is_correct` 로 그대로 재채점 (규칙 변경 없음)")
    A("- **단위정합**: 현행코드 + 위 선언 단위 정합 규칙")
    A("")
    A("## 영향")
    A("")
    A("| 구조 | run | TSR(로그) | TSR(현행코드) | TSR(단위정합) | 로그→현행 | 현행→단위정합 | 문항 TSR(로그→단위정합) |")
    A("|---|---|---|---|---|---|---|---|")
    for s, v in res["by_structure"].items():
        A(f"| {v['label']} | {v['n_runs']} | {v['TSR_log']:.4f} | {v['TSR_cur']:.4f} | "
          f"{v['TSR_fix']:.4f} | {v['delta_log_to_cur']:+d} run | "
          f"{v['delta_cur_to_fix']:+d} run | "
          f"{v['TSR_question_log']:.4f} → {v['TSR_question_fix']:.4f} |")
    A("")
    A("### 과업유형별 (로그 → 현행코드 → 단위정합)")
    A("")
    tts = sorted({tt for v in res["by_structure"].values() for tt in v["by_task_type"]})
    A("| 구조 | " + " | ".join(tts) + " |")
    A("|---|" + "---|" * len(tts))
    for s, v in res["by_structure"].items():
        cells = []
        for tt in tts:
            d = v["by_task_type"].get(tt)
            cells.append(f"{d['TSR_log']:.3f} → {d['TSR_cur']:.3f} → **{d['TSR_fix']:.3f}**"
                         if d else "—")
        A(f"| {v['label']} | " + " | ".join(cells) + " |")
    A("")
    A("### 데이터셋별 (로그 → 현행코드 → 단위정합)")
    A("")
    dss = sorted({ds for v in res["by_structure"].values() for ds in v["by_dataset"]})
    A("| 구조 | " + " | ".join(dss) + " |")
    A("|---|" + "---|" * len(dss))
    for s, v in res["by_structure"].items():
        cells = []
        for ds in dss:
            d = v["by_dataset"].get(ds)
            cells.append(f"{d['TSR_log']:.3f} → {d['TSR_cur']:.3f} → **{d['TSR_fix']:.3f}**"
                         if d else "—")
        A(f"| {v['label']} | " + " | ".join(cells) + " |")
    A("")
    ch = res["changed_runs"]
    n_code = sum(1 for c in ch if c["cause"] == "채점기 코드 갱신 미반영")
    n_unit = len(ch) - n_code
    A(f"### 판정이 바뀐 run: {len(ch)}건 "
      f"(채점기 코드 갱신 미반영 {n_code}건 · 선언 단위 정합 규칙 {n_unit}건)")
    A("")
    if ch:
        A("| 구조 | 문항 | 반복 | 유형 | gold | 단위 | pred | 로그 | 현행 | 단위정합 | 원인 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in ch:
            A(f"| {c['structure']} | {c['question_id']} | {c['run_index']} | {c['task_type']} | "
              f"`{c['gold']}` | {c['unit']} | `{c['pred']}` | {c['log']} | {c['cur']} | "
              f"{c['fix']} | {c['cause']} |")
    A("")
    A("---")
    A("")
    A("## 저자 결정 사항")
    A("")
    A("1. **로그 재채점 여부** — 현행 채점기로 재채점하는 것은 규칙 변경이 아니라 이미 고친")
    A("   버그를 로그에 반영하는 것이므로, 하지 않으면 논문 수치와 공개 코드가 어긋난다.")
    A("   추가 LLM 호출 없이 로그 재집계만으로 가능하다.")
    A("2. **선언 단위 정합 규칙 채택 여부** — 채택하면 `metrics.is_correct` 를 수정하고")
    A("   회귀 테스트를 추가한 뒤 전 실험을 재채점한다.")
    A("3. 채택 시 논문의 Table 3·4·8, 4.2 과업유형 서술, 6장 T1 관련 논의가 모두 갱신된다.")
    A("   특히 T1 의 구조 간 격차(CrewAI 0.667 vs LangGraph 0.200)는 **소멸에 가깝게 축소**된다")
    A("   — 두 구조가 같은 숫자를 답했고 표기만 달랐기 때문이다.")
    A("4. 채택하지 않는다면 이 현상을 **채점 프로토콜의 한계**로 5장에 명시해야 한다.")
    A("")

    path = out_dir / "단위정합_재채점_영향.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="선언 단위 정합 재채점(영향 분석 전용)")
    ap.add_argument("--results-dir", default="results_main")
    ap.add_argument("--out", default="results_phase1")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = rescore(Path(args.results_dir))
    (out_dir / "unit_fix_impact.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    path = write_report(res, out_dir)
    print(f"완료 - {path}")
    for s, v in res["by_structure"].items():
        print(f"  {s}: TSR log {v['TSR_log']:.4f} -> cur {v['TSR_cur']:.4f} "
              f"-> fix {v['TSR_fix']:.4f}  ({v['delta_log_to_cur']:+d} / {v['delta_cur_to_fix']:+d} run)")
    print(f"  판정 변화 run: {len(res['changed_runs'])}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
