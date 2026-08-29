"""FinQA 표본 확대 — 난이도 층화 표집으로 80문항 추가 (실험계획서 §3-C).

배경: 60문항 규모에서는 McNemar 의 불일치쌍이 4개뿐이어서 어떤 현실적 효과도 검출할 수
없다(실험계획서 §1.3 검정력 산출). 확대가 필수인데 K-DART-QA 추가 생성은 저자 검수가
필요해 일정이 길다. 계획서 §6 은 이 리스크에 대해 "영어 데이터셋 확대를 먼저 진행해 본
실험을 착수하고, 한국어 확대분은 2차 라운드로 분리"할 것을 정해 두었다. 이 스크립트가
그 1차 라운드다.

층화 설계 (계획서 §3-C): 계산 단계 수(FinQA program 의 steps 수)로 3층을 만들고 층별
문항 수를 **균형 배치**한다. 자연 분포(1홉 654 / 2홉 409 / 3홉+ 84)를 그대로 쓰면 쉬운
문항이 과대표집되어 불일치율이 오르지 않는다. 균형 배치는 (i) H5(난이도 조절효과) 검정과
(ii) 불일치율 상승을 동시에 달성한다.

  층 L1 단일조회(steps<=1) / 층 L2 2홉(steps==2) / 층 L3 3홉이상(steps>=3)

통제:
- 기존 canonical 20문항(prepare_finqa.DEFAULT_IDS)은 **제외**한다. 기존 60문항 실험의
  재현성을 보존하고, 확대분을 독립 표본으로 쓸 수 있게 하기 위함이다.
- 선택은 시드 고정 무작위(층 내 id 정렬 후 표집)이므로 재현 가능하다.
- gold 는 기존과 동일하게 qa.exe_ans 를 쓰고, 컨텍스트 구성도 prepare_finqa 와 동일
  함수(build_context_chunks)를 재사용한다 → 파이프라인 완전 동일.

  python -m src.prepare_finqa_ext                  # 기본 80문항(층별 27/27/26)
  python -m src.prepare_finqa_ext --per-stratum 27 27 26
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import config
from .prepare_finqa import DEFAULT_IDS, build_context_chunks, n_steps

SEED = 20260821
DEFAULT_TEST_JSON = config.DATA_DIR / "finqa" / "FinQA-main" / "dataset" / "test.json"
OUT_PATH = config.DATA_DIR / "finqa" / "finqa_ext.jsonl"


def stratum_of(item: dict) -> str:
    n = n_steps(item)
    if n <= 1:
        return "L1"
    if n == 2:
        return "L2"
    return "L3"


def to_record(item: dict) -> dict:
    qa = item["qa"]
    gold = qa.get("exe_ans", qa.get("answer"))
    ns = n_steps(item)
    return {
        "id": item["id"],
        "question": qa["question"],
        "gold_answer": str(gold),
        "gold_answer_unit": None,
        "evidence_chunks": list((qa.get("gold_inds") or {}).keys()),
        # task_type 은 기존 규약 유지(T3=단일단계, T4=다단계) — 기존 60문항과 집계 호환.
        "task_type": "T3" if ns <= 1 else "T4",
        "difficulty_stratum": stratum_of(item),
        "context_chunks": build_context_chunks(item),
        "n_steps": ns,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="FinQA 난이도 층화 확대 표집")
    ap.add_argument("--finqa-test", default=str(DEFAULT_TEST_JSON))
    ap.add_argument("--per-stratum", nargs=3, type=int, default=[27, 27, 26],
                    metavar=("L1", "L2", "L3"), help="층별 표집 문항 수 (기본 27/27/26 = 80)")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--max-context-chunks", type=int, default=50,
                    help="컨텍스트 청크 수 상한(기본 50 = config top_k). 초과 문항은 "
                         "동일 층의 다음 후보로 결정적으로 교체한다 — 아래 주석 참조.")
    args = ap.parse_args(argv)

    test_path = Path(args.finqa_test)
    if not test_path.exists():
        raise FileNotFoundError(
            f"FinQA-main test.json 이 없습니다: {test_path}\n"
            "데이터셋/finqa_main.zip 의 dataset/test.json 을 이 경로로 추출하세요.")
    data = json.load(open(test_path, encoding="utf-8"))

    excluded = set(DEFAULT_IDS)
    pools: dict[str, list[dict]] = {"L1": [], "L2": [], "L3": []}
    for it in data:
        if it["id"] in excluded:
            continue
        # gold 실행값이 없는 문항(추출형 답 등)은 채점기 정합을 보장할 수 없어 제외.
        if it["qa"].get("exe_ans") is None:
            continue
        pools[stratum_of(it)].append(it)

    # 컨텍스트 청크 수 상한 (= config/experiment.yaml retrieval.top_k, 기본 50).
    #
    # 왜 필요한가: 파이프라인은 컨텍스트가 top_k 이하면 전량을 그대로 투입하고(임베딩 호출
    # 없음), top_k 를 넘으면 임베딩 코사인 검색으로 상위 k 개를 고른다(src/tools.py).
    # 본 실험 설정은 두 데이터셋 모두 self-contained 라서 임베딩 경로를 쓰지 않도록
    # 구성되어 있고(.env 에 임베딩 키가 없다), 기존 canonical 20문항은 최대 38청크라
    # 이 경로를 탄 적이 없다. 확대 표집에서 50청크를 넘는 문항이 섞이면 그 문항만
    # **다른 검색 조건**(top-50 선별)에 놓여 통제가 깨지고, 키가 없으면 실행 자체가 실패한다.
    #
    # 처리 방식: 1차 표집 결과에서 상한 초과 문항을 **동일 층의 다음 후보로 교체**한다.
    # 후보 순서는 같은 시드로 섞은 결정적 순서이므로 재현 가능하며, 상한을 만족하는 나머지
    # 문항의 선택은 **바뀌지 않는다**(부분 재실행 낭비 방지). 이 제약은 원천 풀에서
    # L1 24개 / L2 20개 / L3 0개만 제외하므로 층별 대표성에 실질적 영향이 없다.
    def _n_chunks(it):
        return len(build_context_chunks(it))

    rng = random.Random(SEED)
    picked: list[dict] = []
    n_replaced = 0
    for stratum, k in zip(["L1", "L2", "L3"], args.per_stratum):
        pool = sorted(pools[stratum], key=lambda x: x["id"])   # 결정적 정렬 후 표집
        if k > len(pool):
            print(f"  [경고] 층 {stratum} 요청 {k} > 가용 {len(pool)} → {len(pool)}개만 선택")
            k = len(pool)
        primary = rng.sample(pool, k)
        chosen_ids = {it["id"] for it in primary}
        # 교체 후보: 1차 표집에 안 뽑힌 문항을 같은 시드로 섞은 결정적 순서
        spare = [it for it in pool if it["id"] not in chosen_ids
                 and _n_chunks(it) <= args.max_context_chunks]
        random.Random(SEED + 1).shuffle(spare)
        spare_iter = iter(spare)
        for it in primary:
            if _n_chunks(it) <= args.max_context_chunks:
                picked.append(it)
                continue
            sub = next(spare_iter, None)
            if sub is None:
                print(f"  [경고] 층 {stratum}: {it['id']} 상한 초과지만 교체 후보 소진 → 제외")
                continue
            print(f"  [교체] 층 {stratum}: {it['id']} ({_n_chunks(it)}청크 > "
                  f"{args.max_context_chunks}) → {sub['id']} ({_n_chunks(sub)}청크)")
            picked.append(sub)
            n_replaced += 1

    records = [to_record(it) for it in picked]
    records.sort(key=lambda r: (r["difficulty_stratum"], r["id"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[prepare_finqa_ext] {len(records)} 문항 -> {out}")
    print("  층 분포     :", dict(sorted(Counter(r["difficulty_stratum"] for r in records).items())))
    print("  task_type   :", dict(sorted(Counter(r["task_type"] for r in records).items())))
    print("  n_steps     :", dict(sorted(Counter(r["n_steps"] for r in records).items())))
    ctx = [len(r["context_chunks"]) for r in records]
    print(f"  컨텍스트 청크: 중위 {sorted(ctx)[len(ctx)//2]} / 최소 {min(ctx)} / 최대 {max(ctx)}")
    print(f"  가용 풀     : L1 {len(pools['L1'])}, L2 {len(pools['L2'])}, L3 {len(pools['L3'])}")
    print(f"  청크 상한   : {args.max_context_chunks} (초과 교체 {n_replaced}건)")
    print(f"  시드        : {SEED} (재현 가능)")
    return records


if __name__ == "__main__":
    main()
