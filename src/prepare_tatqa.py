"""TAT-QA 제3 데이터셋 준비 — 난이도 층화 표집으로 100문항 (실험계획서 §3-C).

배경: 계획서(2026-09-07 제출)는 표본을 "300문항(한국어 1종·영어 2종)"으로 약속하였다.
영어 2종 중 하나는 FinQA(100문항, prepare_finqa + prepare_finqa_ext)이고, 나머지 한 종이
이 스크립트가 만드는 TAT-QA 100문항이다. 목적은 새 결과를 얻는 것이 아니라 **결과가 특정
데이터셋의 성질이 아님을 보이는 것**이다(단일 데이터셋 의존 회피).

TAT-QA 를 고른 이유: 표와 텍스트가 섞인 근거를 짧게 제공하므로 본 연구의 self-contained
설정으로 바꾸기 쉽다. DocFinQA 는 FinQA 문항에 전체 문서를 덧붙인 자료여서 문항이
FinQA 와 겹치고(독립 검증이 되지 않음), 평균 123k 단어라 컨텍스트 청크 상한에 걸린다.

--------------------------------------------------------------------------------
단위(scale) 처리 — 본 스크립트의 핵심 설계 결정
--------------------------------------------------------------------------------
TAT-QA 는 답의 자릿수 단위를 `scale` 필드에 따로 둔다(million/thousand/billion/percent).
그런데 본 연구의 채점기(src/metrics.py)가 아는 자릿수 단위는 **한국어뿐**이다
(조원·억원·백만원·만원·천원). 영어 단위를 gold_answer_unit 에 넣으면 채점이 깨진다.

세 가지 선택지 중 **(가) 질문에 단위를 명시**를 채택하였다.
  (가) 질문 끝에 "(Answer in millions.)" 를 붙이고 gold_answer_unit 은 비운다.
  (나) 채점기에 영어 단위를 추가한다.
  (다) scale 이 빈 문항만 쓴다.

(나)를 피한 이유: 논문 3.2.4절 (7)이 "채점 규칙이 개정되면 기존 실행 로그를 전량
재채점한다"고 못박고 있어, 지금 채점기를 고치면 이미 끝난 전 실험을 다시 채점해야 한다.
(다)를 피한 이유: 표본이 줄고 난이도 층이 한쪽으로 치우친다.

percent 만은 예외로 gold_answer_unit="%" 를 준다. 채점기가 PERCENT_UNITS 를 이미
알고 있고(3.2.4절 (3) 비율–퍼센트 정합), 이 경로가 모델이 0.2222 로 답하든 22.22 로
답하든 정합을 맞춰 주기 때문이다.

--------------------------------------------------------------------------------
층화 설계 (계획서 §3-C)
--------------------------------------------------------------------------------
derivation 의 산술 연산자 개수로 3층을 만들고 층별로 **균형 배치**한다. 자연 분포를
그대로 쓰면 쉬운 문항이 과대표집되어 불일치율이 오르지 않는다(실험계획서 §1.3).

  L1 연산자 1개 / L2 연산자 2개 / L3 연산자 3개 이상

통제:
- answer_type 은 arithmetic 만 쓴다. span·multi-span 은 추출 과업이라 본 연구의
  "다단계 수치 추론" 범위 밖이고, count 는 32문항뿐이라 층화가 되지 않는다.
- 컨텍스트는 self-contained: 표 1개 청크(마크다운) + 문단 1개씩. K-DART-QA 의 청크
  구성 방식과 같다.
- 선택은 시드 고정 무작위이므로 재현 가능하다.

  python -m src.prepare_tatqa                       # 기본 100문항(층별 34/33/33)
  python -m src.prepare_tatqa --per-stratum 34 33 33
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from . import config

SEED = 20260821
DEFAULT_SRC = config.DATA_DIR / "tatqa" / "tatqa_dataset_dev.json"
OUT_PATH = config.DATA_DIR / "tatqa" / "tatqa.jsonl"

# scale → 질문에 덧붙일 문구. percent 는 gold_answer_unit 으로 처리하므로 여기에 없다.
SCALE_HINT = {
    "thousand": "Answer in thousands.",
    "million": "Answer in millions.",
    "billion": "Answer in billions.",
}


def n_ops(q: dict) -> int:
    """derivation 의 산술 연산자 개수. 난이도 층의 기준이다."""
    return len(re.findall(r"[-+*/]", q.get("derivation") or ""))


def stratum_of(q: dict) -> str:
    n = n_ops(q)
    if n <= 1:
        return "L1"
    if n == 2:
        return "L2"
    return "L3"


def table_to_markdown(rows: list[list[str]]) -> str:
    """표를 마크다운 한 덩어리로 변환한다(K-DART-QA 와 동일한 보존 방식)."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[(c or "").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    head, body = norm[0], norm[1:]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def build_context_chunks(passage: dict) -> list[dict]:
    """표 1청크 + 문단 각 1청크. 검색 단계를 두지 않는 self-contained 구성이다."""
    chunks: list[dict] = []
    md = table_to_markdown(passage.get("table", {}).get("table") or [])
    if md:
        chunks.append({"chunk_id": "table_0", "text": md})
    for p in passage.get("paragraphs") or []:
        txt = (p.get("text") or "").strip()
        if txt:
            chunks.append({"chunk_id": f"para_{p.get('order')}", "text": txt})
    return chunks


def evidence_chunk_ids(passage: dict, q: dict) -> list[str]:
    """근거 청크 id.

    주의: TAT-QA 는 FinQA 의 gold_inds 처럼 셀 단위 정답 근거를 주지 않는다. 여기서는
    answer_from 과 rel_paragraphs 로 **청크 수준까지만** 지정한다. 오류 유형 E1/E4 판정에
    쓰기에는 FinQA·K-DART 보다 해상도가 낮다는 점을 논문 한계에 적어야 한다.
    """
    ids: list[str] = []
    if "table" in (q.get("answer_from") or ""):
        ids.append("table_0")
    for o in q.get("rel_paragraphs") or []:
        ids.append(f"para_{o}")
    return ids


def to_record(passage: dict, q: dict) -> dict:
    scale = (q.get("scale") or "").strip()
    question = (q.get("question") or "").strip()
    unit = None
    if scale == "percent":
        unit = "%"
    elif scale in SCALE_HINT:
        question = f"{question} ({SCALE_HINT[scale]})"

    ops = n_ops(q)
    return {
        "id": f"TATQA_{q['uid']}",
        "question": question,
        "gold_answer": str(q.get("answer")),
        "gold_answer_unit": unit,
        "evidence_chunks": evidence_chunk_ids(passage, q),
        # task_type 은 FinQA 확대분과 같은 규약(T3=단일단계, T4=다단계)으로 두어 집계 호환.
        "task_type": "T3" if ops <= 1 else "T4",
        "difficulty_stratum": stratum_of(q),
        "context_chunks": build_context_chunks(passage),
        "n_steps": ops,
        "tatqa_scale": scale or None,
        "tatqa_answer_from": q.get("answer_from"),
        "derivation": q.get("derivation"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="TAT-QA 난이도 층화 표집 (제3 데이터셋)")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--per-stratum", nargs=3, type=int, default=[34, 33, 33],
                    metavar=("L1", "L2", "L3"), help="층별 표집 문항 수 (기본 34/33/33 = 100)")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--max-context-chunks", type=int, default=50,
                    help="컨텍스트 청크 수 상한. 초과 문항은 같은 층의 다음 후보로 교체한다.")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(
            f"TAT-QA 원본이 없습니다: {src}\n"
            "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/"
            "tatqa_dataset_dev.json 을 이 경로로 내려받으세요.")
    data = json.load(open(src, encoding="utf-8"))

    pools: dict[str, list[tuple[dict, dict]]] = {"L1": [], "L2": [], "L3": []}
    skipped = Counter()
    for passage in data:
        for q in passage.get("questions") or []:
            if q.get("answer_type") != "arithmetic":
                skipped[q.get("answer_type")] += 1
                continue
            if q.get("answer") in (None, ""):
                skipped["답 없음"] += 1
                continue
            if not (q.get("derivation") or "").strip():
                skipped["derivation 없음"] += 1
                continue
            pools[stratum_of(q)].append((passage, q))

    def _n_chunks(pair) -> int:
        return len(build_context_chunks(pair[0]))

    rng = random.Random(SEED)
    picked: list[tuple[dict, dict]] = []
    n_replaced = 0
    for stratum, k in zip(["L1", "L2", "L3"], args.per_stratum):
        pool = sorted(pools[stratum], key=lambda pr: pr[1]["uid"])   # 결정적 정렬 후 표집
        if k > len(pool):
            print(f"  [경고] 층 {stratum} 요청 {k} > 가용 {len(pool)} → {len(pool)}개만 선택")
            k = len(pool)
        primary = rng.sample(pool, k)
        chosen = {pr[1]["uid"] for pr in primary}
        spare = [pr for pr in pool
                 if pr[1]["uid"] not in chosen and _n_chunks(pr) <= args.max_context_chunks]
        random.Random(SEED + 1).shuffle(spare)
        spare_iter = iter(spare)
        for pr in primary:
            if _n_chunks(pr) <= args.max_context_chunks:
                picked.append(pr)
                continue
            sub = next(spare_iter, None)
            if sub is None:
                print(f"  [경고] 층 {stratum}: {pr[1]['uid']} 상한 초과, 교체 후보 소진 → 제외")
                continue
            print(f"  [교체] 층 {stratum}: {pr[1]['uid']} ({_n_chunks(pr)}청크) → "
                  f"{sub[1]['uid']} ({_n_chunks(sub)}청크)")
            picked.append(sub)
            n_replaced += 1

    records = [to_record(p, q) for p, q in picked]
    records.sort(key=lambda r: (r["difficulty_stratum"], r["id"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[prepare_tatqa] {len(records)} 문항 -> {out}")
    print("  층 분포     :", dict(sorted(Counter(r["difficulty_stratum"] for r in records).items())))
    print("  task_type   :", dict(sorted(Counter(r["task_type"] for r in records).items())))
    print("  scale       :", dict(sorted(Counter(str(r["tatqa_scale"]) for r in records).items())))
    print("  단위 선언(%) :", sum(1 for r in records if r["gold_answer_unit"] == "%"))
    print("  질문에 단위 명시:", sum(1 for r in records if r["tatqa_scale"] in SCALE_HINT))
    ctx = [len(r["context_chunks"]) for r in records]
    print(f"  컨텍스트 청크: 중위 {sorted(ctx)[len(ctx)//2]} / 최소 {min(ctx)} / 최대 {max(ctx)}")
    print(f"  가용 풀     : L1 {len(pools['L1'])}, L2 {len(pools['L2'])}, L3 {len(pools['L3'])}")
    print(f"  제외        : {dict(skipped)}")
    print(f"  청크 상한   : {args.max_context_chunks} (초과 교체 {n_replaced}건)")
    print(f"  시드        : {SEED} (재현 가능)")
    return records


if __name__ == "__main__":
    main()
