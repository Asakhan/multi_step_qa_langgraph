"""FinQA-main test.json -> data/finqa/finqa.jsonl (영어 20문항).

문근영 외(2026) / 자매 실험(multi_step_qa_langchain)과 **동일한 20문항**을 사용하기 위해
canonical id 목록(DEFAULT_IDS)을 고정한다. 각 문항은 self-contained: FinQA-main 의
pre_text/post_text/table 을 context_chunks(text_N / table_N) 로 함께 저장하여, 검색기가
문항별 RAG(코사인 top_k) 를 수행한다. K-DART 전역 인덱스와 동일 임베딩 모델을 쓴다.

  python -m src.prepare_finqa            # 기본: FinQA-main test.json + canonical 20 ids
  python -m src.prepare_finqa --ids id1,id2,...   # 명시 선택
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config

# 자매 실험 finqa.jsonl 과 동일한 20문항 (단일근거 10[T3] + 다단계 10[T4])
DEFAULT_IDS = [
    # T3 (single-step)
    "AAL/2010/page_72.pdf-1", "AAL/2010/page_72.pdf-3", "AAL/2013/page_172.pdf-1",
    "AAL/2013/page_172.pdf-2", "AAL/2013/page_172.pdf-3", "AAL/2013/page_172.pdf-4",
    "AAL/2013/page_18.pdf-4", "AAL/2014/page_15.pdf-1", "AAL/2014/page_15.pdf-2",
    "AAL/2014/page_15.pdf-3",
    # T4 (multi-step)
    "AAL/2010/page_72.pdf-2", "AAL/2010/page_72.pdf-4", "AAL/2013/page_18.pdf-1",
    "AAL/2013/page_18.pdf-2", "AAL/2013/page_18.pdf-3", "ABMD/2005/page_29.pdf-1",
    "ABMD/2005/page_29.pdf-3", "ABMD/2005/page_29.pdf-4", "ABMD/2007/page_78.pdf-2",
    "ABMD/2007/page_78.pdf-3",
]

DEFAULT_TEST_JSON = config.DATA_DIR / "finqa" / "FinQA-main" / "dataset" / "test.json"
OUT_PATH = config.DATA_DIR / "finqa" / "finqa.jsonl"


def _remove_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def table_row_to_text(header: list[str], row: list[str]) -> str:
    """FinQA general_utils.table_row_to_text 와 동일한 행 선형화."""
    res = ""
    if header and header[0]:
        res += header[0] + " "
    for head, cell in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return _remove_space(res).strip()


def build_context_chunks(item: dict) -> list[dict]:
    chunks: list[dict] = []
    pre = item.get("pre_text", []) or []
    for i, t in enumerate(pre):
        if t and t.strip():
            chunks.append({"chunk_id": f"text_{i}", "text": _remove_space(t)})
    table = item.get("table", []) or []
    if table:
        header = table[0]
        for r, row in enumerate(table[1:], start=1):
            txt = table_row_to_text(header, row)
            if txt:
                chunks.append({"chunk_id": f"table_{r}", "text": txt})
    for j, t in enumerate(item.get("post_text", []) or []):
        if t and t.strip():
            chunks.append({"chunk_id": f"text_{len(pre) + j}", "text": _remove_space(t)})
    return chunks


def n_steps(item: dict) -> int:
    qa = item["qa"]
    steps = qa.get("steps")
    if isinstance(steps, list):
        return len(steps)
    prog = qa.get("program", "") or ""
    return max(1, prog.count("("))


def to_record(item: dict) -> dict:
    qa = item["qa"]
    gold = qa.get("exe_ans", qa.get("answer"))
    return {
        "id": item["id"],
        "question": qa["question"],
        "gold_answer": str(gold),
        "gold_answer_unit": None,
        "evidence_chunks": list((qa.get("gold_inds") or {}).keys()),
        "task_type": "T3" if n_steps(item) <= 1 else "T4",
        "context_chunks": build_context_chunks(item),
        "n_steps": n_steps(item),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="FinQA-main -> finqa.jsonl")
    ap.add_argument("--finqa-test", default=str(DEFAULT_TEST_JSON))
    ap.add_argument("--ids", default="", help="쉼표구분 id (미지정 시 canonical 20 ids)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)

    test_path = Path(args.finqa_test)
    if not test_path.exists():
        raise FileNotFoundError(
            f"FinQA-main test.json 이 없습니다: {test_path}\n"
            "data/finqa/FinQA-main/dataset/test.json 을 확인하세요.")
    data = json.load(open(test_path, encoding="utf-8"))
    by_id = {d["id"]: d for d in data}

    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids.strip() else DEFAULT_IDS
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"FinQA-main 에 없는 id: {missing}")

    records = [to_record(by_id[i]) for i in ids]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[prepare_finqa] {len(records)} 문항 -> {out}")
    print("  task_type:", dict(Counter(r["task_type"] for r in records)))
    print("  ctx chunk sizes:", [len(r["context_chunks"]) for r in records])
    return records


if __name__ == "__main__":
    main()
