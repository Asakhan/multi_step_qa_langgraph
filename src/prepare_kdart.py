"""K-DART-QA (FinQA 포맷) kdart_qa_finqa_format.json -> data/kdart/kdart_finqa.jsonl.

[주의 — 재현 경로 아님] 실험에 쓰인 정본은 저장소에 커밋된 data/kdart/kdart_finqa.jsonl
이며, 이는 별도 구축 저장소(https://github.com/Asakhan/kdart_qa, 논문 §3.3.4)의 산출물이다.
이 스크립트는 참고용 파생 코드로, 청크 라벨링(text_N/table_N)·gold 추출 방식이 정본과
다를 수 있어 **정본을 재현하지 않는다**. 따라서 재현 시에는 실행하지 말고 커밋된 정본을
그대로 사용하라(build_index 는 이 스크립트를 호출하지 않는다).

새 데이터셋은 FinQA-main test.json 과 **동일한 구조**(pre_text/post_text/table/qa)에
kdart_meta 를 더한 자체완결(self-contained) 포맷이다. 각 문항이 자기 근거 문서를 통째로
포함하므로 전역 ChromaDB 인덱스도, 임베딩 검색도 필요 없다. FinQA 와 동일하게 문항별
context_chunks(text_N / table_N) 로 직렬화해 모델에 전체 컨텍스트를 투입한다 → 두 데이터셋이
완전히 대칭(동일 파이프라인)이 되어 더 객관적인 비교가 된다.

  python -m src.prepare_kdart            # 기본: data/kdart/kdart_qa_finqa_format.json -> kdart_finqa.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config

DEFAULT_SRC = config.DATA_DIR / "kdart" / "kdart_qa_finqa_format.json"
OUT_PATH = config.DATA_DIR / "kdart" / "kdart_finqa.jsonl"


def _remove_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _row_to_text(row: list) -> str:
    """테이블 한 행을 파이프 연결 텍스트로 직렬화(한국어 표 보존, gold_inds 표기와 정합)."""
    cells = [_remove_space(c) for c in row]
    return " | ".join(c for c in cells)


def build_context_chunks(item: dict) -> list[dict]:
    """pre_text -> table 행들 -> post_text 순으로 self-contained 청크 구성.

    - 텍스트: text_{i} (pre 는 0..; post 는 len(pre)+j 로 연속 인덱싱 — FinQA 규약)
    - 테이블: table_{r} (원본 행 인덱스 r, 0-base — gold_inds 의 table_N 과 정합)
    """
    chunks: list[dict] = []
    pre = item.get("pre_text", []) or []
    post = item.get("post_text", []) or []

    for i, t in enumerate(pre):
        if t and str(t).strip():
            chunks.append({"chunk_id": f"text_{i}", "text": _remove_space(t)})

    for r, row in enumerate(item.get("table", []) or []):
        txt = _row_to_text(row)
        if txt.replace("|", "").strip():  # 전부 공백인 행은 건너뜀(인덱스 r 은 보존)
            chunks.append({"chunk_id": f"table_{r}", "text": txt})

    for j, t in enumerate(post):
        if t and str(t).strip():
            chunks.append({"chunk_id": f"text_{len(pre) + j}", "text": _remove_space(t)})

    return chunks


def _norm_match(s: str) -> str:
    return re.sub(r"[\s|]+", "", str(s or ""))


def map_gold_chunk_ids(gold_inds: dict, chunks: list[dict]) -> list[str]:
    """gold_inds 의 각 근거를 실제 context 청크 id 로 매핑.

    파일 작성자의 gold_inds 키(table_N/text_N)는 텍스트의 경우 자체 일련번호라 위치
    인덱싱과 어긋날 수 있으므로, 근거 본문을 context 청크 본문과 내용 대조해 매핑한다.
    매핑 실패 시 원래 키를 보존. (검색을 끄면 전체 청크가 투입되므로 gold_chunk_retrieved
    지표가 의미를 갖도록 gold id 가 context id 의 부분집합이 되게 한다.)
    """
    out: list[str] = []
    for key, val in gold_inds.items():
        nv = _norm_match(val)
        matched = None
        if nv:
            for c in chunks:
                nc = _norm_match(c["text"])
                if nv and (nv in nc or nc in nv):
                    matched = c["chunk_id"]
                    break
        out.append(matched or key)
    # 중복 제거(순서 보존)
    seen, uniq = set(), []
    for cid in out:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq


def to_record(item: dict) -> dict:
    qa = item["qa"]
    meta = item.get("kdart_meta", {}) or {}
    gold_inds = qa.get("gold_inds") or {}
    chunks = build_context_chunks(item)

    exe = qa.get("exe_ans")
    answer = qa.get("answer")
    if isinstance(exe, bool):
        gold = answer
    elif isinstance(exe, (int, float)):
        gold = exe                       # 수치 정답: 실행값
    elif exe not in (None, ""):
        gold = exe                       # 문자열 정답(항목명/사업부문명)
    else:
        gold = answer

    return {
        "id": item["id"],
        "question": qa["question"].strip(),
        "gold_answer": str(gold),
        "gold_answer_unit": meta.get("gold_answer_unit"),
        "task_type": meta.get("task_type", "T?"),
        "task_type_design": meta.get("task_type_design", meta.get("task_type", "")),
        "reasoning_hops": meta.get("reasoning_hops"),
        "evidence_quotes": list(gold_inds.values()),
        "evidence_chunks": map_gold_chunk_ids(gold_inds, chunks),
        "context_chunks": chunks,
        "source_company": meta.get("source_company", ""),
        "source_report": meta.get("source_report", ""),
        "source_section": meta.get("source_section", ""),
        "filename": item.get("filename", ""),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="kdart_qa_finqa_format.json -> kdart_finqa.jsonl")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {src}")
    data = json.load(open(src, encoding="utf-8"))

    records = [to_record(it) for it in data]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"[prepare_kdart] {len(records)} 문항 -> {out}")
    print("  task_type:", dict(Counter(r["task_type"] for r in records)))
    print("  ctx chunk sizes: min/max",
          min(len(r["context_chunks"]) for r in records),
          max(len(r["context_chunks"]) for r in records))
    n_gold_ok = sum(1 for r in records
                    if all(any(c["chunk_id"] == g for c in r["context_chunks"])
                           for g in r["evidence_chunks"]))
    print(f"  gold evidence ⊆ context: {n_gold_ok}/{len(records)} 문항")
    return records


if __name__ == "__main__":
    main()
