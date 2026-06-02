"""FinQA(20) + K-DART-QA(40) 를 통일된 Question 레코드 60개로 로딩.

두 데이터셋 모두 자매 실험(multi_step_qa_langchain)과 동일한 공통 스키마를 쓴다:
  K-DART-QA: data/kdart/kdart_finqa.jsonl (FinQA 포맷 자체완결 40문항, T1~T5)
  FinQA    : data/finqa/finqa.jsonl       (FinQA-main 에서 prepare_finqa 로 구성한 20문항)

두 데이터셋 모두 self-contained: 문항별 context_chunks(text_N/table_N)를 함께 보유한다.
K-DART-QA 도 FinQA 와 동일하게 문항이 자기 근거 문서를 통째로 포함하므로(난이도도 FinQA
수준으로 보정됨) 전역 ChromaDB 인덱스·임베딩 검색이 불필요하다 → 두 데이터셋이 완전히
대칭(동일 파이프라인)이 되어 더 객관적인 비교가 된다. (src/prepare_kdart.py 로 생성)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from functools import lru_cache

from . import config

CONFIG = config.experiment_cfg()


@dataclass
class Question:
    qid: str
    dataset: str            # 'finqa' | 'kdart'
    lang: str               # 'en' | 'ko'
    task_type: str          # T1~T5
    orig_task_type: str
    query: str
    gold_answer: str
    gold_answer_unit: str | None
    evidence_quotes: list[str] = field(default_factory=list)
    evidence_chunks: list[str] = field(default_factory=list)
    reasoning_hops: int | None = None
    context_chunks: list[dict] = field(default_factory=list)  # FinQA 문항별 컨텍스트
    source: dict = field(default_factory=dict)

    def as_meta(self) -> dict:
        d = asdict(self)
        d.pop("context_chunks", None)  # 메타 로깅에는 컨텍스트 본문 제외(부피)
        return d


def _load_finqa() -> list[Question]:
    path = CONFIG["datasets"]["finqa"]["questions_jsonl"]
    out: list[Question] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tt = r.get("task_type", "T?")
            out.append(Question(
                qid=r["id"],
                dataset="finqa",
                lang="en",
                task_type=tt,
                orig_task_type=tt,
                query=r["question"].strip(),
                gold_answer=str(r.get("gold_answer", "")).strip(),
                gold_answer_unit=r.get("gold_answer_unit"),
                evidence_quotes=[],
                evidence_chunks=list(r.get("evidence_chunks") or []),
                reasoning_hops=r.get("n_steps"),
                context_chunks=list(r.get("context_chunks") or []),
                source={"n_steps": r.get("n_steps")},
            ))
    return out


def _load_kdart() -> list[Question]:
    path = CONFIG["datasets"]["kdart"]["questions_jsonl"]
    out: list[Question] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Question(
                qid=r["id"],
                dataset="kdart",
                lang="ko",
                task_type=r.get("task_type", "T?"),
                orig_task_type=r.get("task_type_design", r.get("task_type", "")),
                query=r["question"].strip(),
                gold_answer=str(r.get("gold_answer", "")).strip(),
                gold_answer_unit=r.get("gold_answer_unit"),
                evidence_quotes=list(r.get("evidence_quotes") or []),
                evidence_chunks=list(r.get("evidence_chunks") or []),
                reasoning_hops=r.get("reasoning_hops"),
                context_chunks=list(r.get("context_chunks") or []),  # FinQA 포맷: 문항별 self-contained 컨텍스트
                source={
                    "source_company": r.get("source_company", ""),
                    "source_report": r.get("source_report", ""),
                    "source_section": r.get("source_section", ""),
                },
            ))
    return out


@lru_cache(maxsize=1)
def load_questions() -> list[Question]:
    return _load_finqa() + _load_kdart()


def questions_by_dataset() -> dict[str, list[Question]]:
    out: dict[str, list[Question]] = {"finqa": [], "kdart": []}
    for q in load_questions():
        out[q.dataset].append(q)
    return out


def sample_questions(n_per_dataset: int = 2) -> list[Question]:
    by = questions_by_dataset()
    out = []
    for ds in ("finqa", "kdart"):
        out.extend(by[ds][:n_per_dataset])
    return out


if __name__ == "__main__":
    qs = load_questions()
    from collections import Counter
    print("총", len(qs), "문항")
    print("dataset:", dict(Counter(q.dataset for q in qs)))
    print("task_type:", dict(Counter(q.task_type for q in qs)))
    print("lang:", dict(Counter(q.lang for q in qs)))
    finqa = [q for q in qs if q.dataset == "finqa"]
    print("finqa ctx sizes:", [len(q.context_chunks) for q in finqa][:5], "...")
