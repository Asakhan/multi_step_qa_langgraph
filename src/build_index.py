"""데이터 준비 + 검증 (재현용 진입점).

두 데이터셋(FinQA 20 + K-DART-QA 40)은 모두 self-contained 포맷이며, 실험에 쓰인
정본(dataset-of-record)이 저장소에 이미 커밋되어 있다:
  - data/finqa/finqa.jsonl              (FinQA 영어 20문항)
  - data/kdart/kdart_finqa.jsonl        (K-DART-QA 한국어 40문항, 논문 §3.3 최종본)
따라서 clone 직후 별도 준비 없이 곧바로 실험을 실행할 수 있다.

**외부 API 키·임베딩·ChromaDB 인덱스가 필요 없다** — 두 데이터셋 모두 문항이 근거를
통째로 포함하므로 전역 검색 없이 전체 컨텍스트를 투입한다.

이 스크립트는 재현/검수를 위해:
  1) FinQA : 원천(FinQA-main/dataset/test.json)이 있으면 finqa.jsonl 을 재생성한다.
             재생성은 결정적이라 커밋본과 동일하다. FinQA-main 은 대용량이라 미커밋이며
             없으면 커밋된 finqa.jsonl 을 그대로 사용한다(prepare_finqa 의 canonical 20 id).
  2) K-DART: 검증만 한다. 정본 kdart_finqa.jsonl 은 별도 구축 저장소
             (https://github.com/Asakhan/kdart_qa, 논문 §3.3.4)의 산출물이므로 여기서
             재생성하지 않는다. (참고용 파생 코드 src/prepare_kdart.py 는 청크 라벨링·
             추출 방식이 정본과 다를 수 있어 재현 경로에서 사용하지 않는다.)
  3) 검증  : datasets.load_questions() 로 60문항(finqa 20 + kdart 40) 로드 확인.

  python -m src.build_index            # finqa 재생성(가능 시) + 60문항 검증
  python -m src.build_index --verify   # 검증만(재생성 생략)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from . import config


def regenerate_finqa() -> dict:
    """FinQA-main test.json 이 있으면 finqa.jsonl 재생성(결정적, 커밋본과 동일).
    없으면 커밋된 finqa.jsonl 을 그대로 사용."""
    from . import prepare_finqa as pf
    test_json = config.DATA_DIR / "finqa" / "FinQA-main" / "dataset" / "test.json"
    out = config.DATA_DIR / "finqa" / "finqa.jsonl"
    if not test_json.exists():
        return {
            "finqa_jsonl": str(out),
            "regenerated": False,
            "note": ("FinQA-main(대용량 원천)이 없어 재생성 생략 — 커밋된 finqa.jsonl 사용. "
                     "원천에서 재생성하려면 https://github.com/czyssrs/FinQA 를 "
                     "data/finqa/FinQA-main 으로 clone 후 다시 실행."),
        }
    recs = pf.main(["--finqa-test", str(test_json), "--out", str(out)])
    return {"finqa_jsonl": str(out), "regenerated": True, "n": len(recs)}


def verify_datasets() -> dict:
    """60문항(finqa 20 + kdart 40)이 정상 로드되는지 검증."""
    from .datasets import load_questions
    qs = load_questions()
    by_ds = Counter(q.dataset for q in qs)
    by_tt = Counter(q.task_type for q in qs)
    expect = {"finqa": 20, "kdart": 40}
    problems = [f"{ds}: {by_ds.get(ds, 0)}개(기대 {n})"
                for ds, n in expect.items() if by_ds.get(ds, 0) != n]
    if problems:
        raise config.ConfigError(
            "데이터셋 문항 수 불일치: " + ", ".join(problems)
            + "\ndata/finqa/finqa.jsonl, data/kdart/kdart_finqa.jsonl 를 확인하세요.")
    return {
        "n_total": len(qs),
        "by_dataset": dict(by_ds),
        "by_task_type": dict(by_tt),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="데이터 준비 + 검증 (재현용)")
    ap.add_argument("--verify", action="store_true", help="검증만(재생성 생략)")
    args = ap.parse_args(argv)

    summary = {}
    if not args.verify:
        summary["finqa"] = regenerate_finqa()
    summary["verify"] = verify_datasets()

    print("\n=== 데이터 준비/검증 완료 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
    sys.exit(0)
