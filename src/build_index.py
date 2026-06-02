"""데이터 준비 + 검색 코퍼스 검증.

이 실험은 검색 코퍼스를 자매 실험(multi_step_qa_langchain)과 동일하게 재사용한다:
  - FinQA : 전역 인덱스 불필요. FinQA-main 에서 data/finqa/finqa.jsonl 을 구성(문항별 컨텍스트).
  - K-DART: 전역 ChromaDB 인덱스(kdart_chunks, 2,137 청크)를 data/chroma 에 복사해 사용.

따라서 본 스크립트는 (1) FinQA finqa.jsonl 준비 (2) K-DART 인덱스 존재·개수 검증을 한다.
새로 임베딩/색인을 만들지 않으므로 API 키가 없어도 동작한다.

  python -m src.build_index            # finqa.jsonl 준비 + kdart 인덱스 검증
  python -m src.build_index --verify   # 검증만
"""
from __future__ import annotations

import argparse
import sys

from . import config


def prepare_finqa() -> dict:
    from . import prepare_finqa as pf
    out = config.DATA_DIR / "finqa" / "finqa.jsonl"
    recs = pf.main(["--out", str(out)])
    return {"finqa_jsonl": str(out), "n": len(recs)}


def verify_kdart_index() -> dict:
    import chromadb
    name = config.experiment_cfg()["datasets"]["kdart"]["collection"]
    cdir = config.chroma_dir()
    if not (cdir / "chroma.sqlite3").exists():
        raise config.ConfigError(
            f"K-DART ChromaDB 인덱스가 없습니다: {cdir}\n"
            "자매 실험 multi_step_qa_langchain/data/index 를 data/chroma 로 복사하세요.")
    client = chromadb.PersistentClient(path=str(cdir))
    names = [c.name for c in client.list_collections()]
    if name not in names:
        raise config.ConfigError(
            f"컬렉션 '{name}' 이 인덱스에 없습니다. 발견된 컬렉션: {names}")
    col = client.get_collection(name)
    n = col.count()
    if n == 0:
        raise config.ConfigError(f"컬렉션 '{name}' 이 비어 있습니다.")
    return {"collection": name, "n_chunks": n, "chroma_dir": str(cdir)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="데이터 준비 + 코퍼스 검증")
    ap.add_argument("--verify", action="store_true", help="검증만(준비 생략)")
    args = ap.parse_args(argv)

    summary = {}
    if not args.verify:
        summary["finqa"] = prepare_finqa()
    summary["kdart_index"] = verify_kdart_index()

    print("\n=== 데이터 준비/검증 완료 ===")
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
    sys.exit(0)
