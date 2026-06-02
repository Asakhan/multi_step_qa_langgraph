"""공통 도구: RAG 검색 retrieve(query) 와 수치 계산 calc(expression).

모든 구조가 동일한 시그니처를 공유한다. 두 데이터셋 모두 문항별 self-contained
context_chunks(text_N/table_N)를 쓴다(전역 인덱스 불필요):

  K-DART-QA : FinQA 포맷으로 문항이 근거 문서를 통째로 포함 → 전체 청크를 그대로 투입
              (임베딩·검색 없음. src/prepare_kdart.py 로 kdart_finqa.jsonl 생성)
  FinQA     : 문항별 context_chunks 를 in-memory 코사인 검색(top_k=50; 컨텍스트가 top_k
              이하면 전체 반환 → 실측상 임베딩 미발생). 임베딩은 OpenAI text-embedding-3-small.

전역 ChromaDB(_retrieve_kdart / _kdart_collection)는 구(舊) 실험 호환을 위해 남겨두나
현재 파이프라인에서는 호출되지 않는다(레거시).

calc 는 ast 기반 안전 평가(eval 금지).
"""
from __future__ import annotations

import ast
import operator as op
from functools import lru_cache

from . import config

# ---------------------------------------------------------------------------
# RAG 검색
# ---------------------------------------------------------------------------

_TOP_K = config.experiment_cfg()["retrieval"]["top_k"]


@lru_cache(maxsize=None)
def _embedder():
    """OpenAI 임베딩 클라이언트(쿼리/청크 공통)."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.embedding_model())


@lru_cache(maxsize=None)
def _kdart_collection():
    """복사해온 prebuilt Chroma 인덱스의 kdart_chunks 컬렉션."""
    import chromadb
    name = config.experiment_cfg()["datasets"]["kdart"]["collection"]
    client = chromadb.PersistentClient(path=str(config.chroma_dir()))
    return client.get_collection(name)


def _retrieve_kdart(query: str, k: int) -> list[dict]:
    col = _kdart_collection()
    qemb = _embedder().embed_query(query)
    res = col.query(query_embeddings=[qemb], n_results=k,
                    include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    out = []
    for i, text in enumerate(docs):
        md = dict(metas[i] or {}) if i < len(metas) else {}
        out.append({
            "text": text,
            "chunk_id": md.get("chunk_id", ids[i] if i < len(ids) else ""),
            "distance": float(dists[i]) if i < len(dists) else None,
            "source": {
                "company": md.get("company", ""),
                "year": md.get("year", ""),
                "section": md.get("section", ""),
                "doc_id": md.get("report_code", ""),
            },
        })
    return out


# FinQA 문항별 컨텍스트 임베딩 캐시 (반복 실행 간 재사용)
_finqa_emb_cache: dict[str, object] = {}


def _retrieve_finqa(query: str, k: int, context_chunks: list[dict]) -> list[dict]:
    if not context_chunks:
        return []
    # top_k 가 컨텍스트 수 이상이면 전부 반환(순서 보존) — 임베딩 호출 불필요.
    if k >= len(context_chunks):
        return [{"text": c["text"], "chunk_id": c.get("chunk_id", ""),
                 "distance": None, "source": {}} for c in context_chunks]
    import numpy as np
    key = "|".join(c.get("chunk_id", "") for c in context_chunks)
    unit = _finqa_emb_cache.get(key)
    if unit is None:
        embs = _embedder().embed_documents([c["text"] for c in context_chunks])
        mat = np.array(embs, dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = mat / norms
        _finqa_emb_cache[key] = unit
    q = np.array(_embedder().embed_query(query), dtype=float)
    qn = q / (np.linalg.norm(q) or 1.0)
    sims = unit @ qn
    order = np.argsort(-sims)[:k]
    out = []
    for i in order:
        c = context_chunks[int(i)]
        out.append({"text": c["text"], "chunk_id": c.get("chunk_id", ""),
                    "distance": float(1.0 - sims[int(i)]), "source": {}})
    return out


def _retrieve_self_contained(context_chunks: list[dict]) -> list[dict]:
    """자체완결 문항 컨텍스트 전체를 그대로 반환(검색·임베딩 없음, 순서 보존)."""
    return [{"text": c["text"], "chunk_id": c.get("chunk_id", ""),
             "distance": None, "source": {}} for c in (context_chunks or [])]


def retrieve(query: str, dataset: str, top_k: int | None = None,
             context_chunks: list[dict] | None = None) -> list[dict]:
    """질의에 대한 컨텍스트 청크 반환. 각 청크는 text + chunk_id + 출처 메타.

    두 데이터셋 모두 문항별 self-contained 컨텍스트를 사용한다:
      - kdart: FinQA 포맷으로 문항이 근거 문서를 통째로 포함 → 전체 청크 투입(임베딩 없음).
      - finqa: 문항별 context_chunks 를 코사인 top_k 검색(컨텍스트가 top_k 이하면 전체 반환).
    """
    k = top_k or _TOP_K
    if dataset == "kdart":
        return _retrieve_self_contained(context_chunks or [])
    return _retrieve_finqa(query, k, context_chunks or [])


def format_evidence(chunks: list[dict]) -> str:
    """검색 청크들을 프롬프트 EVIDENCE 슬롯용 문자열로 직렬화."""
    lines = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", {})
        tag = c.get("chunk_id") or f"{src.get('company','')}/{src.get('section','')}".strip("/")
        head = f"[{i}] ({tag})" if tag else f"[{i}]"
        lines.append(f"{head}\n{c['text']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 수치 계산 (안전 평가)
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


class CalcError(ValueError):
    pass


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalcError(f"허용되지 않는 상수: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Paren) if hasattr(ast, "Paren") else False:  # pragma: no cover
        return _eval_node(node.value)
    raise CalcError(f"허용되지 않는 식 노드: {ast.dump(node)}")


def calc(expression: str) -> float:
    """LLM 이 생성한 산술식을 안전하게 평가. eval 금지, ast 기반.

    허용: 숫자, + - * / ** % //, 단항 +/-, 괄호. 그 외(이름/호출 등)는 거부.
    """
    expr = (expression or "").strip()
    # 'ARITHMETIC = RESULT' 형태면 산술(연산자 포함) 쪽을 평가해 결과를 재계산.
    if "=" in expr:
        parts = [p.strip() for p in expr.split("=") if p.strip()]
        with_op = [p for p in parts if any(o in p for o in "+-*/")]
        expr = (with_op[0] if with_op else (parts[-1] if parts else expr))
    expr = expr.rstrip("%").strip()
    if not expr:
        raise CalcError("빈 식")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"파싱 실패: {e}") from e
    result = _eval_node(tree)
    return float(result)


def safe_calc(expression: str):
    """예외 없이 (value, error) 반환. failures.csv 의 expr_recompute 산출용."""
    try:
        return calc(expression), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)
