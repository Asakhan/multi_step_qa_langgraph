"""청크 분할: 512 토큰 / overlap 50, tiktoken cl100k_base.
표(마크다운)는 단일 청크로 보존한다.
"""
from __future__ import annotations

from functools import lru_cache

import tiktoken

from . import config

_CH = config.experiment_cfg()["chunking"]


@lru_cache(maxsize=None)
def _enc():
    return tiktoken.get_encoding(_CH["encoding"])


def _is_table_line(ln: str) -> bool:
    s = ln.strip()
    return s.startswith("|") or s.count("|") >= 2


def _segments(text: str):
    """(mode, segment) 리스트로 분할. mode in {'table','prose'}."""
    lines = text.split("\n")
    segs = []
    buf: list[str] = []
    mode: str | None = None
    for ln in lines:
        m = "table" if _is_table_line(ln) else "prose"
        if mode is None:
            mode = m
        if m != mode:
            segs.append((mode, "\n".join(buf)))
            buf = []
            mode = m
        buf.append(ln)
    if buf:
        segs.append((mode, "\n".join(buf)))
    return segs


def chunk_text(text: str) -> list[str]:
    """텍스트를 청크 리스트로. 표 블록은 통째로 단일 청크."""
    chunk_tokens = int(_CH["chunk_tokens"])
    overlap = int(_CH["overlap_tokens"])
    table_single = bool(_CH["table_as_single_chunk"])
    enc = _enc()

    chunks: list[str] = []
    for mode, seg in _segments(text or ""):
        seg = seg.strip()
        if not seg:
            continue
        if mode == "table" and table_single:
            chunks.append(seg)
            continue
        toks = enc.encode(seg)
        if len(toks) <= chunk_tokens:
            chunks.append(seg)
            continue
        i = 0
        step = max(1, chunk_tokens - overlap)
        while i < len(toks):
            window = toks[i: i + chunk_tokens]
            chunks.append(enc.decode(window))
            if i + chunk_tokens >= len(toks):
                break
            i += step
    return chunks
