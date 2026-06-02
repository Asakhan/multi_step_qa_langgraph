"""저자 확정 시스템 프롬프트 로더.

각 프롬프트 파일 첫 줄의 `[[AUTHOR-LOCKED ...]]` 마커(+이어지는 빈 줄)는 제거하고
그 아래 본문을 바이트 동일하게 반환한다. {common_rules} 슬롯만 치환하며, 나머지
{query}/{evidence}/... 슬롯은 그대로 남겨 노드가 user 메시지 구성 시 .format 으로 채운다.

프롬프트 본문은 절대 변형/요약/언어변경하지 않는다 (실험 통제).
"""
from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR

_LOCK_PREFIX = "[[AUTHOR-LOCKED"


def _read_body(name: str) -> str:
    path = PROMPTS_DIR / name
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    # 첫 줄이 lock 마커면 제거하고, 바로 다음의 빈 줄 1개도 제거.
    if lines and lines[0].startswith(_LOCK_PREFIX):
        lines = lines[1:]
        if lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines)


@lru_cache(maxsize=None)
def _common_rules() -> str:
    # common_rules 본문은 끝의 개행을 제거해 {common_rules} 치환 시 깔끔하게 들어가도록.
    return _read_body("common_rules.txt").rstrip("\n")


@lru_cache(maxsize=None)
def system_prompt(name: str) -> str:
    """노드 시스템 프롬프트를 반환. {common_rules} 만 치환한다.

    name 예: 'generate.txt', 'verify.txt', 'vote_generate.txt',
             'decompose.txt', 'solve.txt', 'aggregate_s4.txt'
    """
    body = _read_body(name)
    if "{common_rules}" in body:
        body = body.replace("{common_rules}", _common_rules())
    return body
