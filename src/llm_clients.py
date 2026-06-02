"""모델별 클라이언트 + 토큰/단가 계측 래퍼.

모든 LLM 호출은 반드시 이 모듈을 통해서만 수행한다. 래퍼가 (모델, input_tokens,
output_tokens, node_name) 을 토큰 레코드로 반환하고, 노드가 이를 state.token_usage
(reducer 누적) 에 넣는다. 그래프 종료 시 runner 가 노드별 합을 검산한다.

provider 별 LangChain chat 모델을 사용하며, 생성 파라미터(temperature 등)는 모델
기본값을 그대로 쓴다(통제 변수). 일시적 API 오류는 지수 백오프로 재시도한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from . import config


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    logical_key: str
    node: str

    def token_record(self) -> dict:
        return {
            "node": self.node,
            "model": config.model_pricing_key(self.logical_key),
            "model_string": config.model_string(self.logical_key),
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
        }


@lru_cache(maxsize=None)
def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


@lru_cache(maxsize=None)
def _chat_model(logical_key: str):
    """provider 에 맞는 LangChain chat 모델을 생성(캐시). 생성 파라미터는 기본값."""
    provider = config.model_provider(logical_key)
    model = config.model_string(logical_key)
    timeout = config.experiment_cfg()["robustness"]["request_timeout_seconds"]

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, timeout=timeout, max_retries=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # o3 등 reasoning 모델은 temperature 커스터마이즈 불가 → 기본값만 사용.
        return ChatOpenAI(model=model, timeout=timeout, max_retries=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # ChatAnthropic 은 max_tokens 필수. 답변+추론 JSON 수용 위해 넉넉히.
        return ChatAnthropic(model=model, timeout=timeout, max_retries=0, max_tokens=4096)
    raise config.ConfigError(f"알 수 없는 provider: {provider} (logical_key={logical_key})")


def _usage_from_response(resp, system: str, user: str) -> tuple[int, int]:
    """AIMessage 의 usage_metadata 를 우선 사용, 없으면 tiktoken 추정."""
    meta = getattr(resp, "usage_metadata", None)
    if meta:
        it = int(meta.get("input_tokens") or 0)
        ot = int(meta.get("output_tokens") or 0)
        if it or ot:
            return it, ot
    # fallback
    it = _estimate_tokens(system) + _estimate_tokens(user)
    ot = _estimate_tokens(_text_of(resp))
    return it, ot


def _text_of(resp) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content
    # content 가 블록 리스트인 경우(Anthropic 등) 텍스트만 결합
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", ""))
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content)


def call(logical_key: str, system_prompt: str, user_message: str, node_name: str) -> LLMResult:
    """단일 LLM 호출(지수 백오프 재시도 포함). 토큰 계측 결과를 LLMResult 로 반환."""
    from langchain_core.messages import SystemMessage, HumanMessage

    rb = config.experiment_cfg()["robustness"]
    max_retries = int(rb["max_retries"])
    base = float(rb["backoff_base_seconds"])

    model = _chat_model(logical_key)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = model.invoke(messages)
            it, ot = _usage_from_response(resp, system_prompt, user_message)
            return LLMResult(
                text=_text_of(resp),
                input_tokens=it,
                output_tokens=ot,
                logical_key=logical_key,
                node=node_name,
            )
        except Exception as exc:  # noqa: BLE001 - 일시 오류 백오프 재시도
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_s = base * (2 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(
        f"LLM 호출 영구 실패 (logical_key={logical_key}, node={node_name}): {last_exc}"
    ) from last_exc
