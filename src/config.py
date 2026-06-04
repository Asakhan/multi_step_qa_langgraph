"""환경 변수 로딩 + 검증, config/*.yaml 로딩, 경로 상수.

시작 시 .env 를 로드하고 필수 키가 비어 있으면 어떤 키가 빠졌는지 명시하며 즉시 중단한다.
모델 문자열/단가/실험 파라미터는 전부 파일(config/*.yaml, .env)에서만 읽는다.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
# 결과 디렉토리는 RESULTS_DIR 환경변수로 오버라이드 가능(실험별 결과 분리용).
# 기본값 results/. 예: RESULTS_DIR=results_finqa_format 로 새 실험을 기존과 분리.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(ROOT / "results"))).resolve()
LOGS_DIR = ROOT / "logs"
PROMPTS_DIR = ROOT / "src" / "prompts"

# .env 로드 (있으면). 없으면 OS 환경변수만 사용.
load_dotenv(ROOT / ".env")

ALL_STRUCTURES = ["s0", "s1", "s2", "s3", "s4", "s5"]

# 구조별로 실제 호출에 필요한 provider (필수 env 키 검증에 사용)
_PROVIDER_KEY_ENV = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pricing() -> dict:
    return load_yaml("pricing.yaml")


def models_cfg() -> dict:
    return load_yaml("models.yaml")


def experiment_cfg() -> dict:
    return load_yaml("experiment.yaml")


def model_string(logical_key: str) -> str:
    """논리 모델 키 -> 실제 API 모델 문자열 (.env MODEL_* 에서)."""
    spec = models_cfg()["models"][logical_key]
    env_name = spec["env"]
    val = os.environ.get(env_name, "").strip()
    if not val:
        raise ConfigError(
            f"환경변수 {env_name} 가 비어 있습니다 (논리 모델 '{logical_key}'). "
            f".env 에 실제 모델 문자열을 입력하세요."
        )
    return val


def model_provider(logical_key: str) -> str:
    return models_cfg()["models"][logical_key]["provider"]


def model_pricing_key(logical_key: str) -> str:
    return models_cfg()["models"][logical_key]["pricing"]


class ConfigError(RuntimeError):
    pass


def _logical_keys_for_structures(structures: list[str]) -> set[str]:
    cfg = models_cfg()["structures"]
    keys: set[str] = set()
    for s in structures:
        for _role, logical in cfg.get(s, {}).items():
            keys.add(logical)
    return keys


def required_env_for(structures: list[str], *, need_email: bool, need_embedding: bool = True) -> list[str]:
    """주어진 구조 실행에 반드시 필요한 env 키 목록을 산출."""
    required: set[str] = set()
    # 모델 문자열 + provider API 키
    for logical in _logical_keys_for_structures(structures):
        spec = models_cfg()["models"][logical]
        required.add(spec["env"])  # MODEL_*
        prov = spec["provider"]
        required.add(_PROVIDER_KEY_ENV[prov])  # API 키
    if need_embedding:
        # 검색 임베딩(text-embedding-3-small)은 OpenAI
        required.add("EMBEDDING_MODEL")
        required.add("OPENAI_API_KEY")
    if need_email:
        for k in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_FROM", "MAIL_TO"]:
            required.add(k)
    return sorted(required)


def validate_env(structures: list[str], *, need_email: bool, need_embedding: bool = True) -> None:
    """필수 키가 비어 있으면 친절한 에러와 함께 즉시 중단."""
    missing = [k for k in required_env_for(structures, need_email=need_email, need_embedding=need_embedding)
               if not os.environ.get(k, "").strip()]
    if missing:
        msg = (
            "\n[설정 오류] 다음 필수 환경변수가 비어 있습니다. .env 를 작성하세요:\n"
            + "\n".join(f"  - {k}" for k in missing)
            + "\n\n.env.example 을 복사해 .env 를 만들고 값을 채운 뒤 다시 실행하세요.\n"
        )
        print(msg, file=sys.stderr)
        raise ConfigError(msg)


def chroma_dir() -> Path:
    return Path(os.environ.get("CHROMA_DIR", str(DATA_DIR / "chroma"))).resolve()


def embedding_model() -> str:
    val = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not val:
        raise ConfigError("EMBEDDING_MODEL 환경변수가 비어 있습니다.")
    return val


def cost_cap_usd() -> float | None:
    raw = os.environ.get("COST_CAP_USD", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
