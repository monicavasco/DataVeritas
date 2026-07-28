from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"

PROVIDER_ENV = "LLM_PROVIDER"
PROVIDER_OLLAMA = "ollama"
PROVIDER_GEMINI = "gemini"
PROVIDER_OPENAI = "openai"
PROVIDER_CUSTOM = "custom"
SUPPORTED_PROVIDERS = (PROVIDER_OLLAMA, PROVIDER_GEMINI, PROVIDER_OPENAI, PROVIDER_CUSTOM)

OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

GEMINI_MODEL_ENV = "GEMINI_MODEL"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

CUSTOM_MODEL_ENV = "CUSTOM_LLM_MODEL"
CUSTOM_API_KEY_ENV = "CUSTOM_LLM_API_KEY"
CUSTOM_BASE_URL_ENV = "CUSTOM_LLM_BASE_URL"
DEFAULT_CUSTOM_MODEL = ""

PROVIDER_API_KEY_ENV_ALIASES = {
    "openai": OPENAI_API_KEY_ENV,
    "gemini": GEMINI_API_KEY_ENV,
    "google": GOOGLE_API_KEY_ENV,
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "hosted_vllm": "VLLM_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "ollama_chat": "OLLAMA_API_KEY",
}
CUSTOM_PROVIDERS_WITH_OPTIONAL_KEY = {"ollama", "ollama_chat", "hosted_vllm"}


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    model_name: str
    temperature: float = 0.2
    timeout: int = 240
    max_tokens: int = 1400
    api_key: str | None = None
    api_key_source: str | None = None
    base_url: str | None = None

    @property
    def display_name(self) -> str:
        return f"{provider_label(self.provider)} / {self.model_name}"

    @property
    def is_ready(self) -> bool:
        has_model = bool(self.model_name.strip())
        has_key = bool(self.api_key) or not provider_requires_api_key(self.provider, self.model_name)
        return has_model and has_key

    @property
    def missing_configuration_message(self) -> str | None:
        if not self.model_name.strip():
            return "Informe o modelo antes de executar os agentes."
        if provider_requires_api_key(self.provider, self.model_name) and not self.api_key:
            env_names = " ou ".join(api_key_env_names_for(self.provider, self.model_name))
            return (
                f"Defina {env_names} no arquivo .env ou nas variaveis de ambiente "
                f"para usar {provider_label(self.provider)}."
            )
        return None

    def to_crewai_kwargs(self) -> dict:
        kwargs = {
            "model": crewai_model_name(self.provider, self.model_name),
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
        }
        if self.provider == PROVIDER_OLLAMA:
            kwargs["base_url"] = self.base_url or DEFAULT_OLLAMA_BASE_URL
        elif self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs


def load_local_env(path: Path = LOCAL_ENV_PATH, override: bool = True) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env without adding a runtime dependency."""
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue

        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return loaded


def normalize_provider(provider: str | None = None) -> str:
    load_local_env()
    selected = (provider or os.getenv(PROVIDER_ENV) or PROVIDER_OLLAMA).strip().lower()
    aliases = {
        "local": PROVIDER_OLLAMA,
        "ollama-local": PROVIDER_OLLAMA,
        "google": PROVIDER_GEMINI,
        "google-gemini": PROVIDER_GEMINI,
        "chatgpt": PROVIDER_OPENAI,
        "gpt": PROVIDER_OPENAI,
        "api": PROVIDER_CUSTOM,
        "custom-api": PROVIDER_CUSTOM,
        "openai-compatible": PROVIDER_CUSTOM,
    }
    selected = aliases.get(selected, selected)
    if selected not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Provedor LLM nao suportado: {selected}. Use: {supported}.")
    return selected


def provider_label(provider: str) -> str:
    labels = {
        PROVIDER_OLLAMA: "Ollama local",
        PROVIDER_GEMINI: "Gemini API",
        PROVIDER_OPENAI: "OpenAI API",
        PROVIDER_CUSTOM: "API custom",
    }
    return labels.get(provider, provider)


def default_model_for(provider: str) -> str:
    provider = normalize_provider(provider)
    if provider == PROVIDER_GEMINI:
        return os.getenv(GEMINI_MODEL_ENV, DEFAULT_GEMINI_MODEL)
    if provider == PROVIDER_OPENAI:
        return os.getenv(OPENAI_MODEL_ENV, DEFAULT_OPENAI_MODEL)
    if provider == PROVIDER_CUSTOM:
        return os.getenv(CUSTOM_MODEL_ENV, DEFAULT_CUSTOM_MODEL)
    return os.getenv(OLLAMA_MODEL_ENV, DEFAULT_OLLAMA_MODEL)


def ollama_base_url() -> str:
    load_local_env()
    return os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def base_url_for_provider(provider: str) -> str | None:
    provider = normalize_provider(provider)
    if provider == PROVIDER_OLLAMA:
        return ollama_base_url()
    if provider == PROVIDER_OPENAI:
        return os.getenv(OPENAI_BASE_URL_ENV)
    if provider == PROVIDER_CUSTOM:
        return os.getenv(CUSTOM_BASE_URL_ENV)
    return None


def provider_prefix(model_name: str | None) -> str | None:
    if not model_name or "/" not in model_name:
        return None
    return model_name.split("/", 1)[0].strip().lower() or None


def api_key_env_names_for(provider: str, model_name: str | None = None) -> tuple[str, ...]:
    provider = normalize_provider(provider)
    if provider == PROVIDER_GEMINI:
        return (GEMINI_API_KEY_ENV, GOOGLE_API_KEY_ENV)
    if provider == PROVIDER_OPENAI:
        return (OPENAI_API_KEY_ENV,)
    if provider == PROVIDER_CUSTOM:
        names = [CUSTOM_API_KEY_ENV]
        prefix = provider_prefix(model_name)
        inferred = PROVIDER_API_KEY_ENV_ALIASES.get(prefix or "")
        if inferred:
            names.append(inferred)
        return tuple(dict.fromkeys(names))
    return ()


def provider_requires_api_key(provider: str, model_name: str | None = None) -> bool:
    provider = normalize_provider(provider)
    if provider == PROVIDER_OLLAMA:
        return False
    if provider == PROVIDER_CUSTOM:
        prefix = provider_prefix(model_name)
        return prefix not in CUSTOM_PROVIDERS_WITH_OPTIONAL_KEY
    return True


def api_key_for_provider(
    provider: str,
    explicit_api_key: str | None = None,
    model_name: str | None = None,
) -> tuple[str | None, str | None]:
    provider = normalize_provider(provider)
    if explicit_api_key:
        return explicit_api_key, "entrada da interface"
    if not provider_requires_api_key(provider, model_name):
        return None, None
    for env_name in api_key_env_names_for(provider, model_name):
        value = os.getenv(env_name)
        if value:
            return value, env_name
    return None, None


def build_llm_settings(
    provider: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.2,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LlmSettings:
    selected_provider = normalize_provider(provider)
    selected_model = (model_name or default_model_for(selected_provider)).strip()
    selected_key, selected_key_source = api_key_for_provider(
        selected_provider,
        api_key,
        selected_model,
    )
    return LlmSettings(
        provider=selected_provider,
        model_name=selected_model,
        temperature=temperature,
        api_key=selected_key,
        api_key_source=selected_key_source,
        base_url=(base_url.strip() if base_url else None) or base_url_for_provider(selected_provider),
    )


def crewai_model_name(provider: str, model_name: str) -> str:
    provider = normalize_provider(provider)
    model_name = model_name.strip()
    if provider == PROVIDER_CUSTOM:
        return model_name
    if "/" in model_name:
        return model_name
    return f"{provider}/{model_name}"


def configured_provider() -> str:
    return normalize_provider(os.getenv(PROVIDER_ENV))


def redact_secret(value: str | None) -> str:
    if not value:
        return "ausente"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
