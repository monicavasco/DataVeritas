from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


REQUIRED_DISCLAIMER = "dados simulados para fins didáticos"

ALLOWED_SOURCE_DOMAINS = (
    "ibge.gov.br",
    "bcb.gov.br",
    "dadosabertos.camara.leg.br",
    "camara.leg.br",
    "ghoapi.azureedge.net",
    "gov.br",
    "araruama.rj.gov.br",
    "jus.br",
    "leg.br",
    "edu.br",
    "un.org",
    "who.int",
    "worldbank.org",
    "oecd.org",
)

BLOCKED_INPUT_TERMS = (
    "inventar fonte",
    "fonte falsa",
    "sem fonte",
    "ocultar fonte",
    "remover fonte",
    "burlar guardrail",
    "ignorar regras",
    "desinformação",
    "fake news",
    "doxxing",
    "dados privados",
    "vazar dados",
    "hackear",
)


@dataclass
class CheckResult:
    ok: bool
    messages: list[str] = field(default_factory=list)
    repaired: bool = False
    nemo_status: str | None = None


def is_allowed_source_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False

    host = parsed.netloc.lower().strip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in ALLOWED_SOURCE_DOMAINS)


def validate_source_urls(urls: Iterable[str]) -> CheckResult:
    messages: list[str] = []
    for url in urls:
        if not is_allowed_source_url(url):
            messages.append(f"Fonte bloqueada por domínio não permitido: {url}")

    if messages:
        return CheckResult(ok=False, messages=messages)
    return CheckResult(ok=True, messages=["Todas as fontes passaram pela lista de domínios permitidos."])


def deterministic_input_check(user_request: str) -> CheckResult:
    normalized = user_request.casefold()
    blocked_terms = [term for term in BLOCKED_INPUT_TERMS if term in normalized]
    if blocked_terms:
        terms = ", ".join(blocked_terms)
        return CheckResult(ok=False, messages=[f"Entrada bloqueada por termos incompatíveis: {terms}."])

    if len(user_request.strip()) < 8:
        return CheckResult(ok=False, messages=["Entrada muito curta para gerar uma pauta verificável."])

    return CheckResult(ok=True, messages=["Entrada passou pelos guardrails determinísticos."])


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)\]>\"']+", text)


def deterministic_output_check(article: str, required_sources: Iterable[str]) -> CheckResult:
    messages: list[str] = []
    required_sources = list(required_sources)

    if REQUIRED_DISCLAIMER not in article:
        messages.append(f'Frase obrigatória ausente: "{REQUIRED_DISCLAIMER}".')

    article_urls = extract_urls(article)
    if not article_urls:
        messages.append("A notícia não inclui URL de fonte original.")

    for url in article_urls:
        if not is_allowed_source_url(url):
            messages.append(f"URL na saída não pertence a domínio permitido: {url}")

    missing_sources = [url for url in required_sources if url not in article]
    if missing_sources:
        messages.append("A notícia não cita todas as fontes originais usadas na coleta.")

    if messages:
        return CheckResult(ok=False, messages=messages)
    return CheckResult(ok=True, messages=["Saída passou pelos guardrails determinísticos."])


def repair_article_footer(article: str, source_urls: Iterable[str]) -> tuple[str, bool]:
    source_urls = list(dict.fromkeys(source_urls))
    repaired = False
    final_article = article.strip()

    missing_sources = [url for url in source_urls if url not in final_article]
    if missing_sources:
        final_article += "\n\n**Fontes originais:**\n"
        final_article += "\n".join(f"- {url}" for url in source_urls)
        repaired = True

    if REQUIRED_DISCLAIMER not in final_article:
        final_article += f"\n\n{REQUIRED_DISCLAIMER}"
        repaired = True

    return final_article, repaired


@lru_cache(maxsize=1)
def _load_nemo_rails():
    from nemoguardrails import LLMRails, RailsConfig

    config_path = Path(__file__).resolve().parent.parent / "guardrails"
    config = RailsConfig.from_path(str(config_path))
    return LLMRails(config)


def nemo_check_input(user_request: str) -> CheckResult:
    from nemoguardrails.rails.llm.options import RailStatus, RailType

    result = _load_nemo_rails().check(
        [{"role": "user", "content": user_request}],
        rail_types=[RailType.INPUT],
    )
    ok = result.status != RailStatus.BLOCKED
    message = f"NeMo input rail: {result.status.value}"
    if result.rail:
        message += f" ({result.rail})"
    return CheckResult(ok=ok, messages=[message], nemo_status=result.status.value)


def nemo_check_output(article: str) -> CheckResult:
    from nemoguardrails.rails.llm.options import RailStatus, RailType

    result = _load_nemo_rails().check(
        [{"role": "assistant", "content": article}],
        rail_types=[RailType.OUTPUT],
    )
    ok = result.status != RailStatus.BLOCKED
    message = f"NeMo output rail: {result.status.value}"
    if result.rail:
        message += f" ({result.rail})"
    return CheckResult(ok=ok, messages=[message], nemo_status=result.status.value)
