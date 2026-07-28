from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from dataveritas.guardrails import validate_source_urls

ARARUAMA_NEWS_URL = "https://www.araruama.rj.gov.br/noticias"


@dataclass(frozen=True)
class AraruamaRecord:
    data: str
    titulo: str
    url: str


@dataclass(frozen=True)
class AraruamaDataset:
    tipo: str
    consulta: str
    fonte_nome: str
    registros_encontrados: int
    registros: list[AraruamaRecord]
    fontes: list[str]
    resumo_numerico: str
    nota_metodologica: str
    coletado_em: str

    def to_prompt_dict(self) -> dict:
        data = asdict(self)
        data["registros"] = [asdict(row) for row in self.registros]
        return data


def _session() -> requests.Session:
    from urllib3.util import Retry
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DataVeritas didactic prototype"
        }
    )
    return session


def build_araruama_news_dataset(query: str) -> AraruamaDataset:
    validate_source_urls([ARARUAMA_NEWS_URL])

    response = _session().get(ARARUAMA_NEWS_URL, timeout=25)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=lambda h: h and "/noticia/" in h)

    records = []
    seen_urls = set()

    for l in links:
        href = l.get("href")
        if not href or href in seen_urls:
            continue
        seen_urls.add(href)

        text = " ".join(l.text.split())
        match = re.match(
            r"^(\d{2}\.\d{2}\.\d{4})\s+(.*?)(?:\s+Notícias)?$", text, re.IGNORECASE
        )
        if match:
            date_str = match.group(1)
            title = match.group(2).strip()
        else:
            date_str = ""
            title = text

        records.append(AraruamaRecord(data=date_str, titulo=title, url=href))

    # Optional query keyword filter (helps target specific pautas like "saúde" or "turismo")
    normalized_query = query.lower()
    stop_terms = {
        "noticia",
        "noticias",
        "araruama",
        "prefeitura",
        "gerar",
        "sobre",
        "do",
        "da",
        "de",
        "em",
        "para",
    }
    query_words = [
        w
        for w in re.findall(r"\b\w{3,}\b", normalized_query)
        if w not in stop_terms
    ]

    if query_words:
        filtered = []
        for r in records:
            title_lower = r.titulo.lower()
            if any(word in title_lower for word in query_words):
                filtered.append(r)
        if filtered:
            records = filtered

    # Limit to latest 10 matches
    records = records[:10]

    # Gather sources used
    fontes = list(dict.fromkeys([ARARUAMA_NEWS_URL] + [r.url for r in records]))
    validate_source_urls(fontes)

    collected_at = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    latest_date = "desconhecida"
    if records:
        dates = [r.data for r in records if r.data]
        if dates:
            latest_date = dates[0]

    resumo = (
        f"A busca por notícias de Araruama para '{query}' retornou {len(records)} "
        f"registro(s) do portal oficial da prefeitura. A notícia mais recente é de {latest_date}."
    )

    return AraruamaDataset(
        tipo="araruama_news",
        consulta=query,
        fonte_nome="Prefeitura de Araruama - Portal de Notícias",
        registros_encontrados=len(records),
        registros=records,
        fontes=fontes,
        resumo_numerico=resumo,
        nota_metodologica=(
            "As informações são de caráter jornalístico e institucional público obtidas por "
            "varredura direta do portal de notícias da Prefeitura Municipal de Araruama. "
            "Esses dados devem ser analisados como comunicados de comunicação institucional oficial."
        ),
        coletado_em=collected_at,
    )
