from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dataveritas.guardrails import is_allowed_source_url, validate_source_urls


CAMARA_PROPOSITIONS_URL = "https://dadosabertos.camara.leg.br/api/v2/proposicoes"
DADOS_GOV_DATASETS_URL = "https://dados.gov.br/dados/api/publico/conjuntos-dados"
DADOS_GOV_DOCS_URL = "https://dados.gov.br/swagger-ui/index.html"
WHO_GHO_DOCS_URL = "https://www.who.int/data/gho/info/gho-odata-api"
WHO_GHO_API_BASE_URL = "https://ghoapi.azureedge.net/api"
WHO_GHO_INDICATORS_URL = f"{WHO_GHO_API_BASE_URL}/Indicator"
UN_SDG_DOCS_URL = "https://unstats.un.org/SDGAPI/swagger/"
UN_SDG_INDICATORS_URL = "https://unstats.un.org/SDGAPI/v1/sdg/Indicator/List"

STOPWORDS = {
    "com",
    "das",
    "dados",
    "ache",
    "busca",
    "buscar",
    "busque",
    "descubra",
    "dos",
    "encontre",
    "fonte",
    "fontes",
    "gerar",
    "indicador",
    "indicadores",
    "noticia",
    "para",
    "por",
    "publica",
    "publicas",
    "sobre",
    "taxa",
}

ORGANIZATION_TERMS = {"oms", "who", "onu", "ods", "sdg", "un", "unsd"}

QUERY_EXPANSIONS = {
    "agua": ["water", "drinking water"],
    "aids": ["hiv", "aids"],
    "alcool": ["alcohol"],
    "cancer": ["cancer"],
    "crianca": ["child", "children"],
    "criancas": ["child", "children"],
    "covid": ["covid"],
    "desenvolvimento": ["development", "sustainable development"],
    "diabetes": ["diabetes"],
    "educacao": ["education"],
    "expectativa": ["life expectancy"],
    "fome": ["hunger", "undernourishment"],
    "hiv": ["hiv", "aids"],
    "imunizacao": ["immunization", "vaccination"],
    "infantil": ["child", "neonatal", "under-five"],
    "malaria": ["malaria"],
    "materna": ["maternal"],
    "materno": ["maternal"],
    "mortalidade": ["mortality", "death"],
    "mortes": ["mortality", "death"],
    "obesidade": ["obesity"],
    "obitos": ["mortality", "death"],
    "ods": ["sdg"],
    "oms": ["who", "health", "global health"],
    "onu": ["un", "sdg"],
    "pobreza": ["poverty"],
    "poluicao": ["pollution", "air pollution"],
    "saneamento": ["sanitation"],
    "saude": ["health"],
    "tabagismo": ["tobacco", "smoking"],
    "tuberculose": ["tuberculosis"],
    "vacinacao": ["vaccination", "immunization"],
    "vacina": ["vaccination", "immunization"],
    "vacinas": ["vaccination", "immunization"],
}


@dataclass(frozen=True)
class OpenDataRecord:
    origem: str
    titulo: str
    descricao: str
    data: str
    tipo: str
    url: str


@dataclass(frozen=True)
class PortalSearchStatus:
    portal: str
    ok: bool
    mensagem: str
    url: str


@dataclass(frozen=True)
class OpenDataDiscoveryDataset:
    tipo: str
    consulta: str
    fonte_nome: str
    registros_encontrados: int
    portais_consultados: list[PortalSearchStatus]
    registros: list[OpenDataRecord]
    fontes: list[str]
    resumo_numerico: str
    nota_metodologica: str
    coletado_em: str

    def to_prompt_dict(self) -> dict:
        data = asdict(self)
        data["portais_consultados"] = [asdict(row) for row in self.portais_consultados]
        data["registros"] = [asdict(row) for row in self.registros]
        return data


def _session() -> requests.Session:
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
            "Accept": "application/json",
            "User-Agent": "DataVeritas didactic prototype",
        }
    )
    return session


def _validate_url(url: str) -> None:
    source_check = validate_source_urls([url])
    if not source_check.ok:
        raise ValueError("; ".join(source_check.messages))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").strip())
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.casefold()


def _query_terms(query: str, limit: int = 5) -> list[str]:
    normalized_query = _normalize_text(query)
    tokens = re.findall(r"[a-z0-9]{3,}", normalized_query)
    subject_terms: list[str] = []
    organization_terms: list[str] = []

    for token in tokens:
        target = organization_terms if token in ORGANIZATION_TERMS else subject_terms
        target.extend(QUERY_EXPANSIONS.get(token, []))
        if token not in STOPWORDS and token not in ORGANIZATION_TERMS:
            target.append(token)

    terms = subject_terms + organization_terms
    deduped = list(dict.fromkeys(terms))
    return deduped[:limit] or [normalized_query[:80]]


def _score_text(query: str, text: str) -> int:
    normalized_text = _normalize_text(text)
    text_tokens = set(re.findall(r"[a-z0-9]{2,}", normalized_text))
    score = 0
    for term in _query_terms(query, limit=12):
        normalized_term = _normalize_text(term)
        if len(normalized_term) <= 2:
            if normalized_term in text_tokens:
                score += 1
            continue
        if normalized_term and normalized_term in normalized_text:
            score += 2 if " " in normalized_term else 1
    return score


def _shorten(value: str, limit: int = 420) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1].rstrip()}..."


def _odata_literal(value: str) -> str:
    return value.replace("'", "''")


def _take_diverse_records(records: list[OpenDataRecord], limit: int) -> list[OpenDataRecord]:
    buckets: dict[str, list[OpenDataRecord]] = {}
    seen_keys: set[str] = set()

    for record in records:
        key = f"{record.origem}:{record.titulo}:{record.url}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        buckets.setdefault(record.origem, []).append(record)

    selected: list[OpenDataRecord] = []
    origins = list(buckets)
    while len(selected) < limit and any(buckets.values()):
        for origin in origins:
            if buckets[origin]:
                selected.append(buckets[origin].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def _search_camara(query: str, limit: int) -> tuple[list[OpenDataRecord], PortalSearchStatus]:
    params = {
        "keywords": query,
        "ordem": "DESC",
        "ordenarPor": "id",
        "itens": limit,
    }
    url = f"{CAMARA_PROPOSITIONS_URL}?{urlencode(params)}"
    _validate_url(url)

    response = _session().get(CAMARA_PROPOSITIONS_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    records = []

    for item in payload.get("dados", [])[:limit]:
        title = f"{item.get('siglaTipo', 'Proposição')} {item.get('numero', '')}/{item.get('ano', '')}"
        records.append(
            OpenDataRecord(
                origem="Dados Abertos da Câmara dos Deputados",
                titulo=title.strip(),
                descricao=item.get("ementa") or "Sem ementa informada.",
                data=item.get("dataApresentacao", ""),
                tipo=item.get("siglaTipo", "proposicao"),
                url=item.get("uri", ""),
            )
        )

    source_urls = [row.url for row in records if row.url]
    source_check = validate_source_urls(source_urls or [url])
    if not source_check.ok:
        raise ValueError("; ".join(source_check.messages))

    status = PortalSearchStatus(
        portal="Dados Abertos da Câmara dos Deputados",
        ok=True,
        mensagem=f"{len(records)} registro(s) recuperado(s).",
        url=url,
    )
    return records, status


def _search_who_gho(query: str, limit: int) -> tuple[list[OpenDataRecord], PortalSearchStatus]:
    records: list[OpenDataRecord] = []
    seen_codes: set[str] = set()
    consulted_urls: list[str] = []

    for term in _query_terms(query, limit=4):
        params = {
            "$filter": f"contains(IndicatorName,'{_odata_literal(term)}')",
            "$top": max(limit, 5),
        }
        url = f"{WHO_GHO_INDICATORS_URL}?{urlencode(params)}"
        _validate_url(url)
        consulted_urls.append(url)

        response = _session().get(WHO_GHO_INDICATORS_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        for item in payload.get("value", []):
            code = str(item.get("IndicatorCode") or "").strip()
            name = str(item.get("IndicatorName") or "").strip()
            if not code or not name or code in seen_codes:
                continue
            seen_codes.add(code)
            records.append(
                OpenDataRecord(
                    origem="OMS Global Health Observatory",
                    titulo=name,
                    descricao=(
                        f"Indicador GHO codigo {code}. Use a URL do indicador para recuperar "
                        "a serie numerica oficial antes de afirmar valores."
                    ),
                    data="",
                    tipo="indicador-oms-gho",
                    url=f"{WHO_GHO_API_BASE_URL}/{code}",
                )
            )
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break

    source_urls = [row.url for row in records] or consulted_urls or [WHO_GHO_DOCS_URL]
    source_check = validate_source_urls(source_urls)
    if not source_check.ok:
        raise ValueError("; ".join(source_check.messages))

    status = PortalSearchStatus(
        portal="OMS Global Health Observatory",
        ok=bool(records),
        mensagem=(
            f"{len(records)} indicador(es) GHO recuperado(s)."
            if records
            else "Nenhum indicador GHO encontrado para os termos da pauta."
        ),
        url=consulted_urls[0] if consulted_urls else WHO_GHO_DOCS_URL,
    )
    return records, status


def _search_un_sdg(query: str, limit: int) -> tuple[list[OpenDataRecord], PortalSearchStatus]:
    _validate_url(UN_SDG_INDICATORS_URL)
    response = _session().get(UN_SDG_INDICATORS_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()

    candidates: list[tuple[int, str, dict]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        series = item.get("series") if isinstance(item.get("series"), list) else []
        series_text = " ".join(
            str(series_item.get("description", ""))
            for series_item in series
            if isinstance(series_item, dict)
        )
        searchable_text = " ".join(
            [
                str(item.get("code", "")),
                str(item.get("description", "")),
                str(item.get("goal", "")),
                str(item.get("target", "")),
                series_text,
            ]
        )
        score = _score_text(query, searchable_text)
        if score > 0:
            candidates.append((score, str(item.get("code", "")), item))

    candidates.sort(key=lambda row: (-row[0], row[1]))
    records = []
    for _, _, item in candidates[:limit]:
        code = str(item.get("code") or "").strip()
        description = str(item.get("description") or "Indicador ODS").strip()
        series = item.get("series") if isinstance(item.get("series"), list) else []
        series_summary = "; ".join(
            f"{series_item.get('code')}: {series_item.get('description')}"
            for series_item in series[:3]
            if isinstance(series_item, dict)
        )
        first_release = next(
            (
                str(series_item.get("release"))
                for series_item in series
                if isinstance(series_item, dict) and series_item.get("release")
            ),
            "",
        )
        records.append(
            OpenDataRecord(
                origem="ONU/UNSD SDG API",
                titulo=f"ODS {code}: {description}",
                descricao=_shorten(
                    f"Indicador oficial dos Objetivos de Desenvolvimento Sustentavel. "
                    f"Series associadas: {series_summary or 'nao informadas no retorno resumido'}."
                ),
                data=first_release,
                tipo="indicador-onu-ods",
                url=UN_SDG_INDICATORS_URL,
            )
        )

    status = PortalSearchStatus(
        portal="ONU/UNSD SDG API",
        ok=bool(records),
        mensagem=(
            f"{len(records)} indicador(es) ODS recuperado(s)."
            if records
            else "Nenhum indicador ODS encontrado para os termos da pauta."
        ),
        url=UN_SDG_INDICATORS_URL,
    )
    return records, status


def _search_dados_gov(query: str, limit: int) -> tuple[list[OpenDataRecord], PortalSearchStatus]:
    params = {
        "nomeConjuntoDados": query,
        "dadosAbertos": "true",
        "isPrivado": "false",
        "pagina": 1,
    }
    url = f"{DADOS_GOV_DATASETS_URL}?{urlencode(params)}"
    _validate_url(url)

    token = os.getenv("DADOS_GOV_BR_TOKEN")
    if not token:
        return [], PortalSearchStatus(
            portal="Portal Brasileiro de Dados Abertos",
            ok=False,
            mensagem=(
                "Endpoint de catálogo exige autenticação neste ambiente; defina "
                "DADOS_GOV_BR_TOKEN para habilitar esta busca."
            ),
            url=DADOS_GOV_DOCS_URL,
        )

    headers = {"Authorization": f"Bearer {token}"}
    response = _session().get(DADOS_GOV_DATASETS_URL, params=params, headers=headers, timeout=20)
    if response.status_code == 401:
        return [], PortalSearchStatus(
            portal="Portal Brasileiro de Dados Abertos",
            ok=False,
            mensagem="Token ausente, expirado ou sem permissão para consultar o catálogo.",
            url=DADOS_GOV_DOCS_URL,
        )
    response.raise_for_status()

    payload = response.json()
    raw_items = _extract_dados_gov_items(payload)
    records = []
    for item in raw_items[:limit]:
        dataset_id = str(item.get("id") or item.get("nome") or item.get("name") or "")
        detail_url = f"{DADOS_GOV_DATASETS_URL}/{dataset_id}" if dataset_id else DADOS_GOV_DOCS_URL
        records.append(
            OpenDataRecord(
                origem="Portal Brasileiro de Dados Abertos",
                titulo=str(item.get("titulo") or item.get("title") or item.get("nome") or "Conjunto de dados"),
                descricao=str(item.get("descricao") or item.get("description") or "Sem descrição informada."),
                data=str(item.get("dataAtualizacao") or item.get("modified") or ""),
                tipo="conjunto-dados",
                url=detail_url,
            )
        )

    status = PortalSearchStatus(
        portal="Portal Brasileiro de Dados Abertos",
        ok=True,
        mensagem=f"{len(records)} conjunto(s) de dados recuperado(s).",
        url=url,
    )
    return records, status


def _extract_dados_gov_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("content", "dados", "items", "resultados", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    nested = payload.get("result")
    if isinstance(nested, dict):
        return _extract_dados_gov_items(nested)
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, dict)]

    return []


def build_open_data_discovery_dataset(query: str, limit: int = 5) -> OpenDataDiscoveryDataset:
    statuses: list[PortalSearchStatus] = []
    records: list[OpenDataRecord] = []

    provider_fallback_urls = {
        _search_who_gho: WHO_GHO_DOCS_URL,
        _search_un_sdg: UN_SDG_DOCS_URL,
        _search_dados_gov: DADOS_GOV_DOCS_URL,
        _search_camara: CAMARA_PROPOSITIONS_URL,
    }

    for search_fn in (_search_who_gho, _search_un_sdg, _search_dados_gov, _search_camara):
        try:
            found_records, status = search_fn(query, limit)
        except Exception as exc:
            status = PortalSearchStatus(
                portal=search_fn.__name__.replace("_search_", ""),
                ok=False,
                mensagem=f"Erro ao consultar portal: {exc}",
                url=provider_fallback_urls[search_fn],
            )
            found_records = []
        statuses.append(status)
        records.extend(found_records)

    records = _take_diverse_records(records, limit)
    source_urls = list(dict.fromkeys([row.url for row in records if row.url]))
    if not source_urls:
        source_urls = [status.url for status in statuses if status.url]

    allowed_source_urls = [url for url in source_urls if is_allowed_source_url(url)]
    blocked_source_urls = [url for url in source_urls if not is_allowed_source_url(url)]
    for blocked_url in blocked_source_urls:
        statuses.append(
            PortalSearchStatus(
                portal="Validação de domínio",
                ok=False,
                mensagem=f"Fonte descartada por domínio não permitido: {blocked_url}",
                url=blocked_url,
            )
        )

    source_check = validate_source_urls(allowed_source_urls)
    if not source_check.ok or not allowed_source_urls:
        statuses.append(
            PortalSearchStatus(
                portal="Validação de domínio",
                ok=False,
                mensagem="Nenhuma URL verificável passou pela lista de domínios permitidos.",
                url=DADOS_GOV_DOCS_URL,
            )
        )
        allowed_source_urls = [DADOS_GOV_DOCS_URL]

    successful_portals = [status.portal for status in statuses if status.ok]
    collected_at = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S %Z")

    return OpenDataDiscoveryDataset(
        tipo="open_data_discovery",
        consulta=query,
        fonte_nome="Descoberta em portais públicos",
        registros_encontrados=len(records),
        portais_consultados=statuses,
        registros=records,
        fontes=allowed_source_urls,
        resumo_numerico=(
            f"A busca por '{query}' retornou {len(records)} registro(s) validado(s) "
            f"em {len(successful_portals)} portal(is) público(s): "
            f"{', '.join(successful_portals) if successful_portals else 'nenhum portal com retorno válido'}."
        ),
        nota_metodologica=(
            "Este coletor realiza descoberta de dados abertos e metadados em portais públicos. "
            "Ele consulta APIs internacionais da OMS e da ONU/UNSD, além de portais nacionais. "
            "Ele não deve transformar metadados em estatísticas factuais sem um coletor estruturado "
            "para baixar, parsear e validar a base numérica original."
        ),
        coletado_em=collected_at,
    )
