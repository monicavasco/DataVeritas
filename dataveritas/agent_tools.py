from __future__ import annotations

import json
import io
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from crewai.tools import BaseTool
from pydantic import PrivateAttr

from dataveritas.collectors import collector_definitions
from dataveritas.guardrails import REQUIRED_DISCLAIMER, extract_urls, validate_source_urls
from dataveritas.open_data import build_open_data_discovery_dataset


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, sort_keys=True)


def _compact(value: Any, max_text: int = 800) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact(child, max_text=max_text) for key, child in value.items()}
    if isinstance(value, list):
        return [_compact(item, max_text=max_text) for item in value[:12]]
    if isinstance(value, str) and len(value) > max_text:
        return f"{value[: max_text - 3].rstrip()}..."
    return value


def _numeric_fields(value: Any, prefix: str = "", limit: int = 60) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []

    def visit(node: Any, path: str) -> None:
        if len(fields) >= limit:
            return
        if isinstance(node, bool):
            return
        if isinstance(node, int | float):
            fields.append({"campo": path, "valor": node})
            return
        if isinstance(node, dict):
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                visit(child, child_path)
                if len(fields) >= limit:
                    break
            return
        if isinstance(node, list):
            for index, child in enumerate(node[:8]):
                child_path = f"{path}.{index}" if path else str(index)
                visit(child, child_path)
                if len(fields) >= limit:
                    break

    visit(value, prefix)
    return fields


def _lookup_path(data: Any, field_path: str) -> tuple[bool, Any]:
    current = data
    for part in field_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


def _candidate_fields(data: dict, field_path: str) -> list[str]:
    normalized = field_path.casefold()
    candidates = [
        key
        for key in data
        if normalized in key.casefold() or key.casefold() in normalized
    ]
    return candidates[:12]


def _source_domains(source_urls: list[str]) -> list[str]:
    domains = []
    for url in source_urls:
        parsed = urlparse(url)
        if parsed.netloc:
            domains.append(parsed.netloc.lower())
    return sorted(set(domains))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _tabular_url_candidates(dataset: dict) -> list[str]:
    urls: list[str] = []
    for record in dataset.get("registros", []):
        if isinstance(record, dict):
            urls.append(str(record.get("url") or ""))
    for url in dataset.get("fontes", []):
        urls.append(str(url or ""))
    for status in dataset.get("portais_consultados", []):
        if isinstance(status, dict):
            urls.append(str(status.get("url") or ""))
    return _dedupe(urls)


def _find_rows(data: dict, preferred_path: str = "") -> tuple[str, list[dict[str, Any]]]:
    if preferred_path:
        found, value = _lookup_path(data, preferred_path)
        if found and isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return preferred_path, value

    preferred_keys = ("registros", "observacoes", "ranking_2024", "top_municipios")
    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return key, value

    for key, value in data.items():
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
            return str(key), value

    return "", []


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        source_rows = payload
    elif isinstance(payload, dict):
        source_rows = []
        for key in ("value", "dados", "result", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                source_rows = value
                break
        if not source_rows:
            source_rows = [payload]
    else:
        source_rows = [{"valor": payload}]

    rows: list[dict[str, Any]] = []
    for row in source_rows:
        rows.append(row if isinstance(row, dict) else {"valor": row})
    return rows


def _frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def _dataframe_sample(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sample = frame.head(limit).where(pd.notna(frame.head(limit)), None)
    return sample.to_dict(orient="records")


def _profile_frame(frame: pd.DataFrame, table_path: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "tabela": table_path,
        "linhas": int(len(frame)),
        "colunas": [str(column) for column in frame.columns[:40]],
        "total_colunas": int(len(frame.columns)),
        "amostra": _compact(_dataframe_sample(frame), max_text=300),
    }
    if frame.empty:
        return profile

    numeric_stats: list[dict[str, Any]] = []
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    for column in numeric_frame.columns:
        series = numeric_frame[column].dropna()
        if series.empty:
            continue
        numeric_stats.append(
            {
                "coluna": str(column),
                "contagem": int(series.count()),
                "minimo": round(float(series.min()), 4),
                "maximo": round(float(series.max()), 4),
                "media": round(float(series.mean()), 4),
                "soma": round(float(series.sum()), 4),
                "ausentes": int(frame[column].isna().sum()),
            }
        )
        if len(numeric_stats) >= 12:
            break

    profile["colunas_numericas"] = numeric_stats
    profile["total_colunas_numericas"] = int(sum(not numeric_frame[column].dropna().empty for column in numeric_frame.columns))
    return profile


def _http_get_limited(url: str, max_bytes: int = 5_000_000) -> tuple[bytes, str]:
    source_check = validate_source_urls([url])
    if not source_check.ok:
        raise ValueError("; ".join(source_check.messages))

    response = requests.get(
        url,
        headers={"Accept": "*/*", "User-Agent": "DataVeritas didactic prototype"},
        stream=True,
        timeout=30,
    )
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise ValueError(f"Recurso maior que o limite de {max_bytes} bytes.")

    content = bytearray()
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        content.extend(chunk)
        if len(content) > max_bytes:
            raise ValueError(f"Recurso maior que o limite de {max_bytes} bytes.")

    return bytes(content), response.headers.get("content-type", "")


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", nrows=1000)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="latin1", nrows=1000)


def _frame_from_remote_resource(url: str) -> tuple[pd.DataFrame, str]:
    content, content_type = _http_get_limited(url)
    lowered_url = urlparse(url).path.lower()
    lowered_type = content_type.lower()

    if lowered_url.endswith(".zip") or "zip" in lowered_type:
        with zipfile.ZipFile(io.BytesIO(content)) as zip_file:
            csv_names = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("ZIP nao contem CSV tabular.")
            with zip_file.open(csv_names[0]) as csv_file:
                frame = pd.read_csv(csv_file, sep=None, engine="python", nrows=1000)
        return frame, "zip_csv"

    if lowered_url.endswith((".xlsx", ".xls")) or "spreadsheet" in lowered_type or "excel" in lowered_type:
        return pd.read_excel(io.BytesIO(content), nrows=1000), "excel"

    if lowered_url.endswith(".csv") or "csv" in lowered_type or "text/plain" in lowered_type:
        return _read_csv_bytes(content), "csv"

    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Formato remoto nao reconhecido como CSV, Excel, ZIP ou JSON.") from exc
    return _frame_from_rows(_json_rows(payload)), "json"


def _numbers_from_text(text: str) -> list[dict[str, Any]]:
    without_urls = re.sub(r"https?://[^\s)\]>\"']+", " ", text)
    pattern = re.compile(r"(?<![\w/])[-+]?\d+(?:[.\s]\d{3})*(?:[,.]\d+)?")
    numbers: list[dict[str, Any]] = []
    for match in pattern.finditer(without_urls):
        raw = match.group(0).strip()
        compact = raw.replace(" ", "")
        if "," in compact:
            normalized = compact.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})+", compact):
            normalized = compact.replace(".", "")
        else:
            normalized = compact
        try:
            value = float(normalized)
        except ValueError:
            continue
        if math.isfinite(value):
            numbers.append({"texto": raw, "valor": value})
    return numbers


def _known_dataset_numbers(dataset: dict) -> list[float]:
    serialized = json.dumps(dataset, ensure_ascii=False, default=str)
    values = [float(item["valor"]) for item in _numbers_from_text(serialized)]
    values.extend(float(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", serialized))
    return values


def _matching_numeric_fields(dataset: dict, value: float, limit: int = 8) -> list[dict[str, Any]]:
    matches = []
    if value.is_integer() and int(value) in _dataset_years(dataset):
        matches.append({"campo": "ano_detectado_no_pacote", "valor": int(value)})
    for field in _numeric_fields(dataset, limit=200):
        field_value = float(field["valor"])
        if _is_supported_number(value, [field_value]):
            matches.append(field)
        if len(matches) >= limit:
            break
    return matches


def _is_supported_number(value: float, known_values: list[float]) -> bool:
    for known in known_values:
        if abs(value - known) <= 0.01:
            return True
        denominator = max(abs(value), abs(known), 1.0)
        if abs(value - known) / denominator <= 0.001:
            return True
    return False


def _normalize_check_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.casefold()


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", compact) if sentence.strip()]


def _has_negation_context(sentence: str) -> bool:
    normalized = _normalize_check_text(sentence)
    negation_markers = (
        "nao",
        "sem",
        "nao permite",
        "nao e possivel",
        "nao deve",
        "nao indica",
        "nao prova",
        "nao comprova",
        "evita inferir",
        "sem evidenciar",
        "sem evidencia",
    )
    return any(marker in normalized for marker in negation_markers)


def _article_years(text: str) -> set[int]:
    return {int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", str(text or ""))}


def _dataset_years(dataset: dict) -> set[int]:
    serialized = json.dumps(dataset, ensure_ascii=False, default=str)
    return _article_years(serialized)


def _dataset_reference_year(dataset: dict) -> int | None:
    years = _dataset_years(dataset)
    if years:
        return max(years)
    return None


def _field_text(dataset: dict, *field_names: str) -> str:
    return " ".join(str(dataset.get(field_name, "") or "") for field_name in field_names)


def _numeric_value(data: dict, *field_paths: str) -> float | None:
    for field_path in field_paths:
        found, value = _lookup_path(data, field_path)
        if not found or isinstance(value, bool) or value in (None, ""):
            continue
        try:
            parsed = float(str(value).replace(".", "").replace(",", ".")) if isinstance(value, str) else float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _auto_variation_fields(dataset: dict) -> tuple[str, str]:
    dataset_type = dataset.get("tipo")
    if dataset_type == "population_ibge":
        return "populacao_inicial", "populacao_final"
    if dataset_type == "selic_bcb":
        return "valor_inicial", "valor_final"
    return "valor_inicial", "valor_final"


def _auto_rate_fields(dataset: dict) -> tuple[str, str, float, str]:
    dataset_type = dataset.get("tipo")
    if dataset_type == "mortality_rj":
        return "obitos_residentes", "populacao_residente", 1000.0, "taxa bruta por mil habitantes"
    return "numerador", "denominador", 100.0, "taxa percentual"


def _auto_ranking_fields(dataset: dict) -> tuple[str, str]:
    dataset_type = dataset.get("tipo")
    if dataset_type == "population_ibge":
        return "ranking_2024", "populacao_2024"
    if dataset_type == "mortality_rj":
        return "top_municipios", "obitos_residentes"
    if dataset_type == "open_data_discovery":
        return "registros", "data"
    return "observacoes", "valor"


def _public_source_quality(dataset: dict, catalog_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    source_urls = [str(url) for url in dataset.get("fontes", []) if url]
    source_check = validate_source_urls(source_urls)
    current_source_name = str(dataset.get("fonte_nome", "") or dataset.get("source_label", ""))
    structured = dataset.get("tipo") not in {"open_data_discovery", None}
    cataloged = bool(catalog_candidates)
    official_domain_count = len(_source_domains(source_urls))
    score = 0
    reasons = []

    if source_check.ok and source_urls:
        score += 30
        reasons.append("URLs passam pela lista de dominios oficiais permitidos.")
    if structured:
        score += 25
        reasons.append("Pacote veio de coletor estruturado com campos factuais.")
    else:
        reasons.append("Pacote atual e descoberta de metadados; valores numericos exigem coleta estruturada.")
    if cataloged:
        score += 20
        reasons.append("Tema possui fonte candidata no catalogo RAG/Chroma.")
    if official_domain_count:
        score += min(official_domain_count * 5, 15)
        reasons.append(f"{official_domain_count} dominio(s) oficial(is) identificado(s).")
    if dataset.get("nota_metodologica") or dataset.get("nota_escopo"):
        score += 10
        reasons.append("Pacote inclui nota metodologica ou de escopo.")

    if score >= 80:
        rating = "alta"
    elif score >= 55:
        rating = "media"
    else:
        rating = "baixa"

    return {
        "score": min(score, 100),
        "classificacao": rating,
        "fonte_atual": current_source_name,
        "dominios": _source_domains(source_urls),
        "fonte_catalogada": cataloged,
        "coletor_estruturado": structured,
        "motivos": reasons,
        "mensagens_validacao": source_check.messages,
    }


def _classify_topic(query: str) -> dict[str, Any]:
    normalized = _normalize_check_text(query)
    topic_rules = {
        "eleicoes": ("eleicao", "eleicoes", "tse", "votos", "candidatos", "abstencao"),
        "educacao": ("educacao", "inep", "enem", "ideb", "escola", "matricula", "pisa"),
        "saude": ("saude", "sus", "datasus", "mortalidade", "obitos", "vacinacao", "internacoes"),
        "economia": ("selic", "juros", "bcb", "ipea", "ipeadata", "renda", "pib", "inflacao"),
        "transparencia": ("transparencia", "gastos", "despesas", "orcamento", "contratos", "emendas", "cgu"),
        "legislativo": ("camara", "senado", "legislacao", "projeto de lei", "votacao", "proposicao"),
        "demografia": ("populacao", "habitantes", "ibge", "sidra", "censo", "pnad"),
        "internacional": ("onu", "oms", "who", "world bank", "banco mundial", "ocde", "oecd", "sdg", "ods"),
    }
    scores = {}
    for topic, terms in topic_rules.items():
        scores[topic] = sum(1 for term in terms if term in normalized)
    best_topic, best_score = max(scores.items(), key=lambda item: item[1])
    return {
        "tipo_pauta": best_topic if best_score else "geral",
        "confianca": min(round(best_score / 3, 2), 1.0),
        "pontuacoes": scores,
    }


def _best_catalog_candidate(catalog_output: str) -> dict[str, Any] | None:
    try:
        data = json.loads(catalog_output)
    except json.JSONDecodeError:
        return None
    candidates = data.get("fontes_candidatas", [])
    if not candidates:
        return None
    return candidates[0]


class DatasetProfileTool(BaseTool):
    name: str = "perfil_pacote_dados"
    description: str = (
        "Resume o pacote de dados ja coletado e aprovado para a pauta atual. "
        "Use para confirmar tipo, periodo, escopo, fontes, resumo numerico e limitacoes."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=3, **kwargs)
        self._dataset = dataset

    def _run(self) -> str:
        numeric_fields = _numeric_fields(self._dataset)
        profile = {
            "tipo": self._dataset.get("tipo"),
            "fonte_nome": self._dataset.get("fonte_nome"),
            "escopo_geografico": self._dataset.get("escopo_geografico"),
            "periodo_inicial": self._dataset.get("periodo_inicial"),
            "periodo_final": self._dataset.get("periodo_final"),
            "coletado_em": self._dataset.get("coletado_em"),
            "resumo_numerico": self._dataset.get("resumo_numerico"),
            "nota_metodologica": self._dataset.get("nota_metodologica"),
            "nota_escopo": self._dataset.get("nota_escopo"),
            "fontes": self._dataset.get("fontes", []),
            "campos_numericos": numeric_fields,
        }
        return _json(profile)


class DatasetFieldLookupTool(BaseTool):
    name: str = "consultar_campo_pacote"
    description: str = (
        "Consulta um campo factual do pacote de dados por caminho com ponto, por exemplo "
        "'taxa_mortalidade_por_mil', 'resumo_numerico' ou 'top_municipios.0.municipio'. "
        "Use para verificar numeros antes de escrever uma afirmacao."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=8, **kwargs)
        self._dataset = dataset

    def _run(self, field_path: str) -> str:
        found, value = _lookup_path(self._dataset, field_path.strip())
        if found:
            return _json({"encontrado": True, "campo": field_path, "valor": _compact(value)})

        return _json(
            {
                "encontrado": False,
                "campo": field_path,
                "campos_semelhantes": _candidate_fields(self._dataset, field_path),
                "campos_topo_disponiveis": list(self._dataset.keys()),
            }
        )


class SourcePolicyAuditTool(BaseTool):
    name: str = "auditar_fontes_publicas"
    description: str = (
        "Valida as URLs do pacote contra a politica local de dominios permitidos. "
        "Use antes de afirmar que as fontes sao publicas e citaveis."
    )
    _source_urls: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, source_urls: list[str], **kwargs: Any) -> None:
        super().__init__(max_usage_count=3, **kwargs)
        self._source_urls = source_urls

    def _run(self) -> str:
        result = validate_source_urls(self._source_urls)
        return _json(
            {
                "ok": result.ok,
                "mensagens": result.messages,
                "fontes": self._source_urls,
                "dominios": _source_domains(self._source_urls),
            }
        )


class RagContextTool(BaseTool):
    name: str = "ver_contexto_rag"
    description: str = (
        "Mostra a recomendacao RAG usada para escolher o coletor da pauta atual, "
        "incluindo fonte sugerida, similaridade e documentos recuperados."
    )
    _rag_context: dict | None = PrivateAttr(default=None)

    def __init__(self, rag_context: dict | None, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._rag_context = rag_context

    def _run(self) -> str:
        if not self._rag_context:
            return _json(
                {
                    "usado": False,
                    "mensagem": "RAG automatico nao foi usado; a fonte foi escolhida manualmente.",
                }
            )
        return _json({"usado": True, "rag": _compact(self._rag_context, max_text=500)})


class CollectorRegistryTool(BaseTool):
    name: str = "listar_coletores_configurados"
    description: str = (
        "Lista os coletores configurados, seus temas, limites metodologicos e se sao "
        "coletores estruturados ou apenas descoberta de fontes."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)

    def _run(self) -> str:
        collectors = []
        for definition in collector_definitions():
            collectors.append(
                {
                    "source_type": definition.source_type,
                    "source_label": definition.source_label,
                    "collector": definition.collector_name,
                    "input_kind": definition.input_kind,
                    "structured": definition.structured,
                    "limitations": definition.limitations,
                    "documents": [
                        {
                            "id": document.id,
                            "domains": list(document.domains),
                            "themes": list(document.themes),
                        }
                        for document in definition.source_documents
                    ],
                }
            )
        return _json({"coletores": collectors})


class SourceCatalogLookupTool(BaseTool):
    name: str = "consultar_catalogo_fontes"
    description: str = (
        "Consulta o catalogo RAG/Chroma de fontes publicas conhecidas para a pauta atual. "
        "Retorna fontes candidatas, temas, dominios, documentacao, status de coletor e "
        "limitacoes metodologicas. Use para ampliar a area de conhecimento do Coletor "
        "sem inventar fontes."
    )
    _default_query: str = PrivateAttr(default="")

    def __init__(self, default_query: str | None = None, **kwargs: Any) -> None:
        super().__init__(max_usage_count=3, **kwargs)
        self._default_query = default_query or ""

    def _run(self, query: str = "", limit: int = 5) -> str:
        effective_query = (query or self._default_query).strip()
        if not effective_query:
            return _json(
                {
                    "ok": False,
                    "mensagem": "Nenhuma pauta foi informada para consultar o catalogo de fontes.",
                }
            )

        try:
            safe_limit = min(max(int(limit), 1), 10)
        except (TypeError, ValueError):
            safe_limit = 5

        from dataveritas.rag import search_source_catalog

        matches = search_source_catalog(effective_query, limit=safe_limit)
        return _json(
            {
                "ok": bool(matches),
                "consulta": effective_query,
                "fontes_candidatas": [
                    {
                        "id": match.id,
                        "score": match.score,
                        "nome": match.metadata.get("source_label"),
                        "organizacao": match.metadata.get("organization"),
                        "dominios": match.metadata.get("domain"),
                        "temas": match.metadata.get("themes", []),
                        "api_url": match.metadata.get("api_url"),
                        "docs_url": match.metadata.get("docs_url"),
                        "cobertura": match.metadata.get("coverage"),
                        "atualizacao": match.metadata.get("update_frequency"),
                        "status_coletor": match.metadata.get("collector_status"),
                        "limitacoes": match.metadata.get("limitations"),
                        "coletor_recomendado": match.metadata.get("collector"),
                    }
                    for match in matches
                ],
                "mensagem": (
                    "Estas fontes sao candidatas do catalogo. Use como orientacao de coleta; "
                    "numeros factuais ainda precisam vir de API/base validada."
                ),
            }
        )


class SourceQualityAssessmentTool(BaseTool):
    name: str = "avaliar_qualidade_fonte"
    description: str = (
        "Avalia a qualidade da fonte atual combinando dominio permitido, coletor estruturado, "
        "catalogo de fontes, quantidade de dominios oficiais e notas metodologicas."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)
    _default_query: str = PrivateAttr(default="")

    def __init__(self, dataset: dict, default_query: str | None = None, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset
        self._default_query = default_query or ""

    def _run(self, query: str = "") -> str:
        from dataveritas.rag import search_source_catalog

        effective_query = (query or self._default_query or self._dataset.get("consulta") or "").strip()
        catalog_candidates = [
            {
                "nome": match.metadata.get("source_label"),
                "score": match.score,
                "dominios": match.metadata.get("domain"),
                "coletor": match.metadata.get("collector"),
                "limitacoes": match.metadata.get("limitations"),
            }
            for match in search_source_catalog(effective_query, limit=3)
        ] if effective_query else []
        quality = _public_source_quality(self._dataset, catalog_candidates)
        return _json(
            {
                "ok": quality["classificacao"] != "baixa",
                "consulta": effective_query,
                "qualidade": quality,
                "fontes_candidatas_catalogo": catalog_candidates,
            }
        )


class SourceFreshnessCheckTool(BaseTool):
    name: str = "verificar_atualizacao_fonte"
    description: str = (
        "Verifica sinais de atualizacao temporal no pacote e no catalogo: coletado_em, "
        "periodo final, data de referencia e frequencia de atualizacao esperada."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)
    _default_query: str = PrivateAttr(default="")

    def __init__(self, dataset: dict, default_query: str | None = None, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset
        self._default_query = default_query or ""

    def _run(self, query: str = "") -> str:
        from dataveritas.rag import search_source_catalog

        effective_query = (query or self._default_query or self._dataset.get("consulta") or "").strip()
        catalog_matches = search_source_catalog(effective_query, limit=1) if effective_query else []
        catalog_frequency = catalog_matches[0].metadata.get("update_frequency") if catalog_matches else None
        temporal_fields = {
            "coletado_em": self._dataset.get("coletado_em"),
            "data_referencia": self._dataset.get("data_referencia"),
            "periodo_inicial": self._dataset.get("periodo_inicial"),
            "periodo_final": self._dataset.get("periodo_final"),
            "ano": self._dataset.get("ano"),
        }
        known_years = sorted(_dataset_years(self._dataset))
        warnings = []
        if not any(temporal_fields.values()) and not known_years:
            warnings.append("Pacote nao declara campos temporais claros.")
        if self._dataset.get("tipo") == "open_data_discovery":
            warnings.append("Descoberta de metadados pode nao refletir a atualizacao da base numerica original.")
        return _json(
            {
                "ok": not warnings,
                "consulta": effective_query,
                "campos_temporais": temporal_fields,
                "anos_detectados": known_years,
                "frequencia_catalogo": catalog_frequency,
                "avisos": warnings,
            }
        )


class TopicClassificationTool(BaseTool):
    name: str = "classificar_tipo_de_pauta"
    description: str = (
        "Classifica a pauta em area tematica provavel para orientar a escolha de fonte "
        "publica: saude, educacao, eleicoes, economia, transparencia, legislativo, "
        "demografia, internacional ou geral."
    )
    _default_query: str = PrivateAttr(default="")

    def __init__(self, default_query: str | None = None, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._default_query = default_query or ""

    def _run(self, query: str = "") -> str:
        effective_query = (query or self._default_query).strip()
        classification = _classify_topic(effective_query)
        return _json({"ok": True, "consulta": effective_query, **classification})


class PrimarySourceSelectionTool(BaseTool):
    name: str = "selecionar_fonte_primaria"
    description: str = (
        "Seleciona a fonte primaria mais adequada entre o pacote atual e o catalogo RAG, "
        "explicando motivo, limitacao e status do coletor."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)
    _default_query: str = PrivateAttr(default="")

    def __init__(self, dataset: dict, default_query: str | None = None, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset
        self._default_query = default_query or ""

    def _run(self, query: str = "") -> str:
        from dataveritas.rag import search_source_catalog

        effective_query = (query or self._default_query or self._dataset.get("consulta") or "").strip()
        catalog_matches = search_source_catalog(effective_query, limit=3) if effective_query else []
        primary_candidate = catalog_matches[0] if catalog_matches else None
        dataset_type = self._dataset.get("tipo")
        structured = dataset_type not in {"open_data_discovery", None}

        if structured:
            selected = {
                "nome": self._dataset.get("fonte_nome"),
                "origem": "pacote_atual",
                "coletor": "coletor estruturado ja executado",
                "status": "fonte primaria operacional nesta execucao",
                "motivo": "O pacote atual ja contem dados estruturados de fonte oficial.",
                "limitacao": self._dataset.get("nota_metodologica") or self._dataset.get("nota_escopo"),
            }
        elif primary_candidate is not None:
            selected = {
                "nome": primary_candidate.metadata.get("source_label"),
                "origem": "catalogo_rag",
                "coletor": primary_candidate.metadata.get("collector"),
                "status": primary_candidate.metadata.get("collector_status"),
                "motivo": "Fonte catalogada tem maior similaridade com a pauta.",
                "limitacao": primary_candidate.metadata.get("limitations"),
                "api_url": primary_candidate.metadata.get("api_url"),
                "docs_url": primary_candidate.metadata.get("docs_url"),
                "score": primary_candidate.score,
            }
        else:
            selected = {
                "nome": self._dataset.get("fonte_nome") or "Fonte publica nao determinada",
                "origem": "pacote_atual",
                "coletor": "descoberta em portais publicos",
                "status": "sem fonte primaria catalogada",
                "motivo": "Nao houve candidato claro no catalogo; manter cautela.",
                "limitacao": "E necessario localizar base primaria estruturada antes de conclusoes numericas.",
            }

        return _json(
            {
                "ok": True,
                "consulta": effective_query,
                "fonte_escolhida": selected,
                "candidatas": [
                    {
                        "nome": match.metadata.get("source_label"),
                        "score": match.score,
                        "dominios": match.metadata.get("domain"),
                        "status": match.metadata.get("collector_status"),
                    }
                    for match in catalog_matches
                ],
            }
        )


class SecondarySourceDetectionTool(BaseTool):
    name: str = "detectar_fonte_secundaria_ou_inadequada"
    description: str = (
        "Sinaliza se o pacote atual parece ser apenas descoberta de metadados, fonte "
        "secundaria ou base insuficiente para afirmacoes numericas fortes."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self) -> str:
        dataset_type = self._dataset.get("tipo")
        issues = []
        if dataset_type == "open_data_discovery":
            issues.append(
                "Pacote e descoberta de metadados; nao e base numerica estruturada para conclusoes factuais."
            )
        if not self._dataset.get("fontes"):
            issues.append("Pacote nao contem URLs de fontes citaveis.")
        if not self._dataset.get("resumo_numerico"):
            issues.append("Pacote nao contem resumo numerico.")
        if dataset_type == "open_data_discovery" and self._dataset.get("registros_encontrados", 0) == 0:
            issues.append("Descoberta nao encontrou registros, apenas status/portais consultados.")

        return _json(
            {
                "ok": not issues,
                "tipo_dataset": dataset_type,
                "fonte_nome": self._dataset.get("fonte_nome"),
                "riscos": issues,
                "orientacao": (
                    "Se houver riscos, trate a fonte como candidata/metadado e evite numeros "
                    "ou conclusoes ate haver coletor estruturado."
                ),
            }
        )


class PublicPortalSearchTool(BaseTool):
    name: str = "buscar_portais_publicos"
    description: str = (
        "Busca ou reaproveita resultados de descoberta em portais publicos oficiais "
        "para a pauta atual, incluindo dados.gov.br, Camara dos Deputados, OMS/WHO "
        "GHO API e ONU/UNSD SDG API. Use apenas quando a pauta exigir descoberta "
        "de fontes abertas, nao para substituir coletores estruturados ja escolhidos."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)
    _default_query: str = PrivateAttr(default="")

    def __init__(self, dataset: dict, default_query: str | None = None, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset
        self._default_query = default_query or ""

    def _run(self, query: str = "", limit: int = 5) -> str:
        effective_query = (query or self._default_query or self._dataset.get("consulta") or "").strip()
        if not effective_query:
            return _json(
                {
                    "ok": False,
                    "mensagem": "Nenhuma pauta/consulta foi informada para buscar em portais publicos.",
                }
            )

        try:
            safe_limit = min(max(int(limit), 1), 10)
        except (TypeError, ValueError):
            safe_limit = 5

        if self._dataset.get("tipo") == "open_data_discovery":
            return _json(
                {
                    "ok": True,
                    "consulta": effective_query,
                    "limite": safe_limit,
                    "reutilizou_pacote_atual": True,
                    "mensagem": (
                        "A descoberta em portais publicos ja foi executada pelo coletor "
                        "open_data_discovery antes da etapa dos agentes."
                    ),
                    "resultado": _compact(self._dataset, max_text=600),
                }
            )

        discovery = build_open_data_discovery_dataset(effective_query, limit=safe_limit)
        return _json(
            {
                "ok": True,
                "consulta": effective_query,
                "limite": safe_limit,
                "reutilizou_pacote_atual": False,
                "resultado": _compact(discovery.to_prompt_dict(), max_text=600),
            }
        )


class TabularResourceDownloadTool(BaseTool):
    name: str = "baixar_recurso_tabular"
    description: str = (
        "Tenta baixar e materializar um recurso tabular oficial relacionado ao pacote "
        "atual. Suporta JSON, CSV, Excel e ZIP com CSV, respeitando a lista local de "
        "dominios permitidos. Se nao houver recurso remoto tabular direto, reutiliza "
        "a tabela de metadados ja presente no pacote."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, resource_url: str = "") -> str:
        attempts: list[dict[str, Any]] = []
        candidate_urls = [resource_url] if resource_url else _tabular_url_candidates(self._dataset)

        for url in _dedupe(candidate_urls):
            try:
                frame, detected_format = _frame_from_remote_resource(url)
            except Exception as exc:
                attempts.append({"url": url, "ok": False, "erro": str(exc)})
                continue

            if frame.empty:
                attempts.append({"url": url, "ok": False, "erro": "Recurso tabular vazio."})
                continue

            return _json(
                {
                    "ok": True,
                    "baixado": True,
                    "url": url,
                    "formato_detectado": detected_format,
                    "perfil": _profile_frame(frame, table_path=url),
                    "tentativas": attempts,
                }
            )

        table_path, rows = _find_rows(self._dataset)
        frame = _frame_from_rows(rows)
        if not frame.empty:
            return _json(
                {
                    "ok": True,
                    "baixado": False,
                    "reutilizou_pacote_atual": True,
                    "tabela_origem": table_path,
                    "mensagem": (
                        "Nenhum recurso remoto tabular direto foi materializado; "
                        "a tool usou a tabela ja presente no pacote coletado."
                    ),
                    "perfil": _profile_frame(frame, table_path=table_path),
                    "tentativas": attempts,
                }
            )

        return _json(
            {
                "ok": False,
                "mensagem": "Nao foi possivel baixar nem materializar uma tabela para este pacote.",
                "tentativas": attempts,
                "urls_candidatas": _tabular_url_candidates(self._dataset),
            }
        )


class DatasetStatisticalProfileTool(BaseTool):
    name: str = "perfil_estatistico_dataset"
    description: str = (
        "Gera um perfil estatistico deterministico da tabela principal do pacote: "
        "linhas, colunas, amostra e estatisticas de colunas numericas. Use para "
        "identificar maximos, minimos, medias e totais sem depender da LLM."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=4, **kwargs)
        self._dataset = dataset

    def _run(self, table_path: str = "") -> str:
        resolved_path, rows = _find_rows(self._dataset, preferred_path=table_path)
        frame = _frame_from_rows(rows)
        if not frame.empty:
            return _json(
                {
                    "ok": True,
                    "modo": "tabela",
                    "perfil": _profile_frame(frame, table_path=resolved_path),
                }
            )

        numeric_fields = _numeric_fields(self._dataset)
        return _json(
            {
                "ok": bool(numeric_fields),
                "modo": "campos_numericos_do_pacote",
                "mensagem": (
                    "Nenhuma lista tabular de dicionarios foi encontrada; "
                    "a tool retornou campos numericos recursivos do pacote."
                ),
                "campos_numericos": numeric_fields,
            }
        )


class PercentageVariationTool(BaseTool):
    name: str = "calcular_variacao_percentual"
    description: str = (
        "Calcula variacao absoluta e percentual entre dois campos numericos do pacote. "
        "Se os campos nao forem informados, tenta inferir campos centrais do dataset."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=4, **kwargs)
        self._dataset = dataset

    def _run(self, initial_field: str = "", final_field: str = "") -> str:
        if not initial_field or not final_field:
            initial_field, final_field = _auto_variation_fields(self._dataset)
        initial = _numeric_value(self._dataset, initial_field)
        final = _numeric_value(self._dataset, final_field)
        if initial is None or final is None:
            return _json(
                {
                    "ok": False,
                    "campos_esperados": [initial_field, final_field],
                    "mensagem": "Nao foi possivel localizar os dois campos numericos para variacao.",
                }
            )
        absolute = final - initial
        percent = (absolute / initial) * 100 if initial else None
        return _json(
            {
                "ok": percent is not None,
                "campo_inicial": initial_field,
                "campo_final": final_field,
                "valor_inicial": initial,
                "valor_final": final,
                "variacao_absoluta": round(absolute, 4),
                "variacao_percentual": round(percent, 4) if percent is not None else None,
                "formula": "(valor_final - valor_inicial) / valor_inicial * 100",
            }
        )


class RateCalculationTool(BaseTool):
    name: str = "calcular_taxa"
    description: str = (
        "Calcula taxa deterministica como numerador dividido pelo denominador, multiplicado "
        "por um fator. Por padrao, mortalidade usa obitos/populacao*1000."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=4, **kwargs)
        self._dataset = dataset

    def _run(
        self,
        numerator_field: str = "",
        denominator_field: str = "",
        multiplier: float | int | str = "",
    ) -> str:
        default_numerator, default_denominator, default_multiplier, label = _auto_rate_fields(self._dataset)
        numerator_field = numerator_field or default_numerator
        denominator_field = denominator_field or default_denominator
        try:
            factor = float(multiplier) if multiplier not in ("", None) else default_multiplier
        except (TypeError, ValueError):
            factor = default_multiplier
        numerator = _numeric_value(self._dataset, numerator_field)
        denominator = _numeric_value(self._dataset, denominator_field)
        if numerator is None or denominator is None or denominator == 0:
            return _json(
                {
                    "ok": False,
                    "campos_esperados": [numerator_field, denominator_field],
                    "multiplicador": factor,
                    "mensagem": "Nao foi possivel calcular taxa por falta de numerador/denominador ou denominador zero.",
                }
            )
        rate = (numerator / denominator) * factor
        return _json(
            {
                "ok": True,
                "rotulo": label,
                "numerador_campo": numerator_field,
                "denominador_campo": denominator_field,
                "numerador": numerator,
                "denominador": denominator,
                "multiplicador": factor,
                "taxa": round(rate, 6),
                "formula": "numerador / denominador * multiplicador",
            }
        )


class PeriodComparisonTool(BaseTool):
    name: str = "comparar_periodos"
    description: str = (
        "Compara periodo inicial e final do pacote usando campos centrais ou campos "
        "informados, retornando diferenca absoluta e percentual."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=4, **kwargs)
        self._dataset = dataset

    def _run(self, initial_field: str = "", final_field: str = "") -> str:
        variation_tool = PercentageVariationTool(self._dataset)
        variation = json.loads(variation_tool._run(initial_field=initial_field, final_field=final_field))
        return _json(
            {
                "ok": variation.get("ok", False),
                "periodo_inicial": self._dataset.get("periodo_inicial"),
                "periodo_final": self._dataset.get("periodo_final"),
                "comparacao": variation,
            }
        )


class RankingGenerationTool(BaseTool):
    name: str = "gerar_ranking"
    description: str = (
        "Gera ranking deterministico a partir de uma lista tabular do pacote. "
        "Por padrao usa ranking_2024/populacao_2024, top_municipios/obitos_residentes "
        "ou observacoes/valor conforme o tipo do dataset."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=4, **kwargs)
        self._dataset = dataset

    def _run(self, table_path: str = "", value_field: str = "", descending: bool = True, limit: int = 10) -> str:
        if not table_path or not value_field:
            table_path, value_field = _auto_ranking_fields(self._dataset)
        found, rows = _lookup_path(self._dataset, table_path)
        if not found or not isinstance(rows, list):
            return _json(
                {
                    "ok": False,
                    "tabela": table_path,
                    "campo_valor": value_field,
                    "mensagem": "Tabela de ranking nao encontrada.",
                }
            )
        ranked = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = _numeric_value(row, value_field)
            if value is None:
                continue
            label = (
                row.get("nome")
                or row.get("municipio")
                or row.get("titulo")
                or row.get("data")
                or row.get("origem")
                or "item"
            )
            ranked.append({"item": label, "valor": value, "linha": _compact(row, max_text=250)})
        ranked.sort(key=lambda item: item["valor"], reverse=bool(descending))
        safe_limit = min(max(int(limit), 1), 50)
        return _json(
            {
                "ok": bool(ranked),
                "tabela": table_path,
                "campo_valor": value_field,
                "ordem": "decrescente" if descending else "crescente",
                "ranking": ranked[:safe_limit],
                "total_itens_ranqueados": len(ranked),
            }
        )


class OutlierDetectionTool(BaseTool):
    name: str = "detectar_outliers"
    description: str = (
        "Detecta outliers simples em uma coluna numerica usando escore z. "
        "Se tabela/campo nao forem informados, tenta inferir campos centrais do dataset."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=4, **kwargs)
        self._dataset = dataset

    def _run(self, table_path: str = "", value_field: str = "", z_threshold: float | int | str = 2.0) -> str:
        if not table_path or not value_field:
            table_path, value_field = _auto_ranking_fields(self._dataset)
        found, rows = _lookup_path(self._dataset, table_path)
        if not found or not isinstance(rows, list):
            return _json({"ok": False, "mensagem": "Tabela para outliers nao encontrada.", "tabela": table_path})
        values = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = _numeric_value(row, value_field)
            if value is None:
                continue
            label = row.get("nome") or row.get("municipio") or row.get("titulo") or row.get("data") or "item"
            values.append({"item": label, "valor": value})
        if len(values) < 3:
            return _json(
                {
                    "ok": True,
                    "outliers": [],
                    "mensagem": "Menos de 3 valores; outliers nao foram avaliados.",
                    "total_valores": len(values),
                }
            )
        try:
            threshold = float(z_threshold)
        except (TypeError, ValueError):
            threshold = 2.0
        mean = sum(item["valor"] for item in values) / len(values)
        variance = sum((item["valor"] - mean) ** 2 for item in values) / len(values)
        stddev = math.sqrt(variance)
        outliers = []
        if stddev:
            for item in values:
                z_score = (item["valor"] - mean) / stddev
                if abs(z_score) >= threshold:
                    outliers.append({**item, "z_score": round(z_score, 4)})
        return _json(
            {
                "ok": True,
                "tabela": table_path,
                "campo_valor": value_field,
                "media": round(mean, 4),
                "desvio_padrao": round(stddev, 4),
                "limiar_z": threshold,
                "outliers": outliers,
                "total_valores": len(values),
            }
        )


class IndicatorFormulaValidationTool(BaseTool):
    name: str = "validar_formula_indicador"
    description: str = (
        "Valida formulas centrais de indicadores do pacote, como variacao percentual "
        "ou taxa de mortalidade, comparando resultado calculado com campo publicado no pacote."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=4, **kwargs)
        self._dataset = dataset

    def _run(self, formula: str = "auto") -> str:
        dataset_type = self._dataset.get("tipo")
        checks = []
        if formula in {"auto", "variacao_percentual"}:
            initial_field, final_field = _auto_variation_fields(self._dataset)
            variation = json.loads(
                PercentageVariationTool(self._dataset)._run(
                    initial_field=initial_field,
                    final_field=final_field,
                )
            )
            expected = _numeric_value(self._dataset, "variacao_percentual")
            if variation.get("ok") and expected is not None:
                difference = abs(float(variation["variacao_percentual"]) - expected)
                checks.append(
                    {
                        "formula": "variacao_percentual",
                        "ok": difference <= 0.05,
                        "calculado": variation["variacao_percentual"],
                        "campo_pacote": expected,
                        "diferenca": round(difference, 6),
                    }
                )
        if formula in {"auto", "taxa"} and dataset_type == "mortality_rj":
            rate = json.loads(RateCalculationTool(self._dataset)._run())
            expected = _numeric_value(self._dataset, "taxa_mortalidade_por_mil")
            if rate.get("ok") and expected is not None:
                difference = abs(float(rate["taxa"]) - expected)
                checks.append(
                    {
                        "formula": "taxa_mortalidade_por_mil",
                        "ok": difference <= 0.05,
                        "calculado": rate["taxa"],
                        "campo_pacote": expected,
                        "diferenca": round(difference, 6),
                    }
                )
        return _json(
            {
                "ok": bool(checks) and all(check["ok"] for check in checks),
                "tipo_dataset": dataset_type,
                "formula_solicitada": formula,
                "checagens": checks,
                "mensagem": "Nenhuma formula aplicavel foi encontrada." if not checks else "Formulas avaliadas.",
            }
        )


class ArticleClaimCheckTool(BaseTool):
    name: str = "checador_afirmacoes_artigo"
    description: str = (
        "Extrai numeros do artigo final e verifica se eles aparecem no pacote de dados "
        "ou nas fontes registradas. Use depois da redacao para sinalizar possiveis "
        "numeros inventados ou nao rastreaveis."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, article: str) -> str:
        article_numbers = _numbers_from_text(article or "")
        known_values = _known_dataset_numbers(self._dataset)
        unsupported: list[dict[str, Any]] = []
        supported = 0

        seen_values: set[tuple[str, float]] = set()
        for item in article_numbers:
            key = (item["texto"], round(float(item["valor"]), 4))
            if key in seen_values:
                continue
            seen_values.add(key)
            if _is_supported_number(float(item["valor"]), known_values):
                supported += 1
            else:
                unsupported.append(item)
            if len(unsupported) >= 20:
                break

        return _json(
            {
                "ok": not unsupported,
                "numeros_no_artigo": len(article_numbers),
                "numeros_suportados": supported,
                "numeros_sem_correspondencia": unsupported,
                "total_numeros_conhecidos_no_pacote": len(known_values),
                "criterio": (
                    "Cada numero do artigo, ignorando URLs, deve aparecer no pacote de dados "
                    "com tolerancia absoluta de 0,01 ou relativa de 0,1%."
                ),
            }
        )


class CausalityCheckTool(BaseTool):
    name: str = "checador_causalidade"
    description: str = (
        "Detecta linguagem causal forte no artigo quando o pacote de dados nao oferece "
        "evidencia causal. Sinaliza termos como causou, provocou, levou a, prova que "
        "ou por causa de, preservando frases cautelosas com negacao."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, article: str) -> str:
        causal_patterns = (
            r"\bcausou\b",
            r"\bprovocou\b",
            r"\blevou a\b",
            r"\bpor causa de\b",
            r"\bdevido a\b",
            r"\bem razao de\b",
            r"\bexplica\b",
            r"\bexplicam\b",
            r"\bprova que\b",
            r"\bcomprova que\b",
            r"\bdemonstra que\b",
            r"\bresponsavel por\b",
        )
        safe_phrases = ("causa de morte", "causas de morte", "sem causalidade", "nao implica causalidade")
        risky_sentences: list[dict[str, str]] = []

        for sentence in _sentences(article):
            normalized = _normalize_check_text(sentence)
            if any(phrase in normalized for phrase in safe_phrases):
                continue
            if _has_negation_context(sentence):
                continue
            matched = [pattern for pattern in causal_patterns if re.search(pattern, normalized)]
            if matched:
                risky_sentences.append({"trecho": sentence[:500], "padroes": matched})

        return _json(
            {
                "ok": not risky_sentences,
                "trechos_com_causalidade_forte": risky_sentences,
                "criterio": (
                    "O artigo nao deve afirmar causalidade forte se o pacote so traz "
                    "indicadores observacionais ou descoberta de metadados."
                ),
                "tipo_dataset": self._dataset.get("tipo"),
            }
        )


class TemporalConsistencyCheckTool(BaseTool):
    name: str = "checador_temporal"
    description: str = (
        "Verifica se o artigo trata dados observados como previsao/projecao ou usa "
        "anos futuros/fora do pacote sem apoio nos dados."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, article: str) -> str:
        normalized_article = _normalize_check_text(article)
        dataset_text = _normalize_check_text(json.dumps(self._dataset, ensure_ascii=False, default=str))
        projection_terms = (
            "previsao",
            "previsoes",
            "projecao",
            "projecoes",
            "projetado",
            "projetada",
            "projeta",
            "preve",
            "previsto",
            "prevista",
        )
        article_projection_terms = [
            term
            for term in projection_terms
            if term in normalized_article
        ]
        dataset_allows_projection = any(term in dataset_text for term in projection_terms)

        reference_year = _dataset_reference_year(self._dataset)
        dataset_years = _dataset_years(self._dataset)
        unsupported_future_years: list[int] = []
        for year in sorted(_article_years(article)):
            if reference_year is not None and year > reference_year and year not in dataset_years:
                unsupported_future_years.append(year)

        issues = []
        if article_projection_terms and not dataset_allows_projection:
            issues.append(
                {
                    "tipo": "linguagem_de_previsao_sem_suporte",
                    "termos": article_projection_terms,
                }
            )
        if unsupported_future_years:
            issues.append(
                {
                    "tipo": "ano_futuro_ou_fora_do_pacote",
                    "anos": unsupported_future_years,
                    "ano_referencia_do_pacote": reference_year,
                }
            )

        return _json(
            {
                "ok": not issues,
                "problemas_temporais": issues,
                "anos_do_pacote": sorted(dataset_years),
                "ano_referencia_do_pacote": reference_year,
                "criterio": (
                    "Datas e periodos da noticia devem estar apoiados no pacote. "
                    "Dados observados nao devem ser chamados de previsao ou projecao."
                ),
            }
        )


class GeographicScopeCheckTool(BaseTool):
    name: str = "checador_escopo_geografico"
    description: str = (
        "Verifica se a noticia respeita o escopo geografico do pacote, evitando "
        "confundir UF/estado com cidade, capital, municipio ou regiao metropolitana."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, article: str) -> str:
        dataset_type = self._dataset.get("tipo")
        normalized_article = _normalize_check_text(article)
        issues: list[dict[str, str]] = []

        if dataset_type == "population_ibge":
            state_name = _normalize_check_text(self._dataset.get("estado_nome", ""))
            state_sigla = _normalize_check_text(self._dataset.get("estado_sigla", ""))
            forbidden_terms = ("cidade", "municipio", "capital", "regiao metropolitana", "metropole")
            for sentence in _sentences(article):
                normalized = _normalize_check_text(sentence)
                mentions_state = bool(state_name and state_name in normalized) or bool(
                    state_sigla and re.search(rf"\b{re.escape(state_sigla)}\b", normalized)
                )
                if mentions_state and any(term in normalized for term in forbidden_terms):
                    if _has_negation_context(sentence):
                        continue
                    issues.append(
                        {
                            "trecho": sentence[:500],
                            "problema": "Pauta de populacao IBGE e por UF; texto parece tratar UF como cidade/municipio.",
                        }
                    )

        if dataset_type == "mortality_rj":
            local_type = _normalize_check_text(self._dataset.get("localidade_tipo", ""))
            for sentence in _sentences(article):
                normalized = _normalize_check_text(sentence)
                if _has_negation_context(sentence):
                    continue
                if "unidade da federacao" in local_type and (
                    "cidade do rio" in normalized
                    or "municipio do rio" in normalized
                    or "capital fluminense" in normalized
                ):
                    issues.append(
                        {
                            "trecho": sentence[:500],
                            "problema": "Pacote de mortalidade esta no escopo estadual, mas texto parece falar do municipio/capital.",
                        }
                    )
                if "municipio" in local_type and (
                    "estado do rio" in normalized
                    or "unidade da federacao" in normalized
                    or "erj" in normalized
                ):
                    issues.append(
                        {
                            "trecho": sentence[:500],
                            "problema": "Pacote de mortalidade esta no escopo municipal, mas texto parece falar do estado.",
                        }
                    )

        return _json(
            {
                "ok": not issues,
                "problemas_de_escopo": issues,
                "tipo_dataset": dataset_type,
                "escopo_geografico": self._dataset.get("escopo_geografico"),
                "criterio": "O texto deve preservar o escopo geografico declarado no pacote de dados.",
            }
        )


class TransparencyReviewTool(BaseTool):
    name: str = "revisor_transparencia"
    description: str = (
        "Confere requisitos editoriais de transparencia: secao Como checamos, Fontes "
        "originais, URLs do pacote, disclaimer e linguagem de limitacao metodologica."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, article: str) -> str:
        normalized = _normalize_check_text(article)
        required_sources = [str(url) for url in self._dataset.get("fontes", []) if url]
        missing_sources = [url for url in required_sources if url not in article]
        missing_items: list[str] = []

        if "como checamos" not in normalized:
            missing_items.append("secao Como checamos")
        if "auditoria factual" not in normalized:
            missing_items.append("secao Auditoria factual")
        if "fontes originais" not in normalized:
            missing_items.append("secao Fontes originais")
        if REQUIRED_DISCLAIMER not in article:
            missing_items.append("frase obrigatoria de isencao")
        if missing_sources:
            missing_items.append("todas as URLs originais do pacote")

        limitation_markers = (
            "limite",
            "limitacao",
            "limitacoes",
            "metodolog",
            "cautela",
            "nao permite",
            "nao deve",
            "descoberta de metadados",
            "taxa bruta",
        )
        if not any(marker in normalized for marker in limitation_markers):
            missing_items.append("limitacao ou cautela metodologica")

        return _json(
            {
                "ok": not missing_items,
                "itens_ausentes_ou_incompletos": missing_items,
                "fontes_exigidas": required_sources,
                "fontes_ausentes": missing_sources,
                "criterio": (
                    "A noticia final deve explicar como foi checada, citar fontes originais, "
                    "incluir disclaimer e explicitar limitacoes metodologicas."
                ),
            }
        )


class SourceCitationCheckTool(BaseTool):
    name: str = "checador_fontes_citadas"
    description: str = (
        "Confere se o artigo cita exatamente as fontes originais do pacote, sem omitir "
        "URLs obrigatorias e sem introduzir URLs externas nao presentes na coleta."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, article: str) -> str:
        required_sources = [str(url) for url in self._dataset.get("fontes", []) if url]
        article_urls = extract_urls(article or "")
        missing_sources = [url for url in required_sources if url not in article_urls and url not in (article or "")]
        extra_urls = [
            url
            for url in article_urls
            if not any(url == source or source.startswith(url) or url.startswith(source) for source in required_sources)
        ]
        validation = validate_source_urls(article_urls)
        issues = []
        if missing_sources:
            issues.append("fontes_obrigatorias_ausentes")
        if extra_urls:
            issues.append("urls_extras_nao_presentes_no_pacote")
        if not validation.ok:
            issues.append("url_em_dominio_nao_permitido")

        return _json(
            {
                "ok": not issues,
                "problemas": issues,
                "fontes_exigidas": required_sources,
                "fontes_ausentes": missing_sources,
                "urls_no_artigo": article_urls,
                "urls_extras": extra_urls,
                "validacao_dominios": validation.messages,
                "criterio": (
                    "O Redator deve citar as fontes originais do pacote e nao adicionar "
                    "fontes externas que nao foram coletadas/auditadas."
                ),
            }
        )


class WritingEvidenceMatrixTool(BaseTool):
    name: str = "matriz_evidencias_redacao"
    description: str = (
        "Audita se o Redator respondeu no artigo: quais numeros usou, de onde vieram, "
        "qual e a limitacao e o que nao pode concluir. Tambem mapeia numeros do texto "
        "para campos numericos do pacote quando possivel."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self, article: str) -> str:
        normalized = _normalize_check_text(article or "")
        required_answers = {
            "quais_numeros_usei": bool(re.search(r"quais.{0,60}usei", normalized)),
            "de_onde_vieram": bool(re.search(r"de onde.{0,60}vieram", normalized)),
            "qual_e_a_limitacao": "qual" in normalized and "limitacao" in normalized,
            "o_que_nao_posso_concluir": "o que" in normalized and "concluir" in normalized,
        }
        article_numbers = _numbers_from_text(article or "")
        evidence_rows = []
        seen: set[tuple[str, float]] = set()
        for item in article_numbers:
            value = float(item["valor"])
            key = (item["texto"], round(value, 4))
            if key in seen:
                continue
            seen.add(key)
            matches = _matching_numeric_fields(self._dataset, value)
            evidence_rows.append(
                {
                    "numero_no_texto": item["texto"],
                    "valor_normalizado": value,
                    "campos_do_pacote": matches,
                    "rastreado": bool(matches),
                }
            )
            if len(evidence_rows) >= 30:
                break

        missing_answers = [key for key, present in required_answers.items() if not present]
        unsupported_numbers = [row for row in evidence_rows if not row["rastreado"]]
        return _json(
            {
                "ok": not missing_answers and not unsupported_numbers,
                "respostas_obrigatorias_presentes": required_answers,
                "respostas_obrigatorias_ausentes": missing_answers,
                "evidencias_numericas": evidence_rows,
                "numeros_sem_campo_rastreado": unsupported_numbers,
                "fontes_do_pacote": self._dataset.get("fontes", []),
                "limitacao_do_pacote": (
                    self._dataset.get("nota_metodologica")
                    or self._dataset.get("nota_escopo")
                    or self._dataset.get("nota_participacao")
                    or self._dataset.get("observacao_temporal")
                    or ""
                ),
                "criterio": (
                    "O artigo deve conter a auditoria factual com as quatro respostas "
                    "e cada numero relevante deve ser rastreavel ao pacote."
                ),
            }
        )


class CrossUfComparisonTool(BaseTool):
    name: str = "comparar_ufs"
    description: str = (
        "Compara o estado selecionado com os demais no ranking do pacote population_ibge. "
        "Retorna media, mediana, posicao relativa e distancia da media do estado escolhido. "
        "Use apenas quando o dataset for population_ibge e houver ranking_2024 disponivel."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=3, **kwargs)
        self._dataset = dataset

    def _run(self, comparison_field: str = "populacao_2024") -> str:
        if self._dataset.get("tipo") != "population_ibge":
            return _json(
                {
                    "ok": False,
                    "mensagem": "Esta tool e valida apenas para datasets do tipo population_ibge.",
                    "tipo_dataset": self._dataset.get("tipo"),
                }
            )

        rows = self._dataset.get("ranking_2024", [])
        if not rows or not isinstance(rows, list):
            return _json({"ok": False, "mensagem": "Campo ranking_2024 nao encontrado ou vazio no pacote."})

        values: list[float] = []
        selected_item: dict[str, Any] | None = None

        for row in rows:
            if not isinstance(row, dict):
                continue
            value = _numeric_value(row, comparison_field)
            if value is None:
                continue
            values.append(value)
            if row.get("selecionado") or row.get("id") == self._dataset.get("estado_id"):
                selected_item = row

        if len(values) < 2:
            return _json({"ok": False, "mensagem": "Dados insuficientes para comparacao entre UFs."})

        mean = sum(values) / len(values)
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2 if len(sorted_values) % 2 == 0 else sorted_values[mid]

        selected_value = _numeric_value(selected_item, comparison_field) if selected_item else None
        selected_name = selected_item.get("nome") if selected_item else None
        selected_rank = _numeric_value(selected_item, "rank_2024") if selected_item else None

        comparison: dict[str, Any] = {
            "campo_comparado": comparison_field,
            "total_ufs_no_ranking": len(values),
            "media_nacional": round(mean, 2),
            "mediana_nacional": round(median, 2),
            "maior_valor": round(max(values), 2),
            "menor_valor": round(min(values), 2),
        }

        if selected_item is not None and selected_value is not None:
            distancia_media = selected_value - mean
            pct_acima_media = (distancia_media / mean) * 100 if mean else None
            comparison["uf_selecionada"] = {
                "nome": selected_name,
                "rank": selected_rank,
                "valor": round(selected_value, 2),
                "distancia_da_media": round(distancia_media, 2),
                "pct_em_relacao_a_media": round(pct_acima_media, 2) if pct_acima_media is not None else None,
                "acima_da_media": distancia_media > 0,
            }
        else:
            comparison["aviso"] = "Estado selecionado nao identificado no ranking; use DatasetFieldLookupTool para localizar estado_id."

        return _json({"ok": True, **comparison})


class DatasetCompletionCheckTool(BaseTool):
    name: str = "verificar_completude_pacote"
    description: str = (
        "Verifica se os campos obrigatorios do pacote estao presentes: tipo, fontes, "
        "coletado_em, resumo_numerico, nota_metodologica. Para tipos especificos verifica "
        "campos adicionais (ex.: taxa_mortalidade_por_mil para mortality_rj). "
        "Use no inicio da coleta e antes da redacao para saber o que falta."
    )
    _dataset: dict = PrivateAttr(default_factory=dict)

    _REQUIRED_BASE = ("tipo", "fontes", "coletado_em", "resumo_numerico", "nota_metodologica")
    _REQUIRED_BY_TYPE: dict[str, tuple[str, ...]] = {
        "mortality_rj": ("taxa_mortalidade_por_mil", "obitos_residentes", "populacao_residente"),
        "population_ibge": ("populacao_final", "variacao_percentual", "rank_uf_2024", "ranking_2024"),
        "selic_bcb": ("valor_final", "valor_inicial", "observacoes"),
        "ipca_ibge": ("variacao_acumulada_12m", "media_periodo", "observacoes"),
        "emendas_cgu": ("total_empenhado", "total_pago", "registros"),
        "worldbank_wdi": ("indicador_codigo", "valor_mais_recente", "observacoes"),
        "araruama_news": ("registros",),
    }

    def __init__(self, dataset: dict, **kwargs: Any) -> None:
        super().__init__(max_usage_count=2, **kwargs)
        self._dataset = dataset

    def _run(self) -> str:
        dataset_type = self._dataset.get("tipo") or ""
        required_fields = list(self._REQUIRED_BASE)
        extra_required = list(self._REQUIRED_BY_TYPE.get(dataset_type, ()))
        all_required = required_fields + extra_required

        present: list[str] = []
        absent: list[str] = []
        for field in all_required:
            value = self._dataset.get(field)
            has_value = value is not None and value != "" and value != [] and value != {}
            if has_value:
                present.append(field)
            else:
                absent.append(field)

        completude_pct = round(len(present) / len(all_required) * 100, 1) if all_required else 100.0
        warnings: list[str] = []
        if not self._dataset.get("fontes"):
            warnings.append("Pacote sem URLs de fontes citaveis.")
        if dataset_type == "open_data_discovery":
            warnings.append("Tipo open_data_discovery nao contem dados numericos estruturados; use coletor especifico.")

        return _json(
            {
                "ok": not absent,
                "tipo_dataset": dataset_type,
                "completude_pct": completude_pct,
                "campos_presentes": present,
                "campos_ausentes": absent,
                "campos_obrigatorios_base": required_fields,
                "campos_obrigatorios_por_tipo": extra_required,
                "avisos": warnings,
            }
        )


@dataclass(frozen=True)
class NewsroomToolset:
    collector: list[BaseTool]
    analyst: list[BaseTool]
    writer: list[BaseTool]


@dataclass(frozen=True)
class ToolExecutionRecord:
    role: str
    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "papel": self.role,
            "tool": self.tool_name,
            "argumentos": _compact(self.arguments, max_text=500),
            "ok": self.ok,
            "saida": self.output,
        }


def build_newsroom_toolset(
    dataset: dict,
    source_urls: list[str],
    rag_context: dict | None,
    user_request: str | None = None,
) -> NewsroomToolset:
    collector_tools: list[BaseTool] = [
        DatasetProfileTool(dataset),
        SourcePolicyAuditTool(source_urls),
        RagContextTool(rag_context),
        CollectorRegistryTool(),
        SourceCatalogLookupTool(default_query=user_request),
        SourceQualityAssessmentTool(dataset, default_query=user_request),
        SourceFreshnessCheckTool(dataset, default_query=user_request),
        TopicClassificationTool(default_query=user_request),
        PrimarySourceSelectionTool(dataset, default_query=user_request),
        SecondarySourceDetectionTool(dataset),
    ]
    if dataset.get("tipo") == "open_data_discovery":
        collector_tools.append(PublicPortalSearchTool(dataset, default_query=user_request))
        collector_tools.append(TabularResourceDownloadTool(dataset))

    collector_tools.append(DatasetCompletionCheckTool(dataset))

    return NewsroomToolset(
        collector=collector_tools,
        analyst=[
            DatasetProfileTool(dataset),
            DatasetFieldLookupTool(dataset),
            DatasetStatisticalProfileTool(dataset),
            PercentageVariationTool(dataset),
            RateCalculationTool(dataset),
            PeriodComparisonTool(dataset),
            RankingGenerationTool(dataset),
            OutlierDetectionTool(dataset),
            IndicatorFormulaValidationTool(dataset),
            CrossUfComparisonTool(dataset),
            DatasetCompletionCheckTool(dataset),
        ],
        writer=[
            DatasetFieldLookupTool(dataset),
            SourcePolicyAuditTool(source_urls),
            ArticleClaimCheckTool(dataset),
            CausalityCheckTool(dataset),
            TemporalConsistencyCheckTool(dataset),
            GeographicScopeCheckTool(dataset),
            SourceCitationCheckTool(dataset),
            TransparencyReviewTool(dataset),
            WritingEvidenceMatrixTool(dataset),
            DatasetCompletionCheckTool(dataset),
        ],
    )


def _find_tool(tools: list[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise LookupError(f"Tool nao encontrada: {tool_name}")


def _maybe_find_tool(tools: list[BaseTool], tool_name: str) -> BaseTool | None:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


def _tool_output_ok(output: str) -> bool:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return True

    if isinstance(data, dict) and data.get("ok") is False:
        return False
    return True


def _run_tool(
    role: str,
    tool: BaseTool,
    arguments: dict[str, Any] | None = None,
    required_ok: bool = True,
) -> ToolExecutionRecord:
    arguments = arguments or {}
    try:
        result = tool.run(**arguments)
        output = result if isinstance(result, str) else _json(result)
        return ToolExecutionRecord(
            role=role,
            tool_name=tool.name,
            arguments=arguments,
            ok=_tool_output_ok(output) if required_ok else True,
            output=output,
        )
    except Exception as exc:
        return ToolExecutionRecord(
            role=role,
            tool_name=tool.name,
            arguments=arguments,
            ok=False,
            output=_json({"erro": str(exc)}),
        )


def run_required_tools(toolset: NewsroomToolset | None, user_request: str = "") -> list[ToolExecutionRecord]:
    if toolset is None:
        return []

    execution_plan: list[tuple[str, list[BaseTool], str, dict[str, Any], bool]] = [
        ("coletor", toolset.collector, "perfil_pacote_dados", {}, True),
        ("coletor", toolset.collector, "auditar_fontes_publicas", {}, True),
        ("coletor", toolset.collector, "ver_contexto_rag", {}, True),
        ("coletor", toolset.collector, "listar_coletores_configurados", {}, True),
        ("coletor", toolset.collector, "consultar_catalogo_fontes", {"query": user_request, "limit": 5}, True),
        ("coletor", toolset.collector, "avaliar_qualidade_fonte", {"query": user_request}, True),
        ("coletor", toolset.collector, "verificar_atualizacao_fonte", {"query": user_request}, False),
        ("coletor", toolset.collector, "classificar_tipo_de_pauta", {"query": user_request}, True),
        ("coletor", toolset.collector, "selecionar_fonte_primaria", {"query": user_request}, True),
        ("coletor", toolset.collector, "detectar_fonte_secundaria_ou_inadequada", {}, False),
        ("analista", toolset.analyst, "perfil_pacote_dados", {}, True),
        ("analista", toolset.analyst, "consultar_campo_pacote", {"field_path": "resumo_numerico"}, True),
        ("analista", toolset.analyst, "perfil_estatistico_dataset", {}, True),
        ("analista", toolset.analyst, "calcular_variacao_percentual", {}, False),
        ("analista", toolset.analyst, "calcular_taxa", {}, False),
        ("analista", toolset.analyst, "comparar_periodos", {}, False),
        ("analista", toolset.analyst, "gerar_ranking", {}, False),
        ("analista", toolset.analyst, "detectar_outliers", {}, False),
        ("analista", toolset.analyst, "validar_formula_indicador", {}, False),
        ("analista", toolset.analyst, "verificar_completude_pacote", {}, False),
        ("analista", toolset.analyst, "comparar_ufs", {}, False),
        ("redator", toolset.writer, "verificar_completude_pacote", {}, False),
        ("redator", toolset.writer, "consultar_campo_pacote", {"field_path": "fontes"}, True),
        ("redator", toolset.writer, "auditar_fontes_publicas", {}, True),
    ]

    records: list[ToolExecutionRecord] = []
    for role, tools, tool_name, arguments, required_ok in execution_plan:
        records.append(_run_tool(role, _find_tool(tools, tool_name), arguments, required_ok=required_ok))

    public_portal_tool = _maybe_find_tool(toolset.collector, "buscar_portais_publicos")
    if public_portal_tool is not None:
        records.append(
            _run_tool(
                "coletor",
                public_portal_tool,
                {"query": user_request, "limit": 5},
            )
        )

    tabular_download_tool = _maybe_find_tool(toolset.collector, "baixar_recurso_tabular")
    if tabular_download_tool is not None:
        records.append(_run_tool("coletor", tabular_download_tool, {}))
    return records


def run_post_article_tools(dataset: dict, article: str) -> list[ToolExecutionRecord]:
    return [
        _run_tool(
            "redator",
            ArticleClaimCheckTool(dataset),
            {"article": article},
        ),
        _run_tool(
            "redator",
            CausalityCheckTool(dataset),
            {"article": article},
        ),
        _run_tool(
            "redator",
            TemporalConsistencyCheckTool(dataset),
            {"article": article},
        ),
        _run_tool(
            "redator",
            GeographicScopeCheckTool(dataset),
            {"article": article},
        ),
        _run_tool(
            "redator",
            SourceCitationCheckTool(dataset),
            {"article": article},
        ),
        _run_tool(
            "redator",
            TransparencyReviewTool(dataset),
            {"article": article},
        ),
        _run_tool(
            "redator",
            WritingEvidenceMatrixTool(dataset),
            {"article": article},
        ),
    ]


def format_tool_executions_for_prompt(records: list[ToolExecutionRecord], max_output_chars: int = 1200) -> str:
    if not records:
        return "Tools deterministicas desabilitadas nesta execucao."

    sections = []
    for index, record in enumerate(records, start=1):
        status = "ok" if record.ok else "falha"
        arguments = _json(record.arguments)
        output = record.output
        if len(output) > max_output_chars:
            output = f"{output[: max_output_chars - 3].rstrip()}..."
        sections.append(
            "\n".join(
                [
                    f"{index}. papel={record.role}; tool={record.tool_name}; status={status}",
                    f"argumentos={arguments}",
                    f"saida={output}",
                ]
            )
        )
    return "\n\n".join(sections)


def tool_execution_dicts(records: list[ToolExecutionRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def tool_names(toolset: NewsroomToolset | None) -> list[str]:
    if toolset is None:
        return []

    names = []
    for tool in [*toolset.collector, *toolset.analyst, *toolset.writer]:
        if tool.name not in names:
            names.append(tool.name)
    return names
