from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import chromadb

from dataveritas.collectors import SOURCE_COLLECTORS, SOURCE_LABELS, collector_definitions
from dataveritas.source_catalog import catalog_entries, search_catalog


VECTOR_DIMENSIONS = 384
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"
DEFAULT_COLLECTION_NAME = "dataveritas_rag"


@dataclass(frozen=True)
class RagDocument:
    id: str
    content: str
    metadata: dict


@dataclass(frozen=True)
class RagMatch:
    id: str
    score: float
    content: str
    metadata: dict


@dataclass(frozen=True)
class RagRecommendation:
    query: str
    source_type: str
    source_label: str
    collector: str
    confidence: float
    matches: list[RagMatch]


def _collector_source_documents() -> list[RagDocument]:
    documents: list[RagDocument] = []
    for definition in collector_definitions():
        for source_document in definition.source_documents:
            documents.append(
                RagDocument(
                    id=source_document.id,
                    content=source_document.content,
                    metadata={
                        "doc_kind": "source",
                        "source_type": definition.source_type,
                        "source_label": definition.source_label,
                        "collector": definition.collector_name,
                        "domain": "; ".join(source_document.domains),
                        "themes": list(source_document.themes),
                        "structured": definition.structured,
                        "limitations": definition.limitations,
                    },
                )
            )
    return documents


def _catalog_source_documents() -> list[RagDocument]:
    documents: list[RagDocument] = []
    for entry in catalog_entries():
        documents.append(
            RagDocument(
                id=entry.id,
                content=(
                    f"Catalogo de fonte publica: {entry.name}, mantida por {entry.organization}. "
                    f"Temas: {', '.join(entry.themes)}. Cobertura: {entry.coverage}. "
                    f"API ou portal: {entry.api_url}. Documentacao: {entry.docs_url}. "
                    f"Exemplos de pauta: {', '.join(entry.examples)}. "
                    f"Status do coletor: {entry.collector_status}. "
                    f"Limitacoes metodologicas: {entry.limitations}"
                ),
                metadata={
                    "doc_kind": "source_catalog",
                    "source_type": "open_data_discovery",
                    "source_label": entry.name,
                    "collector": SOURCE_COLLECTORS["open_data_discovery"],
                    "catalog_id": entry.id,
                    "organization": entry.organization,
                    "domain": "; ".join(entry.domains),
                    "themes": list(entry.themes),
                    "api_url": entry.api_url,
                    "docs_url": entry.docs_url,
                    "coverage": entry.coverage,
                    "update_frequency": entry.update_frequency,
                    "collector_status": entry.collector_status,
                    "limitations": entry.limitations,
                    "structured": False,
                },
            )
        )
    return documents


SOURCE_DOCUMENTS = [
    RagDocument(
        id="method:transparency",
        content=(
            "Politica editorial DataVeritas: toda noticia automatica deve incluir fonte "
            "original, URL verificavel, limitacoes metodologicas e a frase dados simulados "
            "para fins didaticos. O RAG recupera contexto e fontes candidatas; os numeros "
            "devem vir das APIs oficiais e passar por guardrails."
        ),
        metadata={
            "doc_kind": "method",
            "source_type": "policy",
            "source_label": "Politica editorial DataVeritas",
            "collector": "guardrails",
            "domain": "local",
            "themes": ["guardrails", "transparencia", "fontes", "checagem"],
        },
    ),
    *_collector_source_documents(),
    *_catalog_source_documents(),
]

SYNONYMS = {
    "habitante": ["populacao", "morador", "demografia", "ibge"],
    "habitantes": ["populacao", "moradores", "demografia", "ibge"],
    "morador": ["populacao", "habitante", "demografia", "ibge"],
    "moradores": ["populacao", "habitantes", "demografia", "ibge"],
    "pessoas": ["populacao", "habitantes", "demografia"],
    "populacional": ["populacao", "demografia", "ibge"],
    "demografico": ["populacao", "demografia", "ibge"],
    "demografica": ["populacao", "demografia", "ibge"],
    "estado": ["uf", "unidade", "federacao"],
    "estados": ["uf", "unidades", "federacao"],
    "juros": ["selic", "copom", "banco", "central", "economia"],
    "juro": ["selic", "copom", "banco", "central", "economia"],
    "taxa": ["selic", "juros"],
    "selic": ["juros", "copom", "bcb", "banco", "central"],
    "copom": ["selic", "juros", "banco", "central"],
    "monetaria": ["selic", "juros", "copom", "economia"],
    "monetario": ["selic", "juros", "copom", "economia"],
    "mortalidade": ["obitos", "mortes", "saude", "sim", "taxa", "rj"],
    "mortal": ["mortalidade", "obitos", "mortes", "saude"],
    "obito": ["mortalidade", "morte", "saude", "sim"],
    "obitos": ["mortalidade", "mortes", "saude", "sim"],
    "morte": ["mortalidade", "obito", "saude", "sim"],
    "mortes": ["mortalidade", "obitos", "saude", "sim"],
    "falecimento": ["mortalidade", "obito", "saude"],
    "falecimentos": ["mortalidade", "obitos", "saude"],
    "rj": ["rio", "janeiro"],
    "janeiro": ["rio", "rj"],
    "saude": ["dados", "abertos", "portal", "publico"],
    "educacao": ["dados", "abertos", "portal", "publico"],
    "seguranca": ["dados", "abertos", "portal", "publico"],
    "ambiente": ["dados", "abertos", "portal", "publico"],
    "transparencia": ["dados", "abertos", "portal", "publico"],
    "legislacao": ["dados", "abertos", "portal", "publico", "camara"],
    "propostas": ["dados", "abertos", "portal", "publico", "camara"],
    "proposicoes": ["dados", "abertos", "portal", "publico", "camara"],
    "orcamento": ["dados", "abertos", "portal", "publico"],
    "cultura": ["dados", "abertos", "portal", "publico"],
    "transporte": ["dados", "abertos", "portal", "publico"],
    "oms": ["who", "saude", "global", "gho", "api"],
    "who": ["oms", "saude", "global", "gho", "api"],
    "onu": ["un", "nacoes", "unidas", "ods", "sdg", "api"],
    "un": ["onu", "nacoes", "unidas", "ods", "sdg", "api"],
    "ods": ["onu", "un", "sdg", "agenda", "2030", "indicadores"],
    "sdg": ["onu", "un", "ods", "agenda", "2030", "indicators"],
    "unsd": ["onu", "un", "sdg", "ods", "statistics"],
    "global": ["internacional", "oms", "who", "onu", "un"],
    "internacional": ["global", "oms", "who", "onu", "un"],
    "datasus": ["saude", "sus", "opendatasus"],
    "sus": ["saude", "datasus", "opendatasus"],
    "inep": ["educacao", "enem", "ideb", "censo", "escolar"],
    "enem": ["inep", "educacao", "prova"],
    "ideb": ["inep", "educacao", "indicador"],
    "tse": ["eleicoes", "votos", "candidatos"],
    "eleicao": ["tse", "votos", "candidatos"],
    "eleicoes": ["tse", "votos", "candidatos"],
    "votos": ["tse", "eleicoes"],
    "ipea": ["ipeadata", "economia", "social"],
    "ipeadata": ["ipea", "economia", "indicadores"],
    "cgu": ["transparencia", "gastos", "orcamento"],
    "despesas": ["transparencia", "gastos", "orcamento"],
    "gastos": ["transparencia", "despesas", "orcamento"],
    "senado": ["legislativo", "legislacao", "votacoes"],
    "sidra": ["ibge", "pnad", "censo"],
    "pnad": ["ibge", "sidra", "trabalho", "renda"],
    "censo": ["ibge", "sidra", "populacao"],
    "ocde": ["oecd", "internacional", "pisa"],
    "oecd": ["ocde", "internacional", "pisa"],
    "pisa": ["ocde", "oecd", "educacao"],
    "worldbank": ["banco", "mundial", "internacional", "wdi"],
    "wdi": ["banco", "mundial", "worldbank"],
}


def _now() -> str:
    return datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S %Z")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.casefold()


def _tokens(text: str) -> list[str]:
    normalized = _normalize(text)
    base_tokens = re.findall(r"[a-z0-9]{2,}", normalized)
    expanded: list[str] = []

    for token in base_tokens:
        expanded.append(token)
        expanded.extend(SYNONYMS.get(token, []))

    expanded.extend(
        f"{left}_{right}"
        for left, right in zip(base_tokens, base_tokens[1:])
        if len(left) > 2 and len(right) > 2
    )
    return expanded


def _source_keyword_boosts(query: str) -> dict[str, float]:
    normalized = _normalize(query)
    base_tokens = set(re.findall(r"[a-z0-9]{2,}", normalized))
    boosts: dict[str, float] = {}

    international_phrases = (
        "nacoes unidas",
        "organizacao das nacoes unidas",
        "organizacao mundial da saude",
        "world health organization",
        "sustainable development",
    )
    if base_tokens.intersection({"oms", "who", "onu", "ods", "sdg", "unsd"}) or any(
        phrase in normalized for phrase in international_phrases
    ):
        boosts["open_data_discovery"] = 0.82

    population_terms = {
        "populacao",
        "populacional",
        "habitante",
        "habitantes",
        "morador",
        "moradores",
        "demografia",
        "demografico",
        "demografica",
    }
    municipality_terms = {"cidade", "municipio", "capital"}
    mentions_population = base_tokens.intersection(population_terms) or any(
        token.startswith("popula") for token in base_tokens
    )
    if mentions_population:
        if base_tokens.intersection(municipality_terms):
            boosts["open_data_discovery"] = max(boosts.get("open_data_discovery", 0.0), 0.45)
        else:
            boosts["population_ibge"] = 0.78

    selic_terms = {"selic", "juros", "juro", "copom"}
    if base_tokens.intersection(selic_terms) or "taxa basica" in normalized:
        boosts["selic_bcb"] = 0.78

    mortality_terms = {"mortalidade", "mortal", "obito", "obitos", "morte", "mortes"}
    rio_terms = {"rj", "janeiro"}
    if base_tokens.intersection(mortality_terms) and (
        "rio de janeiro" in normalized or base_tokens.intersection(rio_terms)
    ):
        boosts["mortality_rj"] = 0.86
    elif base_tokens.intersection(mortality_terms):
        boosts["open_data_discovery"] = max(boosts.get("open_data_discovery", 0.0), 0.45)

    catalog_open_data_terms = {
        "datasus",
        "opendatasus",
        "sus",
        "internacoes",
        "vacinacao",
        "sinasc",
        "cnes",
        "inep",
        "enem",
        "ideb",
        "escola",
        "escolas",
        "educacao",
        "matriculas",
        "tse",
        "eleicao",
        "eleicoes",
        "votos",
        "candidatos",
        "urna",
        "abstencao",
        "ipea",
        "ipeadata",
        "renda",
        "pobreza",
        "desigualdade",
        "transparencia",
        "cgu",
        "gastos",
        "despesas",
        "orcamento",
        "contratos",
        "emendas",
        "senado",
        "legislativo",
        "votacoes",
        "sidra",
        "pnad",
        "censo",
        "worldbank",
        "wdi",
        "ocde",
        "oecd",
        "pisa",
    }
    catalog_phrases = (
        "portal da transparencia",
        "banco mundial",
        "world bank",
        "dados abertos do tse",
        "censo escolar",
        "dados abertos do senado",
    )
    if base_tokens.intersection(catalog_open_data_terms) or any(
        phrase in normalized for phrase in catalog_phrases
    ):
        boosts["open_data_discovery"] = max(boosts.get("open_data_discovery", 0.0), 0.62)

    return boosts


def _mentions_rio_de_janeiro(query: str) -> bool:
    normalized = _normalize(query)
    base_tokens = set(re.findall(r"[a-z0-9]{2,}", normalized))
    return "rio de janeiro" in normalized or bool(
        base_tokens.intersection({"rj", "erj", "fluminense", "janeiro"})
    )


def _hash_index(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, "big")
    index = integer % VECTOR_DIMENSIONS
    sign = 1.0 if (integer >> 8) % 2 == 0 else -1.0
    return index, sign


def embed_text(text: str) -> list[float]:
    vector = [0.0] * VECTOR_DIMENSIONS
    for token in _tokens(text):
        index, sign = _hash_index(token)
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _to_chroma_metadata(metadata: dict) -> dict:
    chroma_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool) or value is None:
            chroma_metadata[key] = "" if value is None else value
        else:
            chroma_metadata[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return chroma_metadata


def _from_chroma_metadata(metadata: dict) -> dict:
    restored = {}
    for key, value in metadata.items():
        if isinstance(value, str) and value[:1] in {"[", "{"}:
            try:
                restored[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        restored[key] = value
    return restored


class VectorStore:
    def __init__(
        self,
        db_path: Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, document: RagDocument) -> None:
        embedding = embed_text(_document_text(document))
        metadata = _to_chroma_metadata(document.metadata)
        metadata["updated_at"] = _now()
        self.collection.upsert(
            ids=[document.id],
            documents=[document.content],
            metadatas=[metadata],
            embeddings=[embedding],
        )

    def search(self, query: str, limit: int = 4) -> list[RagMatch]:
        count = self.count()
        if count == 0:
            return []

        result = self.collection.query(
            query_embeddings=[embed_text(query)],
            n_results=min(limit, count),
            include=["documents", "metadatas", "distances"],
        )

        matches: list[RagMatch] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            matches.append(
                RagMatch(
                    id=doc_id,
                    score=round(score, 4),
                    content=content or "",
                    metadata=_from_chroma_metadata(metadata or {}),
                )
            )

        return matches

    def count(self) -> int:
        return int(self.collection.count())


def _document_text(document: RagDocument) -> str:
    metadata_text = " ".join(
        str(value)
        for key, value in sorted(document.metadata.items())
        if key in {"source_label", "collector", "domain", "source_type", "themes", "limitations"}
    )
    return f"{document.content}\n{metadata_text}"


def bootstrap_vector_store(db_path: Path = DEFAULT_CHROMA_PATH) -> VectorStore:
    store = VectorStore(db_path)
    for document in SOURCE_DOCUMENTS:
        store.upsert(document)
    return store


def search_source_catalog(
    query: str,
    limit: int = 5,
    db_path: Path = DEFAULT_CHROMA_PATH,
) -> list[RagMatch]:
    store = bootstrap_vector_store(db_path)
    matches = store.search(query, limit=max(limit * 4, limit))
    catalog_matches = [
        match
        for match in matches
        if match.metadata.get("doc_kind") == "source_catalog"
    ][:limit]

    if catalog_matches:
        return catalog_matches

    fallback_matches = []
    for item in search_catalog(query, limit=limit):
        fallback_matches.append(
            RagMatch(
                id=item["id"],
                score=round(float(item["score"]) / 10, 4),
                content=(
                    f"Catalogo de fonte publica: {item['name']}. "
                    f"Temas: {', '.join(item['themes'])}. "
                    f"API: {item['api_url']}. Documentacao: {item['docs_url']}."
                ),
                metadata={
                    "doc_kind": "source_catalog",
                    "source_type": "open_data_discovery",
                    "source_label": item["name"],
                    "collector": SOURCE_COLLECTORS["open_data_discovery"],
                    "catalog_id": item["id"],
                    "organization": item["organization"],
                    "domain": "; ".join(item["domains"]),
                    "themes": item["themes"],
                    "api_url": item["api_url"],
                    "docs_url": item["docs_url"],
                    "coverage": item["coverage"],
                    "update_frequency": item["update_frequency"],
                    "collector_status": item["collector_status"],
                    "limitations": item["limitations"],
                    "structured": False,
                },
            )
        )
    return fallback_matches


def recommend_source(query: str, db_path: Path = DEFAULT_CHROMA_PATH) -> RagRecommendation:
    store = bootstrap_vector_store(db_path)
    matches = store.search(query, limit=5)

    source_scores: dict[str, float] = {}
    best_by_source: dict[str, RagMatch] = {}
    for match in matches:
        source_type = match.metadata.get("source_type")
        if source_type not in SOURCE_LABELS:
            continue
        score = max(0.0, match.score)
        if source_type == "mortality_rj" and not _mentions_rio_de_janeiro(query):
            score *= 0.35
        source_scores[source_type] = max(source_scores.get(source_type, 0.0), score)
        best_by_source.setdefault(source_type, match)

    for source_type, boost in _source_keyword_boosts(query).items():
        source_scores[source_type] = max(source_scores.get(source_type, 0.0), boost)

    if not source_scores:
        source_type = "population_ibge"
        confidence = 0.0
    else:
        source_type, confidence = max(source_scores.items(), key=lambda item: item[1])

    best_match = best_by_source.get(source_type)
    source_label = (
        best_match.metadata.get("source_label")
        if best_match
        else SOURCE_LABELS[source_type]
    )
    collector = (
        best_match.metadata.get("collector")
        if best_match
        else SOURCE_COLLECTORS[source_type]
    )

    return RagRecommendation(
        query=query,
        source_type=source_type,
        source_label=source_label,
        collector=collector,
        confidence=round(confidence, 4),
        matches=matches,
    )


def remember_newsroom_run(
    query: str,
    source_type: str,
    article: str,
    source_urls: list[str],
    model_name: str,
    db_path: Path = DEFAULT_CHROMA_PATH,
) -> None:
    store = bootstrap_vector_store(db_path)
    normalized_query = _normalize(query)
    digest = hashlib.sha256(f"{source_type}:{normalized_query}".encode("utf-8")).hexdigest()[:16]
    article_preview = article.strip().replace("\n", " ")[:1200]
    document = RagDocument(
        id=f"run:{source_type}:{digest}",
        content=(
            f"Pauta anterior: {query}\n"
            f"Fonte usada: {SOURCE_LABELS.get(source_type, source_type)}\n"
            f"Resumo da noticia gerada: {article_preview}"
        ),
        metadata={
            "doc_kind": "run",
            "source_type": source_type,
            "source_label": SOURCE_LABELS.get(source_type, source_type),
            "collector": SOURCE_COLLECTORS.get(source_type, "unknown"),
            "model_name": model_name,
            "source_urls": source_urls,
            "created_at": _now(),
        },
    )
    store.upsert(document)
