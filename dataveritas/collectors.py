from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dataveritas.bcb import build_selic_dataset
from dataveritas.ibge import StateOption, build_population_dataset
from dataveritas.mortality import build_mortality_dataset
from dataveritas.open_data import build_open_data_discovery_dataset
from dataveritas.araruama import build_araruama_news_dataset


CollectorInputKind = Literal["state", "query", "none"]


@dataclass(frozen=True)
class SourceDocumentDefinition:
    id: str
    content: str
    domains: tuple[str, ...]
    themes: tuple[str, ...]


@dataclass(frozen=True)
class CollectorDefinition:
    source_type: str
    source_label: str
    collector_name: str
    input_kind: CollectorInputKind
    cache_ttl_seconds: int
    structured: bool
    limitations: str
    source_documents: tuple[SourceDocumentDefinition, ...]


POPULATION_SOURCE_TYPE = "population_ibge"
SELIC_SOURCE_TYPE = "selic_bcb"
MORTALITY_RJ_SOURCE_TYPE = "mortality_rj"
OPEN_DATA_SOURCE_TYPE = "open_data_discovery"
ARARUAMA_NEWS_SOURCE_TYPE = "araruama_news"


COLLECTOR_DEFINITIONS: tuple[CollectorDefinition, ...] = (
    CollectorDefinition(
        source_type=POPULATION_SOURCE_TYPE,
        source_label="Populacao estadual (IBGE)",
        collector_name="build_population_dataset",
        input_kind="state",
        cache_ttl_seconds=1800,
        structured=True,
        limitations=(
            "Dados representam Unidades da Federacao; nao devem ser tratados como cidade, "
            "capital, municipio ou regiao metropolitana."
        ),
        source_documents=(
            SourceDocumentDefinition(
                id="source:ibge:population_uf",
                content=(
                    "Fonte publica IBGE para jornalismo de dados sobre populacao, habitantes, "
                    "moradores, demografia, estimativa populacional, estados, UF, unidades da "
                    "federacao e ranking populacional. Usa a API de agregados 6579, variavel "
                    "9324, no dominio servicodados.ibge.gov.br. O coletor adequado e "
                    "build_population_dataset. Limite metodologico: os dados representam "
                    "Unidades da Federacao, nao cidade, capital, municipio ou regiao metropolitana."
                ),
                domains=("servicodados.ibge.gov.br",),
                themes=("populacao", "habitantes", "demografia", "uf", "estado"),
            ),
        ),
    ),
    CollectorDefinition(
        source_type=SELIC_SOURCE_TYPE,
        source_label="Taxa Selic (Banco Central)",
        collector_name="build_selic_dataset",
        input_kind="none",
        cache_ttl_seconds=1800,
        structured=True,
        limitations="A serie informa a meta Selic publicada; nao inferir causa economica sem outra fonte.",
        source_documents=(
            SourceDocumentDefinition(
                id="source:bcb:selic",
                content=(
                    "Fonte publica Banco Central do Brasil para jornalismo de dados sobre Selic, "
                    "juros, taxa basica de juros, Copom, politica monetaria, economia brasileira "
                    "e serie temporal economica. Usa o Sistema Gerenciador de Series Temporais "
                    "SGS, serie 432, no dominio api.bcb.gov.br. O coletor adequado e "
                    "build_selic_dataset. Limite metodologico: a serie informa a meta Selic "
                    "publicada; nao inferir causa economica sem outra fonte."
                ),
                domains=("api.bcb.gov.br",),
                themes=("selic", "juros", "copom", "economia", "banco central"),
            ),
        ),
    ),
    CollectorDefinition(
        source_type=MORTALITY_RJ_SOURCE_TYPE,
        source_label="Mortalidade RJ (SIM/SES-RJ + IBGE)",
        collector_name="build_mortality_dataset",
        input_kind="query",
        cache_ttl_seconds=1800,
        structured=True,
        limitations=(
            "Taxa bruta por mil habitantes, sem ajuste por idade, sexo, causa de morte "
            "ou estrutura demografica. Coletor especifico para Rio de Janeiro."
        ),
        source_documents=(
            SourceDocumentDefinition(
                id="source:ses_rj:sim_mortality",
                content=(
                    "Fonte publica Secretaria de Estado de Saude do Rio de Janeiro, SIM e IBGE "
                    "para jornalismo de dados sobre obitos, mortes, mortalidade geral, taxa bruta "
                    "de mortalidade, saude publica, residentes no estado do Rio de Janeiro, RJ, "
                    "cidade do Rio de Janeiro e ano de 2021. Usa o download CSV oficial de obitos "
                    "nao fetais do SIM/RJ, conta obitos de residentes e cruza com a populacao "
                    "residente estimada pelo IBGE. O coletor adequado e build_mortality_dataset. "
                    "Limite metodologico: a taxa e bruta por mil habitantes, sem ajuste por idade, "
                    "sexo, causa de morte ou estrutura demografica."
                ),
                domains=("saude.rj.gov.br", "sistemas.saude.rj.gov.br", "servicodados.ibge.gov.br"),
                themes=("mortalidade", "obitos", "mortes", "saude", "rio de janeiro", "rj", "taxa bruta"),
            ),
        ),
    ),
    CollectorDefinition(
        source_type=OPEN_DATA_SOURCE_TYPE,
        source_label="Descoberta em portais publicos",
        collector_name="build_open_data_discovery_dataset",
        input_kind="query",
        cache_ttl_seconds=900,
        structured=False,
        limitations=(
            "Rota de descoberta de metadados e indicadores; valores numericos exigem coleta "
            "estruturada da serie especifica antes de conclusoes factuais."
        ),
        source_documents=(
            SourceDocumentDefinition(
                id="source:open_data:public_portals",
                content=(
                    "Descoberta em portais publicos para pautas abertas de jornalismo de dados "
                    "que nao se encaixam nos coletores estruturados de populacao ou Selic. "
                    "Usa provedores de dados abertos como dados.gov.br quando autenticado e "
                    "Dados Abertos da Camara dos Deputados para localizar registros publicos "
                    "sobre saude, educacao, seguranca, meio ambiente, legislacao, propostas, "
                    "orcamento, transparencia, cultura, transporte e outros temas de interesse "
                    "publico. O coletor adequado e build_open_data_discovery_dataset. "
                    "Limite metodologico: esta rota descobre metadados e registros publicos; "
                    "nao deve inventar estatisticas numericas sem baixar e validar uma base estruturada."
                ),
                domains=("dados.gov.br", "dadosabertos.camara.leg.br"),
                themes=(
                    "dados abertos",
                    "portal publico",
                    "saude",
                    "educacao",
                    "seguranca",
                    "meio ambiente",
                    "legislacao",
                    "transparencia",
                ),
            ),
            SourceDocumentDefinition(
                id="source:open_data:international_apis",
                content=(
                    "Descoberta em APIs internacionais para jornalismo de dados com fontes da "
                    "OMS, WHO Global Health Observatory, ONU, United Nations Statistics Division, "
                    "UNSD, Objetivos de Desenvolvimento Sustentavel, ODS e SDG indicators. "
                    "Usa a API OData da OMS para encontrar indicadores globais de saude e a "
                    "UNSD SDG API para localizar indicadores oficiais da Agenda 2030 sobre "
                    "saude, pobreza, educacao, fome, agua, saneamento, desigualdade, clima, "
                    "instituicoes e desenvolvimento sustentavel. O coletor adequado e "
                    "build_open_data_discovery_dataset. Limite metodologico: esta rota recupera "
                    "metadados e URLs de indicadores; valores numericos exigem uma coleta "
                    "estruturada da serie especifica antes de escrever conclusoes factuais."
                ),
                domains=("ghoapi.azureedge.net", "who.int", "unstats.un.org"),
                themes=(
                    "oms",
                    "who",
                    "onu",
                    "un",
                    "unsd",
                    "ods",
                    "sdg",
                    "saude global",
                    "desenvolvimento sustentavel",
                ),
            ),
        ),
    ),
    CollectorDefinition(
        source_type=ARARUAMA_NEWS_SOURCE_TYPE,
        source_label="Noticias de Araruama (Prefeitura)",
        collector_name="build_araruama_news_dataset",
        input_kind="query",
        cache_ttl_seconds=900,
        structured=False,
        limitations="Noticias jornalisticas e institucionais oficiais da prefeitura; devem ser cruzadas com dados estatisticos estruturados.",
        source_documents=(
            SourceDocumentDefinition(
                id="source:araruama:news",
                content=(
                    "Fonte oficial de noticias da Prefeitura Municipal de Araruama para jornalismo "
                    "de dados sobre o municipio, prefeitura, Araruama, regiao dos lagos, turismo, "
                    "educacao local, infraestrutura urbana, saude publica local e atos oficiais. "
                    "Usa varredura direta no portal de noticias da prefeitura no dominio araruama.rj.gov.br. "
                    "O coletor adequado e build_araruama_news_dataset."
                ),
                domains=("araruama.rj.gov.br",),
                themes=("araruama", "noticias", "prefeitura", "regiao dos lagos", "saude local", "turismo local", "escola local"),
            ),
        ),
    ),
)

COLLECTORS_BY_SOURCE_TYPE = {
    definition.source_type: definition for definition in COLLECTOR_DEFINITIONS
}
SOURCE_LABELS = {
    definition.source_type: definition.source_label for definition in COLLECTOR_DEFINITIONS
}
SOURCE_COLLECTORS = {
    definition.source_type: definition.collector_name for definition in COLLECTOR_DEFINITIONS
}


def collector_definitions() -> tuple[CollectorDefinition, ...]:
    return COLLECTOR_DEFINITIONS


def get_collector_definition(source_type: str) -> CollectorDefinition:
    try:
        return COLLECTORS_BY_SOURCE_TYPE[source_type]
    except KeyError as exc:
        raise ValueError(f"Coletor desconhecido: {source_type}") from exc


def collector_requires_state(source_type: str) -> bool:
    return get_collector_definition(source_type).input_kind == "state"


def collect_dataset(source_type: str, user_request: str, state: StateOption | None = None):
    definition = get_collector_definition(source_type)
    if definition.input_kind == "state":
        if state is None:
            raise ValueError(f"O coletor {source_type} exige uma UF selecionada.")
        return build_population_dataset(state)
    if definition.input_kind == "none":
        return build_selic_dataset()
    if source_type == MORTALITY_RJ_SOURCE_TYPE:
        return build_mortality_dataset(user_request)
    if source_type == OPEN_DATA_SOURCE_TYPE:
        return build_open_data_discovery_dataset(user_request)
    if source_type == ARARUAMA_NEWS_SOURCE_TYPE:
        return build_araruama_news_dataset(user_request)
    raise ValueError(f"Coletor sem funcao de execucao configurada: {source_type}")
