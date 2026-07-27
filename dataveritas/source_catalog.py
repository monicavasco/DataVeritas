from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceCatalogEntry:
    id: str
    name: str
    organization: str
    domains: tuple[str, ...]
    themes: tuple[str, ...]
    api_url: str
    docs_url: str
    coverage: str
    update_frequency: str
    collector_status: str
    limitations: str
    examples: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


SOURCE_CATALOG: tuple[SourceCatalogEntry, ...] = (
    SourceCatalogEntry(
        id="catalog:datasus:health_systems",
        name="DATASUS / OpenDataSUS",
        organization="Ministerio da Saude",
        domains=("datasus.saude.gov.br", "opendatasus.saude.gov.br", "gov.br"),
        themes=(
            "saude",
            "sus",
            "mortalidade",
            "nascimentos",
            "internacoes",
            "vacinacao",
            "cnes",
            "hospital",
        ),
        api_url="https://opendatasus.saude.gov.br/",
        docs_url="https://datasus.saude.gov.br/",
        coverage="Brasil, estados, municipios e estabelecimentos de saude, conforme a base consultada.",
        update_frequency="Varia por sistema; algumas bases sao mensais e outras podem ter defasagem.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "Bases de saude exigem cuidado com defasagem, mudanca de layout, subregistro "
            "e diferencas entre residencia, ocorrencia e estabelecimento."
        ),
        examples=(
            "internacoes hospitalares por estado",
            "nascimentos por municipio",
            "cobertura vacinal",
            "estabelecimentos CNES",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:inep:education",
        name="INEP Dados Abertos",
        organization="Instituto Nacional de Estudos e Pesquisas Educacionais Anisio Teixeira",
        domains=("inep.gov.br", "gov.br"),
        themes=(
            "educacao",
            "escola",
            "censo escolar",
            "enem",
            "ideb",
            "educacao basica",
            "ensino superior",
        ),
        api_url="https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos",
        docs_url="https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos",
        coverage="Brasil, unidades da federacao, municipios, escolas e provas, conforme a base.",
        update_frequency="Geralmente anual para censos, avaliacoes e indicadores.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "Microdados podem ser grandes e exigir dicionario de variaveis; indicadores educacionais "
            "nao devem ser comparados sem respeitar etapa, rede, ano e metodologia."
        ),
        examples=(
            "IDEB por municipio",
            "matriculas no ensino medio",
            "perfil do ENEM",
            "escolas sem infraestrutura",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:ipea:ipeadata",
        name="Ipeadata",
        organization="Instituto de Pesquisa Economica Aplicada",
        domains=("ipeadata.gov.br", "ipea.gov.br", "gov.br"),
        themes=(
            "economia",
            "social",
            "renda",
            "pobreza",
            "trabalho",
            "precos",
            "desigualdade",
            "indicadores",
        ),
        api_url="http://www.ipeadata.gov.br/api/odata4/",
        docs_url="http://www.ipeadata.gov.br/",
        coverage="Series economicas, sociais e regionais brasileiras.",
        update_frequency="Varia por serie e fonte original.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "Series podem ter fonte primaria externa, mudanca metodologica ou frequencias diferentes; "
            "sempre verificar unidade, periodicidade e fonte original."
        ),
        examples=(
            "desigualdade de renda",
            "serie historica de pobreza",
            "indicadores regionais",
            "mercado de trabalho",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:tse:elections",
        name="TSE Dados Abertos",
        organization="Tribunal Superior Eleitoral",
        domains=("dadosabertos.tse.jus.br", "tse.jus.br", "jus.br"),
        themes=(
            "eleicoes",
            "votos",
            "candidatos",
            "partidos",
            "urna",
            "comparecimento",
            "abstencao",
            "prestacao de contas",
        ),
        api_url="https://dadosabertos.tse.jus.br/",
        docs_url="https://dadosabertos.tse.jus.br/",
        coverage="Brasil, unidades da federacao, municipios, zonas e secoes eleitorais, conforme a base.",
        update_frequency="Varia por base; dados eleitorais sao atualizados conforme calendario eleitoral.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "Dados eleitorais exigem filtro por turno, cargo, ano, municipio e situacao da candidatura; "
            "evitar concluir causalidade politica sem outras evidencias."
        ),
        examples=(
            "abstencao nas eleicoes",
            "votos por municipio",
            "perfil de candidaturas",
            "prestacao de contas eleitorais",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:cgu:transparency",
        name="Portal da Transparencia / CGU",
        organization="Controladoria-Geral da Uniao",
        domains=("portaldatransparencia.gov.br", "api.portaldatransparencia.gov.br", "gov.br"),
        themes=(
            "transparencia",
            "gastos publicos",
            "despesas",
            "orcamento",
            "contratos",
            "convenios",
            "emendas",
            "beneficios sociais",
        ),
        api_url="https://api.portaldatransparencia.gov.br/",
        docs_url="https://portaldatransparencia.gov.br/api-de-dados",
        coverage="Governo federal, orgaos, programas, favorecidos e transferencias, conforme endpoint.",
        update_frequency="Varia por conjunto; muitos dados seguem ciclos mensais ou diarios.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "APIs podem exigir chave; valores orcamentarios precisam distinguir empenhado, liquidado, "
            "pago, favorecido, orgao e periodo."
        ),
        examples=(
            "gastos por orgao",
            "emendas parlamentares",
            "transferencias a municipios",
            "beneficios sociais",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:senado:legislative",
        name="Dados Abertos do Senado Federal",
        organization="Senado Federal",
        domains=("legis.senado.leg.br", "senado.leg.br", "leg.br"),
        themes=(
            "senado",
            "legislativo",
            "projetos de lei",
            "votacoes",
            "senadores",
            "comissoes",
            "legislacao",
        ),
        api_url="https://legis.senado.leg.br/dadosabertos/",
        docs_url="https://legis.senado.leg.br/dadosabertos/docs/",
        coverage="Atividade legislativa do Senado, parlamentares, materias e votacoes.",
        update_frequency="Atualizacao conforme tramitacao legislativa.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "Materias legislativas exigem separar autoria, relatoria, tramitacao, votacao e situacao atual."
        ),
        examples=(
            "projetos em tramitacao",
            "votacoes no Senado",
            "perfil de senadores",
            "materias por tema",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:ibge:sidra",
        name="IBGE SIDRA",
        organization="Instituto Brasileiro de Geografia e Estatistica",
        domains=("sidra.ibge.gov.br", "servicodados.ibge.gov.br", "ibge.gov.br"),
        themes=(
            "ibge",
            "sidra",
            "pnad",
            "censo",
            "agropecuaria",
            "inflacao",
            "trabalho",
            "renda",
            "municipios",
        ),
        api_url="https://apisidra.ibge.gov.br/",
        docs_url="https://apisidra.ibge.gov.br/home/ajuda",
        coverage="Brasil, regioes, UFs e municipios, conforme tabela SIDRA.",
        update_frequency="Varia por pesquisa.",
        collector_status="catalogado; alguns casos usam coletores estruturados, outros usam descoberta.",
        limitations=(
            "SIDRA exige identificar tabela, variavel, classificacoes e niveis territoriais; "
            "comparacoes devem preservar unidade, periodo e metodologia da pesquisa."
        ),
        examples=(
            "PNAD Continua",
            "Censo Demografico",
            "producao agricola municipal",
            "indicadores por municipio",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:worldbank:wdi",
        name="World Bank Open Data",
        organization="World Bank",
        domains=("api.worldbank.org", "data.worldbank.org", "worldbank.org"),
        themes=(
            "banco mundial",
            "world bank",
            "wdi",
            "pib",
            "pobreza",
            "desenvolvimento",
            "educacao",
            "saude",
            "internacional",
        ),
        api_url="https://api.worldbank.org/v2/",
        docs_url="https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information",
        coverage="Paises e indicadores internacionais publicados pelo Banco Mundial.",
        update_frequency="Varia por indicador e pais.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "Indicadores internacionais podem ter defasagem, revisoes historicas e metodologias "
            "diferentes das fontes nacionais."
        ),
        examples=(
            "PIB por pais",
            "pobreza internacional",
            "expectativa de vida",
            "indicadores de desenvolvimento",
        ),
    ),
    SourceCatalogEntry(
        id="catalog:oecd:data",
        name="OECD Data",
        organization="Organisation for Economic Co-operation and Development",
        domains=("data.oecd.org", "oecd.org"),
        themes=(
            "ocde",
            "oecd",
            "pisa",
            "educacao",
            "economia",
            "produtividade",
            "mercado de trabalho",
            "internacional",
        ),
        api_url="https://sdmx.oecd.org/public/rest/",
        docs_url="https://data.oecd.org/api/",
        coverage="Paises membros e parceiros da OCDE, conforme indicador.",
        update_frequency="Varia por indicador.",
        collector_status="catalogado; use descoberta em portais publicos ate existir coletor estruturado.",
        limitations=(
            "Comparacoes internacionais exigem checar definicao, unidade, populacao de referencia "
            "e cobertura dos paises."
        ),
        examples=(
            "PISA",
            "produtividade",
            "desemprego internacional",
            "indicadores educacionais comparados",
        ),
    ),
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.casefold()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", _normalize(text)))


def _entry_text(entry: SourceCatalogEntry) -> str:
    return " ".join(
        [
            entry.name,
            entry.organization,
            " ".join(entry.themes),
            entry.coverage,
            entry.limitations,
            " ".join(entry.examples),
        ]
    )


def catalog_entries() -> tuple[SourceCatalogEntry, ...]:
    return SOURCE_CATALOG


def search_catalog(query: str, limit: int = 5) -> list[dict]:
    query_tokens = _tokens(query)
    scored: list[tuple[int, SourceCatalogEntry]] = []
    for entry in SOURCE_CATALOG:
        entry_tokens = _tokens(_entry_text(entry))
        score = len(query_tokens.intersection(entry_tokens))
        if score:
            scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "score": score,
            **entry.to_dict(),
        }
        for score, entry in scored[:limit]
    ]
