from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dataveritas.guardrails import validate_source_urls


STATES_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
POPULATION_SOURCE_URL = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/"
    "{periods}/variaveis/9324?localidades=N3[{localidades}]"
)


@dataclass(frozen=True)
class StateOption:
    id: int
    sigla: str
    nome: str
    regiao: str


@dataclass(frozen=True)
class PopulationRow:
    id: int
    nome: str
    populacao_2024: int
    rank_2024: int


@dataclass(frozen=True)
class PopulationDataset:
    tipo: str
    escopo_geografico: str
    estado_id: int
    estado_nome: str
    estado_sigla: str
    periodo_inicial: int
    periodo_final: int
    populacao_inicial: int
    populacao_final: int
    variacao_absoluta: int
    variacao_percentual: float
    rank_uf_2024: int
    total_ufs_2024: int
    participacao_total_ufs_pct: float
    resumo_numerico: str
    nota_metodologica: str
    nota_participacao: str
    ranking_2024: list[PopulationRow]
    fontes: list[str]
    coletado_em: str

    def to_prompt_dict(self) -> dict:
        data = asdict(self)
        data["ranking_2024"] = [asdict(row) for row in self.ranking_2024]
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
    session.headers.update({"User-Agent": "DataVeritas didactic prototype"})
    return session


def _get_json(url: str):
    source_check = validate_source_urls([url])
    if not source_check.ok:
        raise ValueError("; ".join(source_check.messages))

    response = _session().get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def get_states() -> list[StateOption]:
    payload = _get_json(STATES_URL)
    states = [
        StateOption(
            id=int(item["id"]),
            sigla=item["sigla"],
            nome=item["nome"],
            regiao=item["regiao"]["nome"],
        )
        for item in payload
    ]
    return sorted(states, key=lambda state: state.nome)


def _population_url(periods: str, localidades: str) -> str:
    return POPULATION_SOURCE_URL.format(periods=periods, localidades=localidades)


def _parse_population_series(payload: list[dict]) -> list[dict]:
    if not payload:
        return []

    try:
        return payload[0]["resultados"][0]["series"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("Resposta inesperada da API do IBGE para população.") from exc


def _series_value(series: dict, year: int) -> int:
    raw_value = series.get("serie", {}).get(str(year))
    if raw_value in (None, "", "-"):
        raise ValueError(f"Valor de população ausente para {year}.")
    return int(float(str(raw_value).replace(",", ".")))


def fetch_state_population(state_id: int, start_year: int = 2021, end_year: int = 2024) -> dict:
    periods = f"{start_year}|{end_year}"
    url = _population_url(periods=periods, localidades=str(state_id))
    series = _parse_population_series(_get_json(url))
    if not series:
        raise ValueError("A API do IBGE não retornou dados para a UF selecionada.")

    item = series[0]
    return {
        "id": int(item["localidade"]["id"]),
        "nome": item["localidade"]["nome"],
        "populacao_inicial": _series_value(item, start_year),
        "populacao_final": _series_value(item, end_year),
        "fonte": url,
    }


def fetch_all_states_population(year: int = 2024) -> tuple[list[PopulationRow], str]:
    url = _population_url(periods=str(year), localidades="all")
    rows = []

    for item in _parse_population_series(_get_json(url)):
        rows.append(
            {
                "id": int(item["localidade"]["id"]),
                "nome": item["localidade"]["nome"],
                "populacao_2024": _series_value(item, year),
            }
        )

    rows.sort(key=lambda row: row["populacao_2024"], reverse=True)
    ranked = [
        PopulationRow(
            id=row["id"],
            nome=row["nome"],
            populacao_2024=row["populacao_2024"],
            rank_2024=index + 1,
        )
        for index, row in enumerate(rows)
    ]
    return ranked, url


def build_population_dataset(state: StateOption) -> PopulationDataset:
    start_year = 2021
    end_year = 2024
    state_data = fetch_state_population(state.id, start_year=start_year, end_year=end_year)
    ranking, ranking_source = fetch_all_states_population(year=end_year)

    selected_rank = next(row.rank_2024 for row in ranking if row.id == state.id)
    total_population = sum(row.populacao_2024 for row in ranking)
    population_start = state_data["populacao_inicial"]
    population_end = state_data["populacao_final"]
    delta = population_end - population_start
    pct = (delta / population_start) * 100 if population_start else 0
    share = (population_end / total_population) * 100 if total_population else 0

    collected_at = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S %Z")

    return PopulationDataset(
        tipo="population_ibge",
        escopo_geografico=(
            f"Unidade da Federacao {state.nome} ({state.sigla}); "
            "nao e municipio, capital ou regiao metropolitana."
        ),
        estado_id=state.id,
        estado_nome=state.nome,
        estado_sigla=state.sigla,
        periodo_inicial=start_year,
        periodo_final=end_year,
        populacao_inicial=population_start,
        populacao_final=population_end,
        variacao_absoluta=delta,
        variacao_percentual=round(pct, 2),
        rank_uf_2024=selected_rank,
        total_ufs_2024=total_population,
        participacao_total_ufs_pct=round(share, 2),
        resumo_numerico=(
            f"A UF {state.nome} ({state.sigla}) tinha {population_start} habitantes em {start_year} "
            f"e {population_end} habitantes em {end_year}, variacao absoluta de {delta} "
            f"e variacao percentual de {round(pct, 2)}%. Em {end_year}, ocupava a "
            f"{selected_rank}a posicao no ranking de UFs e representava {round(share, 2)}% "
            "da populacao total somada das UFs retornadas pela consulta."
        ),
        nota_metodologica=(
            "Os dados representam estimativas populacionais por Unidade da Federacao. "
            "Nao descreva a UF como cidade, capital, metropole ou municipio."
        ),
        nota_participacao=(
            "O campo participacao_total_ufs_pct representa participacao na populacao total "
            "somada das UFs retornadas, nao participacao na quantidade de UFs."
        ),
        ranking_2024=ranking,
        fontes=list(dict.fromkeys([state_data["fonte"], ranking_source, STATES_URL])),
        coletado_em=collected_at,
    )
