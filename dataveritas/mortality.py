from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dataveritas.guardrails import validate_source_urls


RJ_STATE_ID = 33
RIO_MUNICIPALITY_ID = 3304557
MIN_SIM_YEAR = 1996
RJ_VITAL_STATS_PAGE = (
    "https://www.saude.rj.gov.br/informacao-sus/dados-sus/2020/10/"
    "estatisticas-vitais-obitos-e-nascimentos-sim-e-sinasc"
)
SIM_SOURCE_URL = "https://www.gov.br/saude/pt-br/composicao/svsa/sistemas-de-informacao/sim"
DEATHS_CSV_URL_TEMPLATE = "https://sistemas.saude.rj.gov.br/tabnetbd/sim/DadosCSV/doerj{year}.zip"
IBGE_POPULATION_URL_TEMPLATE = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/"
    "{year}/variaveis/9324?localidades={level}[{locality_id}]"
)


@dataclass(frozen=True)
class MortalityBreakdownRow:
    municipio: str
    obitos_residentes: int
    participacao_obitos_pct: float


@dataclass(frozen=True)
class MortalityDataset:
    tipo: str
    fonte_nome: str
    escopo_geografico: str
    localidade_id: int
    localidade_nome: str
    localidade_tipo: str
    ano: int
    periodo_inicial: int
    periodo_final: int
    obitos_residentes: int
    populacao_residente: int
    taxa_mortalidade_por_mil: float
    taxa_mortalidade_percentual: float
    top_municipios: list[MortalityBreakdownRow]
    fontes: list[str]
    resumo_numerico: str
    nota_metodologica: str
    nota_escopo: str
    coletado_em: str

    def to_prompt_dict(self) -> dict:
        data = asdict(self)
        data["top_municipios"] = [asdict(row) for row in self.top_municipios]
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


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").strip())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


def _validate_urls(urls: list[str]) -> None:
    source_check = validate_source_urls(urls)
    if not source_check.ok:
        raise ValueError("; ".join(source_check.messages))


def _extract_year(user_request: str, fallback_year: int = 2021) -> int:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", user_request)
    if not match:
        return fallback_year

    year = int(match.group(1))
    current_year = datetime.now(ZoneInfo("America/Sao_Paulo")).year
    if year < MIN_SIM_YEAR or year > current_year:
        raise ValueError(
            f"A base de obitos do SIM/RJ neste prototipo cobre anos entre {MIN_SIM_YEAR} e {current_year}."
        )
    return year


def _infer_scope(user_request: str) -> str:
    normalized = _normalize(user_request)
    municipality_terms = (
        "cidade do rio",
        "cidade de rio",
        "municipio do rio",
        "municipio de rio",
        "capital fluminense",
        "capital do rio",
    )
    if any(term in normalized for term in municipality_terms):
        return "municipality"
    return "state"


def _deaths_csv_url(year: int) -> str:
    return DEATHS_CSV_URL_TEMPLATE.format(year=year)


def _population_url(year: int, level: str, locality_id: int) -> str:
    return IBGE_POPULATION_URL_TEMPLATE.format(year=year, level=level, locality_id=locality_id)


def _read_death_counts(year: int) -> tuple[int, dict[str, int], str]:
    csv_url = _deaths_csv_url(year)
    _validate_urls([RJ_VITAL_STATS_PAGE, csv_url, SIM_SOURCE_URL])

    response = _session().get(csv_url, timeout=90)
    if response.status_code == 404:
        raise ValueError(f"O arquivo de obitos do SIM/RJ para {year} nao foi encontrado.")
    response.raise_for_status()

    wanted_columns = {
        "uf de residencia - sigla",
        "municipio de residencia",
        "residente no erj",
        "obito fetal",
    }
    state_total = 0
    municipality_counts: dict[str, int] = {}

    try:
        zip_buffer = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer) as zip_file:
            csv_names = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("O ZIP do SIM/RJ nao contem arquivo CSV.")

            with zip_file.open(csv_names[0]) as csv_file:
                chunks = pd.read_csv(
                    csv_file,
                    sep=";",
                    encoding="latin1",
                    usecols=lambda column: _normalize(column) in wanted_columns,
                    chunksize=50000,
                )
                for chunk in chunks:
                    chunk = chunk.rename(columns=lambda column: _normalize(column))
                    resident_mask = _resident_mask(chunk)
                    non_fetal_mask = _non_fetal_mask(chunk)
                    valid_rows = chunk.loc[resident_mask & non_fetal_mask]
                    state_total += len(valid_rows)

                    counts = (
                        valid_rows["municipio de residencia"]
                        .fillna("Nao informado")
                        .astype(str)
                        .str.strip()
                        .replace("", "Nao informado")
                        .value_counts()
                    )
                    for municipality, count in counts.items():
                        municipality_counts[municipality] = municipality_counts.get(municipality, 0) + int(count)
    except zipfile.BadZipFile as exc:
        raise ValueError("A resposta do SIM/RJ nao estava em formato ZIP valido.") from exc

    if state_total == 0:
        raise ValueError("A base do SIM/RJ nao retornou obitos de residentes no ERJ para o ano solicitado.")

    return state_total, municipality_counts, csv_url


def _resident_mask(frame: pd.DataFrame) -> pd.Series:
    if "residente no erj" in frame.columns:
        return frame["residente no erj"].astype(str).map(_normalize).eq("sim")
    if "uf de residencia - sigla" in frame.columns:
        return frame["uf de residencia - sigla"].astype(str).str.strip().str.upper().eq("RJ")
    raise ValueError("A base do SIM/RJ nao contem coluna de residencia.")


def _non_fetal_mask(frame: pd.DataFrame) -> pd.Series:
    if "obito fetal" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return ~frame["obito fetal"].astype(str).map(_normalize).eq("sim")


def _fetch_population(year: int, level: str, locality_id: int) -> tuple[int, str, str]:
    url = _population_url(year, level, locality_id)
    _validate_urls([url])

    response = _session().get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    try:
        series = payload[0]["resultados"][0]["series"][0]
        local_name = series["localidade"]["nome"]
        raw_value = series["serie"][str(year)]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("Resposta inesperada da API do IBGE para populacao.") from exc

    if raw_value in (None, "", "-"):
        raise ValueError(f"Valor de populacao ausente no IBGE para {year}.")

    population = int(float(str(raw_value).replace(",", ".")))
    return population, local_name, url


def _top_municipalities(
    municipality_counts: dict[str, int],
    denominator: int,
    limit: int = 10,
) -> list[MortalityBreakdownRow]:
    rows = []
    for municipality, deaths in sorted(municipality_counts.items(), key=lambda item: item[1], reverse=True)[:limit]:
        share = (deaths / denominator) * 100 if denominator else 0.0
        rows.append(
            MortalityBreakdownRow(
                municipio=municipality,
                obitos_residentes=deaths,
                participacao_obitos_pct=round(share, 2),
            )
        )
    return rows


def _municipality_deaths(municipality_counts: dict[str, int], municipality_name: str) -> int:
    normalized_target = _normalize(municipality_name)
    for municipality, deaths in municipality_counts.items():
        if _normalize(municipality) == normalized_target:
            return deaths
    return 0


def build_mortality_dataset(user_request: str, fallback_year: int = 2021) -> MortalityDataset:
    year = _extract_year(user_request, fallback_year=fallback_year)
    scope = _infer_scope(user_request)
    state_deaths, municipality_counts, deaths_source_url = _read_death_counts(year)

    if scope == "municipality":
        locality_id = RIO_MUNICIPALITY_ID
        locality_type = "Municipio"
        deaths = _municipality_deaths(municipality_counts, "Rio de Janeiro")
        population, local_name, population_source_url = _fetch_population(year, "N6", locality_id)
        scope_note = (
            "Escopo: municipio do Rio de Janeiro. O numerador usa obitos de residentes no municipio; "
            "o denominador usa a populacao residente estimada pelo IBGE para o municipio."
        )
    else:
        locality_id = RJ_STATE_ID
        locality_type = "Unidade da Federacao"
        deaths = state_deaths
        population, local_name, population_source_url = _fetch_population(year, "N3", locality_id)
        scope_note = (
            "Escopo: estado do Rio de Janeiro. O numerador usa obitos de residentes no ERJ; "
            "o denominador usa a populacao residente estimada pelo IBGE para a UF."
        )

    if deaths == 0:
        raise ValueError("Nao foram encontrados obitos para o escopo solicitado.")

    rate_per_thousand = (deaths / population) * 1000 if population else 0.0
    rate_percent = (deaths / population) * 100 if population else 0.0
    collected_at = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S %Z")
    methodology_note = (
        "Taxa bruta de mortalidade calculada como obitos de residentes divididos pela populacao residente "
        "estimada, multiplicado por 1.000. O indicador nao ajusta por idade, sexo, causa de morte ou "
        "estrutura demografica."
    )
    if year >= datetime.now(ZoneInfo("America/Sao_Paulo")).year - 1:
        methodology_note += " Anos recentes podem estar sujeitos a atualizacoes na base."

    source_urls = list(
        dict.fromkeys([RJ_VITAL_STATS_PAGE, deaths_source_url, population_source_url, SIM_SOURCE_URL])
    )
    _validate_urls(source_urls)

    return MortalityDataset(
        tipo="mortality_rj",
        fonte_nome="SIM/SES-RJ e IBGE",
        escopo_geografico=scope_note,
        localidade_id=locality_id,
        localidade_nome=local_name,
        localidade_tipo=locality_type,
        ano=year,
        periodo_inicial=year,
        periodo_final=year,
        obitos_residentes=deaths,
        populacao_residente=population,
        taxa_mortalidade_por_mil=round(rate_per_thousand, 2),
        taxa_mortalidade_percentual=round(rate_percent, 3),
        top_municipios=_top_municipalities(municipality_counts, state_deaths),
        fontes=source_urls,
        resumo_numerico=(
            f"Em {year}, {local_name} registrou {deaths} obitos de residentes. "
            f"A populacao residente estimada pelo IBGE era de {population} pessoas. "
            f"A taxa bruta de mortalidade foi de {round(rate_per_thousand, 2)} obitos por mil habitantes "
            f"({round(rate_percent, 3)}%)."
        ),
        nota_metodologica=methodology_note,
        nota_escopo=scope_note,
        coletado_em=collected_at,
    )
