from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dataveritas.guardrails import validate_source_urls


SELIC_SERIES_CODE = 432
SELIC_SOURCE_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs."
    f"{SELIC_SERIES_CODE}/dados?formato=json&dataInicial={{start_date}}&dataFinal={{end_date}}"
)


@dataclass(frozen=True)
class SelicObservation:
    data: str
    valor: float


@dataclass(frozen=True)
class SelicDataset:
    tipo: str
    indicador: str
    fonte_nome: str
    codigo_sgs: int
    data_referencia: str
    periodo_inicial: str
    periodo_final: str
    valor_inicial: float
    valor_final: float
    variacao_pontos_percentuais: float
    maior_valor: float
    menor_valor: float
    media_periodo: float
    resumo_numerico: str
    nota_metodologica: str
    observacao_temporal: str
    observacoes: list[SelicObservation]
    fontes: list[str]
    coletado_em: str

    def to_prompt_dict(self) -> dict:
        data = asdict(self)
        data["observacoes"] = [asdict(row) for row in self.observacoes]
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


def _parse_sgs_value(raw_value: str) -> float:
    return float(str(raw_value).replace(",", "."))


def _parse_sgs_date(raw_date: str) -> datetime:
    return datetime.strptime(raw_date, "%d/%m/%Y")


def _compact_change_points(observations: list[SelicObservation]) -> list[SelicObservation]:
    compacted: list[SelicObservation] = []
    last_value: float | None = None
    sorted_observations = sorted(observations, key=lambda item: _parse_sgs_date(item.data))

    for row in sorted_observations:
        if last_value is None or row.valor != last_value:
            compacted.append(row)
            last_value = row.valor

    if sorted_observations and compacted[-1].data != sorted_observations[-1].data:
        compacted.append(sorted_observations[-1])

    return compacted


def build_selic_dataset(days: int = 730) -> SelicDataset:
    today = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    start = today - timedelta(days=days)
    url = SELIC_SOURCE_URL.format(
        start_date=start.strftime("%d/%m/%Y"),
        end_date=today.strftime("%d/%m/%Y"),
    )
    payload = _get_json(url)
    if not payload:
        raise ValueError("A API do Banco Central nao retornou dados para a Selic.")

    daily_observations = [
        SelicObservation(data=item["data"], valor=_parse_sgs_value(item["valor"]))
        for item in payload
    ]
    observations = _compact_change_points(daily_observations)

    values = [row.valor for row in observations]
    first = observations[0]
    last = observations[-1]
    collected_at = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S %Z")

    return SelicDataset(
        tipo="selic_bcb",
        indicador="Meta da taxa Selic definida pelo Copom",
        fonte_nome="Banco Central do Brasil - Sistema Gerenciador de Series Temporais",
        codigo_sgs=SELIC_SERIES_CODE,
        data_referencia=today.isoformat(),
        periodo_inicial=first.data,
        periodo_final=last.data,
        valor_inicial=round(first.valor, 2),
        valor_final=round(last.valor, 2),
        variacao_pontos_percentuais=round(last.valor - first.valor, 2),
        maior_valor=round(max(values), 2),
        menor_valor=round(min(values), 2),
        media_periodo=round(sum(values) / len(values), 2),
        resumo_numerico=(
            f"No recorte, a Selic vai de {round(first.valor, 2)}% a.a. em {first.data} "
            f"para {round(last.valor, 2)}% a.a. em {last.data}, com variacao de "
            f"{round(last.valor - first.valor, 2)} ponto(s) percentual(is). "
            f"O menor valor observado foi {round(min(values), 2)}% a.a.; "
            f"o maior foi {round(max(values), 2)}% a.a.; "
            f"a media dos pontos de mudanca foi {round(sum(values) / len(values), 2)}% a.a."
        ),
        nota_metodologica=(
            "A serie SGS 432 e um dado publicado pelo Banco Central do Brasil. "
            "Nao descreva os valores coletados como previsoes ou projecoes, a menos "
            "que outra fonte do pacote declare explicitamente esse status."
        ),
        observacao_temporal=(
            "A data_referencia do pacote deve ser tratada como a data atual. "
            "O periodo_final nao passa da data_referencia, portanto o pacote nao contem datas futuras."
        ),
        observacoes=observations,
        fontes=[url],
        coletado_em=collected_at,
    )
