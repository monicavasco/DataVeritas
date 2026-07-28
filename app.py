from __future__ import annotations

import hashlib
import importlib
import re
import unicodedata
from dataclasses import asdict

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from dataveritas.bcb import SelicDataset
from dataveritas.collectors import (
    MORTALITY_RJ_SOURCE_TYPE,
    OPEN_DATA_SOURCE_TYPE,
    SELIC_SOURCE_TYPE,
    ARARUAMA_NEWS_SOURCE_TYPE,
    collect_dataset,
    collector_requires_state,
    get_collector_definition,
)
from dataveritas.guardrails import (
    CheckResult,
    deterministic_input_check,
    deterministic_output_check,
    nemo_check_input,
    nemo_check_output,
    validate_source_urls,
)
from dataveritas.ibge import StateOption, get_states
from dataveritas.llm_config import (
    CUSTOM_API_KEY_ENV,
    CUSTOM_BASE_URL_ENV,
    CUSTOM_MODEL_ENV,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_ENV,
    GOOGLE_API_KEY_ENV,
    OPENAI_API_KEY_ENV,
    OPENAI_MODEL_ENV,
    PROVIDER_CUSTOM,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    SUPPORTED_PROVIDERS,
    build_llm_settings,
    configured_provider,
    default_model_for,
    load_local_env,
    ollama_base_url,
    provider_label,
)
from dataveritas.mortality import MortalityDataset
from dataveritas.open_data import OpenDataDiscoveryDataset
from dataveritas.araruama import AraruamaDataset
from dataveritas.rag import (
    RagRecommendation,
    bootstrap_vector_store,
    recommend_source,
    remember_newsroom_run,
)


load_local_env()


st.set_page_config(
    page_title="DataVeritas",
    page_icon="DV",
    layout="wide",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    :root {
      --ink: #f0f6fc;
      --muted: #8b949e;
      --paper: #0d1117;
      --line: rgba(240, 246, 252, 0.12);
      --teal: #00ffcc;
      --amber: #f5a623;
      --brick: #ff0055;
      --mint: rgba(0, 255, 204, 0.1);
      --sidebar: #161b22;
      --gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
      --glow: 0 0 15px rgba(0, 242, 254, 0.35);
    }

    * {
      font-family: 'Outfit', sans-serif !important;
    }

    .stApp {
      background: radial-gradient(circle at 50% 50%, #172033 0%, #0d1117 100%);
      color: var(--ink);
    }

    [data-testid="stHeader"] {
      background: transparent;
    }

    [data-testid="stSidebar"] {
      background-color: var(--sidebar) !important;
      border-right: 1px solid var(--line);
      backdrop-filter: blur(15px);
    }

    .dv-title {
      font-size: 3.2rem;
      font-weight: 800;
      background: var(--gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.3rem;
      letter-spacing: -0.03em;
      text-shadow: 0 0 30px rgba(0, 242, 254, 0.2);
    }

    .dv-subtitle {
      color: var(--muted);
      font-size: 1.15rem;
      font-weight: 400;
      max-width: 800px;
      margin-bottom: 2rem;
      letter-spacing: -0.01em;
    }

    .metric-shell {
      border: 1px solid var(--line);
      background: rgba(22, 27, 34, 0.7);
      border-radius: 12px;
      padding: 1rem 1.25rem;
      min-height: 96px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      backdrop-filter: blur(5px);
    }

    .metric-shell:hover {
      transform: translateY(-4px);
      border-color: rgba(0, 242, 254, 0.4);
      box-shadow: 0 8px 30px rgba(0, 242, 254, 0.25);
    }

    .metric-label {
      color: var(--muted);
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.3rem;
    }

    .metric-value {
      color: var(--teal);
      font-size: 1.6rem;
      font-weight: 700;
      line-height: 1.2;
    }

    .guardrail-pass {
      border-left: 5px solid var(--teal);
      background: rgba(0, 255, 204, 0.06);
      padding: 0.9rem 1.1rem;
      border-radius: 8px;
      margin-bottom: 0.75rem;
      border-top: 1px solid rgba(0, 255, 204, 0.12);
      border-right: 1px solid rgba(0, 255, 204, 0.12);
      border-bottom: 1px solid rgba(0, 255, 204, 0.12);
      box-shadow: 0 4px 15px rgba(0, 255, 204, 0.03);
    }

    .guardrail-block {
      border-left: 5px solid var(--brick);
      background: rgba(255, 0, 85, 0.06);
      padding: 0.9rem 1.1rem;
      border-radius: 8px;
      margin-bottom: 0.75rem;
      border-top: 1px solid rgba(255, 0, 85, 0.12);
      border-right: 1px solid rgba(255, 0, 85, 0.12);
      border-bottom: 1px solid rgba(255, 0, 85, 0.12);
      box-shadow: 0 4px 15px rgba(255, 0, 85, 0.03);
    }

    /* Primary CTA buttons */
    div.stButton > button {
      background: var(--gradient) !important;
      color: #ffffff !important;
      font-weight: 700 !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 0.65rem 1.5rem !important;
      box-shadow: var(--glow) !important;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
      width: 100% !important;
      letter-spacing: 0.03em !important;
    }

    div.stButton > button:hover {
      transform: translateY(-2px) !important;
      box-shadow: 0 0 25px rgba(0, 242, 254, 0.65) !important;
    }

    div.stButton > button:active {
      transform: translateY(1px) !important;
    }

    /* Style tab bars */
    .stTabs [data-baseweb="tab-list"] {
      gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
      background-color: rgba(22, 27, 34, 0.4) !important;
      border: 1px solid var(--line) !important;
      border-radius: 8px 8px 0px 0px !important;
      color: var(--muted) !important;
      padding: 0.5rem 1.5rem !important;
      font-weight: 600 !important;
      transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
      color: var(--ink) !important;
      background-color: rgba(22, 27, 34, 0.7) !important;
    }

    .stTabs [aria-selected="true"] {
      background-color: rgba(22, 27, 34, 0.85) !important;
      border-bottom: 2px solid #00f2fe !important;
      color: #00f2fe !important;
      box-shadow: 0 4px 15px rgba(0, 242, 254, 0.08) !important;
    }

    /* Inputs/select boxes styling */
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > input {
      background-color: rgba(13, 17, 23, 0.8) !important;
    }

    /* Alerts styling */
    .stAlert {
      border-radius: 8px !important;
      border: 1px solid var(--line) !important;
      background-color: rgba(22, 27, 34, 0.6) !important;
    }

    /* Table headers styling */
    div[data-testid="stTable"] th {
      background-color: rgba(22, 27, 34, 0.8) !important;
      color: var(--teal) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def cached_states() -> list[StateOption]:
    return get_states()


@st.cache_data(ttl=1800)
def cached_structured_source_dataset(source_type: str, user_request: str, state: StateOption | None):
    return collect_dataset(source_type=source_type, user_request=user_request, state=state)


@st.cache_data(ttl=900)
def cached_discovery_source_dataset(source_type: str, user_request: str, state: StateOption | None):
    return collect_dataset(source_type=source_type, user_request=user_request, state=state)


def cached_source_dataset(source_type: str, user_request: str, state: StateOption | None):
    definition = get_collector_definition(source_type)
    if definition.cache_ttl_seconds <= 900:
        return cached_discovery_source_dataset(source_type, user_request, state)
    return cached_structured_source_dataset(source_type, user_request, state)


@st.cache_resource
def cached_rag_store():
    return bootstrap_vector_store()


@st.cache_data(ttl=300)
def cached_rag_recommendation(user_request: str) -> RagRecommendation:
    cached_rag_store()
    return recommend_source(user_request)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.casefold()


def state_from_request(
    user_request: str,
    states: list[StateOption],
    fallback: StateOption | None = None,
) -> StateOption | None:
    normalized_request = normalize_text(user_request)

    for state in states:
        if normalize_text(state.nome) in normalized_request:
            return state

    for state in states:
        pattern = rf"\b{re.escape(state.sigla.casefold())}\b"
        if re.search(pattern, normalized_request):
            return state

    return fallback


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def format_pct(value: float) -> str:
    return f"{value:.2f}%".replace(".", ",")


def format_pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f} p.p.".replace(".", ",")


def ollama_status() -> bool:
    try:
        response = requests.get(f"{ollama_base_url()}/api/tags", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def current_run_newsroom():
    import dataveritas.agent_tools as agent_tools
    import dataveritas.crew_pipeline as crew_pipeline

    importlib.reload(agent_tools)
    return importlib.reload(crew_pipeline).run_newsroom


def result_tool_executions(newsroom_result) -> list[dict]:
    executions = getattr(newsroom_result, "tool_executions", [])
    return executions if isinstance(executions, list) else []


def result_tool_names(newsroom_result) -> list[str]:
    names = getattr(newsroom_result, "tool_names", [])
    return names if isinstance(names, list) else []


@st.cache_data(ttl=30)
def available_ollama_models() -> list[str]:
    try:
        response = requests.get(f"{ollama_base_url()}/api/tags", timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return [default_model_for(PROVIDER_OLLAMA)]

    payload = response.json()
    models = sorted(
        model.get("name")
        for model in payload.get("models", [])
        if model.get("name")
    )
    return models or [default_model_for(PROVIDER_OLLAMA)]


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-shell">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_check(title: str, result: CheckResult) -> None:
    css_class = "guardrail-pass" if result.ok else "guardrail-block"
    body = "<br>".join(result.messages)
    st.markdown(
        f"""
        <div class="{css_class}">
          <strong>{title}</strong><br>{body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_ranking_frame(dataset) -> pd.DataFrame:
    rows = [asdict(row) for row in dataset.ranking_2024]
    df = pd.DataFrame(rows)
    df["selecionado"] = df["id"].eq(dataset.estado_id)
    return df


def build_selic_frame(dataset: SelicDataset) -> pd.DataFrame:
    rows = [asdict(row) for row in dataset.observacoes]
    df = pd.DataFrame(rows)
    df["data_dt"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    return df.sort_values("data_dt")


def build_open_data_frame(dataset: OpenDataDiscoveryDataset) -> pd.DataFrame:
    rows = [asdict(row) for row in dataset.registros]
    return pd.DataFrame(rows)


def build_araruama_frame(dataset: AraruamaDataset) -> pd.DataFrame:
    rows = [asdict(row) for row in dataset.registros]
    return pd.DataFrame(rows)


def build_mortality_frame(dataset: MortalityDataset) -> pd.DataFrame:
    rows = [asdict(row) for row in dataset.top_municipios]
    return pd.DataFrame(rows)


def run_pipeline(
    user_request: str,
    dataset,
    use_nemo: bool,
    use_agent_tools: bool,
    model_provider: str,
    model_name: str,
    rag_recommendation: RagRecommendation | None,
    model_api_key: str | None = None,
    model_base_url: str | None = None,
):
    checks: list[tuple[str, CheckResult]] = []

    input_result = deterministic_input_check(user_request)
    checks.append(("Entrada determinística", input_result))
    if not input_result.ok:
        return None, None, checks, rag_recommendation

    if use_nemo:
        try:
            nemo_input = nemo_check_input(user_request)
        except Exception as exc:  # NeMo should not hide deterministic results.
            nemo_input = CheckResult(ok=False, messages=[f"Erro no NeMo input rail: {exc}"])
        checks.append(("Entrada NeMo", nemo_input))
        if not nemo_input.ok:
            return None, None, checks, rag_recommendation

    source_result = validate_source_urls(dataset.fontes)
    checks.append(("Fontes públicas", source_result))
    if not source_result.ok:
        return dataset, None, checks, rag_recommendation

    model_settings = build_llm_settings(
        provider=model_provider,
        model_name=model_name,
        api_key=model_api_key or None,
        base_url=model_base_url or None,
    )
    model_messages = [f"Provedor: {model_settings.display_name}."]
    if model_settings.api_key_source:
        model_messages.append(f"Chave carregada de: {model_settings.api_key_source}.")
    if model_settings.base_url and model_settings.provider != PROVIDER_OLLAMA:
        model_messages.append(f"Base URL configurada: {model_settings.base_url}.")
    if model_settings.missing_configuration_message:
        model_messages.append(model_settings.missing_configuration_message)
    checks.append(("Modelo LLM", CheckResult(ok=model_settings.is_ready, messages=model_messages)))
    if not model_settings.is_ready:
        return dataset, None, checks, rag_recommendation

    run_newsroom = current_run_newsroom()
    try:
        newsroom_result = run_newsroom(
            user_request=user_request,
            dataset=dataset,
            model_name=model_name,
            model_provider=model_provider,
            enable_tools=use_agent_tools,
            rag_context=asdict(rag_recommendation) if rag_recommendation else None,
            api_key=model_api_key or None,
            base_url=model_base_url or None,
        )
    except Exception as exc:
        checks.append(
            (
                "ExecuÃ§Ã£o LLM",
                CheckResult(ok=False, messages=[f"Falha ao executar {model_settings.display_name}: {exc}"]),
            )
        )
        return dataset, None, checks, rag_recommendation
    if use_agent_tools:
        tool_executions = result_tool_executions(newsroom_result)
        failed_tools = [
            execution
            for execution in tool_executions
            if not execution.get("ok", False)
        ]
        if tool_executions:
            messages = [f"{len(tool_executions)} execução(ões) de tool registradas."]
            messages.extend(
                f"{execution.get('tool')} ({execution.get('papel')}) sinalizou falha ou risco."
                for execution in failed_tools[:5]
            )
            checks.append(("Tools determinísticas", CheckResult(ok=not failed_tools, messages=messages)))
        else:
            checks.append(
                (
                    "Tools determinísticas",
                    CheckResult(ok=False, messages=["Toggle ligado, mas nenhuma execução de tool foi registrada."]),
                )
            )

    if newsroom_result.repaired_footer:
        checks.append(
            (
                "Reparo de transparência",
                CheckResult(
                    ok=True,
                    messages=["Rodapé de fonte/isenção foi completado automaticamente."],
                    repaired=True,
                ),
            )
        )

    output_result = deterministic_output_check(newsroom_result.article, dataset.fontes)
    checks.append(("Saída determinística", output_result))
    if not output_result.ok:
        return dataset, newsroom_result, checks, rag_recommendation

    if use_nemo:
        try:
            nemo_output = nemo_check_output(newsroom_result.article)
        except Exception as exc:
            nemo_output = CheckResult(ok=False, messages=[f"Erro no NeMo output rail: {exc}"])
        checks.append(("Saída NeMo", nemo_output))

    try:
        remember_newsroom_run(
            query=user_request,
            source_type=getattr(dataset, "tipo", "unknown"),
            article=newsroom_result.article,
            source_urls=dataset.fontes,
            model_name=f"{model_settings.provider}:{model_settings.model_name}",
        )
        checks.append(
            (
                "Memória RAG",
                CheckResult(ok=True, messages=["Pauta e notícia registradas no banco vetorial local."]),
            )
        )
    except Exception as exc:
        checks.append(
            (
                "Memória RAG",
                CheckResult(ok=False, messages=[f"Não foi possível registrar memória RAG: {exc}"]),
            )
        )

    return dataset, newsroom_result, checks, rag_recommendation


def render_population_overview(dataset) -> None:
    ranking_df = build_ranking_frame(dataset)

    cols = st.columns(4)
    with cols[0]:
        render_metric("População 2024", format_int(dataset.populacao_final))
    with cols[1]:
        render_metric("Variação 2021-2024", format_pct(dataset.variacao_percentual))
    with cols[2]:
        render_metric("Ranking UF", f"{dataset.rank_uf_2024}º")
    with cols[3]:
        render_metric("Participação no total", format_pct(dataset.participacao_total_ufs_pct))

    top_df = ranking_df.head(10)
    fig = px.bar(
        top_df,
        x="populacao_2024",
        y="nome",
        orientation="h",
        color="selecionado",
        color_discrete_map={True: "#c27a28", False: "#1b7f78"},
        labels={"populacao_2024": "População estimada", "nome": ""},
    )
    fig.update_layout(
        showlegend=False,
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(color="#18211f"),
    )
    st.plotly_chart(fig, width="stretch")


def render_selic_overview(dataset: SelicDataset) -> None:
    cols = st.columns(4)
    with cols[0]:
        render_metric("Selic mais recente", format_pct(dataset.valor_final))
    with cols[1]:
        render_metric("Variação no recorte", format_pp(dataset.variacao_pontos_percentuais))
    with cols[2]:
        render_metric("Maior valor", format_pct(dataset.maior_valor))
    with cols[3]:
        render_metric("Menor valor", format_pct(dataset.menor_valor))

    selic_df = build_selic_frame(dataset)
    fig = px.line(
        selic_df,
        x="data_dt",
        y="valor",
        markers=True,
        labels={"data_dt": "Data", "valor": "Meta Selic (% a.a.)"},
    )
    fig.update_traces(line_color="#1b7f78", marker_color="#c27a28")
    fig.update_layout(
        showlegend=False,
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(color="#18211f"),
    )
    st.plotly_chart(fig, width="stretch")


def render_mortality_overview(dataset: MortalityDataset) -> None:
    cols = st.columns(4)
    with cols[0]:
        rate_label = f"{dataset.taxa_mortalidade_por_mil:.2f}".replace(".", ",")
        render_metric("Taxa bruta", f"{rate_label} por mil")
    with cols[1]:
        render_metric("Obitos residentes", format_int(dataset.obitos_residentes))
    with cols[2]:
        render_metric("Populacao usada", format_int(dataset.populacao_residente))
    with cols[3]:
        render_metric("Ano e escopo", f"{dataset.ano} · {dataset.localidade_tipo}")

    mortality_df = build_mortality_frame(dataset)
    if mortality_df.empty:
        st.info("Nao ha recorte municipal disponivel para exibir.")
        return

    fig = px.bar(
        mortality_df,
        x="obitos_residentes",
        y="municipio",
        orientation="h",
        color="participacao_obitos_pct",
        color_continuous_scale=["#e2f0ec", "#1b7f78", "#a54b3d"],
        labels={
            "obitos_residentes": "Obitos de residentes",
            "municipio": "",
            "participacao_obitos_pct": "Participacao (%)",
        },
    )
    fig.update_layout(
        showlegend=False,
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(color="#18211f"),
    )
    st.plotly_chart(fig, width="stretch")


def render_open_data_overview(dataset: OpenDataDiscoveryDataset) -> None:
    successful_portals = sum(1 for status in dataset.portais_consultados if status.ok)
    cols = st.columns(4)
    with cols[0]:
        render_metric("Registros encontrados", str(dataset.registros_encontrados))
    with cols[1]:
        render_metric("Portais com retorno", str(successful_portals))
    with cols[2]:
        render_metric("Fontes citáveis", str(len(dataset.fontes)))
    with cols[3]:
        render_metric("Tipo", "Descoberta")

    status_rows = [asdict(status) for status in dataset.portais_consultados]
    st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)


def render_araruama_overview(dataset: AraruamaDataset) -> None:
    cols = st.columns(4)
    with cols[0]:
        render_metric("Notícias encontradas", str(dataset.registros_encontrados))
    with cols[1]:
        render_metric("Fontes citáveis", str(len(dataset.fontes)))
    with cols[2]:
        render_metric("Tipo", "Notícias Locais")
    with cols[3]:
        render_metric("Status", "OK")


def render_dataset_overview(dataset) -> None:
    if getattr(dataset, "tipo", "") == SELIC_SOURCE_TYPE:
        render_selic_overview(dataset)
        return

    if getattr(dataset, "tipo", "") == MORTALITY_RJ_SOURCE_TYPE:
        render_mortality_overview(dataset)
        return

    if getattr(dataset, "tipo", "") == OPEN_DATA_SOURCE_TYPE:
        render_open_data_overview(dataset)
        return

    if getattr(dataset, "tipo", "") == ARARUAMA_NEWS_SOURCE_TYPE:
        render_araruama_overview(dataset)
        return

    render_population_overview(dataset)


def render_dataset_table(dataset) -> None:
    if getattr(dataset, "tipo", "") == SELIC_SOURCE_TYPE:
        selic_df = build_selic_frame(dataset)
        st.dataframe(
            selic_df[["data", "valor"]],
            width="stretch",
            hide_index=True,
        )
        return

    if getattr(dataset, "tipo", "") == MORTALITY_RJ_SOURCE_TYPE:
        mortality_df = build_mortality_frame(dataset)
        st.markdown("**Metodo de calculo**")
        st.write(dataset.nota_metodologica)
        st.markdown("**Escopo**")
        st.write(dataset.nota_escopo)
        if mortality_df.empty:
            st.info("Nao ha recorte municipal disponivel para exibir.")
        else:
            st.dataframe(
                mortality_df[["municipio", "obitos_residentes", "participacao_obitos_pct"]],
                width="stretch",
                hide_index=True,
        )
        return

    if getattr(dataset, "tipo", "") == OPEN_DATA_SOURCE_TYPE:
        records_df = build_open_data_frame(dataset)
        if records_df.empty:
            st.info("Nenhum registro validado foi encontrado para esta pauta.")
        else:
            st.dataframe(
                records_df[["origem", "titulo", "data", "tipo", "url"]],
                width="stretch",
                hide_index=True,
            )
        return

    if getattr(dataset, "tipo", "") == ARARUAMA_NEWS_SOURCE_TYPE:
        records_df = build_araruama_frame(dataset)
        if records_df.empty:
            st.info("Nenhuma notícia de Araruama foi encontrada para os termos pesquisados.")
        else:
            st.dataframe(
                records_df[["data", "titulo", "url"]],
                width="stretch",
                hide_index=True,
            )
        return

    ranking_df = build_ranking_frame(dataset)
    st.dataframe(
        ranking_df[["rank_2024", "nome", "populacao_2024", "selecionado"]],
        width="stretch",
        hide_index=True,
    )


def render_rag_context(recommendation: RagRecommendation | None) -> None:
    if recommendation is None:
        st.info("Contexto RAG não disponível nesta execução.")
        return

    cols = st.columns(3)
    with cols[0]:
        render_metric("Fonte sugerida", recommendation.source_label)
    with cols[1]:
        render_metric("Coletor", recommendation.collector)
    with cols[2]:
        render_metric("Similaridade", f"{recommendation.confidence:.2f}")

    st.markdown("**Consulta usada pelo RAG**")
    st.write(recommendation.query)

    st.markdown("**Documentos recuperados**")
    for index, match in enumerate(recommendation.matches, start=1):
        label = match.metadata.get("source_label", match.id)
        doc_kind = match.metadata.get("doc_kind", "document")
        with st.expander(f"{index}. {label} · {doc_kind} · score {match.score:.2f}"):
            st.write(match.content)
            st.json(match.metadata)


states = cached_states()

with st.sidebar:
    st.markdown("### DataVeritas")
    st.caption("Modo: RAG automático")
    selected_state = None

    default_request = "Gerar uma notícia de jornalismo de dados sobre juros no Brasil."
    user_request = st.text_area("Pauta", value=default_request, height=120)

    rag_recommendation = cached_rag_recommendation(user_request)
    st.caption(
        f"RAG sugeriu: {rag_recommendation.source_label} "
        f"(similaridade {rag_recommendation.confidence:.2f})"
    )
    active_source_type = rag_recommendation.source_type
    active_state = None
    active_dataset = None
    if collector_requires_state(active_source_type):
        selected_state = state_from_request(user_request, states)
        if selected_state:
            st.caption(f"UF usada pelo coletor: {selected_state.nome} ({selected_state.sigla})")
            active_state = selected_state
            active_dataset = cached_source_dataset(active_source_type, user_request, active_state)
        else:
            st.warning("A pauta precisa citar uma UF para usar o coletor do IBGE.")
    else:
        active_dataset = cached_source_dataset(active_source_type, user_request, active_state)

    use_nemo = st.toggle(
        "NeMo semântico",
        value=False,
        help="Usa o LLM local para checagens semânticas de entrada e saída. Pode levar mais tempo.",
    )
    use_agent_tools = st.toggle(
        "Tools determinísticas",
        value=True,
        help=(
            "Permite que os agentes CrewAI consultem ferramentas locais de auditoria, "
            "perfil do dataset, campos factuais e contexto RAG."
        ),
    )
    provider_options = list(SUPPORTED_PROVIDERS)
    try:
        configured_llm_provider = configured_provider()
    except ValueError as exc:
        st.warning(str(exc))
        configured_llm_provider = PROVIDER_OLLAMA
    default_provider_index = (
        provider_options.index(configured_llm_provider)
        if configured_llm_provider in provider_options
        else 0
    )
    selected_provider = st.selectbox(
        "Provedor LLM",
        options=provider_options,
        index=default_provider_index,
        format_func=provider_label,
        help="Escolha onde os agentes da CrewAI vao executar o modelo.",
    )
    model_api_key = None
    model_base_url = None
    if selected_provider == PROVIDER_OLLAMA:
        ollama_models = available_ollama_models()
        configured_ollama_model = default_model_for(PROVIDER_OLLAMA)
        default_model_index = (
            ollama_models.index(configured_ollama_model)
            if configured_ollama_model in ollama_models
            else 0
        )
        selected_model = st.selectbox(
            "Modelo Ollama",
            options=ollama_models,
            index=default_model_index,
            help="Modelo local usado pelos agentes da CrewAI.",
        )
    elif selected_provider == PROVIDER_GEMINI:
        selected_model = st.text_input(
            "Modelo Gemini",
            value=default_model_for(PROVIDER_GEMINI),
            help=f"Tambem pode ser definido por {GEMINI_MODEL_ENV}.",
        )
        model_api_key = st.text_input(
            "Chave Gemini temporaria",
            value="",
            type="password",
            help=(
                f"Opcional. Se vazio, usa {GEMINI_API_KEY_ENV} ou {GOOGLE_API_KEY_ENV} "
                "do .env/ambiente. Esta chave nao e salva no RAG."
            ),
        )
    elif selected_provider == PROVIDER_OPENAI:
        selected_model = st.text_input(
            "Modelo OpenAI",
            value=default_model_for(PROVIDER_OPENAI),
            help=f"Tambem pode ser definido por {OPENAI_MODEL_ENV}.",
        )
        model_api_key = st.text_input(
            "Chave OpenAI temporaria",
            value="",
            type="password",
            help=(
                f"Opcional. Se vazio, usa {OPENAI_API_KEY_ENV} "
                "do .env/ambiente. Esta chave nao e salva no RAG."
            ),
        )
    else:
        selected_model = st.text_input(
            "Modelo / provider id",
            value=default_model_for(PROVIDER_CUSTOM),
            placeholder="Ex.: openrouter/openai/gpt-5-mini ou deepseek/deepseek-chat",
            help=(
                f"Tambem pode ser definido por {CUSTOM_MODEL_ENV}. Use o prefixo do provider "
                "quando ele existir no CrewAI."
            ),
        )
        model_api_key = st.text_input(
            "Chave API custom temporaria",
            value="",
            type="password",
            help=(
                f"Opcional. Se vazio, usa {CUSTOM_API_KEY_ENV} ou a variavel do prefixo "
                "do modelo, como OPENROUTER_API_KEY ou DEEPSEEK_API_KEY."
            ),
        )
        model_base_url = st.text_input(
            "Base URL custom opcional",
            value="",
            placeholder="Ex.: https://api.exemplo.com/v1",
            help=f"Tambem pode ser definida por {CUSTOM_BASE_URL_ENV}.",
        )

    selected_model = selected_model.strip() or default_model_for(selected_provider)
    llm_preview_settings = build_llm_settings(
        provider=selected_provider,
        model_name=selected_model,
        api_key=model_api_key or None,
        base_url=model_base_url or None,
    )
    generate = st.button(
        "Gerar notícia",
        type="primary",
        width="stretch",
        disabled=active_dataset is None or not llm_preview_settings.is_ready,
    )

    st.divider()
    st.caption("Modelo")
    if selected_provider == PROVIDER_OLLAMA:
        if ollama_status():
            st.success("Ollama ativo")
        else:
            st.error("Ollama indisponível")
        st.caption(f"Endpoint: {ollama_base_url()}")
    elif selected_provider == PROVIDER_GEMINI:
        if llm_preview_settings.is_ready:
            st.success(f"Gemini configurado ({llm_preview_settings.api_key_source})")
        else:
            st.error("Chave Gemini ausente")
        st.caption(f"Configure {GEMINI_API_KEY_ENV} ou {GOOGLE_API_KEY_ENV} no .env.")
    elif selected_provider == PROVIDER_OPENAI:
        if llm_preview_settings.is_ready:
            st.success(f"OpenAI configurado ({llm_preview_settings.api_key_source})")
        else:
            st.error("Chave OpenAI ausente")
        st.caption(f"Configure {OPENAI_API_KEY_ENV} no .env.")
    else:
        if llm_preview_settings.is_ready:
            source = llm_preview_settings.api_key_source or "chave opcional"
            st.success(f"API custom configurada ({source})")
        else:
            st.error("Modelo ou chave custom ausente")
        st.caption(f"Configure {CUSTOM_MODEL_ENV}, {CUSTOM_API_KEY_ENV} e/ou {CUSTOM_BASE_URL_ENV}.")
        if llm_preview_settings.base_url:
            st.caption(f"Base URL: {llm_preview_settings.base_url}")
    st.caption(f"Modelo selecionado: {llm_preview_settings.model_name}")

st.markdown('<div class="dv-title">DataVeritas</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dv-subtitle">Redação automatizada local com CrewAI, Ollama, fontes públicas e guardrails auditáveis.</div>',
    unsafe_allow_html=True,
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

request_key = hashlib.sha1(user_request.encode("utf-8")).hexdigest()[:12]
source_key = active_source_type
selection_key = (
    f"rag_auto:{source_key}:{selected_state.sigla if selected_state else 'br'}:"
    f"{selected_provider}:{selected_model}:{use_agent_tools}:{request_key}"
)
if st.session_state.get("selection_key") != selection_key:
    st.session_state.last_result = None
    st.session_state.selection_key = selection_key

if generate:
    with st.spinner("Coletando dados, acionando agentes e validando a notícia..."):
        st.session_state.last_result = run_pipeline(
            user_request,
            active_dataset,
            use_nemo,
            use_agent_tools,
            selected_provider,
            selected_model,
            rag_recommendation,
            model_api_key or None,
            model_base_url or None,
        )

if not st.session_state.last_result:
    if active_dataset is None:
        st.info(
            "O RAG escolheu um coletor que precisa de UF. Inclua o estado na pauta, "
            "por exemplo: 'população estimada em Minas Gerais'."
        )
    else:
        render_dataset_overview(active_dataset)
else:
    dataset, newsroom_result, checks, result_rag_recommendation = st.session_state.last_result

    tabs = st.tabs(["Notícia", "Dados", "Guardrails", "RAG"])

    with tabs[0]:
        if newsroom_result:
            st.markdown(newsroom_result.article)
        else:
            st.warning("A notícia não foi gerada porque um guardrail bloqueou o fluxo.")

    with tabs[1]:
        if dataset:
            render_dataset_overview(dataset)
            render_dataset_table(dataset)

    with tabs[2]:
        for title, result in checks:
            render_check(title, result)

        if dataset:
            st.markdown("**Fontes usadas**")
            for source in dataset.fontes:
                st.markdown(f"- {source}")

        if newsroom_result:
            tool_executions = result_tool_executions(newsroom_result)
            tool_names = result_tool_names(newsroom_result)
            if tool_executions:
                st.markdown("**Tools determinísticas executadas obrigatoriamente**")
                execution_rows = [
                    {
                        "papel": execution.get("papel"),
                        "tool": execution.get("tool"),
                        "ok": execution.get("ok"),
                        "argumentos": execution.get("argumentos"),
                    }
                    for execution in tool_executions
                ]
                st.dataframe(pd.DataFrame(execution_rows), use_container_width=True, hide_index=True)

                with st.expander("Saídas das tools determinísticas"):
                    for index, execution in enumerate(tool_executions, start=1):
                        st.markdown(
                            f"**{index}. `{execution.get('tool')}` ({execution.get('papel')})**"
                        )
                        st.text(str(execution.get("saida", ""))[:3000])
            elif tool_names:
                st.warning("Tools determinísticas foram habilitadas, mas nenhuma execução obrigatória foi registrada.")
            else:
                st.info("Tools determinísticas não foram habilitadas nesta execução.")

        if newsroom_result and newsroom_result.task_outputs:
            with st.expander("Saídas intermediárias da CrewAI"):
                for index, output in enumerate(newsroom_result.task_outputs, start=1):
                    st.markdown(f"**Tarefa {index}**")
                    st.text(output[:3000])

    with tabs[3]:
        render_rag_context(result_rag_recommendation)
