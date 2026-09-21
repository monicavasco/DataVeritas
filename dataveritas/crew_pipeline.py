from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from crewai import Agent, Crew, LLM, Process, Task

from dataveritas.agent_tools import (
    NewsroomToolset,
    build_newsroom_toolset,
    format_tool_executions_for_prompt,
    generate_web_story,
    run_post_article_tools,
    run_required_tools,
    tool_execution_dicts,
    tool_names,
)
from dataveritas.guardrails import REQUIRED_DISCLAIMER, repair_article_footer
from dataveritas.llm_config import PROVIDER_OLLAMA, build_llm_settings


os.environ.setdefault("OTEL_SDK_DISABLED", "true")


class PromptDataset(Protocol):
    fontes: list[str]

    def to_prompt_dict(self) -> dict:
        ...


@dataclass(frozen=True)
class NewsroomResult:
    article: str
    web_story: dict[str, Any]
    raw_output: str
    task_outputs: list[str]
    repaired_footer: bool
    tool_names: list[str]
    tool_executions: list[dict[str, Any]]


def _build_llm(
    model_provider: str = PROVIDER_OLLAMA,
    model_name: str | None = None,
    temperature: float = 0.2,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLM:
    settings = build_llm_settings(
        provider=model_provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    if not settings.is_ready:
        raise ValueError(settings.missing_configuration_message or "Configuracao de LLM incompleta.")
    return LLM(**settings.to_crewai_kwargs())


def _agents(llm: LLM, toolset: NewsroomToolset | None = None) -> tuple[Agent, Agent, Agent]:
    collector = Agent(
        role="Agente Coletor de Dados Públicos",
        goal="Conferir se o pacote de dados tem fonte pública, verificável e suficiente para uma notícia didática.",
        backstory=(
            "Você trabalha na DataVeritas e só aceita dados de fontes públicas. "
            "Sua prioridade é preservar a rastreabilidade: fonte, período, limitações e origem dos números."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        tools=toolset.collector if toolset else [],
        max_iter=4 if toolset else 2,
    )

    analyst = Agent(
        role="Agente Analista de Jornalismo de Dados",
        goal="Encontrar padrões, variações e contexto quantitativo sem extrapolar além dos dados recebidos.",
        backstory=(
            "Você transforma dados públicos em achados jornalísticos claros. "
            "Quando há limitação metodológica ou temporal, você explicita a limitação."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        tools=toolset.analyst if toolset else [],
        max_iter=4 if toolset else 2,
    )

    writer = Agent(
        role="Agente Redator Jornalístico",
        goal="Escrever uma notícia curta, transparente e baseada apenas nos achados validados.",
        backstory=(
            "Você escreve para leitores gerais, com linguagem jornalística em português do Brasil. "
            "Você nunca inventa números, fontes ou citações."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        tools=toolset.writer if toolset else [],
        max_iter=4 if toolset else 2,
    )

    return collector, analyst, writer


def _temporal_context(data: dict) -> str:
    reference_date = data.get("data_referencia") or data.get("coletado_em")
    final_period = data.get("periodo_final")
    if reference_date and final_period:
        return (
            f"Considere {reference_date} como a data atual desta execução. "
            f"O período final do pacote é {final_period}. "
            "Não trate datas iguais ou anteriores à data de referência como futuro, previsão ou projeção."
        )
    if reference_date:
        return (
            f"Considere {reference_date} como a data atual desta execução. "
            "Não use sua data interna para reclassificar o pacote."
        )
    return "Use apenas a temporalidade declarada no pacote de dados."


def run_newsroom(
    user_request: str,
    dataset: PromptDataset,
    model_name: str | None = None,
    model_provider: str = PROVIDER_OLLAMA,
    temperature: float = 0.2,
    enable_tools: bool = True,
    rag_context: dict | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NewsroomResult:
    llm = _build_llm(
        model_provider=model_provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    dataset_dict = dataset.to_prompt_dict()
    agent_toolset = (
        build_newsroom_toolset(dataset_dict, dataset.fontes, rag_context, user_request=user_request)
        if enable_tools
        else None
    )
    forced_toolset = (
        build_newsroom_toolset(dataset_dict, dataset.fontes, rag_context, user_request=user_request)
        if enable_tools
        else None
    )
    forced_tool_executions = run_required_tools(forced_toolset, user_request=user_request)
    forced_tool_context = format_tool_executions_for_prompt(forced_tool_executions)
    collector, analyst, writer = _agents(llm, agent_toolset)
    data_json = json.dumps(dataset_dict, ensure_ascii=False, indent=2)
    source_list = "\n".join(f"- {url}" for url in dataset.fontes)
    temporal_context = _temporal_context(dataset_dict)
    numeric_summary = dataset_dict.get("resumo_numerico", "Não há resumo numérico textual adicional.")

    collect_task = Task(
        description=(
            "Revise o pacote de dados abaixo para a pauta '{user_request}'. "
            "Contexto temporal obrigatório: {temporal_context} "
            "Atue como o Agente Coletor: confirme nome da fonte, URLs originais, "
            "período disponível, campos coletados, limites metodológicos e se os "
            "domínios respeitam a política de fontes públicas. "
            "Se o contexto de tools indicar execucao habilitada, descreva o resultado "
            "ja executado de auditar_fontes_publicas, perfil_pacote_dados, "
            "ver_contexto_rag, consultar_catalogo_fontes, avaliar_qualidade_fonte, "
            "verificar_atualizacao_fonte, classificar_tipo_de_pauta, "
            "selecionar_fonte_primaria e detectar_fonte_secundaria_ou_inadequada. "
            "Use consultar_catalogo_fontes apenas para indicar fontes candidatas, "
            "cobertura e limitacoes; nao transforme catalogo em estatistica factual. "
            "Informe fonte escolhida, motivo, limitacao e status do coletor quando "
            "selecionar_fonte_primaria estiver disponivel. "
            "Quando aparecer buscar_portais_publicos no contexto, trate essa saida como "
            "a busca auditavel em portais publicos oficiais para a pauta. "
            "Quando aparecer baixar_recurso_tabular no contexto, descreva se a tool "
            "baixou um recurso remoto ou reutilizou a tabela do pacote atual. "
            "Use a data_referencia ou coletado_em do pacote como referência temporal. "
            "Não busque fontes externas e não invente valores.\n\n"
            "Contexto de tools deterministicas desta execucao:\n{forced_tool_context}\n\n"
            "Pacote de dados:\n{data_json}"
        ),
        expected_output=(
            "Relatório objetivo de coleta com fonte original, período, dados disponíveis "
            "e limitações para checagem."
        ),
        agent=collector,
        tools=agent_toolset.collector if agent_toolset else [],
    )

    analysis_task = Task(
        description=(
            "Com base apenas no pacote de dados e no resumo do Coletor, gere insights. "
            "Contexto de tools deterministicas desta execucao:\n{forced_tool_context}\n"
            "Resumo numérico disponível: {numeric_summary} "
            "Identifique padrões compatíveis com o tipo de indicador recebido: variações, "
            "posição relativa, máximos, mínimos, estabilidade ou tendência quando esses "
            "campos existirem no pacote. "
            "Se o contexto de tools indicar execucao habilitada, use os resultados ja "
            "executados de perfil_pacote_dados, consultar_campo_pacote e "
            "perfil_estatistico_dataset como confirmacao factual. Quando existirem, "
            "priorize calcular_variacao_percentual, calcular_taxa, comparar_periodos, "
            "gerar_ranking, detectar_outliers e validar_formula_indicador antes de "
            "redigir qualquer insight quantitativo. "
            "Não use causalidade sem evidência."
        ),
        expected_output=(
            "Lista curta de insights verificáveis, com números e cautelas metodológicas."
        ),
        agent=analyst,
        context=[collect_task],
        tools=agent_toolset.analyst if agent_toolset else [],
    )

    writing_task = Task(
        description=(
            "Escreva uma notícia automática em português sobre a pauta '{user_request}'. "
            "Contexto temporal obrigatório: {temporal_context} "
            "Contexto de tools deterministicas desta execucao:\n{forced_tool_context}\n"
            "Resumo numérico disponível: {numeric_summary} "
            "Use apenas os dados e insights fornecidos. A notícia deve ter título, lead, "
            "dois ou três parágrafos de corpo, seção 'Como checamos', seção "
            "'Auditoria factual' e seção 'Fontes originais'. "
            "Na seção 'Auditoria factual', responda exatamente estes quatro itens: "
            "'Quais números usei?', 'De onde vieram?', 'Qual é a limitação?' e "
            "'O que eu não posso concluir?'. "
            "Não invente citações, causas, projeções, números ou fontes. "
            "Não descreva dados publicados como projeções ou previsões se o pacote não declarar isso. "
            "Nunca diga que valores numéricos estão ausentes quando o pacote contém campos numéricos "
            "como valor_inicial, valor_final, resumo_numerico, maior_valor, menor_valor, media_periodo, "
            "obitos_residentes, populacao_residente ou taxa_mortalidade_por_mil. "
            "Quando resumo_numerico existir, use esses números no texto. "
            "Se o contexto de tools indicar execucao habilitada, use os resultados ja "
            "executados de consultar_campo_pacote e auditar_fontes_publicas para conferir "
            "numeros centrais e URLs antes da versao final. "
            "O artigo final sera auditado por checador_afirmacoes_artigo, checador_causalidade, "
            "checador_temporal, checador_escopo_geografico, checador_fontes_citadas, "
            "revisor_transparencia e matriz_evidencias_redacao; por isso, "
            "inclua apenas numeros presentes no pacote ou nas saidas deterministicas, evite "
            "causalidade sem evidencia, preserve o periodo/escopo e mantenha as secoes de "
            "transparencia completas. "
            "Quando escopo_geografico existir, respeite esse escopo e não confunda UF/estado "
            "com cidade, capital, município ou região metropolitana. "
            "Quando usar participacao_total_ufs_pct, descreva como participação na população total "
            "somada das UFs retornadas, não como participação na quantidade de UFs. "
            "Se o pacote tiver tipo open_data_discovery, trate-o como descoberta de registros e fontes "
            "públicas: descreva o que foi encontrado, os portais consultados e as limitações, sem "
            "inventar estatísticas agregadas ou conclusões causais. "
            "Se o pacote tiver tipo mortality_rj, descreva taxa_mortalidade_por_mil como taxa bruta "
            "de mortalidade por mil habitantes, citando obitos_residentes e populacao_residente como "
            "numerador e denominador. Não trate a taxa bruta como risco individual nem como taxa "
            "ajustada por idade. "
            "Use a data_referencia ou coletado_em do pacote como a referência temporal da notícia. "
            "Se um campo de ranking, participação ou variação não existir no pacote, não mencione esse dado. "
            "Inclua exatamente estas fontes:\n{source_list}\n\n"
            f'Inclua a frase obrigatória exatamente assim: "{REQUIRED_DISCLAIMER}".'
        ),
        expected_output=(
            "Notícia em Markdown, com fontes originais em URL e frase de isenção obrigatória."
        ),
        agent=writer,
        context=[collect_task, analysis_task],
        tools=agent_toolset.writer if agent_toolset else [],
        markdown=True,
    )

    crew = Crew(
        agents=[collector, analyst, writer],
        tasks=[collect_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
    )

    result = crew.kickoff(
        inputs={
            "user_request": user_request,
            "data_json": data_json,
            "source_list": source_list,
            "temporal_context": temporal_context,
            "numeric_summary": numeric_summary,
            "forced_tool_context": forced_tool_context,
        }
    )

    article = getattr(result, "raw", None) or str(result)
    article, repaired = repair_article_footer(article, dataset.fontes)
    post_tool_executions = run_post_article_tools(dataset_dict, article) if enable_tools else []
    tool_executions = [*forced_tool_executions, *post_tool_executions]
    task_outputs = [str(output) for output in getattr(result, "tasks_output", [])]

    return NewsroomResult(
        article=article,
        web_story=generate_web_story(article),
        raw_output=str(result),
        task_outputs=task_outputs,
        repaired_footer=repaired,
        tool_names=tool_names(agent_toolset),
        tool_executions=tool_execution_dicts(tool_executions),
    )
