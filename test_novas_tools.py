"""
Teste isolado das duas novas tools: CrossUfComparisonTool e DatasetCompletionCheckTool.
"""
from __future__ import annotations

import json
import math
import sys


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, sort_keys=True)


def _lookup_path(data, field_path: str):
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


def _numeric_value(data: dict, *field_paths: str):
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


def run_comparar_ufs(dataset: dict, comparison_field: str = "populacao_2024") -> dict:
    if dataset.get("tipo") != "population_ibge":
        return {
            "ok": False,
            "mensagem": "Esta tool e valida apenas para datasets do tipo population_ibge.",
            "tipo_dataset": dataset.get("tipo"),
        }

    rows = dataset.get("ranking_2024", [])
    if not rows or not isinstance(rows, list):
        return {"ok": False, "mensagem": "Campo ranking_2024 nao encontrado ou vazio no pacote."}

    values: list[float] = []
    selected_item = None

    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _numeric_value(row, comparison_field)
        if value is None:
            continue
        values.append(value)
        if row.get("selecionado") or row.get("id") == dataset.get("estado_id"):
            selected_item = row

    if len(values) < 2:
        return {"ok": False, "mensagem": "Dados insuficientes para comparacao entre UFs."}

    mean = sum(values) / len(values)
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    median = (
        (sorted_values[mid - 1] + sorted_values[mid]) / 2
        if len(sorted_values) % 2 == 0
        else sorted_values[mid]
    )

    selected_value = _numeric_value(selected_item, comparison_field) if selected_item else None
    selected_name = selected_item.get("nome") if selected_item else None
    selected_rank = _numeric_value(selected_item, "rank_2024") if selected_item else None

    result: dict = {
        "ok": True,
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
        result["uf_selecionada"] = {
            "nome": selected_name,
            "rank": selected_rank,
            "valor": round(selected_value, 2),
            "distancia_da_media": round(distancia_media, 2),
            "pct_em_relacao_a_media": round(pct_acima_media, 2) if pct_acima_media is not None else None,
            "acima_da_media": distancia_media > 0,
        }
    else:
        result["aviso"] = "Estado selecionado nao identificado no ranking."

    return result


REQUIRED_BASE = ("tipo", "fontes", "coletado_em", "resumo_numerico", "nota_metodologica")
REQUIRED_BY_TYPE: dict[str, tuple] = {
    "mortality_rj": ("taxa_mortalidade_por_mil", "obitos_residentes", "populacao_residente"),
    "population_ibge": ("populacao_final", "variacao_percentual", "rank_uf_2024", "ranking_2024"),
    "selic_bcb": ("valor_final", "valor_inicial", "observacoes"),
    "ipca_ibge": ("variacao_acumulada_12m", "media_periodo", "observacoes"),
    "emendas_cgu": ("total_empenhado", "total_pago", "registros"),
    "worldbank_wdi": ("indicador_codigo", "valor_mais_recente", "observacoes"),
}


def run_verificar_completude(dataset: dict) -> dict:
    dataset_type = dataset.get("tipo") or ""
    required_fields = list(REQUIRED_BASE)
    extra_required = list(REQUIRED_BY_TYPE.get(dataset_type, ()))
    all_required = required_fields + extra_required

    present, absent = [], []
    for field in all_required:
        value = dataset.get(field)
        has_value = value is not None and value != "" and value != [] and value != {}
        (present if has_value else absent).append(field)

    completude_pct = round(len(present) / len(all_required) * 100, 1) if all_required else 100.0
    warnings = []
    if not dataset.get("fontes"):
        warnings.append("Pacote sem URLs de fontes citaveis.")
    if dataset_type == "open_data_discovery":
        warnings.append("Tipo open_data_discovery nao contem dados numericos estruturados.")

    return {
        "ok": not absent,
        "tipo_dataset": dataset_type,
        "completude_pct": completude_pct,
        "campos_presentes": present,
        "campos_ausentes": absent,
        "campos_obrigatorios_base": required_fields,
        "campos_obrigatorios_por_tipo": extra_required,
        "avisos": warnings,
    }


# ---------------------------------------------------------------------------
# Casos de teste
# ---------------------------------------------------------------------------

DATASET_IBGE_COMPLETO = {
    "tipo": "population_ibge",
    "fonte_nome": "IBGE",
    "fontes": ["https://servicodados.ibge.gov.br/api/v3/..."],
    "coletado_em": "2025-07-27 10:00:00 BRT",
    "resumo_numerico": "Populacao de Sao Paulo: 44.420.459 habitantes.",
    "nota_metodologica": "Estimativa IBGE 2024.",
    "estado_id": "35",
    "populacao_final": 44420459,
    "variacao_percentual": 1.2,
    "rank_uf_2024": 1,
    "ranking_2024": [
        {"id": "35", "nome": "Sao Paulo",      "populacao_2024": 44420459, "rank_2024": 1, "selecionado": True},
        {"id": "31", "nome": "Minas Gerais",   "populacao_2024": 21411923, "rank_2024": 2, "selecionado": False},
        {"id": "33", "nome": "Rio de Janeiro", "populacao_2024": 16054524, "rank_2024": 3, "selecionado": False},
        {"id": "29", "nome": "Bahia",          "populacao_2024": 14659023, "rank_2024": 4, "selecionado": False},
        {"id": "41", "nome": "Parana",         "populacao_2024": 11516841, "rank_2024": 5, "selecionado": False},
    ],
}

DATASET_MORTALITY_INCOMPLETO = {
    "tipo": "mortality_rj",
    "fontes": [],
    "coletado_em": "2025-07-27",
    # faltam: resumo_numerico, nota_metodologica, taxa_mortalidade_por_mil, obitos_residentes, populacao_residente
}

DATASET_SELIC_BASICO = {
    "tipo": "selic_bcb",
    "fontes": ["https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"],
    "coletado_em": "2025-07-27",
    "resumo_numerico": "Selic: 13,25%",
    "nota_metodologica": "Meta Selic definida pelo Copom.",
    "valor_final": 13.25,
    "valor_inicial": 10.5,
    "observacoes": [{"data": "01/07/2025", "valor": 13.25}],
}


def cabecalho(titulo: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")


def mostrar(resultado: dict) -> None:
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


def checar(condicao: bool, descricao: str) -> None:
    status = "PASSOU" if condicao else "FALHOU"
    simbolo = "✓" if condicao else "✗"
    print(f"  {simbolo} [{status}] {descricao}")
    if not condicao:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Execucao dos testes
# ---------------------------------------------------------------------------

falhas = 0

cabecalho("Tool A — comparar_ufs: dataset population_ibge completo")
r = run_comparar_ufs(DATASET_IBGE_COMPLETO)
mostrar(r)
checar(r["ok"] is True, "ok == True")
checar(r["total_ufs_no_ranking"] == 5, "5 UFs no ranking")
checar(r["uf_selecionada"]["nome"] == "Sao Paulo", "estado selecionado identificado")
checar(r["uf_selecionada"]["rank"] == 1.0, "rank_uf == 1")
checar(r["uf_selecionada"]["acima_da_media"] is True, "SP acima da media (e o maior)")
media_esperada = (44420459 + 21411923 + 16054524 + 14659023 + 11516841) / 5
checar(abs(r["media_nacional"] - round(media_esperada, 2)) < 1, "media calculada corretamente")

cabecalho("Tool A — comparar_ufs: tipo errado (selic_bcb)")
r2 = run_comparar_ufs({"tipo": "selic_bcb"})
mostrar(r2)
checar(r2["ok"] is False, "ok == False para tipo errado")
checar("population_ibge" in r2["mensagem"], "mensagem explica o tipo esperado")

cabecalho("Tool A — comparar_ufs: ranking vazio")
r3 = run_comparar_ufs({"tipo": "population_ibge", "ranking_2024": []})
mostrar(r3)
checar(r3["ok"] is False, "ok == False para ranking vazio")

cabecalho("Tool A — comparar_ufs: sem estado selecionado marcado")
dataset_sem_sel = {**DATASET_IBGE_COMPLETO}
rows_sem_sel = [
    {**row, "selecionado": False} for row in DATASET_IBGE_COMPLETO["ranking_2024"]
]
dataset_sem_sel = {**DATASET_IBGE_COMPLETO, "ranking_2024": rows_sem_sel, "estado_id": "99"}
r4 = run_comparar_ufs(dataset_sem_sel)
mostrar(r4)
checar(r4["ok"] is True, "ok == True mesmo sem estado marcado")
checar("aviso" in r4, "aviso emitido quando estado nao identificado")

cabecalho("Tool C — verificar_completude: population_ibge completo")
rc1 = run_verificar_completude(DATASET_IBGE_COMPLETO)
mostrar(rc1)
checar(rc1["ok"] is True, "ok == True para pacote completo")
checar(rc1["completude_pct"] == 100.0, "completude 100%")
checar(len(rc1["campos_ausentes"]) == 0, "nenhum campo ausente")

cabecalho("Tool C — verificar_completude: mortality_rj incompleto")
rc2 = run_verificar_completude(DATASET_MORTALITY_INCOMPLETO)
mostrar(rc2)
checar(rc2["ok"] is False, "ok == False para pacote incompleto")
checar("taxa_mortalidade_por_mil" in rc2["campos_ausentes"], "campo especifico de mortality ausente")
checar("resumo_numerico" in rc2["campos_ausentes"], "resumo_numerico ausente")
checar(len(rc2["avisos"]) > 0, "aviso de fontes emitido")
checar(rc2["completude_pct"] < 100.0, "completude abaixo de 100%")

cabecalho("Tool C — verificar_completude: selic_bcb completo")
rc3 = run_verificar_completude(DATASET_SELIC_BASICO)
mostrar(rc3)
checar(rc3["ok"] is True, "ok == True para selic completo")
checar(rc3["completude_pct"] == 100.0, "completude 100%")

cabecalho("Tool C — verificar_completude: tipo desconhecido (sem campos extras)")
rc4 = run_verificar_completude({"tipo": "unknown", "fontes": ["https://gov.br/x"],
                                 "coletado_em": "2025-07-27", "resumo_numerico": "x",
                                 "nota_metodologica": "x"})
mostrar(rc4)
checar(rc4["ok"] is True, "ok == True para tipo sem campos extras obrigatorios")
checar(rc4["campos_obrigatorios_por_tipo"] == [], "sem campos extras para tipo desconhecido")

print(f"\n{'='*60}")
print("  TODOS OS TESTES PASSARAM")
print(f"{'='*60}\n")
