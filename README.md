# DataVeritas

Plataforma de **jornalismo de dados automatizado** com foco em transparência, rastreabilidade e fontes públicas brasileiras. Combina agentes de IA (CrewAI), guardrails determinísticos e um banco vetorial local (ChromaDB) para gerar notícias verificáveis a partir de dados oficiais — sem inventar números, fontes ou causas.


## Finalidade

A partir de uma pauta em texto livre (ex: *"população estimada em Minas Gerais"*), o sistema:

1. Consulta um banco vetorial RAG para identificar a fonte pública mais adequada
2. Coleta os dados via API oficial
3. Passa o pacote por uma cadeia de três agentes CrewAI em sequência:
   - **Coletor** — valida origem, período e domínios permitidos
   - **Analista** — extrai padrões quantitativos sem extrapolar
   - **Redator** — escreve a notícia com seções obrigatórias de transparência
4. Aplica guardrails determinísticos e semânticos (NeMo opcional) na entrada e na saída
5. Exibe a notícia com auditoria completa: fontes, tools executadas e verificações


## Fontes de dados integradas

| Fonte | Tipo | Cobertura |
|---|---|---|
| IBGE | Estruturado | População estimada por UF (2024) |
| Banco Central | Estruturado | Taxa Selic — série histórica |
| SIM / SES-RJ + IBGE | Estruturado | Mortalidade no Rio de Janeiro |
| dados.gov.br, OMS GHO, ONU SDG, Câmara | Descoberta | Metadados de fontes abertas |

O catálogo RAG inclui ainda: DATASUS, INEP, Ipeadata, TSE, Portal da Transparência, Senado, IBGE SIDRA, Banco Mundial e OCDE — prontos para receber coletores estruturados.


## Tools determinísticas dos agentes

Os agentes CrewAI têm acesso a 20+ ferramentas auditáveis que substituem cálculos e verificações que seriam feitos pelo LLM, eliminando alucinações numéricas:

| Tool | Papel | Função |
|---|---|---|
| `perfil_pacote_dados` | Coletor / Analista | Resume tipo, período, campos e fontes do pacote |
| `consultar_campo_pacote` | Analista / Redator | Lookup determinístico de campo por caminho |
| `auditar_fontes_publicas` | Coletor / Redator | Valida URLs contra lista de domínios oficiais permitidos |
| `consultar_catalogo_fontes` | Coletor | Busca no RAG fontes candidatas para a pauta |
| `avaliar_qualidade_fonte` | Coletor | Score de qualidade da fonte (domínio, estrutura, catálogo) |
| `selecionar_fonte_primaria` | Coletor | Escolhe a melhor fonte entre pacote atual e catálogo |
| `perfil_estatistico_dataset` | Analista | Estatísticas da tabela principal (min, max, média, soma) |
| `calcular_variacao_percentual` | Analista | Variação absoluta e percentual entre dois campos |
| `calcular_taxa` | Analista | Taxa determinística (ex: mortalidade por mil habitantes) |
| `gerar_ranking` | Analista | Ranking ordenado a partir de lista tabular do pacote |
| `detectar_outliers` | Analista | Detecção de outliers por escore z |
| `validar_formula_indicador` | Analista | Compara fórmulas calculadas com valores publicados no pacote |
| `comparar_ufs` | Analista | Compara a UF selecionada com média, mediana e ranking nacional |
| `verificar_completude_pacote` | Coletor / Analista / Redator | Audita campos obrigatórios presentes e ausentes por tipo de dataset |
| `checador_afirmacoes_artigo` | Redator | Verifica se números do artigo estão no pacote coletado |
| `checador_causalidade` | Redator | Detecta linguagem causal sem evidência nos dados |
| `checador_temporal` | Redator | Sinaliza anos fora do escopo ou linguagem de projeção sem suporte |
| `checador_escopo_geografico` | Redator | Impede confusão entre UF, município, capital e região metropolitana |
| `revisor_transparencia` | Redator | Confere seções obrigatórias: *Como checamos*, fontes, disclaimer |
| `matriz_evidencias_redacao` | Redator | Mapeia cada número do artigo ao campo correspondente no pacote |


## Guardrails em camadas

```
Entrada do usuário
      ↓
[1] Determinístico — regex de termos proibidos, validação de domínios
      ↓
[2] Semântico NeMo (opcional) — classificação de intenção via LLM local
      ↓
Coleta de dados → Pipeline de agentes → Artigo gerado
      ↓
[3] Tools pós-artigo — checagem de números, causalidade, fontes, temporalidade
      ↓
[4] Determinístico de saída — disclaimer obrigatório, URLs originais presentes
      ↓
[5] Semântico NeMo de saída (opcional)
      ↓
Artigo publicado com auditoria completa
```


## Stack

- **UI:** Streamlit
- **Agentes:** CrewAI (pipeline sequencial, processo local)
- **LLM:** Ollama · LM Studio · OpenAI · Gemini · Custom (via LiteLLM)
- **RAG / vetorial:** ChromaDB (persistência local em `data/chroma/`)
- **Guardrails semânticos:** NeMo Guardrails
- **Visualizações:** Plotly
- **Dados:** pandas, requests


## Como rodar (Windows)

```powershell
# Primeira vez
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

# Rodar a aplicação
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

Acesse `http://localhost:8501`. Configure a chave do provedor LLM no arquivo `.env` gerado pelo setup.

**Para testar as tools sem LLM:**
```bash
python test_novas_tools.py
```

## Estrutura principal

```
dataveritas/
├── agent_tools.py     # 20+ tools determinísticas dos agentes
├── crew_pipeline.py   # Pipeline CrewAI (Coletor → Analista → Redator)
├── collectors.py      # Dispatcher de coletores de dados
├── guardrails.py      # Guardrails determinísticos e domínios permitidos
├── rag.py             # Banco vetorial ChromaDB e recomendação de fonte
├── source_catalog.py  # Catálogo de fontes públicas conhecidas
├── ibge.py            # Coletor IBGE — população por UF
├── bcb.py             # Coletor BCB — taxa Selic
├── mortality.py       # Coletor SIM/SES-RJ — mortalidade
└── open_data.py       # Descoberta em portais abertos
app.py                 # Interface Streamlit
test_novas_tools.py    # Testes isolados (sem dependências externas)
```



*Projeto didático — os dados e notícias gerados têm finalidade educacional.*
