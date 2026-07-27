# DataVeritas - backup portatil

Este pacote foi preparado para rodar em outro Windows sem levar a `.venv` antiga.

## O que vem no backup

- Codigo fonte do app Streamlit.
- `requirements.txt` com as dependencias Python.
- `setup_windows.ps1` para criar `.venv` e instalar dependencias.
- `run_windows.ps1` para iniciar o app.
- `.env.example` para configurar provedor/modelo/chaves.
- `data/chroma`, quando incluido, para preservar a memoria RAG local.

## O que nao vem no backup

- `.venv`
- `.env`
- caches Python
- arquivos temporarios
- chaves de API reais

## Como rodar em outro PC

Abra PowerShell na pasta extraida e rode:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
.\run_windows.ps1
```

Se o Python nao existir no outro PC, instale Python 3.11 ou 3.12.
Com winget:

```powershell
winget install Python.Python.3.12
```

## Como configurar chaves

O setup cria um `.env` baseado em `.env.example` se ele ainda nao existir.

Exemplo com OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-5.6-luna
OPENAI_API_KEY=sua-chave
```

Exemplo com Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash
GEMINI_API_KEY=sua-chave
```

Exemplo com API custom/OpenAI-compatible:

```env
LLM_PROVIDER=custom
CUSTOM_LLM_MODEL=openrouter/openai/gpt-5-mini
CUSTOM_LLM_API_KEY=sua-chave
```

## Ollama

Se voce escolher `LLM_PROVIDER=ollama`, o outro PC tambem precisa ter o Ollama instalado
e o modelo baixado localmente.

Exemplo:

```powershell
ollama pull gemma4:latest
```

Se usar OpenAI, Gemini ou API custom, o Ollama nao e necessario.

## Criar novo backup

Para criar outro pacote depois:

```powershell
.\make_backup_windows.ps1
```

Para criar sem levar o banco vetorial local:

```powershell
.\make_backup_windows.ps1 -NoVectorStore
```
