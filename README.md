# Investment Agent System

Production-oriented multi-agent investment research engine that runs independent specialist analysts, structured cross-examination, rebuttals, and committee synthesis to produce an auditable thesis.

## Workflow

The system runs a single analyst per role:

- `fundamental`
- `quant`
- `macro`
- `sentiment`
- `skeptic`
- `catalyst`
- `portfolio`

Execution stages:

1. Independent memos (no cross-agent visibility first)
2. Structured challenge prompts across predefined challenger/target pairs
3. Rebuttals from challenged agents
4. Committee-chair synthesis into final thesis

## Setup

1. Activate your existing conda env:

```bash
conda activate myenv
```

2. Install package:

```bash
python -m pip install -e ".[dev]"
```

3. Configure environment:

```bash
cp .env.example .env
```

Required:

- `OLLAMA_API_KEY`

Optional:

- `NEWSAPI_KEY`
- `FMP_API_KEY`
- `SEC_USER_AGENT` (recommended to include your real contact)

## CLI

Run a thesis:

```bash
ias run --ticker AAPL --horizon "6-12 months"
```

Limit roles:

```bash
ias run --ticker NVDA --roles "fundamental,quant,skeptic,portfolio"
```

Inspect runs:

```bash
ias list-runs
ias show <run_id>
```

Start API:

```bash
ias api --host 127.0.0.1 --port 8000
```

## API

Create run:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","horizon":"6-12 months"}'
```

Read outputs:

```bash
curl http://127.0.0.1:8000/runs/<run_id>
curl http://127.0.0.1:8000/runs/<run_id>/thesis
curl http://127.0.0.1:8000/runs/<run_id>/agents
curl http://127.0.0.1:8000/runs/<run_id>/discussion
curl http://127.0.0.1:8000/runs/<run_id>/audit
```

## Output Files

Each run writes flat artifacts in `reports/`:

- `<run_id>_<ticker>.md`
- `<run_id>_<ticker>.json`
- `<run_id>_<ticker>_agents.json`
- `<run_id>_<ticker>_discussion.json`

SQL audit data is stored in `data/investment_agent.db`.

## Data Providers

- SEC EDGAR filings/XBRL
- Yahoo Finance market data/options with Stooq fallback
- NewsAPI (optional key), Google News RSS, Yahoo RSS
- FRED macro series
- DuckDuckGo web search
- FMP estimates (optional key)

## Tests

```bash
pytest
```
