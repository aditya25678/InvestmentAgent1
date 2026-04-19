# Implementation Plan

## Objectives

- Build a production-ready multi-agent investment research system.
- Enforce independent research before debate to avoid early convergence.
- Provide live internet evidence ingestion (no dummy datasets).
- Persist full audit trail for evidence, claims, challenges, and decisions.
- Expose operational interfaces (CLI + API) for real usage.

## Execution Roadmap

1. **Core scaffolding**
   - Python package structure
   - dependency management via `pyproject.toml`
   - typed environment configuration and logging

2. **Data ingestion layer**
   - market data and options snapshot (Yahoo Finance)
   - filings/fundamentals from SEC EDGAR
   - macro context from FRED
   - news/sentiment feed from NewsAPI or RSS fallback
   - open web retrieval via DuckDuckGo
   - optional analyst estimate feed via FMP

3. **Persistent memory model**
   - relational schema for:
     - `research_runs`
     - `evidence`
     - `claims`
     - `memos`
     - `challenges`
     - `rebuttals`
     - `final_theses`
   - repository abstraction for writes/reads

4. **Agent framework**
   - strict role definitions (fundamental, quant, macro, sentiment, skeptic, catalyst, portfolio)
   - stage-specific structured output contracts
   - bounded personalities (decision style + evidence preference)
   - anti-hallucination protocol requiring citations and uncertainty disclosure

5. **Orchestration workflow**
   - Stage A: independent memos in parallel
   - Stage B: structured challenge rounds by configured pairs
   - Stage C: rebuttals with confidence updates
   - Stage D: committee synthesis and final recommendation

6. **Output and operability**
   - markdown report generation
   - JSON thesis artifact
   - CLI commands for run/list/show
   - API endpoints for run creation and full audit retrieval

7. **Validation and hardening**
   - schema and scoring tests
   - context filtering tests
   - dependency/runtime compatibility checks

## Design Constraints

- No open-ended unconstrained chat loops.
- Every investment conclusion must be falsifiable.
- Must support `LONG`, `SHORT`, `WATCHLIST`, and `PASS`.
- Preserve process explainability and reproducibility.
