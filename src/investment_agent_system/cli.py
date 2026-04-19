from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich import print
from rich.console import Console
from rich.table import Table

from investment_agent_system.config import get_settings
from investment_agent_system.models.schemas import AgentRole, RunRequest
from investment_agent_system.orchestration.service import ThesisService
from investment_agent_system.storage.db import init_db
from investment_agent_system.utils.logging import setup_logging

app = typer.Typer(help="Investment Agent System CLI")
console = Console()


def _ensure_runtime_env() -> None:
    settings = get_settings()
    try:
        settings.validate_runtime_environment()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _parse_roles(roles_csv: Optional[str]) -> Optional[list[AgentRole]]:
    if not roles_csv:
        return None
    roles: list[AgentRole] = []
    for raw in roles_csv.split(","):
        value = raw.strip().lower()
        if not value:
            continue
        roles.append(AgentRole(value))
    return roles


@app.command()
def run(
    ticker: str = typer.Option(..., help="Ticker symbol, e.g. AAPL"),
    horizon: str = typer.Option("6-12 months", help="Investment horizon"),
    roles: Optional[str] = typer.Option(
        None,
        help="Comma-separated roles. Example: fundamental,quant,skeptic,portfolio",
    ),
    model: Optional[str] = typer.Option(None, help="Optional model override"),
) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    _ensure_runtime_env()
    init_db()

    request = RunRequest(
        ticker=ticker,
        horizon=horizon,
        include_roles=_parse_roles(roles),
        model=model,
    )
    service = ThesisService()
    try:
        output = asyncio.run(service.run(request))
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(f"[red]Run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    report_dir = Path(settings.reports_dir)
    report_stem = f"{output.run_id}_{output.ticker}"
    md_path = report_dir / f"{report_stem}.md"
    json_path = report_dir / f"{report_stem}.json"
    agents_path = report_dir / f"{report_stem}_agents.json"
    discussion_path = report_dir / f"{report_stem}_discussion.json"

    print(f"[bold green]Run completed[/bold green]: {output.run_id}")
    print(f"Recommendation: [bold]{output.final_thesis.recommendation.value}[/bold]")
    print(f"Overall conviction: {output.final_thesis.confidence.overall_conviction:.2f}")
    print(f"Markdown report: {md_path}")
    print(f"JSON thesis: {json_path}")
    print(f"Agent memos: {agents_path}")
    print(f"Discussion log: {discussion_path}")


@app.command()
def list_runs(limit: int = typer.Option(25, help="Number of runs to display")) -> None:
    _ensure_runtime_env()
    init_db()
    service = ThesisService()
    runs = service.list_runs(limit=limit)
    table = Table(title="Recent Investment Thesis Runs")
    table.add_column("Run ID")
    table.add_column("Ticker")
    table.add_column("Status")
    table.add_column("Started At")
    table.add_column("Completed At")
    for row in runs:
        table.add_row(
            row["run_id"],
            row["ticker"],
            row["status"],
            str(row["started_at"]),
            str(row["completed_at"]),
        )
    console.print(table)


@app.command()
def show(run_id: str = typer.Argument(..., help="Run ID")) -> None:
    _ensure_runtime_env()
    init_db()
    service = ThesisService()
    payload = service.get_full_audit(run_id)
    if not payload:
        raise typer.BadParameter(f"Run not found: {run_id}")
    print(json.dumps(payload, indent=2))


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    _ensure_runtime_env()
    init_db()
    uvicorn.run("investment_agent_system.api.main:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    app()
