"""Interactive CLI for DataVantage built with argparse and Rich terminal tables.

Supports subcommands:
- `ingest`: Validate and persist raw B2B digital asset feeds (JSON/CSV).
- `evaluate`: Run multi-factor algorithmic scoring and render institutional scorecards.
- `export`: Export evaluated asset portfolios to JSON or CSV reports.
- `benchmark`: Measure ingestion, scoring, and WAL SQLite throughput & latency percentiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from datavantage.config import AppConfig
from datavantage.ingestor import AssetIngestor, AssetRecord, HistoricalSale
from datavantage.storage import ValuationStorage
from datavantage.valuation_engine import ValuationEngine, ValuationResult

console = Console()


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI argument parser and subcommands.

    Returns:
        argparse.ArgumentParser: Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        prog="datavantage",
        description="DataVantage CLI — Asynchronous B2B Digital Asset Valuation & Market Intelligence Pipeline",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="datavantage_vault.db",
        help="Path to the SQLite3 persistence vault (default: datavantage_vault.db)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Stream-ingest a JSON or CSV digital asset dataset into the SQLite vault",
    )
    ingest_parser.add_argument(
        "source",
        type=str,
        help="Path to input .json or .csv asset dataset (e.g., data/sample_assets.json)",
    )
    ingest_parser.add_argument(
        "--evaluate-now",
        action="store_true",
        help="Immediately run multi-factor valuation on ingested records",
    )

    # Subcommand: evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate persisted or source assets and render Rich valuation scorecards",
    )
    eval_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional input file to ingest before evaluation",
    )
    eval_parser.add_argument(
        "--tld",
        type=str,
        default=None,
        help="Filter displayed valuations by TLD (e.g., ai, com, io)",
    )
    eval_parser.add_argument(
        "--tier",
        type=str,
        default=None,
        choices=["PRIME", "LIQUID", "GROWTH", "ILLIQUID"],
        help="Filter displayed valuations by institutional liquidity tier",
    )
    eval_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum composite score threshold [0-100]",
    )
    eval_parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Maximum number of ranked assets to display (default: 25)",
    )

    # Subcommand: export
    export_parser = subparsers.add_parser(
        "export",
        help="Export evaluated asset portfolio to JSON or CSV",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Destination file path (.json or .csv)",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv"],
        default=None,
        help="Explicit export format (inferred from --output extension if omitted)",
    )
    export_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum composite score filter for exported rows",
    )

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Execute synthetic throughput and latency benchmarks across engine & WAL storage",
    )
    bench_parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=500,
        help="Number of synthetic B2B digital assets to synthesize and evaluate (default: 500)",
    )

    return parser


def _render_valuations_table(results: List[ValuationResult], summary: dict) -> None:
    """Render an institutional Rich table of evaluated B2B digital assets."""
    header_text = (
        f"[bold cyan]Portfolio Assets:[/bold cyan] {summary['total_assets']}   "
        f"[bold green]Aggregate NAV:[/bold green] ${summary['total_valuation_usd']:,.2f}   "
        f"[bold yellow]Mean Composite:[/bold yellow] {summary['avg_composite_score']:.2f}/100   "
        f"[bold magenta]Anomalies Flagged:[/bold magenta] {summary['anomalous_count']}"
    )
    console.print(Panel(header_text, title="[bold white]DataVantage Market Intelligence Vault[/bold white]", border_style="cyan"))

    table = Table(
        title="Multi-Factor B2B Digital Asset Valuation Scorecard",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Rank", justify="right", style="dim", width=5)
    table.add_column("Domain Asset", style="bold white", min_width=18)
    table.add_column("Sector", style="cyan", min_width=14)
    table.add_column("Intent", justify="right")
    table.add_column("Entropy", justify="right")
    table.add_column("TLD Auth", justify="right")
    table.add_column("Liquidity", justify="right")
    table.add_column("Composite", justify="right", style="bold yellow")
    table.add_column("Est. Value (USD)", justify="right", style="bold green")
    table.add_column("Tier", justify="center")

    tier_styles = {
        "PRIME": "[bold bright_green]PRIME[/bold bright_green]",
        "LIQUID": "[bold cyan]LIQUID[/bold cyan]",
        "GROWTH": "[yellow]GROWTH[/yellow]",
        "ILLIQUID": "[dim red]ILLIQUID[/dim red]",
    }

    for rank, item in enumerate(results, start=1):
        domain_label = f"{item.domain} [bold red][!][/bold red]" if item.is_anomalous else item.domain
        table.add_row(
            str(rank),
            domain_label,
            item.sector,
            f"{item.commercial_intent_score:.1f}",
            f"{item.brandability_entropy_score:.1f}",
            f"{item.tld_authority_score:.1f}",
            f"{item.liquidity_index_score:.1f}",
            f"{item.composite_score:.2f}",
            f"${item.estimated_value_usd:,.2f}",
            tier_styles.get(item.liquidity_tier, item.liquidity_tier),
        )

    console.print(table)


def handle_ingest(args: argparse.Namespace, config: AppConfig, storage: ValuationStorage) -> int:
    """Execute the `ingest` CLI subcommand."""
    ingestor = AssetIngestor(config=config)
    report = ingestor.ingest_file(args.source)
    persisted = storage.save_raw_assets(report.valid_records)

    summary_table = Table(title="Stream Ingestion Telemetry", box=box.SIMPLE_HEAVY, header_style="bold green")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count / Status", justify="right", style="bold white")
    summary_table.add_row("Source File", str(args.source))
    summary_table.add_row("Total Records Scanned", str(report.total_processed))
    summary_table.add_row("Validated & Persisted", str(persisted))
    summary_table.add_row("Quarantined (Corrupted)", str(len(report.corrupted_records)))
    summary_table.add_row("Statistical Anomalies Flagged", str(report.anomaly_count))
    summary_table.add_row("SQLite Journal Mode", storage.get_journal_mode().upper())
    console.print(summary_table)

    if args.evaluate_now and report.valid_records:
        engine = ValuationEngine(config=config)
        valuations = engine.evaluate_batch(report.valid_records)
        storage.save_valuations(valuations)
        _render_valuations_table(valuations, storage.get_portfolio_summary())

    return 0


def handle_evaluate(args: argparse.Namespace, config: AppConfig, storage: ValuationStorage) -> int:
    """Execute the `evaluate` CLI subcommand."""
    if args.source:
        ingestor = AssetIngestor(config=config)
        report = ingestor.ingest_file(args.source)
        storage.save_raw_assets(report.valid_records)

    raw_assets = storage.load_raw_assets()
    if not raw_assets:
        console.print("[bold red]No raw assets found in vault. Run `ingest` or pass `--source` first.[/bold red]")
        return 1

    engine = ValuationEngine(config=config)
    valuations = engine.evaluate_batch(raw_assets)
    storage.save_valuations(valuations)

    filtered = storage.query_valuations(
        min_score=args.min_score,
        tld=args.tld,
        tier=args.tier,
        limit=args.top,
    )
    _render_valuations_table(filtered, storage.get_portfolio_summary())
    return 0


def handle_export(args: argparse.Namespace, storage: ValuationStorage) -> int:
    """Execute the `export` CLI subcommand."""
    output_path = Path(args.output)
    fmt = args.format or output_path.suffix.lower().lstrip(".")
    if fmt not in ("json", "csv"):
        console.print(f"[bold red]Unsupported export format '{fmt}'. Choose 'json' or 'csv'.[/bold red]")
        return 1

    results = storage.query_valuations(min_score=args.min_score, limit=10_000)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        payload = {
            "summary": storage.get_portfolio_summary(),
            "valuations": [r.to_dict() for r in results],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        fieldnames = [
            "asset_id",
            "domain",
            "tld",
            "sector",
            "commercial_intent_score",
            "brandability_entropy_score",
            "tld_authority_score",
            "liquidity_index_score",
            "composite_score",
            "estimated_value_usd",
            "confidence_low_usd",
            "confidence_high_usd",
            "liquidity_tier",
            "is_anomalous",
            "evaluated_at",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.to_dict()
                row.pop("confidence_interval_usd", None)
                writer.writerow(row)

    console.print(
        f"[bold green]Exported {len(results)} evaluated assets to {output_path} ({fmt.upper()}).[/bold green]"
    )
    return 0


def handle_benchmark(args: argparse.Namespace, config: AppConfig, storage: ValuationStorage) -> int:
    """Execute the `benchmark` CLI subcommand."""
    count = max(10, args.iterations)
    engine = ValuationEngine(config=config)
    tlds = ["com", "ai", "io", "cloud", "dev", "finance", "security", "co"]
    prefixes = ["quantum", "nexus", "vector", "cipher", "stratos", "pulse", "vertex", "aegis", "synapse", "krypton"]

    synthetic_assets: List[AssetRecord] = []
    for i in range(count):
        tld = tlds[i % len(tlds)]
        sld = f"{prefixes[i % len(prefixes)]}{i // len(prefixes) if i >= len(prefixes) else ''}"
        synthetic_assets.append(
            AssetRecord(
                asset_id=f"BENCH-{i + 1:05d}",
                domain=f"{sld}.{tld}",
                sld=sld,
                tld=tld,
                search_volume=1200 + (i * 379) % 95_000,
                cpc_usd=round(2.5 + ((i * 17) % 450) / 10.0, 2),
                age_years=round(1.0 + (i % 22), 1),
                keyword_count=1 if i % 3 != 0 else 2,
                sector="Synthetic Benchmark",
                historical_sales=[HistoricalSale(year=2023, price_usd=4500.0 + (i * 125) % 35000)],
            )
        )

    latencies_us: List[float] = []
    valuations: List[ValuationResult] = []

    t0 = time.perf_counter()
    for asset in synthetic_assets:
        s0 = time.perf_counter()
        valuations.append(engine.evaluate(asset))
        latencies_us.append((time.perf_counter() - s0) * 1_000_000.0)
    eval_elapsed = max(1e-9, time.perf_counter() - t0)

    db_t0 = time.perf_counter()
    storage.save_raw_assets(synthetic_assets)
    storage.save_valuations(valuations)
    _ = storage.query_valuations(min_score=50.0, limit=100)
    _ = storage.query_valuations(min_score=50.0, limit=100)  # Second read hits LRU cache
    db_elapsed = max(1e-9, time.perf_counter() - db_t0)

    latencies_sorted = sorted(latencies_us)
    p50 = statistics.median(latencies_sorted)
    p99 = latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.99))]
    throughput = count / eval_elapsed

    bench_table = Table(title=f"DataVantage Pipeline Benchmark ({count} Assets)", box=box.DOUBLE_EDGE)
    bench_table.add_column("Benchmark Metric", style="bold cyan")
    bench_table.add_column("Observed Performance", justify="right", style="bold green")
    bench_table.add_row("Algorithmic Scoring Throughput", f"{throughput:,.0f} assets/sec")
    bench_table.add_row("Valuation Median Latency (p50)", f"{p50:.2f} µs")
    bench_table.add_row("Valuation Tail Latency (p99)", f"{p99:.2f} µs")
    bench_table.add_row("SQLite WAL Batch Write + Cached Read", f"{db_elapsed * 1000.0:.2f} ms")
    bench_table.add_row("LRU Cache Hits / Misses", f"{storage.cache_stats['hits']} / {storage.cache_stats['misses']}")
    console.print(bench_table)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint invoked by console script or `python -m datavantage.cli`."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AppConfig(db_path=Path(args.db))
    storage = ValuationStorage(db_path=config.db_path, config=config)

    if args.command == "ingest":
        return handle_ingest(args, config, storage)
    if args.command == "evaluate":
        return handle_evaluate(args, config, storage)
    if args.command == "export":
        return handle_export(args, storage)
    if args.command == "benchmark":
        return handle_benchmark(args, config, storage)

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

