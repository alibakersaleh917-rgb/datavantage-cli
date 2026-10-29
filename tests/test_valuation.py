"""Comprehensive unit tests covering edge cases, boundary scores, malformed schemas, and database locking."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from datavantage.cli import main as cli_main
from datavantage.config import AppConfig, ScoringWeights
from datavantage.ingestor import AssetIngestor, AssetRecord, HistoricalSale
from datavantage.storage import ValuationStorage
from datavantage.valuation_engine import ValuationEngine


# ---------------------------------------------------------------------------
# 1. Configuration & Weight Invariants
# ---------------------------------------------------------------------------


def test_scoring_weights_must_sum_to_one() -> None:
    """Verify ScoringWeights rejects weight distributions that do not sum to 1.0."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ScoringWeights(
            commercial_intent=0.50,
            brandability_entropy=0.25,
            tld_authority=0.20,
            liquidity_index=0.20,
        )


def test_scoring_weights_rejects_negative_values() -> None:
    """Verify ScoringWeights rejects negative factor weights even if sum is 1.0."""
    with pytest.raises(ValueError, match="must be within"):
        ScoringWeights(
            commercial_intent=1.10,
            brandability_entropy=-0.10,
            tld_authority=0.0,
            liquidity_index=0.0,
        )


def test_tld_authority_weight_lookup() -> None:
    """Verify TLD lookup normalizes leading dots and falls back for unknown TLDs."""
    cfg = AppConfig()
    assert cfg.get_tld_weight(".COM") == 1.00
    assert cfg.get_tld_weight("ai") == 0.92
    assert cfg.get_tld_weight(".unknownext") == cfg.default_tld_weight


# ---------------------------------------------------------------------------
# 2. Mathematical Valuation Engine Boundary & Edge Cases
# ---------------------------------------------------------------------------


def test_boundary_scores_zero_and_saturation() -> None:
    """Verify all individual factor scores stay strictly bounded within [0.0, 100.0]."""
    engine = ValuationEngine()

    # Zero boundary
    assert engine.calculate_commercial_intent(search_volume=0, cpc_usd=0.0) == 0.0
    assert engine.calculate_brandability_entropy(sld="", keyword_count=1) == 0.0
    assert engine.calculate_liquidity_index(age_years=0.0, historical_sales_count=0, historical_avg_price=0.0) == 0.0

    # Extreme saturation ceiling boundary
    high_intent = engine.calculate_commercial_intent(search_volume=10_000_000, cpc_usd=500.0)
    assert high_intent == 100.0

    high_liquidity = engine.calculate_liquidity_index(
        age_years=50.0,
        historical_sales_count=25,
        historical_avg_price=1_000_000.0,
    )
    assert 99.0 <= high_liquidity <= 100.0


def test_brandability_entropy_penalizes_repetition_and_hyphens() -> None:
    """Verify high-entropy phonetic domains outscore repetitive or hyphenated strings."""
    engine = ValuationEngine()

    prime_sld = engine.calculate_brandability_entropy(sld="Vortexa", keyword_count=1)
    repetitive_sld = engine.calculate_brandability_entropy(sld="aaaaaaa", keyword_count=1)
    hyphen_digit_sld = engine.calculate_brandability_entropy(sld="b2b-auto-99", keyword_count=3)

    assert prime_sld > 80.0
    assert repetitive_sld < prime_sld
    assert hyphen_digit_sld < 55.0


def test_prime_vs_illiquid_tier_classification() -> None:
    """Verify end-to-end evaluation classifies category-killer .com as PRIME and weak .biz as ILLIQUID/GROWTH."""
    engine = ValuationEngine()

    prime_asset = AssetRecord(
        asset_id="ASSET-PRIME",
        domain="vector.com",
        sld="vector",
        tld="com",
        search_volume=185_000,
        cpc_usd=42.50,
        age_years=24.0,
        keyword_count=1,
        sector="AI Infrastructure",
        historical_sales=[
            HistoricalSale(year=2019, price_usd=85_000.0, venue="Sedo"),
            HistoricalSale(year=2023, price_usd=145_000.0, venue="Private"),
        ],
    )
    weak_asset = AssetRecord(
        asset_id="ASSET-WEAK",
        domain="my-cheap-host-99.biz",
        sld="my-cheap-host-99",
        tld="biz",
        search_volume=10,
        cpc_usd=0.15,
        age_years=0.5,
        keyword_count=4,
        sector="Web Hosting",
        historical_sales=[],
    )

    prime_res = engine.evaluate(prime_asset)
    weak_res = engine.evaluate(weak_asset)

    assert prime_res.liquidity_tier == "PRIME"
    assert prime_res.composite_score > 80.0
    assert prime_res.estimated_value_usd > 50_000.0
    assert prime_res.confidence_interval_usd[0] < prime_res.estimated_value_usd < prime_res.confidence_interval_usd[1]

    assert weak_res.liquidity_tier == "ILLIQUID"
    assert weak_res.composite_score < 35.0
    assert prime_res.estimated_value_usd > weak_res.estimated_value_usd * 20


# ---------------------------------------------------------------------------
# 3. Stream Ingestion, Malformed Schema Quarantine & Anomaly Detection
# ---------------------------------------------------------------------------


def test_ingestor_quarantines_malformed_records_and_recovers(tmp_path: Path) -> None:
    """Verify stream ingestor isolates invalid schemas without halting valid record processing."""
    payload = [
        {
            "asset_id": "VALID-01",
            "domain": "quantumpay.ai",
            "tld": "ai",
            "search_volume": 24000,
            "cpc_usd": 18.75,
            "age_years": 6.5,
            "keyword_count": 2,
            "sector": "FinTech",
            "historical_sales": [{"year": 2022, "price_usd": 14500.0, "venue": "Afternic"}],
        },
        {
            "asset_id": "BAD-DOMAIN",
            "domain": "not a valid domain!!",
            "tld": "com",
            "search_volume": 1000,
            "cpc_usd": 5.0,
            "age_years": 2.0,
            "keyword_count": 1,
        },
        {
            "asset_id": "TLD-MISMATCH",
            "domain": "cloudmesh.io",
            "tld": "com",
            "search_volume": 5000,
            "cpc_usd": 12.0,
            "age_years": 4.0,
            "keyword_count": 2,
        },
        {
            "asset_id": "NEGATIVE-CPC",
            "domain": "ledgerflow.com",
            "tld": "com",
            "search_volume": 8000,
            "cpc_usd": -15.0,
            "age_years": 5.0,
            "keyword_count": 2,
        },
    ]

    json_file = tmp_path / "mixed_feed.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    ingestor = AssetIngestor()
    report = ingestor.ingest_file(json_file)

    assert report.total_processed == 4
    assert len(report.valid_records) == 1
    assert len(report.corrupted_records) == 3
    assert report.valid_records[0].domain == "quantumpay.ai"


def test_ingestor_detects_statistical_cpc_anomaly() -> None:
    """Verify Welford Z-score and extreme CPC outlier rules flag anomalous telemetry."""
    ingestor = AssetIngestor()
    stream_items = []
    for i in range(15):
        stream_items.append(
            (
                i,
                {
                    "asset_id": f"NORM-{i}",
                    "domain": f"saasnode{i}.com",
                    "tld": "com",
                    "search_volume": 5000,
                    "cpc_usd": 10.0,
                    "age_years": 5.0,
                    "keyword_count": 2,
                },
            )
        )
    # Inject extreme CPC anomaly
    stream_items.append(
        (
            99,
            {
                "asset_id": "ANOMALY-99",
                "domain": "insurevault.com",
                "tld": "com",
                "search_volume": 5000,
                "cpc_usd": 295.0,
                "age_years": 12.0,
                "keyword_count": 2,
            },
        )
    )

    report = ingestor.ingest_stream(iter(stream_items))
    assert report.anomaly_count == 1
    flagged = [r for r in report.valid_records if r.is_anomalous]
    assert len(flagged) == 1
    assert flagged[0].asset_id == "ANOMALY-99"
    assert len(flagged[0].anomaly_reasons) >= 1


# ---------------------------------------------------------------------------
# 4. SQLite WAL Persistence, Query Caching & Concurrent Database Locking
# ---------------------------------------------------------------------------


def test_sqlite_wal_mode_and_concurrent_thread_locking(tmp_path: Path) -> None:
    """Verify SQLite storage operates in WAL mode and survives concurrent multithreaded read/write load."""
    db_file = tmp_path / "concurrent_vault.db"
    storage = ValuationStorage(db_path=db_file)
    engine = ValuationEngine()

    assert storage.get_journal_mode() == "wal"

    def worker(worker_id: int) -> int:
        assets = [
            AssetRecord(
                asset_id=f"W{worker_id}-A{idx}",
                domain=f"worker{worker_id}node{idx}.io",
                sld=f"worker{worker_id}node{idx}",
                tld="io",
                search_volume=2500 + idx * 100,
                cpc_usd=8.5 + idx,
                age_years=4.0,
                keyword_count=2,
                sector="DevOps",
            )
            for idx in range(10)
        ]
        storage.save_raw_assets(assets)
        vals = engine.evaluate_batch(assets)
        storage.save_valuations(vals)
        queried = storage.query_valuations(min_score=10.0, limit=50)
        return len(queried)

    with ThreadPoolExecutor(max_workers=6) as pool:
        counts = list(pool.map(worker, range(6)))

    assert all(c > 0 for c in counts)
    summary = storage.get_portfolio_summary()
    assert summary["total_assets"] == 60

    # Verify LRU cache hit increments on repeated identical query
    _ = storage.query_valuations(min_score=25.0, tld="io", limit=20)
    hits_before = storage.cache_stats["hits"]
    _ = storage.query_valuations(min_score=25.0, tld="io", limit=20)
    assert storage.cache_stats["hits"] == hits_before + 1


def test_sample_assets_dataset_and_cli_pipeline(tmp_path: Path) -> None:
    """Verify the bundled 25-record `data/sample_assets.json` ingests, evaluates, and exports cleanly via CLI."""
    sample_path = Path(__file__).resolve().parents[1] / "data" / "sample_assets.json"
    assert sample_path.exists(), "data/sample_assets.json must exist"

    db_path = tmp_path / "cli_test.db"
    export_json = tmp_path / "exported_portfolio.json"

    # 1. CLI Ingest + Evaluate
    rc_ingest = cli_main(["--db", str(db_path), "ingest", str(sample_path), "--evaluate-now"])
    assert rc_ingest == 0

    # 2. CLI Export
    rc_export = cli_main(["--db", str(db_path), "export", "--output", str(export_json)])
    assert rc_export == 0
    assert export_json.exists()

    exported_data = json.loads(export_json.read_text(encoding="utf-8"))
    assert exported_data["summary"]["total_assets"] == 25
    assert len(exported_data["valuations"]) == 25

