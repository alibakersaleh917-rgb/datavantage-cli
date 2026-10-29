"""SQLite3 persistence layer with indexed tables, WAL mode enabled, and LRU query caching.

Manages transactional storage for raw digital asset telemetry and computed multi-factor
valuations, guaranteeing thread-safe lock resilience and sub-millisecond cached reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datavantage.config import AppConfig
from datavantage.ingestor import AssetRecord, HistoricalSale
from datavantage.valuation_engine import ValuationResult


class _LRUCache:
    """Thread-safe bounded Least-Recently-Used (LRU) query cache."""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = max(1, maxsize)
        self._store: OrderedDict[Tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: Tuple[Any, ...]) -> Optional[Any]:
        """Retrieve cached item and move to most-recently-used position."""
        with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]

    def set(self, key: Tuple[Any, ...], value: Any) -> None:
        """Insert or update cached entry, evicting oldest entry when over capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Invalidate all cached query results."""
        with self._lock:
            self._store.clear()


class ValuationStorage:
    """SQLite3 persistence engine with WAL concurrency and indexed analytical queries."""

    def __init__(self, db_path: Optional[Path | str] = None, config: Optional[AppConfig] = None) -> None:
        """Initialize the SQLite storage layer and bootstrap schema + indexes.

        Args:
            db_path: Optional override path for the SQLite database file.
            config: Optional AppConfig instance.
        """
        self.config = config or AppConfig()
        self.db_path = Path(db_path) if db_path else self.config.db_path
        self._cache = _LRUCache(maxsize=self.config.cache_maxsize)
        self._write_lock = threading.RLock()
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Create a configured SQLite connection with WAL mode and busy timeout."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.config.sqlite_busy_timeout_ms / 1000.0,
            isolation_level="DEFERRED",
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout = {self.config.sqlite_busy_timeout_ms};")
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        """Create tables and analytical B-tree indexes if they do not exist."""
        with self._write_lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_assets (
                    asset_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL UNIQUE,
                    sld TEXT NOT NULL,
                    tld TEXT NOT NULL,
                    search_volume INTEGER NOT NULL,
                    cpc_usd REAL NOT NULL,
                    age_years REAL NOT NULL,
                    keyword_count INTEGER NOT NULL,
                    sector TEXT NOT NULL,
                    historical_sales_json TEXT NOT NULL,
                    is_anomalous INTEGER NOT NULL DEFAULT 0,
                    anomaly_reasons_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS asset_valuations (
                    asset_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL UNIQUE,
                    tld TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    commercial_intent_score REAL NOT NULL,
                    brandability_entropy_score REAL NOT NULL,
                    tld_authority_score REAL NOT NULL,
                    liquidity_index_score REAL NOT NULL,
                    composite_score REAL NOT NULL,
                    estimated_value_usd REAL NOT NULL,
                    confidence_low_usd REAL NOT NULL,
                    confidence_high_usd REAL NOT NULL,
                    liquidity_tier TEXT NOT NULL,
                    is_anomalous INTEGER NOT NULL DEFAULT 0,
                    evaluated_at TEXT NOT NULL,
                    FOREIGN KEY (asset_id) REFERENCES raw_assets(asset_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_raw_assets_tld ON raw_assets(tld);
                CREATE INDEX IF NOT EXISTS idx_raw_assets_sector ON raw_assets(sector);
                CREATE INDEX IF NOT EXISTS idx_valuations_composite ON asset_valuations(composite_score DESC);
                CREATE INDEX IF NOT EXISTS idx_valuations_value_usd ON asset_valuations(estimated_value_usd DESC);
                CREATE INDEX IF NOT EXISTS idx_valuations_tier_tld ON asset_valuations(liquidity_tier, tld);
                """
            )

    def get_journal_mode(self) -> str:
        """Return active SQLite journal mode (e.g., 'wal')."""
        with self._connect() as conn:
            row = conn.execute("PRAGMA journal_mode;").fetchone()
            return str(row[0]).lower() if row else "unknown"

    def save_raw_assets(self, assets: List[AssetRecord]) -> int:
        """Upsert a batch of validated AssetRecord instances into `raw_assets`.

        Args:
            assets: List of validated AssetRecord objects.

        Returns:
            int: Number of records persisted.
        """
        if not assets:
            return 0

        rows = [
            (
                a.asset_id,
                a.domain,
                a.sld,
                a.tld,
                a.search_volume,
                a.cpc_usd,
                a.age_years,
                a.keyword_count,
                a.sector,
                json.dumps([{"year": s.year, "price_usd": s.price_usd, "venue": s.venue} for s in a.historical_sales]),
                1 if a.is_anomalous else 0,
                json.dumps(a.anomaly_reasons),
            )
            for a in assets
        ]

        with self._write_lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO raw_assets (
                    asset_id, domain, sld, tld, search_volume, cpc_usd,
                    age_years, keyword_count, sector, historical_sales_json,
                    is_anomalous, anomaly_reasons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    domain = excluded.domain,
                    sld = excluded.sld,
                    tld = excluded.tld,
                    search_volume = excluded.search_volume,
                    cpc_usd = excluded.cpc_usd,
                    age_years = excluded.age_years,
                    keyword_count = excluded.keyword_count,
                    sector = excluded.sector,
                    historical_sales_json = excluded.historical_sales_json,
                    is_anomalous = excluded.is_anomalous,
                    anomaly_reasons_json = excluded.anomaly_reasons_json;
                """,
                rows,
            )
        self._cache.clear()
        return len(rows)

    def load_raw_assets(self) -> List[AssetRecord]:
        """Load all persisted AssetRecord instances from `raw_assets`.

        Returns:
            List[AssetRecord]: Hydrated digital asset telemetry records.
        """
        cache_key = ("raw_assets_all",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM raw_assets ORDER BY asset_id ASC;").fetchall()

        results: List[AssetRecord] = []
        for row in rows:
            sales_raw = json.loads(row["historical_sales_json"])
            sales = [HistoricalSale.from_dict(s) for s in sales_raw]
            results.append(
                AssetRecord(
                    asset_id=row["asset_id"],
                    domain=row["domain"],
                    sld=row["sld"],
                    tld=row["tld"],
                    search_volume=int(row["search_volume"]),
                    cpc_usd=float(row["cpc_usd"]),
                    age_years=float(row["age_years"]),
                    keyword_count=int(row["keyword_count"]),
                    sector=row["sector"],
                    historical_sales=sales,
                    is_anomalous=bool(row["is_anomalous"]),
                    anomaly_reasons=json.loads(row["anomaly_reasons_json"]),
                )
            )

        self._cache.set(cache_key, results)
        return list(results)

    def save_valuations(self, valuations: List[ValuationResult]) -> int:
        """Upsert a batch of ValuationResult scorecards into `asset_valuations`.

        Args:
            valuations: Computed ValuationResult objects.

        Returns:
            int: Number of valuations written.
        """
        if not valuations:
            return 0

        rows = [
            (
                v.asset_id,
                v.domain,
                v.tld,
                v.sector,
                v.commercial_intent_score,
                v.brandability_entropy_score,
                v.tld_authority_score,
                v.liquidity_index_score,
                v.composite_score,
                v.estimated_value_usd,
                v.confidence_interval_usd[0],
                v.confidence_interval_usd[1],
                v.liquidity_tier,
                1 if v.is_anomalous else 0,
                v.evaluated_at,
            )
            for v in valuations
        ]

        with self._write_lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO asset_valuations (
                    asset_id, domain, tld, sector, commercial_intent_score,
                    brandability_entropy_score, tld_authority_score, liquidity_index_score,
                    composite_score, estimated_value_usd, confidence_low_usd,
                    confidence_high_usd, liquidity_tier, is_anomalous, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    domain = excluded.domain,
                    tld = excluded.tld,
                    sector = excluded.sector,
                    commercial_intent_score = excluded.commercial_intent_score,
                    brandability_entropy_score = excluded.brandability_entropy_score,
                    tld_authority_score = excluded.tld_authority_score,
                    liquidity_index_score = excluded.liquidity_index_score,
                    composite_score = excluded.composite_score,
                    estimated_value_usd = excluded.estimated_value_usd,
                    confidence_low_usd = excluded.confidence_low_usd,
                    confidence_high_usd = excluded.confidence_high_usd,
                    liquidity_tier = excluded.liquidity_tier,
                    is_anomalous = excluded.is_anomalous,
                    evaluated_at = excluded.evaluated_at;
                """,
                rows,
            )
        self._cache.clear()
        return len(rows)

    def query_valuations(
        self,
        min_score: float = 0.0,
        tld: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 100,
    ) -> List[ValuationResult]:
        """Query evaluated assets with optional filtering and LRU query caching.

        Args:
            min_score: Minimum composite score filter.
            tld: Optional TLD filter (e.g., 'ai', 'com').
            tier: Optional liquidity tier filter ('PRIME', 'LIQUID', 'GROWTH', 'ILLIQUID').
            limit: Maximum rows to return.

        Returns:
            List[ValuationResult]: Ranked valuation results matching query criteria.
        """
        norm_tld = tld.strip().lower().lstrip(".") if tld else None
        norm_tier = tier.strip().upper() if tier else None
        cache_key = ("query_valuations", round(min_score, 2), norm_tld, norm_tier, limit)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        clauses = ["composite_score >= ?"]
        params: List[Any] = [min_score]

        if norm_tld:
            clauses.append("tld = ?")
            params.append(norm_tld)
        if norm_tier:
            clauses.append("liquidity_tier = ?")
            params.append(norm_tier)

        params.append(limit)
        where_sql = " AND ".join(clauses)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM asset_valuations
                WHERE {where_sql}
                ORDER BY estimated_value_usd DESC, composite_score DESC
                LIMIT ?;
                """,
                params,
            ).fetchall()

        results = [
            ValuationResult(
                asset_id=row["asset_id"],
                domain=row["domain"],
                tld=row["tld"],
                sector=row["sector"],
                commercial_intent_score=float(row["commercial_intent_score"]),
                brandability_entropy_score=float(row["brandability_entropy_score"]),
                tld_authority_score=float(row["tld_authority_score"]),
                liquidity_index_score=float(row["liquidity_index_score"]),
                composite_score=float(row["composite_score"]),
                estimated_value_usd=float(row["estimated_value_usd"]),
                confidence_interval_usd=(
                    float(row["confidence_low_usd"]),
                    float(row["confidence_high_usd"]),
                ),
                liquidity_tier=row["liquidity_tier"],
                is_anomalous=bool(row["is_anomalous"]),
                evaluated_at=row["evaluated_at"],
            )
            for row in rows
        ]

        self._cache.set(cache_key, results)
        return list(results)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Compute aggregate portfolio statistics across all evaluated digital assets."""
        cache_key = ("portfolio_summary",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_assets,
                    COALESCE(SUM(estimated_value_usd), 0.0) AS total_valuation_usd,
                    COALESCE(AVG(estimated_value_usd), 0.0) AS avg_valuation_usd,
                    COALESCE(AVG(composite_score), 0.0) AS avg_composite_score,
                    COALESCE(MAX(estimated_value_usd), 0.0) AS max_valuation_usd,
                    COALESCE(SUM(is_anomalous), 0) AS anomalous_count
                FROM asset_valuations;
                """
            ).fetchone()

        summary = {
            "total_assets": int(row["total_assets"]) if row else 0,
            "total_valuation_usd": round(float(row["total_valuation_usd"]), 2) if row else 0.0,
            "avg_valuation_usd": round(float(row["avg_valuation_usd"]), 2) if row else 0.0,
            "avg_composite_score": round(float(row["avg_composite_score"]), 2) if row else 0.0,
            "max_valuation_usd": round(float(row["max_valuation_usd"]), 2) if row else 0.0,
            "anomalous_count": int(row["anomalous_count"]) if row else 0,
            "cache_hits": self._cache.hits,
            "cache_misses": self._cache.misses,
        }
        self._cache.set(cache_key, summary)
        return dict(summary)

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Expose current LRU query cache hit/miss telemetry."""
        return {"hits": self._cache.hits, "misses": self._cache.misses}

