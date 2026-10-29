"""Configuration management, scoring weights, and environment settings for DataVantage CLI.

Provides strongly-typed Pydantic-style and dataclass configurations governing mathematical
valuation heuristics, TLD authority tiers, anomaly thresholds, and SQLite storage settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYDANTIC_AVAILABLE = False


DEFAULT_TLD_WEIGHTS: Dict[str, float] = {
    "com": 1.00,
    "ai": 0.92,
    "io": 0.85,
    "co": 0.78,
    "cloud": 0.74,
    "dev": 0.72,
    "tech": 0.68,
    "org": 0.65,
    "net": 0.62,
    "app": 0.70,
    "finance": 0.75,
    "security": 0.76,
    "xyz": 0.45,
    "info": 0.35,
    "biz": 0.30,
}


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """Normalized multi-factor weights for composite digital asset valuation.

    Attributes:
        commercial_intent: Weight assigned to search volume and CPC monetization potential.
        brandability_entropy: Weight assigned to lexical conciseness, phonetic balance, and entropy.
        tld_authority: Weight assigned to top-level domain market adoption and trust tier.
        liquidity_index: Weight assigned to historical sales velocity and age maturity.
    """

    commercial_intent: float = 0.35
    brandability_entropy: float = 0.25
    tld_authority: float = 0.20
    liquidity_index: float = 0.20

    def __post_init__(self) -> None:
        """Validate that all component weights are positive and sum to 1.0."""
        total = (
            self.commercial_intent
            + self.brandability_entropy
            + self.tld_authority
            + self.liquidity_index
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ScoringWeights must sum to 1.0, received {total:.4f}")
        for name, val in (
            ("commercial_intent", self.commercial_intent),
            ("brandability_entropy", self.brandability_entropy),
            ("tld_authority", self.tld_authority),
            ("liquidity_index", self.liquidity_index),
        ):
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Weight '{name}' must be within [0.0, 1.0], got {val}")


@dataclass(frozen=True, slots=True)
class ValuationThresholds:
    """Mathematical scaling constants and anomaly detection boundaries.

    Attributes:
        base_valuation_multiplier: Baseline USD multiplier for composite score conversion.
        max_cpc_ceiling_usd: Saturation ceiling for CPC normalization (in USD).
        max_search_volume_ceiling: Logarithmic ceiling for monthly search volume.
        max_age_years_ceiling: Maturity saturation threshold in domain registration years.
        optimal_domain_length: Ideal character count for peak brandability before decay.
        anomaly_z_score_limit: Z-score threshold above which ingestion flags price/CPC anomalies.
    """

    base_valuation_multiplier: float = 1250.0
    max_cpc_ceiling_usd: float = 85.0
    max_search_volume_ceiling: int = 250_000
    max_age_years_ceiling: int = 25
    optimal_domain_length: int = 7
    anomaly_z_score_limit: float = 3.2


@dataclass(slots=True)
class AppConfig:
    """Master runtime configuration for the DataVantage pipeline.

    Attributes:
        db_path: Filesystem path to the SQLite3 persistence store.
        cache_maxsize: Maximum entries retained in the in-memory query cache.
        enable_wal_mode: Whether to enforce SQLite Write-Ahead Logging (WAL).
        sqlite_busy_timeout_ms: Milliseconds SQLite waits on lock contention before raising.
        weights: Multi-factor scoring weight distribution.
        thresholds: Mathematical scaling and anomaly boundaries.
        tld_weights: Mapping of normalized TLD strings to authority scores in [0.0, 1.0].
        default_tld_weight: Fallback authority score for unrecognized TLDs.
    """

    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DATAVANTAGE_DB_PATH", "datavantage_vault.db"))
    )
    cache_maxsize: int = 512
    enable_wal_mode: bool = True
    sqlite_busy_timeout_ms: int = 5000
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: ValuationThresholds = field(default_factory=ValuationThresholds)
    tld_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TLD_WEIGHTS))
    default_tld_weight: float = 0.25

    @classmethod
    def from_env(cls) -> AppConfig:
        """Instantiate configuration with environment variable overrides.

        Returns:
            AppConfig: Configured runtime settings instance.
        """
        db_env = os.getenv("DATAVANTAGE_DB_PATH")
        cache_env = os.getenv("DATAVANTAGE_CACHE_SIZE")
        busy_env = os.getenv("DATAVANTAGE_BUSY_TIMEOUT_MS")

        return cls(
            db_path=Path(db_env) if db_env else Path("datavantage_vault.db"),
            cache_maxsize=int(cache_env) if cache_env else 512,
            sqlite_busy_timeout_ms=int(busy_env) if busy_env else 5000,
        )

    def get_tld_weight(self, tld: str) -> float:
        """Resolve authority weight for a given TLD string.

        Args:
            tld: Top-level domain (with or without leading dot).

        Returns:
            float: Authority multiplier between 0.0 and 1.0.
        """
        normalized = tld.strip().lower().lstrip(".")
        return self.tld_weights.get(normalized, self.default_tld_weight)
