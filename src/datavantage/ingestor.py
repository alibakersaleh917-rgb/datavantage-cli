"""Stream-based CSV/JSON data ingestion with strict schema validation, corruption recovery, and anomaly detection.

Provides memory-efficient generator pipelines that parse heterogeneous digital asset datasets,
quarantine corrupted records with diagnostic provenance, and compute rolling statistical
Z-scores to flag anomalous market observations.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datavantage.config import AppConfig

DOMAIN_REGEX = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.[a-z]{2,24}$")


@dataclass(slots=True)
class HistoricalSale:
    """Single historical secondary-market transaction for a digital asset.

    Attributes:
        year: Calendar year of the transaction.
        price_usd: Gross settlement price in USD.
        venue: Marketplace or registrar where the transaction settled.
    """

    year: int
    price_usd: float
    venue: str = "private"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> HistoricalSale:
        """Validate and construct a HistoricalSale from a raw dictionary.

        Args:
            raw: Unvalidated sale record dictionary.

        Returns:
            HistoricalSale: Validated transaction instance.

        Raises:
            ValueError: If year or price_usd is missing or out of bounds.
        """
        year = int(raw.get("year", 0))
        price_usd = float(raw.get("price_usd", -1.0))
        venue = str(raw.get("venue", "private")).strip() or "private"

        if year < 1985 or year > 2035:
            raise ValueError(f"Invalid historical sale year: {year}")
        if price_usd < 0.0 or math.isnan(price_usd) or math.isinf(price_usd):
            raise ValueError(f"Invalid historical sale price_usd: {price_usd}")

        return cls(year=year, price_usd=price_usd, venue=venue)


@dataclass(slots=True)
class AssetRecord:
    """Validated B2B digital asset telemetry record ready for valuation.

    Attributes:
        asset_id: Unique canonical identifier (e.g., 'ASSET-001').
        domain: Normalized full domain name (e.g., 'quantumpay.ai').
        sld: Second-level domain label (e.g., 'quantumpay').
        tld: Top-level domain extension without leading dot (e.g., 'ai').
        search_volume: Monthly global keyword search volume.
        cpc_usd: Average commercial cost-per-click in USD.
        age_years: Continuous registration age in years.
        keyword_count: Number of semantic root tokens composing the SLD.
        sector: Target B2B industry vertical.
        historical_sales: Chronological list of verified prior sales.
        is_anomalous: True if flagged by statistical Z-score anomaly detection.
        anomaly_reasons: Diagnostic descriptions of any detected statistical anomalies.
    """

    asset_id: str
    domain: str
    sld: str
    tld: str
    search_volume: int
    cpc_usd: float
    age_years: float
    keyword_count: int
    sector: str = "Enterprise Tech"
    historical_sales: List[HistoricalSale] = field(default_factory=list)
    is_anomalous: bool = False
    anomaly_reasons: List[str] = field(default_factory=list)

    @property
    def historical_avg_price(self) -> float:
        """Compute mean historical sale price, or 0.0 if no prior sales exist."""
        if not self.historical_sales:
            return 0.0
        return sum(s.price_usd for s in self.historical_sales) / len(self.historical_sales)


@dataclass(slots=True)
class CorruptedRecordError:
    """Diagnostic record captured when corruption recovery quarantines a malformed row.

    Attributes:
        index: Zero-based record index in the input stream.
        raw_payload: Raw string or dictionary representation of the rejected item.
        reason: Human-readable explanation of the validation or parsing failure.
    """

    index: int
    raw_payload: Any
    reason: str


@dataclass(slots=True)
class IngestionReport:
    """Summary telemetry from a completed stream ingestion run.

    Attributes:
        valid_records: Successfully validated AssetRecord instances.
        corrupted_records: Quarantined records with diagnostic failure reasons.
        anomaly_count: Number of valid records flagged for statistical anomalies.
        total_processed: Total raw records encountered in the stream.
    """

    valid_records: List[AssetRecord] = field(default_factory=list)
    corrupted_records: List[CorruptedRecordError] = field(default_factory=list)
    anomaly_count: int = 0
    total_processed: int = 0


class _WelfordAccumulator:
    """Online Welford algorithm for numerically stable running mean and variance."""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, value: float) -> None:
        """Incorporate a new scalar observation into the running moments."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std_dev(self) -> float:
        """Sample standard deviation, or 0.0 if fewer than 2 observations exist."""
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def z_score(self, value: float) -> float:
        """Calculate absolute Z-score for a candidate observation."""
        sd = self.std_dev
        if sd <= 1e-9:
            return 0.0
        return abs(value - self.mean) / sd


class AssetIngestor:
    """Resilient stream ingestor supporting JSON and CSV digital asset feeds.

    Implements strict schema validation, per-record corruption recovery, and two-pass
    or online Welford Z-score anomaly detection across CPC and search volume metrics.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the ingestor with runtime configuration.

        Args:
            config: Optional AppConfig instance; defaults to standard settings.
        """
        self.config = config or AppConfig()

    # =========================================================================
    # TODO (Phase 2): Asynchronous WHOIS/RDAP Concurrency Pooling & Rate-Limited
    #                 API Backoff State Machine
    # -------------------------------------------------------------------------
    # Architectural Specification for Phase 2 Implementation:
    # 1. Implement `AsyncRDAPEnricher` using `aiohttp.ClientSession` with a
    #    bounded `asyncio.Semaphore(max_concurrency=32)` and per-registrar
    #    connection pooling to hydrate live expiration timestamps, DNSSEC
    #    status, and registrar lock flags during ingestion.
    # 2. Build a deterministic finite-state machine (`BackoffStateMachine`)
    #    transitioning across `CLOSED -> HALF_OPEN -> OPEN` circuit-breaker
    #    states when upstream RDAP/CPC endpoints return HTTP 429 or 503.
    # 3. Incorporate decorrelated jitter exponential backoff:
    #    `sleep_ms = min(cap_ms, random.uniform(base_ms, prev_sleep_ms * 3))`
    #    and honor `Retry-After` headers per RFC 7480 before re-queuing tasks.
    # =========================================================================

    def ingest_file(self, file_path: Path | str) -> IngestionReport:
        """Ingest a JSON or CSV dataset file with corruption recovery and anomaly tagging.

        Args:
            file_path: Path to the `.json` or `.csv` source file.

        Returns:
            IngestionReport: Validated records, quarantined errors, and anomaly counts.

        Raises:
            FileNotFoundError: If `file_path` does not exist.
            ValueError: If the file extension is unsupported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ingestion source file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            raw_stream = self._stream_json_records(path)
        elif suffix == ".csv":
            raw_stream = self._stream_csv_records(path)
        else:
            raise ValueError(f"Unsupported file extension '{suffix}'. Expected .json or .csv")

        return self.ingest_stream(raw_stream)

    def ingest_stream(self, raw_iterable: Iterator[Tuple[int, Any]]) -> IngestionReport:
        """Validate a stream of raw records, isolating corrupted entries and flagging anomalies.

        Args:
            raw_iterable: Iterator yielding `(index, raw_payload)` tuples.

        Returns:
            IngestionReport: Complete ingestion telemetry and clean records.
        """
        report = IngestionReport()
        cpc_stats = _WelfordAccumulator()
        vol_stats = _WelfordAccumulator()

        for idx, payload in raw_iterable:
            report.total_processed += 1
            if isinstance(payload, CorruptedRecordError):
                report.corrupted_records.append(payload)
                continue

            try:
                record = self._validate_and_normalize(payload, idx)
                report.valid_records.append(record)
                cpc_stats.update(record.cpc_usd)
                vol_stats.update(float(record.search_volume))
            except (ValueError, TypeError, KeyError) as exc:
                report.corrupted_records.append(
                    CorruptedRecordError(index=idx, raw_payload=payload, reason=str(exc))
                )

        # Statistical anomaly pass across validated corpus
        z_limit = self.config.thresholds.anomaly_z_score_limit
        for record in report.valid_records:
            cpc_z = cpc_stats.z_score(record.cpc_usd)
            vol_z = vol_stats.z_score(float(record.search_volume))

            if cpc_z > z_limit:
                record.is_anomalous = True
                record.anomaly_reasons.append(
                    f"CPC Z-score {cpc_z:.2f} exceeds threshold {z_limit:.2f}"
                )
            if vol_z > z_limit:
                record.is_anomalous = True
                record.anomaly_reasons.append(
                    f"Search volume Z-score {vol_z:.2f} exceeds threshold {z_limit:.2f}"
                )
            if record.cpc_usd > self.config.thresholds.max_cpc_ceiling_usd * 2.5:
                record.is_anomalous = True
                record.anomaly_reasons.append(
                    f"Extreme outlier CPC (${record.cpc_usd:.2f}) exceeds 2.5x ceiling"
                )

            if record.is_anomalous:
                report.anomaly_count += 1

        return report

    def _stream_json_records(self, path: Path) -> Iterator[Tuple[int, Any]]:
        """Yield records from either a JSON array or JSON Lines (JSONL) file with corruption recovery."""
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return

        # Attempt standard JSON parsing first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for idx, item in enumerate(parsed):
                    yield idx, item
                return
            elif isinstance(parsed, dict) and "assets" in parsed and isinstance(parsed["assets"], list):
                for idx, item in enumerate(parsed["assets"]):
                    yield idx, item
                return
            else:
                yield 0, CorruptedRecordError(
                    index=0,
                    raw_payload=content[:200],
                    reason="Root JSON structure must be an array of asset objects or contain an 'assets' list",
                )
                return
        except json.JSONDecodeError:
            # Fallback to line-by-line JSONL / corrupted JSON recovery
            for idx, line in enumerate(content.splitlines()):
                stripped = line.strip().rstrip(",")
                if not stripped or stripped in ("[", "]", "{", "}"):
                    continue
                try:
                    obj = json.loads(stripped)
                    yield idx, obj
                except json.JSONDecodeError as exc:
                    yield idx, CorruptedRecordError(
                        index=idx,
                        raw_payload=stripped[:200],
                        reason=f"Malformed JSON fragment: {exc.msg}",
                    )

    def _stream_csv_records(self, path: Path) -> Iterator[Tuple[int, Any]]:
        """Stream rows from a CSV file as dictionaries."""
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                yield idx, dict(row)

    def _validate_and_normalize(self, raw: Any, index: int) -> AssetRecord:
        """Enforce strict domain, numeric, and structural constraints on a single raw record.

        Args:
            raw: Unvalidated dictionary payload.
            index: Stream index for fallback ID generation and diagnostics.

        Returns:
            AssetRecord: Clean, normalized digital asset entity.

        Raises:
            ValueError: If any schema constraint is violated.
            TypeError: If `raw` is not a mapping.
        """
        if not isinstance(raw, dict):
            raise TypeError(f"Record at index {index} must be a dict, got {type(raw).__name__}")

        domain_raw = str(raw.get("domain", "")).strip().lower()
        if not domain_raw or not DOMAIN_REGEX.match(domain_raw):
            raise ValueError(f"Invalid or missing RFC-1035 domain name: '{domain_raw}'")

        sld, tld = domain_raw.rsplit(".", 1)
        declared_tld = str(raw.get("tld", tld)).strip().lower().lstrip(".")
        if declared_tld != tld:
            raise ValueError(
                f"TLD mismatch for '{domain_raw}': domain suffix is '.{tld}' but tld field is '.{declared_tld}'"
            )

        asset_id = str(raw.get("asset_id") or f"ASSET-{index + 1:04d}").strip()

        try:
            search_volume = int(raw["search_volume"])
            cpc_usd = float(raw["cpc_usd"])
            age_years = float(raw["age_years"])
            keyword_count = int(raw.get("keyword_count", 1))
        except KeyError as exc:
            raise ValueError(f"Missing required metric field: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric metric value encountered: {exc}") from exc

        if search_volume < 0:
            raise ValueError(f"search_volume must be non-negative, got {search_volume}")
        if cpc_usd < 0.0 or math.isnan(cpc_usd) or math.isinf(cpc_usd):
            raise ValueError(f"cpc_usd must be a finite non-negative float, got {cpc_usd}")
        if age_years < 0.0 or age_years > 60.0 or math.isnan(age_years):
            raise ValueError(f"age_years must be within [0.0, 60.0], got {age_years}")
        if keyword_count < 1 or keyword_count > 12:
            raise ValueError(f"keyword_count must be within [1, 12], got {keyword_count}")

        sector = str(raw.get("sector", "Enterprise Tech")).strip() or "Enterprise Tech"

        raw_sales = raw.get("historical_sales", [])
        if isinstance(raw_sales, str):
            raw_sales = json.loads(raw_sales) if raw_sales.strip() else []
        if not isinstance(raw_sales, list):
            raise ValueError("historical_sales must be a list of transaction objects")

        historical_sales = [HistoricalSale.from_dict(item) for item in raw_sales]

        return AssetRecord(
            asset_id=asset_id,
            domain=domain_raw,
            sld=sld,
            tld=tld,
            search_volume=search_volume,
            cpc_usd=cpc_usd,
            age_years=age_years,
            keyword_count=keyword_count,
            sector=sector,
            historical_sales=historical_sales,
        )

