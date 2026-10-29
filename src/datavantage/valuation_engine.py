"""Multi-factor algorithmic valuation engine for B2B digital assets.

Computes normalized [0.0, 100.0] factor scores and estimated fair-market USD valuations
using mathematical heuristics across four orthogonal dimensions:
1. Commercial Intent Score (search demand + CPC monetization power)
2. Brandability Entropy (Shannon character entropy, phonetic vowel-consonant ratio, length decay)
3. TLD Authority Weight (extension tier + keyword conciseness synergy)
4. Liquidity Index (historical sales velocity, price anchoring, and domain maturity)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from datavantage.config import AppConfig
from datavantage.ingestor import AssetRecord

VOWELS = frozenset("aeiouy")


@dataclass(slots=True)
class ValuationResult:
    """Comprehensive valuation scorecard produced for a single digital asset.

    Attributes:
        asset_id: Canonical asset identifier.
        domain: Evaluated domain name.
        tld: Normalized top-level domain.
        sector: Target B2B industry vertical.
        commercial_intent_score: Normalized score [0.0, 100.0] for commercial demand.
        brandability_entropy_score: Normalized score [0.0, 100.0] for lexical/phonetic quality.
        tld_authority_score: Normalized score [0.0, 100.0] for extension tier & keyword synergy.
        liquidity_index_score: Normalized score [0.0, 100.0] for secondary-market liquidity.
        composite_score: Weighted multi-factor score in [0.0, 100.0].
        estimated_value_usd: Algorithmic fair-market valuation in USD.
        confidence_interval_usd: Tuple of `(low_usd, high_usd)` valuation band.
        liquidity_tier: Categorical classification ('PRIME', 'LIQUID', 'GROWTH', 'ILLIQUID').
        is_anomalous: True if the underlying telemetry triggered anomaly detection.
        evaluated_at: UTC ISO-8601 timestamp of evaluation.
    """

    asset_id: str
    domain: str
    tld: str
    sector: str
    commercial_intent_score: float
    brandability_entropy_score: float
    tld_authority_score: float
    liquidity_index_score: float
    composite_score: float
    estimated_value_usd: float
    confidence_interval_usd: tuple[float, float]
    liquidity_tier: str
    is_anomalous: bool = False
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize valuation result to a JSON-compatible dictionary."""
        data = asdict(self)
        data["confidence_low_usd"] = self.confidence_interval_usd[0]
        data["confidence_high_usd"] = self.confidence_interval_usd[1]
        return data


class ValuationEngine:
    """Quantitative multi-factor scoring engine for B2B digital assets."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the valuation engine with configured weights and thresholds.

        Args:
            config: Optional AppConfig instance; uses default configuration when omitted.
        """
        self.config = config or AppConfig()

    # =========================================================================
    # TODO (Phase 2): Semantic Vector Embedding Clustering & Market Comp Graph
    # -------------------------------------------------------------------------
    # Architectural Specification for Phase 2 Implementation:
    # 1. Integrate dense B2B taxonomy embeddings (768-d sentence-transformers or
    #    OpenAI `text-embedding-3-small`) to project domain tokens into a metric
    #    vector space and compute cosine similarity against high-multiple SaaS,
    #    FinTech, AI Infrastructure, and Cybersecurity category centroids.
    # 2. Construct an in-memory Hierarchical Navigable Small World (HNSW) index
    #    over historical verified comp sales so `calculate_liquidity_index`
    #    incorporates k-NN distance-weighted comparable multiples alongside
    #    direct asset sales history.
    # =========================================================================

    # =========================================================================
    # TODO (Phase 2): Real-Time Orderbook Liquidity Depth Calibration
    # -------------------------------------------------------------------------
    # Architectural Specification for Phase 2 Implementation:
    # 1. Implement a dynamic liquidity discount factor modeled via Ornstein-
    #    Uhlenbeck mean-reverting spread dynamics across active B2B auction
    #    orderbooks (Sedo, Afternic, NameBio live feeds).
    # 2. Adjust `confidence_interval_usd` bands dynamically using conditional
    #    heteroskedastic variance (GARCH(1,1)) calibrated per TLD cohort.
    # =========================================================================

    def evaluate(self, asset: AssetRecord) -> ValuationResult:
        """Execute full multi-factor valuation on a validated AssetRecord.

        Args:
            asset: Validated digital asset telemetry record.

        Returns:
            ValuationResult: Complete quantitative scorecard and USD valuation.
        """
        commercial_score = self.calculate_commercial_intent(
            search_volume=asset.search_volume,
            cpc_usd=asset.cpc_usd,
        )
        brandability_score = self.calculate_brandability_entropy(
            sld=asset.sld,
            keyword_count=asset.keyword_count,
        )
        tld_score = self.calculate_tld_authority(
            tld=asset.tld,
            keyword_count=asset.keyword_count,
        )
        liquidity_score = self.calculate_liquidity_index(
            age_years=asset.age_years,
            historical_sales_count=len(asset.historical_sales),
            historical_avg_price=asset.historical_avg_price,
        )

        weights = self.config.weights
        composite = (
            weights.commercial_intent * commercial_score
            + weights.brandability_entropy * brandability_score
            + weights.tld_authority * tld_score
            + weights.liquidity_index * liquidity_score
        )
        composite = round(self._clamp(composite, 0.0, 100.0), 2)

        estimated_usd = self._compute_usd_valuation(
            composite_score=composite,
            commercial_score=commercial_score,
            tld_score=tld_score,
            asset=asset,
        )

        # Confidence band narrows with higher historical sales depth and age maturity
        uncertainty_pct = max(0.12, 0.35 - 0.04 * min(len(asset.historical_sales), 4))
        low_usd = round(estimated_usd * (1.0 - uncertainty_pct), 2)
        high_usd = round(estimated_usd * (1.0 + uncertainty_pct), 2)

        return ValuationResult(
            asset_id=asset.asset_id,
            domain=asset.domain,
            tld=asset.tld,
            sector=asset.sector,
            commercial_intent_score=round(commercial_score, 2),
            brandability_entropy_score=round(brandability_score, 2),
            tld_authority_score=round(tld_score, 2),
            liquidity_index_score=round(liquidity_score, 2),
            composite_score=composite,
            estimated_value_usd=round(estimated_usd, 2),
            confidence_interval_usd=(low_usd, high_usd),
            liquidity_tier=self._classify_tier(composite, liquidity_score),
            is_anomalous=asset.is_anomalous,
        )

    def evaluate_batch(self, assets: List[AssetRecord]) -> List[ValuationResult]:
        """Evaluate a batch of digital assets and return results sorted by descending value.

        Args:
            assets: Collection of validated AssetRecord objects.

        Returns:
            List[ValuationResult]: Ranked valuation results.
        """
        results = [self.evaluate(asset) for asset in assets]
        results.sort(key=lambda r: r.estimated_value_usd, reverse=True)
        return results

    def calculate_commercial_intent(self, search_volume: int, cpc_usd: float) -> float:
        """Compute Commercial Intent Score in [0.0, 100.0] from search demand and CPC.

        Combines log-normalized monthly search volume with square-root compressed CPC
        and a synergistic cross-product term representing organic traffic replacement value.

        Args:
            search_volume: Monthly keyword search volume (>= 0).
            cpc_usd: Average cost-per-click in USD (>= 0.0).

        Returns:
            float: Normalized Commercial Intent Score in [0.0, 100.0].
        """
        if search_volume <= 0 and cpc_usd <= 0.0:
            return 0.0

        vol_ceiling = max(1, self.config.thresholds.max_search_volume_ceiling)
        cpc_ceiling = max(1.0, self.config.thresholds.max_cpc_ceiling_usd)

        vol_norm = min(1.0, math.log10(max(0, search_volume) + 1.0) / math.log10(vol_ceiling + 1.0))
        cpc_norm = min(1.0, math.sqrt(max(0.0, cpc_usd)) / math.sqrt(cpc_ceiling))
        synergy = math.sqrt(vol_norm * cpc_norm)

        raw_score = 100.0 * (0.45 * vol_norm + 0.40 * cpc_norm + 0.15 * synergy)
        return self._clamp(raw_score, 0.0, 100.0)

    def calculate_brandability_entropy(self, sld: str, keyword_count: int) -> float:
        """Compute Brandability Entropy Score in [0.0, 100.0] using information theory and phonetics.

        Evaluates four lexical signals on the second-level domain (SLD):
        1. Normalized Shannon character entropy $H(X) / \\log_2(|SLD|)$ to penalize repetitive strings.
        2. Gaussian length optimality centered around `optimal_domain_length` (7 chars).
        3. Phonetic pronounceability via deviation from ideal vowel ratio (~0.38).
        4. Structural penalties for hyphens, digits, and excessive keyword fragmentation.

        Args:
            sld: Second-level domain string (without TLD).
            keyword_count: Number of semantic words in the SLD.

        Returns:
            float: Normalized Brandability Entropy Score in [0.0, 100.0].
        """
        clean_sld = sld.strip().lower()
        length = len(clean_sld)
        if length == 0:
            return 0.0

        # 1. Normalized Shannon Entropy
        if length == 1:
            normalized_entropy = 1.0
        else:
            counts = Counter(clean_sld)
            shannon_h = -sum((c / length) * math.log2(c / length) for c in counts.values())
            max_h = math.log2(length)
            normalized_entropy = shannon_h / max_h if max_h > 0 else 0.0

        # 2. Gaussian Length Decay around optimal_domain_length
        opt_len = self.config.thresholds.optimal_domain_length
        sigma = 5.5
        length_factor = math.exp(-((length - opt_len) ** 2) / (2.0 * (sigma**2)))

        # 3. Phonetic Pronounceability (vowel-to-character balance)
        alpha_chars = [ch for ch in clean_sld if ch.isalpha()]
        if alpha_chars:
            vowel_ratio = sum(1 for ch in alpha_chars if ch in VOWELS) / len(alpha_chars)
            phonetic_balance = max(0.0, 1.0 - abs(vowel_ratio - 0.38) / 0.38)
        else:
            phonetic_balance = 0.15

        # 4. Structural Penalties (hyphens, digits, multi-word dilution)
        hyphen_penalty = 0.25 if "-" in clean_sld else 0.0
        digit_penalty = 0.12 if any(ch.isdigit() for ch in clean_sld) else 0.0
        keyword_penalty = max(0.0, (keyword_count - 1) * 0.08)

        base_score = 100.0 * (
            0.35 * normalized_entropy + 0.40 * length_factor + 0.25 * phonetic_balance
        )
        penalized_score = base_score * max(0.15, 1.0 - hyphen_penalty - digit_penalty - keyword_penalty)
        return self._clamp(penalized_score, 0.0, 100.0)

    def calculate_tld_authority(self, tld: str, keyword_count: int = 1) -> float:
        """Compute TLD Authority Score in [0.0, 100.0] weighted by keyword conciseness.

        Single-word and two-word category-defining domains on top-tier extensions (.com, .ai, .io)
        capture a structural authority premium.

        Args:
            tld: Top-level domain string.
            keyword_count: Number of semantic tokens in the domain label.

        Returns:
            float: Normalized TLD Authority Score in [0.0, 100.0].
        """
        base_weight = self.config.get_tld_weight(tld)
        conciseness_modifier = 1.0 if keyword_count <= 1 else max(0.75, 1.0 - 0.07 * (keyword_count - 1))
        score = 100.0 * base_weight * conciseness_modifier
        return self._clamp(score, 0.0, 100.0)

    def calculate_liquidity_index(
        self,
        age_years: float,
        historical_sales_count: int,
        historical_avg_price: float,
    ) -> float:
        """Compute Liquidity Index Score in [0.0, 100.0] from age maturity and sales velocity.

        Args:
            age_years: Continuous registration age in years.
            historical_sales_count: Count of verified historical marketplace sales.
            historical_avg_price: Mean USD price across historical sales.

        Returns:
            float: Normalized Liquidity Index Score in [0.0, 100.0].
        """
        age_ceiling = max(1.0, float(self.config.thresholds.max_age_years_ceiling))
        age_norm = min(1.0, max(0.0, age_years) / age_ceiling)

        velocity_norm = 1.0 - math.exp(-0.55 * max(0, historical_sales_count))
        price_depth_norm = min(
            1.0, math.log10(max(0.0, historical_avg_price) + 1.0) / math.log10(100_000.0)
        )

        score = 100.0 * (0.40 * age_norm + 0.30 * velocity_norm + 0.30 * price_depth_norm)
        return self._clamp(score, 0.0, 100.0)

    def _compute_usd_valuation(
        self,
        composite_score: float,
        commercial_score: float,
        tld_score: float,
        asset: AssetRecord,
    ) -> float:
        """Convert normalized factor scores and historical anchors into a fair-market USD valuation."""
        base_mult = self.config.thresholds.base_valuation_multiplier
        # Non-linear convex appreciation for top-decile composite scores
        exponential_growth = math.exp(composite_score / 22.5)
        monetization_kick = 1.0 + (commercial_score / 100.0) * (tld_score / 100.0) * 1.65
        intrinsic_usd = base_mult * (composite_score / 50.0) * exponential_growth * monetization_kick

        # Blend with historical average sale price when verifiable comps exist on the exact asset
        if asset.historical_sales:
            hist_anchor = asset.historical_avg_price
            # Weight historical anchor up to 35% based on transaction count
            anchor_weight = min(0.35, 0.15 * len(asset.historical_sales))
            return (1.0 - anchor_weight) * intrinsic_usd + anchor_weight * hist_anchor

        return max(100.0, intrinsic_usd)

    @staticmethod
    def _classify_tier(composite_score: float, liquidity_score: float) -> str:
        """Classify asset into an institutional liquidity tier."""
        if composite_score >= 78.0 and liquidity_score >= 65.0:
            return "PRIME"
        if composite_score >= 62.0:
            return "LIQUID"
        if composite_score >= 42.0:
            return "GROWTH"
        return "ILLIQUID"

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        """Clamp a floating-point value to the closed interval [low, high]."""
        return max(low, min(high, value))

