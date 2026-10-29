# DataVantage CLI

**Asynchronous B2B Digital Asset Valuation & Market Intelligence Pipeline**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#quickstart--cli-usage)
[![Architecture](https://img.shields.io/badge/architecture-SQLite3%20WAL%20%2B%20LRU%20Cache-success.svg)](#system-architecture)
[![Code Style](https://img.shields.io/badge/typing-strict%20PEP%20484-informational.svg)](#mathematical-valuation-model)

---

## 1. Project Vision & Problem Statement

Institutional private equity funds, venture studios, and corporate M&A desks acquire and manage thousands of high-value B2B digital assets (category-defining `.com`, `.ai`, `.io`, `.security`, and `.cloud` domains). Despite secondary-market transactions routinely clearing between **$15,000 and $500,000+**, portfolio valuation remains plagued by opaque broker heuristics, fragmented historical comps, and brittle spreadsheet models that fail when exposed to malformed registrar feeds or extreme Cost-Per-Click (CPC) outliers.

**DataVantage CLI** solves this by delivering an institutional-grade, deterministic valuation and market intelligence pipeline engineered for speed, auditability, and fault tolerance:

* **Zero-Halt Stream Ingestion**: Memory-efficient generator pipelines ingest heterogeneous JSON/CSV feeds, automatically quarantining malformed records while computing online **Welford running moments** to flag statistical CPC and search-volume anomalies ($|Z| > 3.2$).
* **Quantitative Multi-Factor Scoring**: Evaluates every digital asset across four orthogonal mathematical dimensions—**Commercial Intent**, **Brandability Entropy**, **TLD Authority Weight**, and **Liquidity Index**—producing normalized $[0, 100]$ factor scores, confidence intervals, and liquidity tier classifications (`PRIME`, `LIQUID`, `GROWTH`, `ILLIQUID`).
* **Low-Latency WAL Storage**: Backed by an embedded **SQLite3 Write-Ahead Logging (WAL)** persistence engine with composite B-tree indexes and a thread-safe **Least-Recently-Used (LRU)** query cache for sub-millisecond portfolio slice queries.

---

## 2. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                            DATAVANTAGE CLI PIPELINE                               |
|                (ingest | evaluate | export | benchmark subcommands)               |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 1. STREAM INGESTION & ANOMALY ENGINE (src/datavantage/ingestor.py)                |
|                                                                                   |
|  [JSON / CSV Feeds] ---> [RFC-1035 & Schema Validator] ---> [Quarantine Buffer]   |
|                                         |                                         |
|                                         v                                         |
|                     [Online Welford Z-Score Anomaly Detector]                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. MULTI-FACTOR VALUATION ENGINE (src/datavantage/valuation_engine.py)            |
|                                                                                   |
|  +-----------------------+  +-----------------------+  +-----------------------+  |
|  | Commercial Intent     |  | Brandability Entropy  |  | TLD Authority &       |  |
|  | Log-Search + Sqrt-CPC |  | Shannon H(X) + Vowel  |  | Liquidity Velocity    |  |
|  | + Synergy Cross-Term  |  | Ratio + Gaussian Len  |  | Decay + Comp Anchor   |  |
|  +-----------+-----------+  +-----------+-----------+  +-----------+-----------+  |
|              |                          |                          |              |
|              +--------------------------+--------------------------+              |
|                                         |                                         |
|                                         v                                         |
|             [Convex USD Fair-Market Estimator & Confidence Banding]               |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. PERSISTENCE & CACHING LAYER (src/datavantage/storage.py)                       |
|                                                                                   |
|  [Thread-Safe LRU Query Cache] <===> [SQLite3 Engine (PRAGMA journal_mode = WAL)] |
|                                      - Table: raw_assets (Indexed by TLD, Sector) |
|                                      - Table: asset_valuations (Indexed by Score) |
+-----------------------------------------------------------------------------------+
```

---

## 3. Mathematical Valuation Model

Each validated `AssetRecord` is scored across four normalized $[0, 100]$ factors before convex conversion into an estimated fair-market USD valuation.

### 3.1 Commercial Intent Score ($S_{\text{intent}}$)
Captures organic traffic replacement value by combining logarithmic search demand, square-root compressed CPC monetization, and a geometric synergy term:

$$
V_{\text{norm}} = \min\left(1, \frac{\log_{10}(\text{SearchVol} + 1)}{\log_{10}(V_{\max} + 1)}\right), \quad C_{\text{norm}} = \min\left(1, \sqrt{\frac{\text{CPC}}{\text{CPC}_{\max}}}\right)
$$

$$
S_{\text{intent}} = 100 \times \left(0.45\,V_{\text{norm}} + 0.40\,C_{\text{norm}} + 0.15\,\sqrt{V_{\text{norm}} \cdot C_{\text{norm}}}\right)
$$

### 3.2 Brandability Entropy Score ($S_{\text{brand}}$)
Combines normalized Shannon character entropy $H(X)$, Gaussian length optimality centered at $L_{\text{opt}} = 7$ characters ($\sigma = 5.5$), and phonetic vowel balance ($\rho^* = 0.38$), penalized for hyphens, digits, and multi-token dilution:

$$
\hat{H}(X) = -\frac{1}{\log_2 L}\sum_{c \in \Sigma} p(c)\log_2 p(c), \quad G(L) = \exp\left(-\frac{(L - L_{\text{opt}})^2}{2\sigma^2}\right)
$$

$$
S_{\text{brand}} = 100 \times \left(0.35\,\hat{H}(X) + 0.40\,G(L) + 0.25\,B_{\text{phonetic}}\right) \times (1 - P_{\text{struct}})
$$

### 3.3 TLD Authority Score ($S_{\text{tld}}$) & Liquidity Index ($S_{\text{liq}}$)
* **TLD Authority ($S_{\text{tld}}$)**: Scales institutional TLD tier weights (`.com` $= 1.00$, `.ai` $= 0.92$, `.io` $= 0.85$, `.security` $= 0.76$, `.cloud` $= 0.74$) by a semantic conciseness modifier that rewards single-word category killers.
* **Liquidity Index ($S_{\text{liq}}$)**: Combines domain registration maturity ($A_{\text{norm}}$), exponential transaction velocity $1 - e^{-0.55 N_{\text{sales}}}$, and log-scaled historical clearing price depth.

### 3.4 Composite Score & Convex USD Conversion
Given normalized weights $\mathbf{w} = (0.35, 0.25, 0.20, 0.20)$:

$$
S_{\text{comp}} = 0.35\,S_{\text{intent}} + 0.25\,S_{\text{brand}} + 0.20\,S_{\text{tld}} + 0.20\,S_{\text{liq}}
$$

Intrinsic USD value is derived via convex exponential scaling anchored against verified historical transactions when prior comp sales exist on the asset.

---

## 4. Quickstart & CLI Usage

### Installation

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 1. Ingest & Immediately Evaluate the Sample B2B Dataset

```bash
python -m datavantage.cli ingest data/sample_assets.json --evaluate-now
```

### 2. Query & Filter Evaluated Portfolio by TLD, Tier, or Minimum Score

```bash
# Display top PRIME-tier assets in the vault
python -m datavantage.cli evaluate --tier PRIME --top 10

# Filter specifically for .ai digital assets with Composite Score >= 70
python -m datavantage.cli evaluate --tld ai --min-score 70
```

### 3. Export Portfolio Valuation Report to JSON or CSV

```bash
python -m datavantage.cli export --output reports/b2b_valuations.json
python -m datavantage.cli export --output reports/b2b_valuations.csv --min-score 65.0
```

### 4. Run Synthetic Throughput & SQLite WAL Latency Benchmark

```bash
python -m datavantage.cli benchmark --iterations 1000
```

---

## 5. Active Development Roadmap

| Phase | Milestone | Architectural Scope | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Core Pipeline & Quantitative Engine** | Stream JSON/CSV ingestor with Welford Z-score anomaly detection, 4-factor valuation engine, SQLite3 WAL persistence + LRU cache, Rich CLI (`ingest`, `evaluate`, `export`, `benchmark`), and unit test suite. | **COMPLETE** |
| **Phase 2.1** | **Async RDAP/WHOIS Concurrency Pool** | `aiohttp` connection pooling with bounded `asyncio.Semaphore(32)` and a `CLOSED -> HALF_OPEN -> OPEN` rate-limit backoff state machine (`src/datavantage/ingestor.py`). | **IN ACTIVE DEV** |
| **Phase 2.2** | **Semantic Vector Embedding Clustering** | 768-d B2B taxonomy embeddings + in-memory HNSW k-NN index for comparable sales clustering across SaaS, FinTech, and AI verticals (`src/datavantage/valuation_engine.py`). | **IN ACTIVE DEV** |
| **Phase 2.3** | **Live Orderbook Liquidity Calibration** | Ornstein-Uhlenbeck spread modeling and GARCH(1,1) conditional volatility bands across live auction feeds (`src/datavantage/valuation_engine.py`). | **PLANNED** |

