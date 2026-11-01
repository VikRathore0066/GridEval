# GridEval: Empirical Benchmarking of Classical GBDTs and Time-Series Foundation Models

GridEval benchmarks domain-trained point-regression and quantile GBDTs against pre-trained zero-shot Time Series Foundation Models (TSFMs) for hourly load forecasting across ERCOT and GEFCom 2017.

**Live Dashboard**: [https://grideval-5x6uktiqevdmtugeqgxm6l.streamlit.app/](https://grideval-5x6uktiqevdmtugeqgxm6l.streamlit.app/)

---

## Datasets & Split Protocol

The benchmark evaluates models across two power grid scales and regions:

1. **ERCOT (Texas Interconnection)**:
   - Macro-scale power grid (~70,000 MW peak demand).
   - Hourly electricity demand spanning 2020 to 2024 (~43,800 timestamps).

2. **GEFCom 2017 (ISO-NE Connecticut Zone)**:
   - Regional-scale power grid (~4,000 MW peak demand).
   - Hourly electricity demand paired with drybulb and dewpoint temperature data (2012 to 2016).

### Data Split Protocol & Measured Observation Counts
A chronological split is applied across both grids to prevent temporal leakage:
- **ERCOT**:
  - Train: 26,304 observations (2020-01-01–2022-12-31)
  - Validation: 8,760 observations (2023-01-01–2023-12-31)
  - Test: 8,784 observations (2024-01-01–2024-12-31)
- **GEFCom**:
  - Train: 26,298 observations (2012-01-01–2014-12-31)
  - Validation: 8,758 observations (2015-01-01–2015-12-31)
  - Test: 8,782 observations (2016-01-01–2016-12-31)

---

## Evaluated Models

1. **Supervised GBDTs (Domain-Trained)**:
   - **Point Regression GBDTs**: LightGBM & XGBoost trained with squared error loss for point forecasts.
   - **Quantile GBDTs**: Separately trained LightGBM & XGBoost with pinball loss across an 11-quantile evaluation grid (alpha = 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95).
   - **Features**: Autoregressive lags (t-1, t-2, t-3, t-24, t-168), rolling statistics (24h, 168h mean/std), calendar encodings, and holiday indicators (weather excluded by default via `include_weather=False`).

2. **Pre-Trained Foundation Models (Zero-Shot TSFM)**:
   - **Amazon Chronos-T5-Small**: Evaluated zero-shot with a 512-observation context in the primary comparison. The separate context-length sweep outputs have not been regenerated under the corrected protocol.

---

## Evaluation Metrics

### Point Forecasting Accuracy
- **MAE**: Mean Absolute Error (MW)
- **RMSE**: Root Mean Squared Error (MW)
- **MAPE**: Mean Absolute Percentage Error (%)

### Probabilistic Calibration & Uncertainty
- **CRPS**: Continuous Ranked Probability Score (probability-weighted quantile loss integral for GBDTs, empirical sample score for Chronos)
- **PICP (90%)**: Prediction Interval Coverage Probability (nominal target: 90.0%)
- **MPIW (90%)**: Mean Prediction Interval Width (sharpness in MW)
- **Quantile Calibration Error**: Mean absolute deviation across tested quantile levels (0.05 to 0.95)

---

## Regenerated Benchmark Results

The tables below use the latest baseline and calibration runs on the canonical 2024 ERCOT and 2016 GEFCom test splits. All three models use the same 24-observation rolling forecast origins and target identities. GBDT forecasts update lag history recursively within each block; Chronos forecasts each block from information available at its origin. The GBDT point-regression and quantile models are separately trained.

### Point Forecast Accuracy

| Grid | Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---:|---:|---:|
| ERCOT | LightGBM | 2,337.55 | 3,560.92 | 4.34 |
| ERCOT | XGBoost | 2,270.14 | 3,383.25 | 4.17 |
| ERCOT | Chronos-T5-Small (zero-shot) | 1,907.38 | 2,947.31 | 3.59 |
| GEFCom | LightGBM | 157.27 | 227.31 | 4.39 |
| GEFCom | XGBoost | 157.73 | 229.21 | 4.33 |
| GEFCom | Chronos-T5-Small (zero-shot) | 155.35 | 241.23 | 4.29 |

### Probabilistic Forecast Evaluation

The 90% interval uses the 5th and 95th percentile forecasts. No post-hoc interval correction was applied; PICP reports how often actual load fell inside these intervals. Lower CRPS and ECE are better, while PICP is compared with the nominal 90% target.

| Grid | Model | CRPS (MW) | PICP 90% (%) | Mean Width (MW) | ECE |
|---|---|---:|---:|---:|---:|
| ERCOT | LightGBM Quantile | 2,493.91 | 68.72 | 11,180.74 | 0.1837 |
| ERCOT | XGBoost Quantile | 2,587.26 | 67.04 | 11,472.15 | 0.1811 |
| ERCOT | Chronos-T5-Small | 1,429.74 | 77.02 | 5,654.04 | 0.0492 |
| GEFCom | LightGBM Quantile | 176.47 | 60.73 | 684.32 | 0.1242 |
| GEFCom | XGBoost Quantile | 183.53 | 61.06 | 706.67 | 0.1195 |
| GEFCom | Chronos-T5-Small | 116.07 | 75.55 | 441.10 | 0.0522 |

### Findings and Limits

- Chronos has the lowest MAE on both test sets. Its GEFCom MAE advantage is small, and its GEFCom RMSE is higher than both GBDTs.
- Chronos has lower CRPS and ECE than the quantile GBDTs on both datasets, but its nominal 90% intervals cover only 77.02% of ERCOT and 75.55% of GEFCom observations. The GBDT intervals cover fewer observations. None of the models meets the 90% coverage target.
- These results evaluate unadjusted forecast intervals; they do not show post-hoc calibration to 90%.
- The default feature policy excludes weather. The results do not measure the effect of adding GEFCom weather features.
- Each dataset uses a three-year training window, though earlier records are available. An expanded-history comparison has not been run.
- The separate Chronos context-length sweep has not been regenerated under this shared evaluation protocol and is not used for these findings. Older sweep outputs and September 2026 run summaries are retained under `results/archive/legacy_pre_shared_protocol/` for provenance; they are excluded from current dashboard findings and experiment skip decisions.

## Dashboard Structure (`app.py`)

The Streamlit dashboard contains 5 analysis tabs:

1. **Leaderboard & Cross-Grid Comparison**: Side-by-side metric tables for point accuracy and calibration across ERCOT and GEFCom.
2. **Forecast Trajectories & Interval Visualizer**: Interactive Plotly time-series showing median predictions (P50), actual load, and 90% uncertainty bands across seasonal regimes (Heatwave, Cold Snap, Shoulder, Full Year).
3. **Calibration & Reliability Analysis**: Empirical vs nominal quantile curves evaluating model coverage and ECE.
4. **Context Length Scaling Analysis**: TSFM accuracy, coverage, and latency sensitivity curves across 72-observation to 1,000-observation context lengths.
5. **Interactive Scenario Simulator**: Real-time stress-testing engine with temperature shock (+/-25 deg F) and structural demand growth (+/-25%) sliders, computing peak demand (MW), integrated energy (GWh), and spinning reserve requirements.

---

## Repository Structure

```
GridEval/
|-- configs/            # Experiment YAML configurations
|-- datasets/           # Raw and yearly partitioned source data
|-- data/processed/     # Cleaned and split Parquet files
|-- notebooks/          # Exploratory Data Analysis (EDA)
|-- results/            # Benchmark outputs (master_table.csv, calibration_master.csv)
|   |-- ercot/          # Active ERCOT metrics and aligned forecasts
|   |-- gefcom/         # Active GEFCom metrics and aligned forecasts
|   |-- archive/        # Superseded pre-protocol result files, retained for provenance
|   `-- plots/          # Reliability and calibration plots
|-- src/                # Modular benchmark codebase
|   |-- baselines.py        # GBDT and heuristic baseline models
|   |-- tsfm_models.py      # Chronos inference pipeline
|   |-- features.py         # Lag, rolling, and temporal feature engineering
|   |-- evaluation.py       # Vectorized accuracy, CRPS, PICP, and ECE routines
|   |-- data_loader.py      # Chronological split validation and data ingestion
|   |-- run_calibration.py  # Probabilistic calibration benchmarking suite
|   |-- run_tsfm_sweep.py   # Context length sweep (72-observation to 1000 observations)
|   |-- baseline_summary.py # Terminal summary printer
|   `-- main.py             # Orchestration and table compilation
|-- tests/              # Unit and integration test suite
|-- app.py              # Streamlit evaluation and scenario dashboard
|-- run_dashboard.py    # Local launcher script
|-- requirements.txt    # Pinned dashboard dependencies
|-- requirements-train.txt # Pinned experiment and notebook dependencies
`-- pyproject.toml      # Core package metadata and optional dependency groups
```

---

## Quick Start

### 1. Environment Setup
Use Python 3.10, 3.11, or 3.12 for the pinned training environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dashboard manifest pins its direct dependencies. To run the full
experiments and edit the notebooks, install the training and notebook
dependencies as well:

```bash
pip install -r requirements-train.txt
```

For Chronos inference on an NVIDIA GPU, install a CUDA-enabled PyTorch wheel
that matches the machine's CUDA driver **before** installing the training
requirements. For example, in Linux or WSL with a CUDA 12.1-compatible setup:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-train.txt
```

The training manifest explicitly includes PyTorch, Transformers, and
Accelerate. The GPU wheel is installed separately because the correct CUDA
wheel depends on the host system; a normal `pip install torch` may install a
CPU-only build. Verify GPU access before a long run with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Run Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Compile or Run Benchmarks
```bash
# Compile summary CSV tables from benchmark runs
python src/main.py --compile-only

# Print baseline summary table to terminal
python src/baseline_summary.py

# Run all benchmark pipelines (skips completed stages)
python src/main.py --skip-existing
```

### 4. Launch Dashboard Locally
```bash
python run_dashboard.py
```
Or directly with Streamlit:
```bash
streamlit run app.py
```

## Data Ingestion & Canonical Workflow

### ERCOT Ingestion & Provenance
To ingest raw ERCOT annual CSV files into the canonical processed dataset with full provenance tracking (source file, row number, global source order, and timestamp/occurrence multi-index):
```bash
python3 ingest_ercot.py
```
- **Canonical Output Path**: `data/processed/ercot_clean.parquet`
- **Split Indices**: `data/processed/ercot_train_idx.parquet`, `ercot_val_idx.parquet`, `ercot_test_idx.parquet`
- **Missing-Row Audit Log**: `results/ercot_missing_audit.json`
- **Timestamp & Occurrence Semantics**: Raw hourly records are parsed chronologically. Autumn DST fallback hours (`02:00:00`) appearing twice are indexed using a MultiIndex `(timestamp, occurrence)` where `occurrence` (0 or 1) preserves distinct source observations and source order without arbitrary minute offsets or unverified timezone labels.
- **Missing-Row Policy**: The completely empty row at `2016-11-07 00:00:00` (source row 7463 in `2016_ercot_hourly_load_data.csv`) is retained in the raw-derived artifact and logged in `results/ercot_missing_audit.json`, while being transparently excluded from target-based training/evaluation (`load_dataset`).

### Running Tests
To run the focused test suite (including ingestion provenance, audit logging, feature preparation, split membership, and exact-identity evaluation alignment):
```bash
PYTHONPATH=. python3 tests/test_ingest.py
PYTHONPATH=. python3 tests/test_baselines.py
```


### Dataset Splits & Measured Observation Counts
- **ERCOT**:
  - Train: 26,304 observations (2020-01-01–2022-12-31)
  - Validation: 8,760 observations (2023-01-01–2023-12-31)
  - Test: 8,784 observations (2024-01-01–2024-12-31)
- **GEFCom**:
  - Train: 26,298 observations (2012-01-01–2014-12-31)
  - Validation: 8,758 observations (2015-01-01–2015-12-31)
  - Test: 8,782 observations (2016-01-01–2016-12-31)

### Forecast Contract & Evaluation Protocol
Both GBDT (LightGBM/XGBoost) and TSFM (Chronos) evaluation pipelines operate on an **observation-wise rolling evaluation protocol**:
- **Horizon & Step**: Evaluated in blocks of **24 target observations** stepping by **24 observations**.
- **Context Length**: Measured in observation steps (e.g. 72, 168, 512, 1000 observations).
- **Dataset Gaps & Repeated Observations**: Due to dataset-specific timestamp gaps and repeated autumn DST fallback hours (represented via MultiIndex `(timestamp, occurrence)`), these 24-observation blocks represent sequential dataset records in chronological order. Because of temporal irregularities, observation counts do not guarantee strict wall-clock hourly intervals across all DST transitions.
- **Fairness & Alignment**: Both model paths share the exact same ordered target identities and rolling evaluation origins derived from the canonical dataset splits, ensuring a consistent and reproducible comparison without data leakage. Primary baseline and calibration results are regenerated under this unified protocol. Context-length sweep artifacts remain pending regeneration.
