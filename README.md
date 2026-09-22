# GridEval: Empirical Benchmarking of Classical ML vs Time Series Foundation Models

GridEval benchmarks domain-trained classical ML models against pre-trained zero-shot Time Series Foundation Models (TSFMs) for electricity load forecasting across two power grid scales (ERCOT and GEFCom 2017).

**Live Dashboard**: [https://grideval-5x6uktiqevdmtugeqgxm6l.streamlit.app/](https://grideval-5x6uktiqevdmtugeqgxm6l.streamlit.app/)

---

## Datasets & Split Protocol

The benchmark evaluates models across two power grids:

1. **ERCOT (Texas Interconnection)**:
   - Macro-scale power grid (~70,000 MW peak demand).
   - Hourly electricity demand spanning 2020 to 2024 (~43,800 timestamps).
   - High industrial load growth and renewable penetration.

2. **GEFCom 2017 (ISO-NE Connecticut Zone)**:
   - Regional-scale power grid (~4,000 MW peak demand).
   - Hourly electricity demand paired with drybulb and dewpoint temperature data (2012 to 2016).
   - Non-linear temperature-load relationship (summer cooling, winter heating).

### Data Split Protocol
A chronological 3-1-1 split is applied across both grids to prevent temporal leakage:
- **Train**: 3 years (26,280 hours)
- **Validation**: 1 year (8,760 hours)
- **Test**: 1 year (8,760 hours)

---

## Evaluated Models

1. **Supervised GBDTs (Domain-Trained)**:
   - LightGBM & XGBoost with quantile loss (alpha = 0.05, 0.50, 0.95).
   - Features: Autoregressive lags (t-1, t-2, t-3, t-24, t-168), rolling statistics (24h, 168h mean/std), calendar encodings, holiday indicators, and temperature interactions (temperature squared, dewpoint depression).

2. **Classical & Heuristic Baselines**:
   - Prophet: Additive model with weekly/yearly Fourier seasonality and holiday effects.
   - Naive Seasonal: 168-hour persistence baseline (t = t-168).

3. **Pre-Trained Foundation Models (Zero-Shot TSFM)**:
   - Amazon Chronos-T5 (Small): Evaluated zero-shot without fine-tuning across context window sweeps from 72h to 1,000h.

---

## Evaluation Metrics

### Point Forecasting Accuracy
- **MAE**: Mean Absolute Error (MW)
- **RMSE**: Root Mean Squared Error (MW)
- **MAPE**: Mean Absolute Percentage Error (%)

### Probabilistic Calibration & Uncertainty
- **CRPS**: Continuous Ranked Probability Score (distributional sharpness and calibration)
- **PICP (90%)**: Prediction Interval Coverage Probability (nominal target: 90.0%)
- **MPIW (90%)**: Mean Prediction Interval Width (sharpness in MW)
- **ECE**: Expected Calibration Error across all quantile levels (0.05 to 0.95)

---

## Empirical Benchmark Results

### 1. Point Forecasting Performance (Test Year)

| Grid | Model | Type | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---|---|---|---|
| **ERCOT** | LightGBM | Classical ML | 505.53 | 798.08 | **0.90%** |
| **ERCOT** | XGBoost | Classical ML | 506.45 | 825.49 | 0.90% |
| **ERCOT** | Chronos-T5-Small (512h) | Zero-Shot TSFM | 1921.32 | 2976.93 | 3.61% |
| **ERCOT** | Naive Seasonal (168h) | Baseline | 4478.53 | 6274.29 | 8.38% |
| **ERCOT** | Prophet | Classical ML | 5304.77 | 6935.90 | 9.75% |
| **GEFCom** | XGBoost | Classical ML | 32.67 | 44.39 | **0.96%** |
| **GEFCom** | LightGBM | Classical ML | 32.73 | 44.67 | 0.96% |
| **GEFCom** | Chronos-T5-Small (512h) | Zero-Shot TSFM | 154.85 | 239.50 | 4.28% |
| **GEFCom** | Naive Seasonal (168h) | Baseline | 343.37 | 491.25 | 9.69% |
| **GEFCom** | Prophet | Classical ML | 586.37 | 736.84 | 16.80% |

### 2. Probabilistic Calibration Performance (Nominal 90% Intervals)

| Grid | Model | CRPS | PICP 90% (Target: 90%) | Width 90% (MW) | ECE |
|---|---|---|---|---|---|
| **ERCOT** | LightGBM Quantile | **827.49** | 88.11% | 5,863 MW | 0.0898 |
| **ERCOT** | XGBoost Quantile | 854.69 | **88.63%** | 6,328 MW | 0.0926 |
| **ERCOT** | Chronos-T5-Small (512h) | 1429.61 | 77.39% (Under-covered) | 5,679 MW | **0.0490** |
| **ERCOT** | Prophet | 4522.44 | 94.71% (Over-wide) | 39,697 MW | 0.0742 |
| **GEFCom** | LightGBM Quantile | **50.43** | **87.80%** | 337 MW | 0.0540 |
| **GEFCom** | XGBoost Quantile | 55.40 | 86.92% | 394 MW | **0.0482** |
| **GEFCom** | Chronos-T5-Small (512h) | 114.37 | 74.87% (Under-covered) | 437 MW | 0.0501 |
| **GEFCom** | Prophet | 467.30 | 94.51% (Over-wide) | 3,964 MW | 0.2018 |

---

## Key Findings

1. **GBDTs outperform Zero-Shot TSFMs on point accuracy**:
   - Feature-engineered LightGBM and XGBoost achieve ~0.90% to 0.96% MAPE across both grids, outperforming zero-shot Chronos-T5 (~3.61% to 4.28% MAPE). Domain-specific lag features and calendar interactions remain critical inductive biases for electrical grid demand.

2. **Zero-shot TSFMs exhibit interval under-coverage**:
   - Chronos-T5 achieves ~74.8% to 77.4% empirical coverage for nominal 90% prediction intervals, indicating overconfidence during peak and regime shifts.
   - Quantile GBDTs provide well-calibrated intervals (~86.9% to 88.6% PICP) without excessive interval inflation.

3. **Context window scaling reaches a plateau at 512 hours**:
   - Extending Chronos context length from 72h to 512h yields noticeable error reduction. Beyond 512h (720h, 1,000h), accuracy gains plateau while inference latency scales linearly.

---

## Dashboard Structure (`app.py`)

The Streamlit dashboard contains 5 analysis tabs:

1. **Leaderboard & Cross-Grid Comparison**: Side-by-side metric tables for point accuracy and calibration across ERCOT and GEFCom.
2. **Forecast Trajectories & Interval Visualizer**: Interactive Plotly time-series showing median predictions (P50), actual load, and 90% uncertainty bands across seasonal regimes (Heatwave, Cold Snap, Shoulder, Full Year).
3. **Calibration & Reliability Analysis**: Empirical vs nominal quantile curves evaluating model coverage and ECE.
4. **Context Length Scaling Analysis**: TSFM accuracy, coverage, and latency sensitivity curves across 72h to 1,000h context lengths.
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
|   |-- ercot/          # ERCOT JSON metrics and runs
|   |-- gefcom/         # GEFCom JSON metrics and runs
|   `-- plots/          # Reliability and calibration plots
|-- src/                # Modular benchmark codebase
|   |-- baselines.py        # GBDT and heuristic baseline models
|   |-- tsfm_models.py      # Chronos inference pipeline
|   |-- features.py         # Lag, rolling, and temporal feature engineering
|   |-- evaluation.py       # Vectorized accuracy, CRPS, PICP, and ECE routines
|   |-- data_loader.py      # Chronological split validation and data ingestion
|   |-- run_calibration.py  # Probabilistic calibration benchmarking suite
|   |-- run_tsfm_sweep.py   # Context length sweep (72h to 1000h)
|   |-- baseline_summary.py # Terminal summary printer
|   `-- main.py             # Orchestration and table compilation
|-- tests/              # Unit and integration test suite
|-- app.py              # Streamlit evaluation and scenario dashboard
|-- run_dashboard.py    # Local launcher script
|-- requirements.txt    # Fast-deploy dependency manifest
`-- pyproject.toml      # Build metadata and development configuration
```

---

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/
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
