# GridEval: Empirical Benchmarking of Classical ML vs Time Series Foundation Models

GridEval is an empirical research and evaluation platform designed to systematically benchmark domain-trained classical machine learning models against pre-trained zero-shot Time Series Foundation Models (TSFMs) for electricity load forecasting.

GridEval provides a standardized evaluation harness across independent power grid scales, coupled with an interactive live analytics and scenario simulation web dashboard.

---

## Research Scope & Datasets

The benchmark evaluates models across two distinct power grid scales:

1. ERCOT (Texas Interconnection):
   - Macro-scale power grid (~70,000 MW peak demand).
   - Hourly electricity demand spanning 2020 to 2024 (~43,800 timestamps).
   - Characterized by rapid industrial load growth and high renewable penetration.

2. GEFCom 2017 (ISO-NE Connecticut Zone):
   - Regional-scale power grid (~4,000 MW peak demand).
   - Hourly electricity demand paired with drybulb and dewpoint temperature data (2012 to 2016).
   - Characterized by distinct non-linear temperature-load relationships (summer cooling, winter heating).

### Data Split Protocol
A symmetrical 3-1-1 chronological split protocol is enforced across both grids to prevent temporal data leakage:
- Training: 3 full years (26,280 hours)
- Validation / Hyperparameter Tuning: 1 full year (8,760 hours)
- Out-of-Sample Test Evaluation: 1 full year (8,760 hours)

---

## Evaluated Models

GridEval compares three paradigms of time-series modeling:

1. Supervised Gradient Boosted Decision Trees (Domain-Trained):
   - LightGBM & XGBoost with quantile pinball loss (alpha = 0.05, 0.50, 0.95).
   - Features: Autoregressive lags (t-1, t-2, t-3, t-24, t-168), rolling statistics (24h, 168h means and standard deviations), calendar encodings, holiday indicators, and non-linear temperature interactions (temperature squared, dewpoint depression).

2. Classical Additive & Heuristic Baselines:
   - Prophet: Decomposable additive model with weekly and yearly Fourier seasonality and holiday effects.
   - Naive Seasonal: Deterministic 168-hour persistence baseline (load at hour t equals load at hour t-168).

3. Pre-Trained Foundation Models (Zero-Shot TSFM):
   - Amazon Chronos-T5 (Small): Transformer-based architecture pre-trained on diverse open and synthetic time-series corpora. Evaluated zero-shot without fine-tuning, sweeping context windows from 72h up to 1,000h.

---

## Evaluation Framework

GridEval benchmarks performance across two critical operational dimensions:

### 1. Point Forecasting Accuracy
- Mean Absolute Error (MAE): Average absolute deviation in Megawatts (MW).
- Root Mean Squared Error (RMSE): Quadratic error metric penalizing extreme outliers.
- Mean Absolute Percentage Error (MAPE): Scale-independent percentage error.

### 2. Probabilistic Calibration & Uncertainty
- Continuous Ranked Probability Score (CRPS): Evaluates full predictive distributions for sharpness and calibration simultaneously.
- Prediction Interval Coverage Probability (PICP): Empirical percentage of test observations falling within the nominal 90% confidence interval (target: 90.0%).
- Mean Prediction Interval Width (MPIW): Measures sharpness (narrowness) of uncertainty bounds.
- Expected Calibration Error (ECE): Quantifies average deviation across all quantile probability levels (5% through 95%).

---

## Interactive Dashboard Features (`app.py`)

The platform includes an interactive Streamlit application structured into five core modules:

1. Tab 1: Leaderboard & Cross-Grid Comparison
   - Side-by-side metric tables for point accuracy (MAE, RMSE, MAPE) and probabilistic calibration (CRPS, 90% PICP, MPIW, ECE).
   - Filterable by single grid (ERCOT or GEFCom) or comprehensive cross-comparison.

2. Tab 2: Forecast Trajectories & Interval Visualizer
   - Interactive Plotly time-series charts showing forecast median (P50), observed actuals, and 90% uncertainty ribbons (P05 to P95).
   - Preset time windows: Whole Test Year (8,760 hours), Summer Peak Heatwave, Winter Cold Snap, Spring Shoulder Season, or Custom Date Picker.
   - Dynamic window metric cards updating MAE, MAPE, interval coverage, and mean interval width on demand.
   - Adaptive downsampling for large spans to maintain sub-50ms canvas rendering.

3. Tab 3: Calibration & Reliability Analysis
   - Empirical vs nominal quantile reliability curves for ERCOT and GEFCom.
   - Quantitative evaluation of model overconfidence (Chronos under-coverage) vs calibrated quantile regression (GBDT coverage).

4. Tab 4: Context Length Scaling Analysis
   - Empirical scaling curves evaluating TSFM accuracy and coverage sensitivity across context lengths from 72h to 1,000h.
   - Visualizes performance plateaus and inference latency trade-offs.

5. Tab 5: Interactive Live Forecasting & Scenario Simulator
   - Real-time scenario stress-testing engine allowing operators to simulate grid dispatch under custom operational conditions.
   - Configurable forecast horizons: 24h (Day-Ahead), 48h (Two-Day), 72h (Three-Day), and 168h (7-Day Outlook).
   - Stress-testing sliders: Temperature shock (-25°F to +25°F) with non-linear heating/cooling elasticity curves, and structural demand growth (-20% to +25%).
   - Model selection: LightGBM Quantiles, XGBoost Quantiles, Chronos-T5 Zero-Shot, or Multi-Model Overlay.
   - Operational KPIs: Simulated Peak Demand (MW), Integrated Energy (GWh/MWh), Required 90% Spinning Reserve Margin (MW), and Net Demand Shift vs Baseline.
   - Tabular export: Full hourly simulated dispatch schedule downloadable as CSV.

---

## Key Empirical Findings

1. Tabular GBDTs Outperform Zero-Shot TSFMs on Point Accuracy:
   - LightGBM and XGBoost achieve ~0.90% MAPE on ERCOT and ~0.96% MAPE on GEFCom CT.
   - Zero-shot Chronos-T5 achieves ~3.61% MAPE on ERCOT and ~4.28% MAPE on GEFCom CT.
   - Domain-specific autoregressive features and calendar encodings provide stronger inductive bias for electrical grids than generalized zero-shot pre-training.

2. Zero-Shot Foundation Models Exhibit Severe Overconfidence:
   - Chronos-T5 achieves only ~75.4% to 77.1% coverage for nominal 90% prediction intervals, underestimating real-world grid volatility.
   - Quantile GBDTs achieve well-calibrated coverage (~87.2% to 88.5% PICP) while maintaining narrow interval widths.

3. Context Window Scaling Plateaus:
   - Increasing TSFM historical context window from 72h to 512h improves accuracy significantly.
   - Beyond 512h (e.g., 720h and 1,000h), accuracy gains plateau and inference latency increases linearly without closing the gap to GBDTs.

---

## Repository Structure

```
GridEval/
|-- configs/            # Experiment YAML configuration files
|-- datasets/           # Raw and yearly partitioned source data
|-- data/processed/     # Cleaned, validated, and split Parquet files
|-- notebooks/          # Exploratory Data Analysis (EDA)
|-- results/            # Benchmark outputs (master_table.csv, calibration_master.csv)
|   |-- ercot/          # ERCOT metrics, JSON runs, and model checkpoints
|   |-- gefcom/         # GEFCom metrics, JSON runs, and model checkpoints
|   `-- plots/          # Reliability diagrams and forecast curves
|-- src/                # Modular benchmark codebase
|   |-- baselines.py        # GBDT and heuristic baseline training
|   |-- tsfm_models.py      # Chronos zero-shot inference pipeline
|   |-- features.py         # Lag, rolling, and temporal feature engineering
|   |-- evaluation.py       # Vectorized accuracy, CRPS, PICP, and ECE routines
|   |-- data_loader.py      # Chronological split validation and data ingestion
|   |-- run_calibration.py  # Probabilistic calibration benchmarking suite
|   |-- run_tsfm_sweep.py   # Context length ablation sweep (72h to 1000h)
|   `-- main.py             # Master orchestration script
|-- tests/              # Automated unit and integration test suite
|-- app.py              # Interactive Streamlit evaluation and simulation dashboard
|-- run_dashboard.py    # 1-Click VS Code and WSL browser bridge launcher
|-- pyproject.toml      # Build system and dependency definitions
`-- README.md
```

---

## Quick Start & Reproduction

### 1. Environment Setup
```bash
# Create and activate Python virtual environment (Python 3.10 - 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install project and dependencies in editable mode
pip install -e .
```

### 2. Run Test Suite
```bash
pytest tests/
```

### 3. Execute Benchmarks
```bash
# Run all benchmark stages and compile master summary tables
python src/main.py

# Or run individual benchmark experiments
python src/baselines.py
python src/run_tsfm_sweep.py
python src/run_calibration.py
```

### 4. Launch Interactive Dashboard
```bash
streamlit run app.py
```
Or execute the 1-click launcher in VS Code:
```bash
python run_dashboard.py
```
Access the dashboard at `http://localhost:8501` to inspect leaderboards, forecast trajectories, calibration curves, context length scaling plots, and the live scenario simulator.
