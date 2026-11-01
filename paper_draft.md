# GridEval: Dual-Grid Benchmarking of GBDTs and Time-Series Foundation Models

## Draft status

This draft reports the completed baseline and probabilistic evaluation runs in results/. The main benchmark uses a shared 24-observation rolling-origin schedule on the ERCOT 2024 and GEFCom 2016 test splits. The separate Chronos context-length sweep remains outside the validated findings.

## Introduction

Short-term load forecasting supports operational planning, while reliable uncertainty estimates help communicate forecast risk. This study compares domain-trained LightGBM and XGBoost models with zero-shot Chronos-T5-Small on two electricity-load datasets. It reports point accuracy and probabilistic forecast quality using a common rolling evaluation schedule.

## Methods

The chronological splits use three training years, one validation year, and one test year. ERCOT has 26,304 training, 8,760 validation, and 8,784 test observations. GEFCom has 26,298 training, 8,758 validation, and 8,782 test observations.

All models share ordered 24-observation forecast origins and target identities. GBDT forecasts recursively update lag and rolling features within each block using prior predictions. Chronos uses a 512-observation context and forecasts each 24-observation block from information available at its origin. The default GBDT feature policy excludes weather. Point-regression GBDTs and quantile GBDTs are separate model fits.

Probabilistic performance is evaluated with CRPS, 90% prediction interval coverage (PICP), mean interval width, and quantile calibration error (ECE). The intervals are evaluated as generated; no post-hoc adjustment is applied to force 90% test coverage.

## Results

### Point accuracy

| Dataset | Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---:|---:|---:|
| ERCOT | LightGBM | 2,337.55 | 3,560.92 | 4.34 |
| ERCOT | XGBoost | 2,270.14 | 3,383.25 | 4.17 |
| ERCOT | Chronos-T5-Small | 1,907.38 | 2,947.31 | 3.59 |
| GEFCom | LightGBM | 157.27 | 227.31 | 4.39 |
| GEFCom | XGBoost | 157.73 | 229.21 | 4.33 |
| GEFCom | Chronos-T5-Small | 155.35 | 241.23 | 4.29 |

Chronos has the lowest MAE on both test sets. On GEFCom, its MAE advantage is small and its RMSE is higher than the GBDTs.

### Probabilistic evaluation

| Dataset | Model | CRPS (MW) | PICP 90% (%) | Mean width (MW) | ECE |
|---|---|---:|---:|---:|---:|
| ERCOT | LightGBM Quantile | 2,493.91 | 68.72 | 11,180.74 | 0.1837 |
| ERCOT | XGBoost Quantile | 2,587.26 | 67.04 | 11,472.15 | 0.1811 |
| ERCOT | Chronos-T5-Small | 1,429.74 | 77.02 | 5,654.04 | 0.0492 |
| GEFCom | LightGBM Quantile | 176.47 | 60.73 | 684.32 | 0.1242 |
| GEFCom | XGBoost Quantile | 183.53 | 61.06 | 706.67 | 0.1195 |
| GEFCom | Chronos-T5-Small | 116.07 | 75.55 | 441.10 | 0.0522 |

Chronos has lower CRPS and ECE on both datasets and higher coverage than the quantile GBDTs. All models fall short of the nominal 90% interval coverage target. These are evaluations of unadjusted intervals, not results after post-hoc calibration.

## Limitations and interpretation

- The fixed training windows use three years, although older history exists in the source datasets. Expanded-history results are not included.
- Weather features are disabled by default, so this comparison does not estimate the benefit of weather inputs on GEFCom.
- Test coverage is below the nominal target for every model.
- The separate context-length sweep has not been rerun under the shared protocol and is not used to support conclusions here.
- Chronos used CUDA-enabled PyTorch on an RTX 3050 for this run. The GPU-enabled runtime used for the run is not yet installed as the project's persistent virtual environment.
