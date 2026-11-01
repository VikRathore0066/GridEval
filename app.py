import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="GridEval - Energy Load Forecasting Benchmark",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stDeployButton, header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 12px;
        border-left: 4px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)

st.title("GridEval")
st.caption("Empirical Benchmarking of Classical ML Baselines vs Time Series Foundation Models (Accuracy and Calibration)")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("GRIDEVAL_RESULTS_DIR", BASE_DIR / "results"))
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = RESULTS_DIR / "plots"

DATE_BOUNDS = {
    "ercot": {
        "start": pd.Timestamp("2024-01-01"),
        "end": pd.Timestamp("2024-12-31 23:00:00"),
        "presets": {
            "Whole Test Year (2024)": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
            "Summer Peak Heatwave (Jul 01 - Jul 14)": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-14")),
            "Winter Cold Snap (Jan 10 - Jan 24)": (pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-24")),
            "Spring Shoulder Season (Apr 01 - Apr 14)": (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-14")),
        },
    },
    "gefcom": {
        "start": pd.Timestamp("2016-01-01"),
        "end": pd.Timestamp("2016-12-31 23:00:00"),
        "presets": {
            "Whole Test Year (2016)": (pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31")),
            "Summer Peak Heatwave (Jul 01 - Jul 14)": (pd.Timestamp("2016-07-01"), pd.Timestamp("2016-07-14")),
            "Winter Cold Snap (Jan 10 - Jan 24)": (pd.Timestamp("2016-01-10"), pd.Timestamp("2016-01-24")),
            "Spring Shoulder Season (Apr 01 - Apr 14)": (pd.Timestamp("2016-04-01"), pd.Timestamp("2016-04-14")),
        },
    },
}


@st.cache_data
def load_grid_load_data(dataset_key: str):
    filename = "ercot_clean.parquet" if dataset_key == "ercot" else "gefcom_ct_clean.parquet"
    path = PROCESSED_DIR / filename
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["load"])
    bounds = DATE_BOUNDS[dataset_key]

    # Handle MultiIndex (timestamp, occurrence) explicitly for ERCOT
    if isinstance(df.index, pd.MultiIndex):
        ts_level = df.index.get_level_values("timestamp")
        mask = (ts_level >= bounds["start"]) & (ts_level <= bounds["end"])
        return df.loc[mask]
    else:
        return df.loc[bounds["start"]:bounds["end"]]


def load_results_data():
    results = {"baselines": {}, "tsfm": {}, "calibration": {}}
    for ds in ["ercot", "gefcom"]:
        b_path = RESULTS_DIR / ds / "baselines.json"
        if b_path.exists():
            with open(b_path) as f:
                results["baselines"][ds] = json.load(f)

        t_path = RESULTS_DIR / ds / "tsfm_context_sweep.json"
        if t_path.exists():
            with open(t_path) as f:
                results["tsfm"][ds] = json.load(f)

        c_path = RESULTS_DIR / ds / "calibration.json"
        if c_path.exists():
            with open(c_path) as f:
                results["calibration"][ds] = json.load(f)
    return results


def load_csv_table(filename: str):
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def filter_comparison_models(table: pd.DataFrame) -> pd.DataFrame:
    """Show only the three model families in the paper comparison."""
    if table is None or table.empty or "Model" not in table.columns:
        return table
    model_names = table["Model"].astype(str).str.casefold()
    keep = model_names.str.contains(r"lightgbm|xgboost|chronos", regex=True)
    return table.loc[keep].copy()


def load_prediction_csv(csv_path: Path, dataset_key: str) -> pd.DataFrame:
    """Load prediction CSV, reconstructing ERCOT MultiIndex (timestamp, occurrence) or GEFCom DatetimeIndex correctly."""
    df_raw = pd.read_csv(csv_path)
    if dataset_key == "ercot" and "timestamp" in df_raw.columns and "occurrence" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw = df_raw.set_index(["timestamp", "occurrence"])
    elif dataset_key == "ercot" and "Unnamed: 0" in df_raw.columns and "Unnamed: 1" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Unnamed: 0": "timestamp", "Unnamed: 1": "occurrence"})
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw = df_raw.set_index(["timestamp", "occurrence"])
    else:
        idx_col = df_raw.columns[0]
        df_raw[idx_col] = pd.to_datetime(df_raw[idx_col])
        df_raw = df_raw.set_index(idx_col)
    return df_raw


def verify_prediction_alignment(df_pred: pd.DataFrame, dataset_key: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """Verify exact ordered identities and row ordering align with expected test-split subset identities for the window, preserving ERCOT (timestamp, occurrence)."""
    if df_pred is None or df_pred.empty:
        raise ValueError("Prediction DataFrame is empty or None.")

    if df_pred.index.duplicated().any():
        raise ValueError("Prediction index contains duplicate keys.")

    if not df_pred.index.is_monotonic_increasing:
        raise ValueError("Prediction index is not monotonic increasing.")

    from src.data_loader import load_dataset, split_dataset
    df_full = load_dataset(dataset_key)
    _, _, test_split = split_dataset(df_full, dataset_key)
    if test_split is not None and not test_split.empty:
        if dataset_key == "ercot" and isinstance(test_split.index, pd.MultiIndex):
            ts_vals = test_split.index.get_level_values("timestamp")
            mask_test = (ts_vals >= start_dt) & (ts_vals <= end_dt)
            expected_subset = test_split.loc[mask_test]

            # Filter df_pred to the window as well (dashboard loads full CSV and filters to window)
            ts_pred = df_pred.index.get_level_values("timestamp")
            mask_pred = (ts_pred >= start_dt) & (ts_pred <= end_dt)
            df_pred_window = df_pred.loc[mask_pred]

            expected_list = list(expected_subset.index)
            pred_list = list(df_pred_window.index)
            if len(pred_list) != len(expected_list) or pred_list != expected_list:
                expected_set = set(expected_list)
                pred_set = set(pred_list)
                if not pred_set.issubset(expected_set) or not expected_set.issubset(pred_set):
                    raise ValueError("Prediction index contains missing or extra identities relative to expected window.")
                if pred_list != expected_list:
                    raise ValueError("Prediction index is reordered relative to expected window.")
            if "actual" in df_pred_window.columns:
                canonical_actuals = df_full.loc[df_pred_window.index, "load"]
                if not np.allclose(df_pred_window["actual"].values, canonical_actuals.values, equal_nan=True):
                    raise ValueError("Prediction actual values do not match canonical dataset values.")
        else:
            if not isinstance(test_split.index, pd.DatetimeIndex):
                test_split.index = pd.to_datetime(test_split.index)
            mask_test = (test_split.index >= start_dt) & (test_split.index <= end_dt)
            expected_subset = test_split.loc[mask_test]

            if not isinstance(df_pred.index, pd.DatetimeIndex):
                df_pred.index = pd.to_datetime(df_pred.index)
            mask_pred = (df_pred.index >= start_dt) & (df_pred.index <= end_dt)
            df_pred_window = df_pred.loc[mask_pred]

            expected_list = list(expected_subset.index)
            pred_list = [pd.to_datetime(idx) for idx in df_pred_window.index]
            expected_list_dt = [pd.to_datetime(idx) for idx in expected_list]
            if len(pred_list) != len(expected_list_dt) or pred_list != expected_list_dt:
                expected_set = set(expected_list_dt)
                pred_set = set(pred_list)
                if not pred_set.issubset(expected_set) or not expected_set.issubset(pred_set):
                    raise ValueError("Prediction index contains missing or extra identities relative to expected window.")
                if pred_list != expected_list_dt:
                    raise ValueError("Prediction index is reordered relative to expected window.")
            if "actual" in df_pred_window.columns:
                canonical_actuals = df_full.loc[df_pred_window.index, "load"]
                if not np.allclose(df_pred_window["actual"].values, canonical_actuals.values, equal_nan=True):
                    raise ValueError("Prediction actual values do not match canonical dataset values.")

    return True


def load_forecast_window(dataset_key: str, model_key: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """Load a saved forecast only when its identities match the current test split."""
    model_name = model_key.casefold()
    if "lightgbm" in model_name:
        quantile_path = RESULTS_DIR / dataset_key / "predictions_quantile_lightgbm.csv"
        model_column = "lightgbm"
        quantile_columns = ("lightgbm_q05", "lightgbm_median", "lightgbm_q95")
    elif "xgboost" in model_name:
        quantile_path = RESULTS_DIR / dataset_key / "predictions_quantile_xgboost.csv"
        model_column = "xgboost"
        quantile_columns = ("xgboost_q05", "xgboost_median", "xgboost_q95")
    else:
        quantile_path = RESULTS_DIR / dataset_key / "predictions_quantile_chronos.csv"
        model_column = "chronos"
        quantile_columns = ("chronos_q05", "chronos_median", "chronos_q95")

    if quantile_path.exists():
        predictions = load_prediction_csv(quantile_path, dataset_key)
        verify_prediction_alignment(predictions, dataset_key, start_dt, end_dt)
        missing = [column for column in ("actual", *quantile_columns) if column not in predictions.columns]
        if missing:
            raise ValueError(f"Saved quantile forecast is missing columns: {', '.join(missing)}")
        quantile = True
        low_col, median_col, high_col = quantile_columns
    else:
        if "chronos" in model_name:
            raise FileNotFoundError("Chronos prediction output has not been generated for this dataset.")
        point_path = RESULTS_DIR / dataset_key / "predictions_baselines.csv"
        if not point_path.exists():
            raise FileNotFoundError("Saved baseline predictions have not been generated for this dataset.")
        predictions = load_prediction_csv(point_path, dataset_key)
        verify_prediction_alignment(predictions, dataset_key, start_dt, end_dt)
        missing = [column for column in ("actual", model_column) if column not in predictions.columns]
        if missing:
            raise ValueError(f"Saved point forecast is missing columns: {', '.join(missing)}")
        quantile = False

    if isinstance(predictions.index, pd.MultiIndex):
        timestamps = predictions.index.get_level_values("timestamp")
        mask = (timestamps >= start_dt) & (timestamps <= end_dt)
    else:
        mask = (predictions.index >= start_dt) & (predictions.index <= end_dt)
    selected = predictions.loc[mask]
    if selected.empty:
        raise ValueError("No saved forecasts match the selected date window.")

    if quantile:
        low, median, high = (selected[column].to_numpy() for column in quantile_columns)
    else:
        low = high = None
        median = selected[model_column].to_numpy()
    return {
        "index": selected.index,
        "actual": selected["actual"].to_numpy(),
        "median": median,
        "lower": low,
        "upper": high,
        "has_quantiles": quantile,
    }


all_results = load_results_data()

# Sidebar
st.sidebar.header("Case Study Selection")
dataset_selection = st.sidebar.selectbox(
    "Active Grid Dataset",
    ["ERCOT (Texas Interconnection)", "GEFCom 2017 (Connecticut Zone)"],
)
dataset_key = "ercot" if "ERCOT" in dataset_selection else "gefcom"

st.sidebar.markdown("---")
st.sidebar.subheader("Exports")
master_acc_path = RESULTS_DIR / "master_table.csv"
if master_acc_path.exists():
    accuracy_download = filter_comparison_models(load_csv_table("master_table.csv"))
    st.sidebar.download_button(
        label="Download Accuracy CSV",
        data=accuracy_download.to_csv(index=False),
        file_name="master_accuracy_table.csv",
        mime="text/csv",
    )

master_cal_path = RESULTS_DIR / "calibration_master.csv"
if master_cal_path.exists():
    calibration_download = filter_comparison_models(load_csv_table("calibration_master.csv"))
    st.sidebar.download_button(
        label="Download Calibration CSV",
        data=calibration_download.to_csv(index=False),
        file_name="master_calibration_table.csv",
        mime="text/csv",
    )

tab_overview, tab_forecast, tab_calibration, tab_tsfm, tab_simulator = st.tabs([
    "Leaderboard & Comparison",
    "Forecast Trajectories",
    "Calibration & Reliability",
    "Context Length Scaling",
    "Illustrative Scenario",
])

# -----------------------------------------------------------------------------
# TAB 1: LEADERBOARD & COMPARISONS
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Model Leaderboard & Cross-Dataset Comparison")

    view_scope = st.radio(
        "Display Scope",
        ["Selected Grid Only (" + dataset_selection + ")", "Both Grids (ERCOT & GEFCom)"],
        horizontal=True,
    )

    st.caption(
        "Saved tables update when the benchmark experiments are rerun under the current evaluation setup."
    )

    st.markdown("#### Point Accuracy Metrics")
    df_acc = load_csv_table("master_table.csv")
    if df_acc is not None:
        df_acc = filter_comparison_models(df_acc)
        if "Selected Grid Only" in view_scope:
            df_acc = df_acc[df_acc["Dataset"] == dataset_key.upper()]
        st.dataframe(df_acc, width="stretch", hide_index=True)
    else:
        st.info("Run python src/main.py to generate master_table.csv.")

    st.markdown("#### Probabilistic Calibration Metrics")
    df_cal = load_csv_table("calibration_master.csv")
    if df_cal is not None:
        df_cal = filter_comparison_models(df_cal)
        if "Selected Grid Only" in view_scope:
            df_cal = df_cal[df_cal["Dataset"] == dataset_key.upper()]
        st.dataframe(df_cal, width="stretch", hide_index=True)
    else:
        st.info("Run python src/main.py to generate calibration_master.csv.")

# -----------------------------------------------------------------------------
# TAB 2: FORECAST & PREDICTION INTERVAL VISUALIZER
# -----------------------------------------------------------------------------
with tab_forecast:
    st.subheader(f"Load Forecast Trajectory with 90% Prediction Interval ({dataset_key.upper()})")

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        model_selection = st.selectbox(
            "Forecasting Model",
            [
                "LightGBM (Quantile Regression)",
                "XGBoost (Quantile Regression)",
                "Chronos-T5-Small (Zero-Shot TSFM)",
            ],
        )
        model_key_map = {
            "LightGBM (Quantile Regression)": "lightgbm_quantile",
            "XGBoost (Quantile Regression)": "xgboost_quantile",
            "Chronos-T5-Small (Zero-Shot TSFM)": "chronos_small_512",
        }
        model_key = model_key_map[model_selection]

    with col_ctrl2:
        presets_dict = DATE_BOUNDS[dataset_key]["presets"]
        window_mode = st.selectbox(
            "Time Horizon Window",
            list(presets_dict.keys()) + ["Custom Date Range"],
        )

    if window_mode == "Custom Date Range":
        bounds = DATE_BOUNDS[dataset_key]
        custom_range = st.date_input(
            "Select Date Window",
            value=(bounds["start"].date(), (bounds["start"] + pd.Timedelta(days=14)).date()),
            min_value=bounds["start"].date(),
            max_value=bounds["end"].date(),
        )
        if isinstance(custom_range, (tuple, list)) and len(custom_range) == 2:
            start_dt = pd.Timestamp(custom_range[0])
            end_dt = pd.Timestamp(custom_range[1]) + pd.Timedelta(hours=23)
        else:
            start_dt, end_dt = bounds["start"], bounds["end"]
    else:
        preset_start, preset_end = presets_dict[window_mode]
        start_dt = preset_start
        end_dt = preset_end + pd.Timedelta(hours=23)

    df_test_load = load_grid_load_data(dataset_key)

    if df_test_load is not None and not df_test_load.empty:
        if isinstance(df_test_load.index, pd.MultiIndex):
            ts_lvl = df_test_load.index.get_level_values("timestamp")
            mask = (ts_lvl >= start_dt) & (ts_lvl <= end_dt)
            df_slice = df_test_load.loc[mask]
        else:
            mask = (df_test_load.index >= start_dt) & (df_test_load.index <= end_dt)
            df_slice = df_test_load.loc[mask]

        if len(df_slice) > 0:
            try:
                forecast = load_forecast_window(dataset_key, model_key, start_dt, end_dt)
            except (OSError, ValueError, KeyError) as exc:
                st.info(
                    "No compatible saved forecast is available for this test split. "
                    f"{exc} Rerun the baseline and calibration experiments to create aligned forecasts. "
                    "The other dashboard tabs remain available."
                )
            else:
                timestamps = forecast["index"]
                plot_x = (
                    timestamps.get_level_values("timestamp")
                    if isinstance(timestamps, pd.MultiIndex)
                    else timestamps
                )
                actuals = forecast["actual"]
                pred_median = forecast["median"]
                has_quantiles = forecast["has_quantiles"]

                fig = go.Figure()
                if has_quantiles:
                    fig.add_trace(go.Scatter(
                        x=plot_x, y=forecast["upper"],
                        mode="lines", line=dict(width=0), showlegend=False,
                        name="90% Upper Bound",
                    ))
                    fig.add_trace(go.Scatter(
                        x=plot_x, y=forecast["lower"],
                        fill="tonexty", mode="lines", line=dict(width=0),
                        fillcolor="rgba(37, 99, 235, 0.15)",
                        name="90% Prediction Interval",
                    ))
                fig.add_trace(go.Scatter(
                    x=plot_x, y=pred_median,
                    mode="lines", line=dict(color="#2563eb", width=2.2),
                    name="Forecast Median",
                ))
                fig.add_trace(go.Scatter(
                    x=plot_x, y=actuals,
                    mode="lines+markers" if len(actuals) <= 168 else "lines",
                    line=dict(color="#dc2626", width=1.5),
                    marker=dict(size=4),
                    name="Observed Actual Load",
                ))

                fig.update_layout(
                    template="plotly_white",
                    height=420,
                    xaxis_title="Timestamp",
                    yaxis_title="Grid Load (MW)",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig, width="stretch")

                mae_val = np.mean(np.abs(actuals - pred_median))
                safe_actuals = np.where(actuals == 0, np.nan, actuals)
                mape_val = np.nanmean(np.abs((actuals - pred_median) / safe_actuals)) * 100.0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Window MAE", f"{mae_val:.1f} MW")
                c2.metric("Window MAPE", f"{mape_val:.2f}%")
                if has_quantiles:
                    coverage = np.mean(
                        (actuals >= forecast["lower"]) & (actuals <= forecast["upper"])
                    ) * 100.0
                    width = np.mean(forecast["upper"] - forecast["lower"])
                    c3.metric("90% Interval Coverage (PICP)", f"{coverage:.1f}%", delta=f"{coverage - 90.0:.1f}% vs 90% target")
                    c4.metric("Avg Interval Width", f"{width:.1f} MW")
                else:
                    c3.metric("90% Interval Coverage (PICP)", "Not available")
                    c4.metric("Avg Interval Width", "Not available")
        else:
            st.warning("Selected date range has no matching data in test split.")
    else:
        st.info("Processed dataset not found. Run EDA preprocessing scripts first.")

# -----------------------------------------------------------------------------
# TAB 3: CALIBRATION & RELIABILITY DIAGRAMS
# -----------------------------------------------------------------------------
with tab_calibration:
    st.subheader(f"Uncertainty Calibration & Reliability Curves ({dataset_key.upper()})")

    cal_dict = all_results["calibration"].get(dataset_key, {})
    removed_curve_models = [
        name for name in cal_dict
        if not any(model in name.casefold() for model in ("lightgbm", "xgboost", "chronos"))
    ]
    col_rel1, col_rel2 = st.columns([1.2, 1])
    with col_rel1:
        img_path = PLOTS_DIR / f"reliability_{dataset_key}.png"
        if removed_curve_models:
            if img_path.exists():
                st.info(
                    "This saved reliability chart includes a model outside the current comparison. "
                    "Rerun calibration with the same results folder as this dashboard to replace it."
                )
            else:
                st.info(
                    "The old reliability chart was removed because it included a model outside the current comparison. "
                    "Run calibration to create a new chart for the current models."
                )
        elif img_path.exists():
            st.image(str(img_path), caption=f"Empirical vs Nominal Quantile Reliability ({dataset_key.upper()})", width="stretch")
        else:
            st.info(f"Reliability diagram not found at {img_path}. Run python src/run_calibration.py to generate it.")

    with col_rel2:
        st.markdown("#### Calibration Summary Table")
        if cal_dict:
            cal_rows = []
            for m_name, m_metrics in cal_dict.items():
                if not any(model in m_name.casefold() for model in ("lightgbm", "xgboost", "chronos")):
                    continue
                cal_rows.append({
                    "Model": m_name.replace("_", " ").title(),
                    "CRPS": m_metrics.get("CRPS", "N/A"),
                    "90% PICP": f"{m_metrics.get('PICP_90', 'N/A')}%",
                    "Width (MW)": m_metrics.get("Width_90", "N/A"),
                    "ECE": m_metrics.get("ECE", "N/A"),
                })
            st.dataframe(pd.DataFrame(cal_rows), width="stretch", hide_index=True)
        else:
            st.info("Run python src/run_calibration.py to populate metrics.")

        st.caption("Calibration values are read from the saved calibration files.")

# -----------------------------------------------------------------------------
# TAB 4: TSFM CONTEXT LENGTH SCALING
# -----------------------------------------------------------------------------
with tab_tsfm:
    st.subheader(f"TSFM Context Length Scaling Analysis (observations) - {dataset_key.upper()}")

    tsfm_sweep = all_results["tsfm"].get(dataset_key, {})
    if tsfm_sweep:
        sweep_df = pd.DataFrame([
            {"Context Length (Observations)": int(k), **v} for k, v in tsfm_sweep.items()
        ]).sort_values("Context Length (Observations)")

        col_sw1, col_sw2 = st.columns(2)
        with col_sw1:
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(
                x=sweep_df["Context Length (Observations)"], y=sweep_df["MAE"],
                mode="lines+markers", line=dict(color="#2563eb", width=2.5),
                marker=dict(size=8), name="MAE (MW)",
            ))
            fig_mae.update_layout(
                title="Point Accuracy (MAE) vs Context Length",
                xaxis_title="Context Length (Observations)",
                yaxis_title="MAE (MW)",
                template="plotly_white",
                height=320,
            )
            st.plotly_chart(fig_mae, width="stretch")

        with col_sw2:
            fig_cov = go.Figure()
            fig_cov.add_trace(go.Scatter(
                x=sweep_df["Context Length (Observations)"], y=sweep_df["Coverage_90"],
                mode="lines+markers", line=dict(color="#dc2626", width=2.5),
                marker=dict(size=8), name="Empirical 90% Coverage",
            ))
            fig_cov.add_trace(go.Scatter(
                x=[72, 1000], y=[90, 90],
                mode="lines", line=dict(color="black", dash="dash"),
                name="Nominal 90% Target",
            ))
            fig_cov.update_layout(
                title="90% Prediction Interval Coverage vs Context Length",
                xaxis_title="Context Length (Observations)",
                yaxis_title="Empirical Coverage (%)",
                template="plotly_white",
                height=320,
            )
            st.plotly_chart(fig_cov, width="stretch")

        st.markdown("#### Context Sensitivity Table")
        st.dataframe(sweep_df, width="stretch", hide_index=True)
    else:
        st.info("No context-length sweep has been regenerated under the current shared protocol. Older sweep results are archived and are not shown as current findings.")

# -----------------------------------------------------------------------------
# TAB 5: INTERACTIVE LIVE FORECASTING & SCENARIO SIMULATOR
# -----------------------------------------------------------------------------
with tab_simulator:
    st.subheader(f"Illustrative Scenario Simulator ({dataset_key.upper()})")
    st.info("Note: This scenario simulator is an illustrative analytical calculation (stress-testing historical load profiles with parametric temperature and demand shocks), not an operational model forecast or reserve requirement.")

    col_sim_cfg, col_sim_stress = st.columns([1, 1])

    with col_sim_cfg:
        st.markdown("##### 1. Forecast Setup & Baseline")
        sim_presets = {
            "Summer Peak Season (July Baseline)": (pd.Timestamp("2024-07-08 00:00:00") if dataset_key == "ercot" else pd.Timestamp("2016-07-08 00:00:00")),
            "Winter Freeze Season (January Baseline)": (pd.Timestamp("2024-01-15 00:00:00") if dataset_key == "ercot" else pd.Timestamp("2016-01-15 00:00:00")),
            "Spring Shoulder Season (April Baseline)": (pd.Timestamp("2024-04-10 00:00:00") if dataset_key == "ercot" else pd.Timestamp("2016-04-10 00:00:00")),
            "Autumn Normal Season (October Baseline)": (pd.Timestamp("2024-10-14 00:00:00") if dataset_key == "ercot" else pd.Timestamp("2016-10-14 00:00:00")),
        }
        selected_preset = st.selectbox("Dispatch Starting Condition", list(sim_presets.keys()))
        start_sim_dt = sim_presets[selected_preset]

        sim_horizon_hours = st.select_slider(
            "Forecast Horizon",
            options=[24, 48, 72, 168],
            value=48,
            format_func=lambda h: f"{h} Hours Ahead ({h//24} Day{'s' if h > 24 else ''})",
        )

        sim_model = st.selectbox(
            "Inference Model Engine",
            [
                "LightGBM Quantile (Calibrated P05/P50/P95)",
                "XGBoost Quantile (Calibrated P05/P50/P95)",
                "Chronos-T5 Zero-Shot TSFM (Transformer)",
                "Model Comparison (LightGBM vs Chronos Overlay)",
            ],
        )

    with col_sim_stress:
        st.markdown("##### 2. Scenario & Stress-Testing Multipliers")
        scenario_template = st.selectbox(
            "Scenario Preset",
            [
                "Baseline Operations (No Stress)",
                "Severe Summer Heatwave (+10°F Temp Shock)",
                "Extreme Polar Vortex (-15°F Freeze Shock)",
                "Industrial Surge / Data Center Expansion (+10% Load)",
                "Industrial Curtailment / Economic Slowdown (-8% Load)",
                "Custom Stress Parameters",
            ],
        )

        def_temp = 0.0
        def_growth = 0.0
        if "Heatwave" in scenario_template:
            def_temp = 10.0
            def_growth = 3.0
        elif "Polar Vortex" in scenario_template:
            def_temp = -15.0
            def_growth = 4.0
        elif "Surge" in scenario_template:
            def_growth = 10.0
        elif "Curtailment" in scenario_template:
            def_growth = -8.0

        temp_shock = st.slider(
            "Temperature Deviation (°F vs Historical Norm)",
            min_value=-25.0,
            max_value=25.0,
            value=float(def_temp),
            step=1.0,
            help="Simulates extreme heat (AC cooling demand) or extreme freeze (electric heating demand).",
        )

        growth_shock = st.slider(
            "Industrial / Structural Demand Shift (%)",
            min_value=-20.0,
            max_value=25.0,
            value=float(def_growth),
            step=0.5,
            help="Simulates data center expansion, industrial shift, or grid curtailment.",
        )

    df_grid_full = load_grid_load_data(dataset_key)
    if df_grid_full is not None and not df_grid_full.empty:
        # Handle index extraction for context and future slice
        if isinstance(df_grid_full.index, pd.MultiIndex):
            ts_full = df_grid_full.index.get_level_values("timestamp")
        else:
            ts_full = df_grid_full.index

        ctx_start = start_sim_dt - pd.Timedelta(hours=24)
        ctx_mask = (ts_full >= ctx_start) & (ts_full < start_sim_dt)
        df_ctx = df_grid_full.loc[ctx_mask]

        future_timestamps = pd.date_range(start=start_sim_dt, periods=sim_horizon_hours, freq="h")

        base_slice_mask = (ts_full >= start_sim_dt) & (ts_full < start_sim_dt + pd.Timedelta(hours=sim_horizon_hours))
        df_future_base = df_grid_full.loc[base_slice_mask]

        if len(df_future_base) == sim_horizon_hours:
            base_curve = df_future_base["load"].values.astype(float)
            baseline_label = "Observed Historical Future Reference"
        else:
            mean_level = df_grid_full["load"].mean()
            std_level = df_grid_full["load"].std()
            h_idx = np.array([ts.hour for ts in future_timestamps])
            base_curve = mean_level + std_level * 0.5 * np.sin(2 * np.pi * (h_idx - 6) / 24.0)
            baseline_label = "Synthetic Seasonal Reference Baseline"

        weather_factor = 1.0
        if temp_shock > 0:
            weather_factor += (temp_shock * 0.022) + (0.0008 * (temp_shock ** 2))
        elif temp_shock < 0:
            abs_cold = abs(temp_shock)
            weather_factor += (abs_cold * 0.018) + (0.0006 * (abs_cold ** 2))

        growth_factor = 1.0 + (growth_shock / 100.0)

        simulated_p50 = base_curve * weather_factor * growth_factor
        baseline_p50 = base_curve.copy()

        grid_scale = np.mean(simulated_p50)
        if "LightGBM" in sim_model or "XGBoost" in sim_model:
            interval_radius = grid_scale * 0.045
            p05 = simulated_p50 - interval_radius * (1.0 + 0.003 * np.arange(sim_horizon_hours))
            p95 = simulated_p50 + interval_radius * (1.0 + 0.003 * np.arange(sim_horizon_hours))
        elif "Chronos" in sim_model:
            interval_radius = grid_scale * 0.032
            p05 = simulated_p50 - interval_radius * (1.0 + 0.005 * np.arange(sim_horizon_hours))
            p95 = simulated_p50 + interval_radius * (1.0 + 0.005 * np.arange(sim_horizon_hours))
        else:
            interval_radius = grid_scale * 0.045
            p05 = simulated_p50 - interval_radius
            p95 = simulated_p50 + interval_radius
            chronos_sim = simulated_p50 * (1.0 + 0.018 * np.sin(np.linspace(0, 3 * np.pi, sim_horizon_hours)))

        fig_sim = go.Figure()

        if len(df_ctx) > 0:
            fig_sim.add_trace(go.Scatter(
                x=df_ctx.index.get_level_values("timestamp") if isinstance(df_ctx.index, pd.MultiIndex) else df_ctx.index,
                y=df_ctx["load"],
                mode="lines", line=dict(color="#64748b", width=2),
                name="Preceding 24h Actual Load",
            ))

        fig_sim.add_trace(go.Scatter(
            x=future_timestamps, y=p95,
            mode="lines", line=dict(width=0), showlegend=False,
            name="90% Upper Bound (P95)",
        ))
        fig_sim.add_trace(go.Scatter(
            x=future_timestamps, y=p05,
            fill="tonexty", mode="lines", line=dict(width=0),
            fillcolor="rgba(37, 99, 235, 0.15)",
            name="Illustrative Parametric Uncertainty Band (Scenario Model - Formula Generated)",
        ))

        fig_sim.add_trace(go.Scatter(
            x=future_timestamps, y=baseline_p50,
            mode="lines", line=dict(color="#94a3b8", width=1.8, dash="dot"),
            name=baseline_label,
        ))

        fig_sim.add_trace(go.Scatter(
            x=future_timestamps, y=simulated_p50,
            mode="lines+markers" if sim_horizon_hours <= 48 else "lines",
            line=dict(color="#2563eb", width=2.6),
            marker=dict(size=4),
            name="Simulated Scenario Forecast P50 (Formula-Generated)",
        ))

        if "Comparison" in sim_model:
            fig_sim.add_trace(go.Scatter(
                x=future_timestamps, y=chronos_sim,
                mode="lines", line=dict(color="#f59e0b", width=2.2, dash="dash"),
                name="Illustrative Formula-Generated Comparison Curve",
            ))

        peak_threshold = np.max(simulated_p50)
        fig_sim.add_shape(
            type="line",
            x0=future_timestamps[0], x1=future_timestamps[-1],
            y0=peak_threshold, y1=peak_threshold,
            line=dict(color="#dc2626", width=1.2, dash="dashdot"),
        )

        fig_sim.update_layout(
            template="plotly_white",
            height=440,
            xaxis_title="Simulation Timeline",
            yaxis_title="Grid Dispatch Load (MW)",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_sim, width="stretch")

        peak_idx = int(np.argmax(simulated_p50))
        peak_time_str = future_timestamps[peak_idx].strftime("%b %d, %H:00")
        total_gwh = np.sum(simulated_p50) / 1000.0 if dataset_key == "ercot" else np.sum(simulated_p50)
        unit_str = "GWh" if dataset_key == "ercot" else "MWh"
        max_reserve = np.max(p95 - simulated_p50)
        delta_mw = np.max(simulated_p50) - np.max(baseline_p50)
        delta_pct = (delta_mw / np.max(baseline_p50)) * 100.0 if np.max(baseline_p50) > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simulated Peak Demand", f"{np.max(simulated_p50):,.0f} MW", help=f"Expected peak at {peak_time_str}")
        m2.metric("Total Horizon Energy", f"{total_gwh:,.1f} {unit_str}", help="Total integrated energy requirement")
        m3.metric("Required Reserve Margin (90%)", f"{max_reserve:,.0f} MW", help="Capacity buffer required between P50 and P95 to prevent blackouts")
        m4.metric("Scenario Demand Shift", f"{delta_mw:+,.0f} MW", delta=f"{delta_pct:+.1f}% vs baseline", delta_color="inverse" if delta_mw > 0 else "normal")

        with st.expander("View Hourly Dispatch Schedule & Export Data"):
            df_sched = pd.DataFrame({
                "Timestamp": future_timestamps,
                "Baseline_Load_MW": np.round(baseline_p50, 1),
                "Simulated_P50_MW": np.round(simulated_p50, 1),
                "Lower_Bound_P05_MW": np.round(p05, 1),
                "Upper_Bound_P95_MW": np.round(p95, 1),
                "Required_Reserve_MW": np.round(p95 - simulated_p50, 1),
            })
            st.dataframe(df_sched, width="stretch", hide_index=True)
            csv_data = df_sched.to_csv(index=False)
            st.download_button(
                "Download Simulated Dispatch CSV",
                data=csv_data,
                file_name=f"grideval_simulation_{dataset_key}_{sim_horizon_hours}h.csv",
                mime="text/csv",
            )
    else:
        st.info("Dataset files not detected in data/processed/. Run pre-processing to enable simulation.")
