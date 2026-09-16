import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags=(1, 2, 3, 24, 168)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows=(24, 168)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        # Shift by 1 hour to prevent lookahead leakage
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=w).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w, min_periods=w).std()
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)

    pre_holidays = holidays - pd.Timedelta(days=1)
    df["is_pre_holiday"] = df.index.normalize().isin(pre_holidays).astype(int)
    return df


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "drybulb" in df.columns:
        df["temp_squared"] = df["drybulb"] ** 2
    if "dewpoint" in df.columns and "drybulb" in df.columns:
        df["dewpoint_depression"] = df["drybulb"] - df["dewpoint"]
    return df


def prepare_features(df: pd.DataFrame, target_col: str, dataset_name: str) -> pd.DataFrame:
    df_feat = add_temporal_features(df)
    df_feat = add_lag_features(df_feat, target_col)
    df_feat = add_rolling_features(df_feat, target_col)
    df_feat = add_holiday_features(df_feat)

    if dataset_name == "gefcom":
        df_feat = add_temperature_features(df_feat)

    return df_feat.dropna()


def get_feature_columns(dataset_name: str) -> list[str]:
    base_features = [
        "hour", "dayofweek", "month", "dayofyear", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
        "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_std_168",
        "is_holiday", "is_pre_holiday"
    ]
    if dataset_name == "gefcom":
        base_features += ["drybulb", "dewpoint", "temp_squared", "dewpoint_depression"]
    return base_features
