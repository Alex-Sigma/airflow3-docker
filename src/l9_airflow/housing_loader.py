from __future__ import annotations

import os
from datetime import date
import pandas as pd
from sqlalchemy import create_engine, text
import openml


def _analytics_pg_url() -> str:
    host = os.environ.get("ANALYTICS_PG_HOST", "analytics-postgres")
    port = os.environ.get("ANALYTICS_PG_PORT", "5432")
    db = os.environ.get("ANALYTICS_PG_DB", "analytics")
    user = os.environ.get("ANALYTICS_PG_USER", "analytics")
    pwd = os.environ.get("ANALYTICS_PG_PASSWORD", "analytics")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def load_boston_like_from_openml() -> pd.DataFrame:
    """
    Boston Housing is not shipped in sklearn anymore.
    We use OpenML to fetch a compatible housing dataset.

    This fetches the dataset and returns a DataFrame with the classic column names:
    crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv
    """
    # Common OpenML dataset name often used for the Boston variant
    ds = openml.datasets.get_dataset("boston")
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)

    df = X.copy()
    df["medv"] = y
    return df


def insert_daily_sample(sample_size: int = 50, run_dt: date | None = None, seed: int = 42) -> dict:
    run_dt = run_dt or date.today()
    df = load_boston_like_from_openml()

    # Normalize OpenML column names (OpenML returns uppercase for Boston)
    df.columns = [c.strip().lower() for c in df.columns]

    expected = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected Boston columns from OpenML. Missing={missing}. Got={df.columns.tolist()}")

    # create stable house_id from row index
    df = df.reset_index(drop=True)
    df["house_id"] = df.index.astype(int)
    df["run_date"] = run_dt  # <- simpler + correct type

    # sample daily batch
    daily = df.sample(n=min(sample_size, len(df)), random_state=seed).copy()

    cols = [
        "crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv",
        "run_date","house_id"
    ]
    daily = daily[cols]

    engine = create_engine(_analytics_pg_url(), future=True)

    daily.to_sql(
        "daily_housing_raw",
        engine,
        schema="public",
        if_exists="append",
        index=False,
        method="multi",
    )

    with engine.connect() as conn:
        cnt = conn.execute(
            text("SELECT COUNT(*) FROM public.daily_housing_raw WHERE run_date = :d"),
            {"d": run_dt},
        ).scalar_one()

    return {"run_date": str(run_dt), "inserted_rows_total_for_day": int(cnt)}