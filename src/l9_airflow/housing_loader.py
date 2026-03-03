from __future__ import annotations

import os
from datetime import date, timedelta
import pandas as pd
import json
from pathlib import Path
from sqlalchemy import create_engine, text
import openml





INSERT_SQL = """
INSERT INTO public.daily_housing_raw
(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv, run_date, house_id)
VALUES
(:crim, :zn, :indus, :chas, :nox, :rm, :age, :dis, :rad, :tax, :ptratio, :b, :lstat, :medv, :run_date, :house_id)
ON CONFLICT (run_date, house_id) DO NOTHING;
"""


def _analytics_pg_url() -> str:
    host = os.environ.get("ANALYTICS_PG_HOST", "analytics-postgres")
    port = os.environ.get("ANALYTICS_PG_PORT", "5432")
    db = os.environ.get("ANALYTICS_PG_DB", "analytics")
    user = os.environ.get("ANALYTICS_PG_USER", "analytics")
    pwd = os.environ.get("ANALYTICS_PG_PASSWORD", "analytics")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

SNAPSHOT_PATH = Path("/opt/airflow/data/boston_snapshot.parquet")
META_PATH = Path("/opt/airflow/data/boston_snapshot.meta.json")

# Optional: pick a stable dataset id instead of name
OPENML_DATASET_ID = int(os.environ.get("OPENML_BOSTON_ID", "0"))  # set later

def ensure_boston_snapshot(force_refresh: bool = False) -> pd.DataFrame:
    """
    Ensures we have a stable local snapshot of the Boston-like dataset.
    - If snapshot exists: load it
    - Else: download from OpenML once, save snapshot, then load it
    """
    if SNAPSHOT_PATH.exists() and not force_refresh:
        df = pd.read_parquet(SNAPSHOT_PATH)
        return df

    # Download once
    if OPENML_DATASET_ID > 0:
        ds = openml.datasets.get_dataset(OPENML_DATASET_ID)
    else:
        ds = openml.datasets.get_dataset("boston")

    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
    df = X.copy()
    df["medv"] = y

    # Normalize columns now, so snapshot is already “clean”
    df.columns = [c.strip().lower() for c in df.columns]

    # Persist snapshot
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SNAPSHOT_PATH, index=False)

    # Optional metadata
    meta = {
        "openml_id": getattr(ds, "dataset_id", None),
        "openml_name": getattr(ds, "name", None),
        "target": getattr(ds, "default_target_attribute", None),
        "rows": int(df.shape[0]),
        "cols": df.columns.tolist(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return df


def load_boston_like_from_openml() -> pd.DataFrame:
    return ensure_boston_snapshot(force_refresh=False)


def insert_daily_sample(sample_size: int = 100, run_dt: date | None = None, seed: int = 42) -> dict:
    run_dt = run_dt or date.today()

    df = load_boston_like_from_openml()
    df.columns = [c.strip().lower() for c in df.columns]

    expected = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected columns from OpenML. Missing={missing}. Got={df.columns.tolist()}")

    df = df.reset_index(drop=True)
    df["house_id"] = df.index.astype(int)
    df["run_date"] = run_dt  # date type is fine

    # deterministic per-day sampling (stable reruns)
    day_seed = seed + int(run_dt.strftime("%Y%m%d"))
    daily = df.sample(n=min(sample_size, len(df)), random_state=day_seed).copy()

    # IMPORTANT: keep only the columns that match INSERT_SQL
    daily = daily[expected + ["run_date", "house_id"]]

    engine = create_engine(_analytics_pg_url(), future=True)

    rows = daily.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(text(INSERT_SQL), rows)

        cnt = conn.execute(
            text("SELECT COUNT(*) FROM public.daily_housing_raw WHERE run_date = :d"),
            {"d": run_dt},
        ).scalar_one()

    return {"run_date": str(run_dt), "rows_present_for_day": int(cnt)}

def backfill_missing_days(
    start_dt: date = date(2025, 1, 1),
    end_dt: date | None = None,
    sample_size: int = 100,
    seed: int = 42,
) -> dict:
    end_dt = end_dt or date.today()
    engine = create_engine(_analytics_pg_url(), future=True)

    # 1) Dates that already exist in DB
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT run_date
                FROM public.daily_housing_raw
                WHERE run_date BETWEEN :start AND :end
                ORDER BY run_date
            """),
            {"start": start_dt, "end": end_dt},
        ).all()

    existing = {r[0] for r in rows}  # set[date]

    # 2) Compute all expected dates
    missing_dates: list[date] = []
    current = start_dt
    while current <= end_dt:
        if current not in existing:
            missing_dates.append(current)
        current += timedelta(days=1)

    # 3) Insert only missing dates
    for d in missing_dates:
        insert_daily_sample(sample_size=sample_size, run_dt=d, seed=seed)

    return {
        "start_dt": str(start_dt),
        "end_dt": str(end_dt),
        "missing_days_filled": len(missing_dates),
        "missing_dates_preview": [str(x) for x in missing_dates[:10]],
    }