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

OPENML_DATASET_ID = int(os.environ.get("OPENML_BOSTON_ID", "0"))


def ensure_boston_snapshot(force_refresh: bool = False) -> pd.DataFrame:
    if SNAPSHOT_PATH.exists() and not force_refresh:
        return pd.read_parquet(SNAPSHOT_PATH)

    if OPENML_DATASET_ID > 0:
        ds = openml.datasets.get_dataset(OPENML_DATASET_ID)
    else:
        ds = openml.datasets.get_dataset("boston")

    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
    df = X.copy()
    df["medv"] = y
    df.columns = [c.strip().lower() for c in df.columns]

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SNAPSHOT_PATH, index=False)

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

    expected = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected columns from OpenML. Missing={missing}. Got={df.columns.tolist()}")

    df = df.reset_index(drop=True)
    df["house_id"] = df.index.astype(int)
    df["run_date"] = run_dt

    day_seed = seed + int(run_dt.strftime("%Y%m%d"))
    daily = df.sample(n=min(sample_size, len(df)), random_state=day_seed).copy()
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

    existing = {r[0] for r in rows}

    missing_dates: list[date] = []
    current = start_dt
    while current <= end_dt:
        if current not in existing:
            missing_dates.append(current)
        current += timedelta(days=1)

    for d in missing_dates:
        insert_daily_sample(sample_size=sample_size, run_dt=d, seed=seed)

    return {
        "start_dt": str(start_dt),
        "end_dt": str(end_dt),
        "missing_days_filled": len(missing_dates),
        "missing_dates_preview": [str(x) for x in missing_dates[:10]],
    }


def backfill_features(
    start_dt: date = date(2025, 1, 1),
    end_dt: date | None = None,
) -> dict:
    """
    Populate public.housing_features for dates missing in features but present in raw.
    """

    end_dt = end_dt or date.today()
    engine = create_engine(_analytics_pg_url(), future=True)

    with engine.connect() as conn:
        raw_dates = conn.execute(
            text("""
                SELECT DISTINCT run_date
                FROM public.daily_housing_raw
                WHERE run_date BETWEEN :s AND :e
                ORDER BY run_date
            """),
            {"s": start_dt, "e": end_dt},
        ).scalars().all()

    with engine.connect() as conn:
        feat_dates = conn.execute(
            text("""
                SELECT DISTINCT run_date
                FROM public.housing_features
                WHERE run_date BETWEEN :s AND :e
            """),
            {"s": start_dt, "e": end_dt},
        ).scalars().all()

    raw_dates_set = set(raw_dates)
    feat_dates_set = set(feat_dates)
    dates_missing = sorted(list(raw_dates_set - feat_dates_set))

    inserted = 0
    with engine.begin() as conn:
        for d in dates_missing:
            conn.execute(
                text("""
INSERT INTO public.housing_features
(
    run_date, house_id,
    crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat,
    rm_sq, crime_tax_ratio, lstat_ptratio_interact, dis_rm_interact, tax_norm,
    medv, log1p_medv
)
SELECT
    raw.run_date,
    raw.house_id,
    raw.crim, raw.zn, raw.indus, raw.chas, raw.nox, raw.rm, raw.age, raw.dis,
    raw.rad, raw.tax, raw.ptratio, raw.b, raw.lstat,

    raw.rm * raw.rm AS rm_sq,
    CASE WHEN raw.tax > 0 THEN raw.crim / raw.tax ELSE NULL END AS crime_tax_ratio,
    raw.lstat * raw.ptratio AS lstat_ptratio_interact,
    raw.dis * raw.rm AS dis_rm_interact,
    raw.tax / 1000.0 AS tax_norm,

    raw.medv,
    LN(1 + raw.medv) AS log1p_medv
FROM public.daily_housing_raw raw
WHERE raw.run_date = :dt
ON CONFLICT (run_date, house_id) DO UPDATE
SET
    crim = EXCLUDED.crim,
    zn = EXCLUDED.zn,
    indus = EXCLUDED.indus,
    chas = EXCLUDED.chas,
    nox = EXCLUDED.nox,
    rm = EXCLUDED.rm,
    age = EXCLUDED.age,
    dis = EXCLUDED.dis,
    rad = EXCLUDED.rad,
    tax = EXCLUDED.tax,
    ptratio = EXCLUDED.ptratio,
    b = EXCLUDED.b,
    lstat = EXCLUDED.lstat,
    rm_sq = EXCLUDED.rm_sq,
    crime_tax_ratio = EXCLUDED.crime_tax_ratio,
    lstat_ptratio_interact = EXCLUDED.lstat_ptratio_interact,
    dis_rm_interact = EXCLUDED.dis_rm_interact,
    tax_norm = EXCLUDED.tax_norm,
    medv = EXCLUDED.medv,
    log1p_medv = EXCLUDED.log1p_medv
                """),
                {"dt": d},
            )
            inserted += 1

    return {
        "start_dt": str(start_dt),
        "end_dt": str(end_dt),
        "dates_checked": len(raw_dates),
        "dates_missing": len(dates_missing),
        "dates_filled": inserted,
        "preview": [str(x) for x in dates_missing[:10]],
    }