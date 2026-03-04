from __future__ import annotations

import os
from pathlib import Path
from datetime import date  # <-- ADD THIS
import duckdb
import pandas as pd
from sqlalchemy import create_engine, text

def _duckdb_path() -> str:
    # Stored on the shared /opt/airflow/data volume (already mounted in your compose)
    return os.environ.get("DUCKDB_PATH", "/opt/airflow/data/duckdb/warehouse.duckdb")


DDL = """
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS ml;

-- Bronze: mirror of Postgres public.housing_features
CREATE TABLE IF NOT EXISTS bronze.housing_features (
  run_date DATE NOT NULL,
  house_id INTEGER NOT NULL,

  crim DOUBLE,
  zn DOUBLE,
  indus DOUBLE,
  chas INTEGER,
  nox DOUBLE,
  rm DOUBLE,
  age DOUBLE,
  dis DOUBLE,
  rad INTEGER,
  tax DOUBLE,
  ptratio DOUBLE,
  b DOUBLE,
  lstat DOUBLE,

  rm_sq DOUBLE,
  crime_tax_ratio DOUBLE,
  lstat_ptratio_interact DOUBLE,
  dis_rm_interact DOUBLE,
  tax_norm DOUBLE,

  medv DOUBLE,

  PRIMARY KEY (run_date, house_id)
);

-- Minimal ML metadata tables (we’ll use these very soon)
CREATE TABLE IF NOT EXISTS ml.model_registry (
  model_id VARCHAR PRIMARY KEY,
  created_at TIMESTAMP,
  algo VARCHAR,
  train_start DATE,
  train_end DATE,
  target VARCHAR,
  feature_source VARCHAR,
  artifact_path VARCHAR,
  params_json VARCHAR
);

CREATE TABLE IF NOT EXISTS ml.metrics (
  model_id VARCHAR,
  computed_at TIMESTAMP,
  metric_name VARCHAR,
  metric_value DOUBLE,
  notes VARCHAR
);

CREATE TABLE IF NOT EXISTS ml.predictions (
  run_date DATE,
  house_id INTEGER,
  model_id VARCHAR,
  predicted_medv DOUBLE,
  created_at TIMESTAMP,
  PRIMARY KEY (run_date, house_id, model_id)
);
"""


def duckdb_init() -> dict:
    path = _duckdb_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(path)
    try:
        con.execute(DDL)
    finally:
        con.close()

    return {"duckdb_path": path, "status": "initialized"}

import pandas as pd
from sqlalchemy import create_engine, text


def _analytics_pg_url() -> str:
    host = os.environ.get("ANALYTICS_PG_HOST", "analytics-postgres")
    port = os.environ.get("ANALYTICS_PG_PORT", "5432")
    db = os.environ.get("ANALYTICS_PG_DB", "analytics")
    user = os.environ.get("ANALYTICS_PG_USER", "analytics")
    pwd = os.environ.get("ANALYTICS_PG_PASSWORD", "analytics")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def sync_features_to_duckdb(start_dt: date = date(2025, 1, 1), end_dt: date | None = None) -> dict:
    """
    Incremental sync:
      Postgres public.housing_features  -> DuckDB bronze.housing_features

    - Finds max(run_date) already present in DuckDB bronze
    - Pulls only missing/newer dates from Postgres
    - Inserts into DuckDB (idempotent via PRIMARY KEY)
    """
    end_dt = end_dt or date.today()
    duck_path = _duckdb_path()

    con = duckdb.connect(duck_path)
    try:
        # Ensure schemas/tables exist
        con.execute(DDL)

        max_dt = con.execute("SELECT MAX(run_date) FROM bronze.housing_features;").fetchone()[0]
        pull_from = start_dt if max_dt is None else max_dt  # include max_dt day for safety

        pg = create_engine(_analytics_pg_url(), future=True)

        q = text("""
            SELECT
              run_date, house_id,
              crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat,
              rm_sq, crime_tax_ratio, lstat_ptratio_interact, dis_rm_interact, tax_norm,
              medv
            FROM public.housing_features
            WHERE run_date BETWEEN :s AND :e
            ORDER BY run_date, house_id
        """)

        df = pd.read_sql(q, pg, params={"s": pull_from, "e": end_dt})

        if df.empty:
            return {
                "duckdb_path": duck_path,
                "status": "noop",
                "range": {"start": str(pull_from), "end": str(end_dt)},
                "pulled_rows": 0,
            }

        # Insert into DuckDB (PK prevents duplicates)
        con.register("incoming_features", df)
        con.execute("""
            INSERT OR IGNORE INTO bronze.housing_features
            SELECT * FROM incoming_features
        """)

        duck_cnt = con.execute("""
            SELECT COUNT(*) FROM bronze.housing_features
            WHERE run_date BETWEEN ? AND ?
        """, [pull_from, end_dt]).fetchone()[0]

        return {
            "duckdb_path": duck_path,
            "status": "ok",
            "range": {"start": str(pull_from), "end": str(end_dt)},
            "pulled_rows_from_pg": int(df.shape[0]),
            "duckdb_rows_in_range_after": int(duck_cnt),
        }
    finally:
        con.close()