from __future__ import annotations

from datetime import date, datetime

from airflow.sdk import dag, task
from l9_airflow.housing_loader import backfill_missing_days, backfill_features
from l9_airflow.duckdb_warehouse import sync_features_to_duckdb, duckdb_init


@dag(
    dag_id="housing_daily_ingest",
    description="Backfill daily housing raw data, materialize features in Postgres, then sync to DuckDB",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # later: "0 6 * * *"
    catchup=False,
    tags=["housing", "ingestion", "postgres", "features", "duckdb"],
)
def housing_daily_ingest():
    @task()
    def raw_ingest():
        return backfill_missing_days(
            start_dt=date(2025, 1, 1),
            end_dt=date.today(),
            sample_size=100,
            seed=42,
        )

    @task()
    def features_ingest():
        return backfill_features(
            start_dt=date(2025, 1, 1),
            end_dt=date.today(),
        )

    @task()
    def duckdb_setup():
        # safe to call every run (DDL is IF NOT EXISTS)
        return duckdb_init()

    @task()
    def duckdb_sync():
        return sync_features_to_duckdb(
            start_dt=date(2025, 1, 1),
            end_dt=date.today(),
        )

    raw = raw_ingest()
    feat = features_ingest()
    setup = duckdb_setup()
    sync = duckdb_sync()

    # Dependency chain
    raw >> feat >> setup >> sync


dag = housing_daily_ingest()