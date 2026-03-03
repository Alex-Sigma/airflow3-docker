from __future__ import annotations
from datetime import date, datetime

from airflow.sdk import dag, task
from l9_airflow.housing_loader import backfill_missing_days, backfill_features

@dag(
    dag_id="housing_daily_ingest",
    description="Backfill daily housing raw data and materialized features",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # later: "0 6 * * *"
    catchup=False,
    tags=["housing", "ingestion", "postgres", "features"],
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

    raw = raw_ingest()
    feat = features_ingest()
    feat.set_upstream(raw)  # ensure features backfill after raw backfill

housing_daily_ingest_dag = housing_daily_ingest()