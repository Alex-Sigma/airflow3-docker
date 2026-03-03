from __future__ import annotations
from datetime import date, datetime
import json
from pathlib import Path
from airflow.sdk import dag, task
from l9_airflow.housing_loader import backfill_missing_days



@dag(
    dag_id="housing_daily_ingest",
    description="Backfill Boston housing daily raw data (fills only missing days) into analytics-postgres",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # later you can set: "0 6 * * *"
    catchup=False,
    tags=["housing", "ingestion", "postgres", "backfill"],
)
def housing_daily_ingest():
    @task()
    def backfill():
        # Always ensure full history exists from 2025-01-01 up to today.
        # Function only inserts missing dates, so re-runs are fast & safe.
        return backfill_missing_days(
            start_dt=date(2025, 1, 1),
            end_dt=date.today(),
            sample_size=100,
            seed=42,
        )

    backfill()


dag = housing_daily_ingest()