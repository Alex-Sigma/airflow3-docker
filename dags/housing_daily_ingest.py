from datetime import datetime

from airflow.sdk import dag, task
from l9_airflow.housing_loader import insert_daily_sample


@dag(
    dag_id="housing_daily_ingest",
    start_date=datetime(2025, 1, 1),
    schedule=None,          # later: "0 6 * * *"
    catchup=False,
    tags=["housing", "postgres", "ingestion"],
)
def housing_daily_ingest():
    @task()
    def ingest():
        info = insert_daily_sample(sample_size=50)
        print("Ingestion result:", info)
        return info

    ingest()


dag = housing_daily_ingest()