from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

from l9_airflow.score_and_publish import score_and_publish_daily
from l9_airflow.export_duckdb_objects import export_duckdb_object


def run_scoring(**context):
    logical_date = context["logical_date"]

    # score yesterday
    target_date = (logical_date - timedelta(days=1)).date()

    res = score_and_publish_daily(
        start_dt=target_date,
        end_dt=target_date,
    )

    print(res)


def run_export():
    res_csv = export_duckdb_object(
        "gold.housing_predictions_daily",
        export_format="csv",
    )
    print(res_csv)

    res_parquet = export_duckdb_object(
        "gold.housing_predictions_daily",
        export_format="parquet",
    )
    print(res_parquet)


def map_scoring_run_to_ingest_run(logical_date, **kwargs):
    """
    scoring logical_date: 2026-03-07 00:15
    ingest logical_date:  2026-03-07 00:00

    So we explicitly map scoring run to same-day midnight ingest run.
    """
    return logical_date.replace(hour=0, minute=0, second=0, microsecond=0)


with DAG(
    dag_id="daily_housing_model_scoring",
    start_date=datetime(2026, 3, 5),
    schedule="15 0 * * *",
    catchup=True,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["ml", "scoring", "export"],
) as dag:

    wait_for_housing_ingest = ExternalTaskSensor(
        task_id="wait_for_housing_ingest",
        external_dag_id="housing_daily_ingest",
        external_task_id=None,   # wait for whole DAG run
        allowed_states=["success"],
        failed_states=["failed"],
        execution_date_fn=map_scoring_run_to_ingest_run,
        mode="reschedule",
        poke_interval=30,
        timeout=60 * 60,
    )

    scoring_task = PythonOperator(
        task_id="score_and_publish_daily",
        python_callable=run_scoring,
    )

    export_task = PythonOperator(
        task_id="export_gold_predictions",
        python_callable=run_export,
    )

    wait_for_housing_ingest >> scoring_task >> export_task