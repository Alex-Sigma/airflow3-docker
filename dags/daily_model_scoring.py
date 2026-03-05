from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from l9_airflow.score_and_publish import score_and_publish_daily


def run_scoring(**context):

    logical_date = context["logical_date"]

    # score yesterday
    target_date = (logical_date - timedelta(days=1)).date()

    res = score_and_publish_daily(
        start_dt=target_date,
        end_dt=target_date,
    )

    print(res)


with DAG(
    dag_id="daily_housing_model_scoring",
    start_date=datetime(2026, 3, 5),
    schedule="@daily",
    catchup=True,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["ml", "scoring"],
) as dag:

    scoring_task = PythonOperator(
        task_id="score_and_publish_daily",
        python_callable=run_scoring,
    )

    scoring_task