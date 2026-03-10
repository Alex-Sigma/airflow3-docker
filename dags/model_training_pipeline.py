from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from l9_airflow.train_windowed_model import train_windowed_model
from l9_airflow.model_lifecycle import (
    compare_challenger_to_active_champion,
    promote_model_to_champion,
)


def compute_training_windows(logical_date):
    """
    For retraining run at logical_date T:

    Train: T-187 to T-8   -> 180 days
    Valid: T-7   to T-1   -> 7 days
    """
    anchor_day = logical_date.date()

    train_start_dt = anchor_day - timedelta(days=187)
    train_end_dt = anchor_day - timedelta(days=8)

    valid_start_dt = anchor_day - timedelta(days=7)
    valid_end_dt = anchor_day - timedelta(days=1)

    return {
        "train_start_dt": train_start_dt,
        "train_end_dt": train_end_dt,
        "valid_start_dt": valid_start_dt,
        "valid_end_dt": valid_end_dt,
    }


def train_challenger(**context):
    logical_date = context["logical_date"]
    windows = compute_training_windows(logical_date)

    res = train_windowed_model(
        train_start_dt=windows["train_start_dt"],
        train_end_dt=windows["train_end_dt"],
        valid_start_dt=windows["valid_start_dt"],
        valid_end_dt=windows["valid_end_dt"],
        target_col="log1p_medv",
        model_name="ridge_retrain_log1p",
        alpha=1.0,
        dry_run=False,
        save_artifact=True,
        write_duckdb=True,
    )

    print(res)
    return res.run_id


def evaluate_and_promote(**context):
    ti = context["ti"]
    challenger_run_id = ti.xcom_pull(task_ids="train_challenger")

    if not challenger_run_id:
        raise ValueError("No challenger_run_id received from train_challenger task")

    comparison = compare_challenger_to_active_champion(
        challenger_run_id=challenger_run_id,
        min_relative_improvement=0.05,
    )

    print(comparison)

    if comparison["qualifies_for_promotion"]:
        promotion = promote_model_to_champion(
            run_id=challenger_run_id,
            reason=(
                "automatic retraining promotion: challenger exceeded active champion "
                "by required relative RMSE improvement threshold"
            ),
        )
        print(promotion)
        return {
            "status": "promoted",
            "challenger_run_id": challenger_run_id,
            "comparison": comparison,
            "promotion": promotion,
        }

    return {
        "status": "kept_existing_champion",
        "challenger_run_id": challenger_run_id,
        "comparison": comparison,
    }


with DAG(
    dag_id="model_training_pipeline",
    start_date=datetime(2026, 3, 10),
    schedule="0 2 * * 1",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["ml", "training", "lifecycle", "champion"],
) as dag:

    train_task = PythonOperator(
        task_id="train_challenger",
        python_callable=train_challenger,
    )

    compare_and_promote_task = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
    )

    train_task >> compare_and_promote_task