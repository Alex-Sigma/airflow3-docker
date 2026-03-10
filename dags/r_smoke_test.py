from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="r_smoke_test",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["r", "smoke-test"],
) as dag:

    run_r_script = BashOperator(
        task_id="run_r_hello",
        bash_command="""
        docker exec r-runtime Rscript /project/R/scripts/hello_r.R
        """,
    )