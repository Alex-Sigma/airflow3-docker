from __future__ import annotations

import os
import pendulum

from airflow.sdk import dag, task


@dag(
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["smoke-test", "ml"],
)
def ml_packages_smoke_test():
    """
    Smoke test DAG:
    - Creates a small pandas DataFrame
    - Trains a tiny sklearn model
    - Saves it with joblib into the mounted logs folder
    """

    @task
    def make_dataframe() -> dict:
        import pandas as pd

        df = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [2, 1, 0, 1, 2],
                "y":  [3, 3, 3, 5, 7],
            }
        )

        # Return as JSON-serializable payload for XCom
        return {
            "rows": df.to_dict(orient="records"),
            "shape": [int(df.shape[0]), int(df.shape[1])],
        }

    @task
    def train_and_save(payload: dict) -> str:
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        import joblib

        df = pd.DataFrame(payload["rows"])

        X = df[["x1", "x2"]]
        y = df["y"]

        model = LinearRegression()
        model.fit(X, y)

        out_dir = "/opt/airflow/logs/artifacts"
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(out_dir, "linreg_model.joblib")
        joblib.dump(model, model_path)

        # Return path so you can see it in task output
        return model_path

    payload = make_dataframe()
    train_and_save(payload)


ml_packages_smoke_test()