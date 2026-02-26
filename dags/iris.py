#iris.py
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task
from l9_airflow.train import train


DATA_DIR = Path("/opt/airflow/data")
MODEL_DIR = Path("/opt/airflow/models")

@dag(
    dag_id="iris_train_dag",
    description="Train a simple LogisticRegression model on the Iris dataset",
    start_date=datetime(2025, 9, 1),
    schedule=None,  # set to "0 9 * * *" to run daily at 09:00
    catchup=False,
    tags=["ml", "example", "iris"],
)
def iris_train_pipeline():
    @task()
    def prepare_data() -> dict:
        """Load Iris, split, and persist train/test CSVs."""
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        import pandas as pd

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        iris = load_iris(as_frame=True)
        df = iris.frame.copy()  # columns: features + "target"

        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_path = DATA_DIR / "iris_train.csv"
        test_path = DATA_DIR / "iris_test.csv"

        pd.concat([X_train, y_train.rename("target")], axis=1).to_csv(
            train_path, index=False
        )
        pd.concat([X_test, y_test.rename("target")], axis=1).to_csv(
            test_path, index=False
        )

        return {"train_path": str(train_path), "test_path": str(test_path)}

    @task()
    def train_model(paths: dict) -> dict:
        """Train LogisticRegression and save as .joblib."""
        return train(paths["train_path"], MODEL_DIR)

    @task()
    def evaluate(paths: dict, model_info: dict) -> dict:
        """Evaluate on the test split and print metrics."""
        import pandas as pd
        from sklearn.metrics import accuracy_score, classification_report
        import joblib

        test_df = pd.read_csv(paths["test_path"])
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        clf = joblib.load(model_info["model_path"])
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        report = classification_report(y_test, y_pred, output_dict=False)
        print("=== Iris Test Metrics ===")
        print(f"Model path: {model_info['model_path']}")
        print(f"Train accuracy: {model_info['train_accuracy']:.4f}")
        print(f"Test  accuracy: {acc:.4f}")
        print(report)

        return {"test_accuracy": acc}

    # Orchestration
    paths = prepare_data()
    model_info = train_model(paths)
    evaluate(paths, model_info)


dag = iris_train_pipeline()
