from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(train_path: str, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X_train, y_train)

    # quick train accuracy for visibility
    train_pred = clf.predict(X_train)
    train_acc = float(accuracy_score(y_train, train_pred))

    model_path = model_dir / "iris_logreg.joblib"
    joblib.dump(clf, model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Train accuracy: {train_acc:.4f}")

    return {"model_path": str(model_path), "train_accuracy": train_acc}


if __name__ == "__main__":
    train("data/iris_train.csv", Path("models"))
