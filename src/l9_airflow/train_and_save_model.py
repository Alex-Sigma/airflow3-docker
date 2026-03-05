from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    run_id: str
    model_path: str
    metrics: Dict[str, float]
    n_train: int
    n_valid: int
    model_name: str
    target_col: str


def _utc_now_ts() -> datetime:
    # store as TIMESTAMP, not string
    return datetime.now(timezone.utc).replace(microsecond=0)


def _ensure_ml_schema(con: duckdb.DuckDBPyConnection) -> None:
    """
    Assumes you already recreated tables via SQL.
    This is just a safety check so the function fails early with a clear message.
    """
    con.execute("CREATE SCHEMA IF NOT EXISTS ml;")
    # quick existence checks
    con.execute("SELECT 1 FROM information_schema.tables WHERE table_schema='ml' AND table_name='predictions';").fetchone()
    con.execute("SELECT 1 FROM information_schema.tables WHERE table_schema='ml' AND table_name='metrics';").fetchone()
    con.execute("SELECT 1 FROM information_schema.tables WHERE table_schema='ml' AND table_name='model_registry';").fetchone()


def _get_view_columns(con: duckdb.DuckDBPyConnection, view_name: str) -> List[str]:
    schema, name = view_name.split(".")
    df = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = ?
          AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema, name],
    ).df()
    return df["column_name"].tolist()


def _infer_features(all_cols: List[str]) -> List[str]:
    # do not ever use ids or any targets as features
    forbidden = {"run_date", "house_id", "medv", "log1p_medv"}
    return [c for c in all_cols if c not in forbidden]


def _load_split(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    cols_sql = ", ".join(["run_date", "house_id"] + feature_cols + [target_col])
    df = con.execute(f"SELECT {cols_sql} FROM {view_name}").df()

    # safety: drop any nulls (your silver should be clean, but keep this anyway)
    df = df.dropna(subset=feature_cols + [target_col])

    id_df = df[["run_date", "house_id"]].copy()
    X_df = df[feature_cols].copy()
    y = df[target_col].astype(float).to_numpy()

    return X_df, y, id_df


def train_and_save_model(
    duckdb_path: str = "/opt/airflow/data/duckdb/warehouse.duckdb",
    train_view: str = "silver.housing_train",
    valid_view: str = "silver.housing_valid",
    target_col: str = "medv",  # or "log1p_medv"
    model_name: str = "ridge_baseline",
    alpha: float = 1.0,
    artifacts_dir: str = "/opt/airflow/models",
    feature_cols: Optional[List[str]] = None,  # if None -> inferred from train view
) -> TrainResult:
    """
    - Reads train/valid from DuckDB silver views
    - Trains Ridge with StandardScaler (good default for Ridge)
    - Saves pickle artifact (pipeline)
    - Writes:
        * ml.predictions (row-level, includes y_true/y_pred)
        * ml.metrics (one row)
        * ml.model_registry (one row with params + feature list + artifact path)
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    run_id = str(uuid.uuid4())
    created_at = _utc_now_ts()

    con = duckdb.connect(duckdb_path)
    try:
        _ensure_ml_schema(con)

        all_cols = _get_view_columns(con, train_view)
        if target_col not in all_cols:
            raise ValueError(f"target_col='{target_col}' not found in {train_view}. Available: {all_cols}")

        if feature_cols is None:
            feature_cols = _infer_features(all_cols)

        # load
        X_train_df, y_train, id_train = _load_split(con, train_view, feature_cols, target_col)
        X_valid_df, y_valid, id_valid = _load_split(con, valid_view, feature_cols, target_col)

        # pipeline: scale numeric -> ridge
        preproc = ColumnTransformer(
            transformers=[("num", StandardScaler(), feature_cols)],
            remainder="drop",
        )
        pipe = Pipeline(
            steps=[
                ("preprocess", preproc),
                ("model", Ridge(alpha=alpha, random_state=42)),
            ]
        )

        pipe.fit(X_train_df, y_train)
        y_pred = pipe.predict(X_valid_df).astype(float)

        mae = float(mean_absolute_error(y_valid, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
        r2 = float(r2_score(y_valid, y_pred))
        metrics = {"mae": mae, "rmse": rmse, "r2": r2}

        # save artifact
        model_path = os.path.join(artifacts_dir, f"{model_name}__{target_col}__{run_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "pipeline": pipe,
                    "model_name": model_name,
                    "target_col": target_col,
                    "feature_cols": feature_cols,
                    "params": {"alpha": alpha},
                    "train_view": train_view,
                    "valid_view": valid_view,
                    "run_id": run_id,
                    "created_at": created_at,
                },
                f,
            )

        # write predictions
        pred_df = pd.DataFrame(
            {
                "run_id": run_id,
                "model_name": model_name,
                "target_col": target_col,
                "run_date": id_valid["run_date"],
                "house_id": id_valid["house_id"],
                "y_true": y_valid.astype(float),
                "y_pred": y_pred.astype(float),
                "created_at": created_at,
            }
        )
        con.register("pred_df", pred_df)
        con.execute(
            """
            INSERT INTO ml.predictions
            SELECT run_id, model_name, target_col, run_date, house_id, y_true, y_pred, created_at
            FROM pred_df
            """
        )
        con.unregister("pred_df")

        # write metrics (one row)
        metrics_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "target_col": target_col,
                    "n_train": int(len(X_train_df)),
                    "n_valid": int(len(X_valid_df)),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "created_at": created_at,
                }
            ]
        )
        con.register("metrics_row", metrics_row)
        con.execute(
            """
            INSERT INTO ml.metrics
            SELECT run_id, model_name, target_col, n_train, n_valid, mae, rmse, r2, created_at
            FROM metrics_row
            """
        )
        con.unregister("metrics_row")

        # write registry
        registry_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "target_col": target_col,
                    "params_json": json.dumps({"alpha": alpha}),
                    "feature_list_json": json.dumps(feature_cols),
                    "train_view": train_view,
                    "valid_view": valid_view,
                    "model_path": model_path,
                    "metrics_json": json.dumps(metrics),
                    "created_at": created_at,
                }
            ]
        )
        con.register("registry_row", registry_row)
        con.execute(
            """
            INSERT INTO ml.model_registry
            SELECT run_id, model_name, target_col, params_json, feature_list_json,
                   train_view, valid_view, model_path, metrics_json, created_at
            FROM registry_row
            """
        )
        con.unregister("registry_row")

        return TrainResult(
            run_id=run_id,
            model_path=model_path,
            metrics=metrics,
            n_train=int(len(X_train_df)),
            n_valid=int(len(X_valid_df)),
            model_name=model_name,
            target_col=target_col,
        )

    finally:
        con.close()