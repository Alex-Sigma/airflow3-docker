# src/l9_airflow/train_windowed_model.py

from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DUCKDB_PATH_DEFAULT = "/opt/airflow/data/duckdb/warehouse.duckdb"


@dataclass
class WindowedTrainResult:
    run_id: str
    model_name: str
    target_col: str
    model_path: Optional[str]
    train_start_dt: str
    train_end_dt: str
    valid_start_dt: str
    valid_end_dt: str
    metrics: Dict[str, float]
    n_train: int
    n_valid: int
    feature_cols: List[str]
    dry_run: bool


def _duckdb_path() -> str:
    return os.environ.get("DUCKDB_PATH", DUCKDB_PATH_DEFAULT)


def _utc_now_ts() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _ensure_ml_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS ml;")

    # fail early if expected tables do not exist
    required = ["predictions", "metrics", "model_registry"]
    for tbl in required:
        row = con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'ml'
              AND table_name = ?
            """,
            [tbl],
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Required table ml.{tbl} does not exist")


def _ensure_scoring_view(con: duckdb.DuckDBPyConnection) -> None:
    """
    Stable source view for both training and scoring.
    """
    con.execute("""
    CREATE SCHEMA IF NOT EXISTS silver;

    CREATE OR REPLACE VIEW silver.housing_score AS
    SELECT *
    FROM bronze.housing_features;
    """)


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
    forbidden = {"run_date", "house_id", "medv", "log1p_medv"}
    return [c for c in all_cols if c not in forbidden]


def _load_window(
    con: duckdb.DuckDBPyConnection,
    source_view: str,
    start_dt: date,
    end_dt: date,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    cols_sql = ", ".join(["run_date", "house_id"] + feature_cols + [target_col])

    df = con.execute(
        f"""
        SELECT {cols_sql}
        FROM {source_view}
        WHERE run_date BETWEEN ? AND ?
        ORDER BY run_date, house_id
        """,
        [start_dt, end_dt],
    ).df()

    df = df.dropna(subset=feature_cols + [target_col])

    id_df = df[["run_date", "house_id"]].copy()
    X_df = df[feature_cols].copy()
    y = df[target_col].astype(float).to_numpy()

    return X_df, y, id_df


def _build_pipeline(feature_cols: List[str], alpha: float) -> Pipeline:
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
    return pipe


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _save_artifact(
    pipe: Pipeline,
    run_id: str,
    model_name: str,
    target_col: str,
    feature_cols: List[str],
    alpha: float,
    train_start_dt: date,
    train_end_dt: date,
    valid_start_dt: date,
    valid_end_dt: date,
    created_at: datetime,
    artifacts_dir: str,
    save_artifact: bool,
) -> Optional[str]:
    if not save_artifact:
        return None

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(
        artifacts_dir,
        f"{model_name}__{target_col}__{run_id}.pkl",
    )

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": pipe,
                "model_name": model_name,
                "target_col": target_col,
                "feature_cols": feature_cols,
                "params": {"alpha": alpha},
                "train_window": [str(train_start_dt), str(train_end_dt)],
                "valid_window": [str(valid_start_dt), str(valid_end_dt)],
                "run_id": run_id,
                "created_at": created_at,
            },
            f,
        )

    return model_path


def _write_predictions(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    model_name: str,
    target_col: str,
    id_valid: pd.DataFrame,
    y_valid: np.ndarray,
    y_pred: np.ndarray,
    created_at: datetime,
) -> None:
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


def _write_metrics(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    model_name: str,
    target_col: str,
    metrics: Dict[str, float],
    n_train: int,
    n_valid: int,
    created_at: datetime,
) -> None:
    metrics_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model_name": model_name,
                "target_col": target_col,
                "n_train": int(n_train),
                "n_valid": int(n_valid),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
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


def _write_registry(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    model_name: str,
    target_col: str,
    alpha: float,
    feature_cols: List[str],
    train_start_dt: date,
    train_end_dt: date,
    valid_start_dt: date,
    valid_end_dt: date,
    model_path: Optional[str],
    metrics: Dict[str, float],
    created_at: datetime,
) -> None:
    registry_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model_name": model_name,
                "target_col": target_col,
                "params_json": json.dumps(
                    {
                        "alpha": alpha,
                        "train_start_dt": str(train_start_dt),
                        "train_end_dt": str(train_end_dt),
                        "valid_start_dt": str(valid_start_dt),
                        "valid_end_dt": str(valid_end_dt),
                    }
                ),
                "feature_list_json": json.dumps(feature_cols),
                "train_view": "silver.housing_score",
                "valid_view": "silver.housing_score",
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


def train_windowed_model(
    train_start_dt: date,
    train_end_dt: date,
    valid_start_dt: date,
    valid_end_dt: date,
    duckdb_path: Optional[str] = None,
    source_view: str = "silver.housing_score",
    target_col: str = "log1p_medv",
    model_name: str = "ridge_retrain_log1p",
    alpha: float = 1.0,
    feature_cols: Optional[List[str]] = None,
    artifacts_dir: str = "/opt/airflow/models",
    dry_run: bool = False,
    save_artifact: bool = True,
    write_duckdb: bool = True,
) -> WindowedTrainResult:
    """
    Train the approved recipe on explicit time windows.

    Approved recipe:
      - target = log1p_medv
      - features = inferred from source view
      - preprocessing = StandardScaler
      - model = Ridge

    dry_run=True:
      - trains and evaluates
      - does NOT write predictions/metrics/registry to DuckDB
      - artifact saving can still be controlled separately via save_artifact
    """
    duckdb_path = duckdb_path or _duckdb_path()
    created_at = _utc_now_ts()
    run_id = str(uuid.uuid4())

    if dry_run:
        write_duckdb = False

    con = duckdb.connect(duckdb_path)
    try:
        _ensure_ml_schema(con)
        _ensure_scoring_view(con)

        all_cols = _get_view_columns(con, source_view)
        if target_col not in all_cols:
            raise ValueError(f"target_col='{target_col}' not found in {source_view}. Available: {all_cols}")

        if feature_cols is None:
            feature_cols = _infer_features(all_cols)

        X_train_df, y_train, id_train = _load_window(
            con=con,
            source_view=source_view,
            start_dt=train_start_dt,
            end_dt=train_end_dt,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        X_valid_df, y_valid, id_valid = _load_window(
            con=con,
            source_view=source_view,
            start_dt=valid_start_dt,
            end_dt=valid_end_dt,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        if len(X_train_df) == 0:
            raise ValueError("Training window returned 0 rows")
        if len(X_valid_df) == 0:
            raise ValueError("Validation window returned 0 rows")

        pipe = _build_pipeline(feature_cols=feature_cols, alpha=alpha)
        pipe.fit(X_train_df, y_train)

        y_pred = pipe.predict(X_valid_df).astype(float)
        metrics = _compute_metrics(y_true=y_valid, y_pred=y_pred)

        model_path = _save_artifact(
            pipe=pipe,
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            feature_cols=feature_cols,
            alpha=alpha,
            train_start_dt=train_start_dt,
            train_end_dt=train_end_dt,
            valid_start_dt=valid_start_dt,
            valid_end_dt=valid_end_dt,
            created_at=created_at,
            artifacts_dir=artifacts_dir,
            save_artifact=save_artifact,
        )

        if write_duckdb:
            _write_predictions(
                con=con,
                run_id=run_id,
                model_name=model_name,
                target_col=target_col,
                id_valid=id_valid,
                y_valid=y_valid,
                y_pred=y_pred,
                created_at=created_at,
            )

            _write_metrics(
                con=con,
                run_id=run_id,
                model_name=model_name,
                target_col=target_col,
                metrics=metrics,
                n_train=len(X_train_df),
                n_valid=len(X_valid_df),
                created_at=created_at,
            )

            _write_registry(
                con=con,
                run_id=run_id,
                model_name=model_name,
                target_col=target_col,
                alpha=alpha,
                feature_cols=feature_cols,
                train_start_dt=train_start_dt,
                train_end_dt=train_end_dt,
                valid_start_dt=valid_start_dt,
                valid_end_dt=valid_end_dt,
                model_path=model_path,
                metrics=metrics,
                created_at=created_at,
            )

        return WindowedTrainResult(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            model_path=model_path,
            train_start_dt=str(train_start_dt),
            train_end_dt=str(train_end_dt),
            valid_start_dt=str(valid_start_dt),
            valid_end_dt=str(valid_end_dt),
            metrics=metrics,
            n_train=int(len(X_train_df)),
            n_valid=int(len(X_valid_df)),
            feature_cols=feature_cols,
            dry_run=dry_run,
        )

    finally:
        con.close()