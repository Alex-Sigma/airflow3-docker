from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DUCKDB_PATH_DEFAULT = "/opt/airflow/data/duckdb/warehouse.duckdb"


def _duckdb_path() -> str:
    return os.environ.get("DUCKDB_PATH", DUCKDB_PATH_DEFAULT)


def _utc_now_ts() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


LIFECYCLE_DDL = """
CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.model_champion (
    champion_id VARCHAR,
    run_id VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    target_col VARCHAR NOT NULL,
    promoted_at TIMESTAMP NOT NULL,
    active_from TIMESTAMP NOT NULL,
    active_to TIMESTAMP,
    promotion_reason VARCHAR,
    is_active BOOLEAN NOT NULL
);
"""


@dataclass
class ChampionInfo:
    champion_id: str
    run_id: str
    model_name: str
    target_col: str
    promoted_at: datetime
    active_from: datetime
    active_to: Optional[datetime]
    promotion_reason: Optional[str]
    is_active: bool
    model_path: str


def ensure_model_lifecycle_tables(duckdb_path: Optional[str] = None) -> None:
    duckdb_path = duckdb_path or _duckdb_path()
    con = duckdb.connect(duckdb_path)
    try:
        con.execute(LIFECYCLE_DDL)
    finally:
        con.close()


def get_model_registry_row(run_id: str, duckdb_path: Optional[str] = None) -> Dict[str, Any]:
    duckdb_path = duckdb_path or _duckdb_path()
    con = duckdb.connect(duckdb_path)
    try:
        row = con.execute(
            """
            SELECT
                run_id,
                model_name,
                target_col,
                params_json,
                feature_list_json,
                train_view,
                valid_view,
                model_path,
                metrics_json,
                created_at
            FROM ml.model_registry
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

        if row is None:
            raise ValueError(f"run_id='{run_id}' not found in ml.model_registry")

        cols = [
            "run_id",
            "model_name",
            "target_col",
            "params_json",
            "feature_list_json",
            "train_view",
            "valid_view",
            "model_path",
            "metrics_json",
            "created_at",
        ]
        return dict(zip(cols, row))
    finally:
        con.close()


def get_active_champion(duckdb_path: Optional[str] = None) -> ChampionInfo:
    duckdb_path = duckdb_path or _duckdb_path()
    con = duckdb.connect(duckdb_path)
    try:
        con.execute(LIFECYCLE_DDL)

        row = con.execute(
            """
            SELECT
                c.champion_id,
                c.run_id,
                c.model_name,
                c.target_col,
                c.promoted_at,
                c.active_from,
                c.active_to,
                c.promotion_reason,
                c.is_active,
                r.model_path
            FROM ml.model_champion c
            JOIN ml.model_registry r
              ON c.run_id = r.run_id
            WHERE c.is_active = TRUE
            ORDER BY c.promoted_at DESC
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            raise RuntimeError("No active champion found in ml.model_champion")

        return ChampionInfo(
            champion_id=row[0],
            run_id=row[1],
            model_name=row[2],
            target_col=row[3],
            promoted_at=row[4],
            active_from=row[5],
            active_to=row[6],
            promotion_reason=row[7],
            is_active=row[8],
            model_path=row[9],
        )
    finally:
        con.close()


def deactivate_current_champion(con: duckdb.DuckDBPyConnection) -> None:
    now_ts = _utc_now_ts()
    con.execute(
        """
        UPDATE ml.model_champion
        SET
            is_active = FALSE,
            active_to = ?
        WHERE is_active = TRUE
        """,
        [now_ts],
    )


def promote_model_to_champion(
    run_id: str,
    reason: str = "manual promotion",
    duckdb_path: Optional[str] = None,
) -> Dict[str, Any]:
    duckdb_path = duckdb_path or _duckdb_path()
    ensure_model_lifecycle_tables(duckdb_path)

    registry_row = get_model_registry_row(run_id, duckdb_path)
    now_ts = _utc_now_ts()
    champion_id = str(uuid.uuid4())

    con = duckdb.connect(duckdb_path)
    try:
        deactivate_current_champion(con)

        con.execute(
            """
            INSERT INTO ml.model_champion (
                champion_id,
                run_id,
                model_name,
                target_col,
                promoted_at,
                active_from,
                active_to,
                promotion_reason,
                is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                champion_id,
                registry_row["run_id"],
                registry_row["model_name"],
                registry_row["target_col"],
                now_ts,
                now_ts,
                None,
                reason,
                True,
            ],
        )

        return {
            "status": "ok",
            "champion_id": champion_id,
            "run_id": registry_row["run_id"],
            "model_name": registry_row["model_name"],
            "target_col": registry_row["target_col"],
            "promoted_at": str(now_ts),
            "reason": reason,
        }
    finally:
        con.close()


def _load_predictions_for_run(run_id: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT run_id, model_name, target_col, run_date, house_id, y_true, y_pred, created_at
        FROM ml.predictions
        WHERE run_id = ?
        ORDER BY run_date, house_id
        """,
        [run_id],
    ).df()

    if df.empty:
        raise ValueError(f"No predictions found in ml.predictions for run_id='{run_id}'")

    return df


def _compute_medv_scale_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    model_name = str(df["model_name"].iloc[0])
    target_col = str(df["target_col"].iloc[0])
    run_id = str(df["run_id"].iloc[0])

    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    if target_col == "log1p_medv":
        y_true_medv = np.expm1(y_true)
        y_pred_medv = np.expm1(y_pred)
    elif target_col == "medv":
        y_true_medv = y_true
        y_pred_medv = y_pred
    else:
        raise ValueError(f"Unsupported target_col='{target_col}'")

    mae_medv = float(mean_absolute_error(y_true_medv, y_pred_medv))
    rmse_medv = float(np.sqrt(mean_squared_error(y_true_medv, y_pred_medv)))

    return {
        "run_id": run_id,
        "model_name": model_name,
        "target_col": target_col,
        "mae_medv": mae_medv,
        "rmse_medv": rmse_medv,
    }


def compare_challenger_vs_champion(
    challenger_run_id: str,
    champion_run_id: str,
    duckdb_path: Optional[str] = None,
    min_relative_improvement: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare challenger against champion in ORIGINAL medv scale.

    Promotion rule:
        challenger wins only if

            challenger_rmse < champion_rmse * (1 - min_relative_improvement)

    Example:
        champion_rmse = 4.00
        min_relative_improvement = 0.05
        promotion_cutoff = 3.80

        challenger must be < 3.80 to qualify.
    """
    duckdb_path = duckdb_path or _duckdb_path()
    con = duckdb.connect(duckdb_path)
    try:
        df_challenger = _load_predictions_for_run(challenger_run_id, con)
        df_champion = _load_predictions_for_run(champion_run_id, con)

        challenger_metrics = _compute_medv_scale_metrics(df_challenger)
        champion_metrics = _compute_medv_scale_metrics(df_champion)

        challenger_rmse = challenger_metrics["rmse_medv"]
        champion_rmse = champion_metrics["rmse_medv"]

        promotion_cutoff = champion_rmse * (1 - min_relative_improvement)
        qualifies_for_promotion = challenger_rmse < promotion_cutoff

        winner = challenger_metrics if qualifies_for_promotion else champion_metrics

        return {
            "challenger": challenger_metrics,
            "champion": champion_metrics,
            "min_relative_improvement": float(min_relative_improvement),
            "promotion_cutoff_rmse": float(promotion_cutoff),
            "qualifies_for_promotion": bool(qualifies_for_promotion),
            "winner_run_id": winner["run_id"],
            "winner_model_name": winner["model_name"],
            "winner_target_col": winner["target_col"],
        }
    finally:
        con.close()


def compare_challenger_to_active_champion(
    challenger_run_id: str,
    duckdb_path: Optional[str] = None,
    min_relative_improvement: float = 0.05,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
      - loads active champion from ml.model_champion
      - compares challenger to active champion
    """
    duckdb_path = duckdb_path or _duckdb_path()
    champion = get_active_champion(duckdb_path)

    return compare_challenger_vs_champion(
        challenger_run_id=challenger_run_id,
        champion_run_id=champion.run_id,
        duckdb_path=duckdb_path,
        min_relative_improvement=min_relative_improvement,
    )


def choose_best_run(
    run_ids: List[str],
    duckdb_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rank candidate runs by medv-scale RMSE only.
    This selects the best challenger among several new candidates.
    It does NOT compare them to the active champion.
    """
    if len(run_ids) < 1:
        raise ValueError("run_ids must contain at least one run_id")

    duckdb_path = duckdb_path or _duckdb_path()
    con = duckdb.connect(duckdb_path)
    try:
        rows = []
        for run_id in run_ids:
            df = _load_predictions_for_run(run_id, con)
            rows.append(_compute_medv_scale_metrics(df))

        results_df = pd.DataFrame(rows).sort_values("rmse_medv", ascending=True).reset_index(drop=True)
        best = results_df.iloc[0].to_dict()

        return {
            "best_run_id": best["run_id"],
            "best_model_name": best["model_name"],
            "best_target_col": best["target_col"],
            "best_rmse_medv": float(best["rmse_medv"]),
            "best_mae_medv": float(best["mae_medv"]),
            "ranking": results_df.to_dict(orient="records"),
        }
    finally:
        con.close()