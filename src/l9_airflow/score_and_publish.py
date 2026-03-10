from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# NEW: scoring should use the explicitly promoted champion
from l9_airflow.model_lifecycle import get_active_champion


DUCKDB_PATH_DEFAULT = "/opt/airflow/data/duckdb/warehouse.duckdb"


def _duckdb_path() -> str:
    return os.environ.get("DUCKDB_PATH", DUCKDB_PATH_DEFAULT)


def _analytics_pg_url() -> str:
    host = os.environ.get("ANALYTICS_PG_HOST", "analytics-postgres")
    port = os.environ.get("ANALYTICS_PG_PORT", "5432")
    db = os.environ.get("ANALYTICS_PG_DB", "analytics")
    user = os.environ.get("ANALYTICS_PG_USER", "analytics")
    pwd = os.environ.get("ANALYTICS_PG_PASSWORD", "analytics")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def _utc_now_ts() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


GOLD_DDL = """
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.housing_predictions_daily (
  run_date DATE NOT NULL,
  house_id BIGINT NOT NULL,

  champion_run_id TEXT NOT NULL,
  model_name TEXT NOT NULL,
  target_col TEXT NOT NULL,

  y_pred_log1p DOUBLE,
  predicted_medv DOUBLE,

  created_at TIMESTAMP NOT NULL,

  PRIMARY KEY (run_date, house_id)
);
"""


@dataclass
class ScorePublishResult:
    start_dt: str
    end_dt: str
    champion_run_id: str
    model_name: str
    target_col: str
    scored_rows: int
    duckdb_upserted: int
    postgres_upserted: int


# -------------------------------------------------------------------
# OLD LOGIC (COMMENTED OUT)
# -------------------------------------------------------------------
# def _get_latest_champion(con: duckdb.DuckDBPyConnection) -> Tuple[str, str, str, str]:
#     """
#     Stage 1 rule (simple and deterministic):
#       - Champion = latest run for target_col='log1p_medv'
#       - If none exists, fallback to latest run for target_col='medv'
#     Returns: (run_id, model_name, target_col, model_path)
#     """
#     row = con.execute(
#         """
#         SELECT run_id, model_name, target_col, model_path
#         FROM ml.model_registry
#         WHERE target_col='log1p_medv'
#         ORDER BY created_at DESC
#         LIMIT 1
#         """
#     ).fetchone()
#
#     if row is None:
#         row = con.execute(
#             """
#             SELECT run_id, model_name, target_col, model_path
#             FROM ml.model_registry
#             WHERE target_col='medv'
#             ORDER BY created_at DESC
#             LIMIT 1
#             """
#         ).fetchone()
#
#     if row is None:
#         raise RuntimeError("No champion found in ml.model_registry (expected at least one trained model).")
#
#     run_id, model_name, target_col, model_path = row
#     return str(run_id), str(model_name), str(target_col), str(model_path)
# -------------------------------------------------------------------


def _load_artifact(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    if "pipeline" not in obj or "feature_cols" not in obj:
        raise ValueError(f"Unexpected model artifact format in {model_path}. Keys={list(obj.keys())}")
    return obj


def _ensure_scoring_view(con: duckdb.DuckDBPyConnection) -> None:
    """
    A stable scoring view = rows usable for inference.
    """
    con.execute("""
    CREATE SCHEMA IF NOT EXISTS silver;

    CREATE OR REPLACE VIEW silver.housing_score AS
    SELECT *
    FROM bronze.housing_features;
    """)


def score_and_publish_daily(
    start_dt: date,
    end_dt: date,
    duckdb_path: Optional[str] = None,
) -> ScorePublishResult:
    """
    End-to-end production scoring:

      DuckDB (silver score rows)
        -> load ACTIVE champion from ml.model_champion
        -> predict
        -> write DuckDB gold
        -> publish Postgres

    Important:
    This no longer uses "latest trained model".
    It uses the explicitly promoted active champion.
    """
    duckdb_path = duckdb_path or _duckdb_path()
    created_at = _utc_now_ts()

    con = duckdb.connect(duckdb_path)
    try:
        con.execute(GOLD_DDL)
        _ensure_scoring_view(con)

        # ------------------------------------------------------------
        # NEW CHAMPION LOGIC
        # ------------------------------------------------------------
        # OLD:
        # champion_run_id, model_name, target_col, model_path = _get_latest_champion(con)
        #
        # NEW:
        # Read the currently active champion from the lifecycle table.
        # This makes scoring governance-aware and prevents accidental
        # auto-promotion of the latest trained model.
        # ------------------------------------------------------------
        champion = get_active_champion(duckdb_path)

        champion_run_id = champion.run_id
        model_name = champion.model_name
        target_col = champion.target_col
        model_path = champion.model_path
        # ------------------------------------------------------------

        artifact = _load_artifact(model_path)
        pipe = artifact["pipeline"]
        feature_cols = artifact["feature_cols"]

        df = con.execute(
            f"""
            SELECT run_date, house_id, {", ".join(feature_cols)}
            FROM silver.housing_score
            WHERE run_date BETWEEN ? AND ?
            ORDER BY run_date, house_id
            """,
            [start_dt, end_dt],
        ).df()

        if df.empty:
            return ScorePublishResult(
                start_dt=str(start_dt),
                end_dt=str(end_dt),
                champion_run_id=champion_run_id,
                model_name=model_name,
                target_col=target_col,
                scored_rows=0,
                duckdb_upserted=0,
                postgres_upserted=0,
            )

        ids = df[["run_date", "house_id"]].copy()
        X = df[feature_cols].copy()

        y_pred = pipe.predict(X).astype(float)

        if target_col == "log1p_medv":
            y_pred_log1p = y_pred
            predicted_medv = np.expm1(y_pred_log1p)
        elif target_col == "medv":
            y_pred_log1p = np.array([None] * len(y_pred), dtype=object)
            predicted_medv = y_pred
        else:
            raise ValueError(f"Unsupported target_col={target_col}")

        out = pd.DataFrame(
            {
                "run_date": ids["run_date"],
                "house_id": ids["house_id"],
                "champion_run_id": champion_run_id,
                "model_name": model_name,
                "target_col": target_col,
                "y_pred_log1p": y_pred_log1p,
                "predicted_medv": predicted_medv.astype(float),
                "created_at": created_at,
            }
        )

        con.register("preds", out)
        con.execute(
            """
            INSERT INTO gold.housing_predictions_daily AS t
            SELECT * FROM preds
            ON CONFLICT (run_date, house_id) DO UPDATE SET
              champion_run_id = excluded.champion_run_id,
              model_name      = excluded.model_name,
              target_col      = excluded.target_col,
              y_pred_log1p    = excluded.y_pred_log1p,
              predicted_medv  = excluded.predicted_medv,
              created_at      = excluded.created_at
            """
        )
        con.unregister("preds")

        duckdb_upserted = int(out.shape[0])

    finally:
        con.close()

    pg = create_engine(_analytics_pg_url(), future=True)

    with pg.begin() as c:
        rows = out.to_dict(orient="records")
        c.execute(
            text(
                """
                INSERT INTO public.housing_predictions_daily (
                  run_date, house_id,
                  champion_run_id, model_name, target_col,
                  y_pred_log1p, predicted_medv,
                  created_at
                )
                VALUES (
                  :run_date, :house_id,
                  :champion_run_id, :model_name, :target_col,
                  :y_pred_log1p, :predicted_medv,
                  :created_at
                )
                ON CONFLICT (run_date, house_id) DO UPDATE SET
                  champion_run_id = EXCLUDED.champion_run_id,
                  model_name      = EXCLUDED.model_name,
                  target_col      = EXCLUDED.target_col,
                  y_pred_log1p    = EXCLUDED.y_pred_log1p,
                  predicted_medv  = EXCLUDED.predicted_medv,
                  created_at      = EXCLUDED.created_at
                """
            ),
            rows,
        )

    return ScorePublishResult(
        start_dt=str(start_dt),
        end_dt=str(end_dt),
        champion_run_id=champion_run_id,
        model_name=model_name,
        target_col=target_col,
        scored_rows=int(out.shape[0]),
        duckdb_upserted=duckdb_upserted,
        postgres_upserted=int(out.shape[0]),
    )