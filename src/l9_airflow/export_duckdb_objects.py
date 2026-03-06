from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import duckdb


def _duckdb_path() -> str:
    return os.environ.get("DUCKDB_PATH", "/opt/airflow/data/duckdb/warehouse.duckdb")


def export_duckdb_object(
    object_name: str,
    export_format: Literal["csv", "parquet"] = "csv",
    base_export_dir: str = "/opt/airflow/data/exports",
    filename: Optional[str] = None,
) -> dict:
    """
    Export a DuckDB table or view like:
      gold.housing_predictions_daily
      silver.housing_train
      bronze.housing_features

    into:
      /opt/airflow/data/exports/<schema>/<object>.<ext>
    """
    if "." not in object_name:
        raise ValueError("object_name must be schema.object, e.g. gold.housing_predictions_daily")

    schema_name, table_name = object_name.split(".", 1)

    export_dir = Path(base_export_dir) / schema_name
    export_dir.mkdir(parents=True, exist_ok=True)

    ext = "csv" if export_format == "csv" else "parquet"
    filename = filename or f"{table_name}.{ext}"
    export_path = export_dir / filename

    con = duckdb.connect(_duckdb_path())
    try:
        if export_format == "csv":
            con.execute(
                f"""
                COPY (
                  SELECT *
                  FROM {object_name}
                )
                TO '{export_path.as_posix()}'
                WITH (HEADER, DELIMITER ',');
                """
            )
        elif export_format == "parquet":
            con.execute(
                f"""
                COPY (
                  SELECT *
                  FROM {object_name}
                )
                TO '{export_path.as_posix()}'
                (FORMAT PARQUET);
                """
            )
        else:
            raise ValueError("export_format must be 'csv' or 'parquet'")

        return {
            "status": "ok",
            "object_name": object_name,
            "export_format": export_format,
            "export_path": export_path.as_posix(),
        }
    finally:
        con.close()