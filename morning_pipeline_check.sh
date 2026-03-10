#!/bin/bash
set -e

echo "========================================"
echo "1) AIRFLOW DAG RUNS"
echo "========================================"
docker compose exec airflow-scheduler airflow dags list-runs housing_daily_ingest | head -n 10 || true
echo
docker compose exec airflow-scheduler airflow dags list-runs daily_housing_model_scoring | head -n 10 || trueompose exec airflow-scheduler airflow dags list-runs -d daily_housing_model_scoring | head -n 10 || true

echo
echo "========================================"
echo "2) DUCKDB DATE COVERAGE"
echo "========================================"
docker compose exec airflow-worker python -c "
import os, duckdb
con = duckdb.connect(os.environ['DUCKDB_PATH'])

print('BRONZE:')
print(con.execute(\"\"\"
SELECT run_date, COUNT(*) AS rows_per_day
FROM bronze.housing_features
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 5
\"\"\").fetchall())

print('\\nSILVER:')
print(con.execute(\"\"\"
SELECT run_date, COUNT(*) AS rows_per_day
FROM silver.housing_score
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 5
\"\"\").fetchall())

print('\\nGOLD:')
print(con.execute(\"\"\"
SELECT run_date, COUNT(*) AS preds_per_day
FROM gold.housing_predictions_daily
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 5
\"\"\").fetchall())

con.close()
"

echo
echo "========================================"
echo "3) POSTGRES DATE COVERAGE"
echo "========================================"
docker compose exec analytics-postgres psql -U analytics -d analytics -c "
SELECT 'daily_housing_raw' AS table_name, run_date, COUNT(*) AS rows_per_day
FROM public.daily_housing_raw
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 5;

SELECT 'housing_features' AS table_name, run_date, COUNT(*) AS rows_per_day
FROM public.housing_features
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 5;

SELECT 'housing_predictions_daily' AS table_name, run_date, COUNT(*) AS preds_per_day
FROM public.housing_predictions_daily
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 5;
"

echo
echo "========================================"
echo "4) CSV EXPORT FILE CHECK"
echo "========================================"
ls -lah data/exports/gold/housing_predictions_daily.csv data/exports/gold/housing_predictions_daily.parquet 2>/dev/null || true

echo
echo "CSV HEAD:"
head -n 5 data/exports/gold/housing_predictions_daily.csv 2>/dev/null || true

echo
echo "========================================"
echo "5) AIRFLOW IMPORT ERRORS"
echo "========================================"
docker compose exec airflow-scheduler airflow dags list-import-errors || true

echo
echo "========================================"
echo "DONE"
echo "========================================"