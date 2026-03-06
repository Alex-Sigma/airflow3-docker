#!/bin/bash
set -e

echo "Restarting Airflow scheduler..."
docker compose restart airflow-scheduler

echo "Restarting Airflow API server..."
docker compose restart airflow-apiserver

echo "Waiting for Airflow to reload DAGs..."
sleep 5

echo "Relevant DAGs:"
docker compose exec airflow-scheduler airflow dags list | grep -E "housing|scoring|export" || true