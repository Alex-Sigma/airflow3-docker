#!/bin/bash

set -e

echo "Stopping containers..."
docker compose down

echo "Removing old custom image (if exists)..."
docker image rm airflow3-custom:3.1.7 2>/dev/null || true

echo "Building custom image..."
docker compose build --no-cache airflow-apiserver

echo "Initializing Airflow..."
docker compose up airflow-init

echo "Starting Airflow..."
docker compose up -d

echo "Done. Airflow is running at http://localhost:8080"