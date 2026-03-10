# MLOps Housing Pipeline

## Part 1. What this infrastructure does

This project is a lightweight analytical and MLOps platform built around DuckDB, Postgres, and Apache Airflow.

Its purpose is to automate the full path from raw data to model predictions, and to do so in a way that is understandable, testable, and ready to be extended.

### Business functionality

For a non-technical stakeholder, the system currently does four main things.

#### 1. It collects and structures housing data every day

Every day, the pipeline loads housing records, prepares analytical features, and stores them in a structured warehouse.

#### 2. It produces daily model predictions automatically

A production scoring pipeline uses the active champion model to calculate housing price predictions for the newest available data.

These predictions are written into:

- DuckDB for local analytical use
- Postgres for concurrent access and BI consumption
- CSV / Parquet exports for practical business use in Excel and similar tools

#### 3. It manages the model lifecycle

The system does not only score data. It also retrains a challenger model on a rolling historical window and compares it to the current champion.

A new model is promoted only if it is materially better than the active champion according to a strict improvement rule.

This means the infrastructure supports:

- model retraining
- model comparison
- champion/challenger governance
- controlled model replacement

#### 4. It creates simple delivery paths for business users

Because many business teams work in Excel rather than directly in databases, the platform also exports the prediction tables into portable files.

This gives a practical bridge between analytical infrastructure and day-to-day operational consumption.

### Current value of the system

At the current stage, the infrastructure already provides:

- automated daily data ingestion
- automated daily scoring
- automated export of predictions
- model registry and metrics tracking
- active champion tracking
- challenger retraining and evaluation
- controlled future model replacement

In short, this is no longer just a model script. It is a functioning analytical and MLOps workflow.

---

## Part 2. How the infrastructure works

### Main components

#### DuckDB

DuckDB is the main local analytical warehouse.

It stores data in layered schemas:

- bronze — ingested raw data /
- silver structured, feature-level data for warehouse use
- gold — production-ready prediction outputs
- ml — model registry, metrics, predictions, champion lifecycle tables

DuckDB is used as the central analytical computation layer.

#### Postgres

Postgres acts as a serving and concurrency layer.

It stores:

- daily raw housing data
- feature tables
- production prediction tables

It is useful when multiple consumers need access to the same prediction outputs.

#### Apache Airflow

Airflow orchestrates the workflows.

It schedules and runs the DAGs that:

- ingest data
- materialize features
- sync data to DuckDB
- score daily predictions
- export prediction files
- retrain challenger models
- evaluate challenger vs champion
- promote a new champion if qualified

#### Export layer

The system exports key prediction outputs into:

- CSV
- Parquet

This enables direct consumption in:

- Excel
- Power Query
- future BI tooling

---

## Data flow

### 1. Daily ingestion pipeline

DAG: `housing_daily_ingest`

This pipeline:

1. fills missing raw housing dates
2. creates feature-engineered data
3. initializes DuckDB objects if needed
4. syncs the prepared features into DuckDB

Result:

- fresh daily data is available in Postgres and DuckDB

### 2. Daily scoring pipeline

DAG: `daily_housing_model_scoring`

This pipeline:

1. waits for daily ingestion to finish
2. reads the active champion model from the lifecycle table
3. scores the newest daily data
4. writes predictions into DuckDB Gold
5. publishes predictions into Postgres
6. exports the current prediction table to CSV and Parquet

Result:

- daily predictions are available for analytics and business consumption

### 3. Model lifecycle pipeline

DAG: `model_training_pipeline`

This pipeline:

1. computes a rolling time window from Airflow logical date
2. trains a challenger model on a 180-day training window
3. validates it on the following 7-day validation window
4. writes model artifact, metrics, registry entry, and validation predictions
5. compares challenger against the current active champion in original target scale
6. promotes challenger only if it exceeds the required relative improvement threshold

Result:

- the current champion is monitored and can be replaced in a controlled way

---

## Model lifecycle logic

### Current champion governance

The active production model is stored in:

- `ml.model_champion`

This table defines which model is officially active.

The scoring pipeline does not automatically use the latest trained model.
It uses only the model that is officially promoted as champion.

This is important because it prevents uncontrolled promotion of newly trained candidates.

### Challenger training

A challenger model is trained using the approved recipe:

- target: `log1p_medv`
- same feature logic as the approved baseline
- Ridge regression with standardized features

### Promotion rule

A challenger is promoted only if it materially outperforms the champion.

Current rule:

- the challenger must improve RMSE by at least 5% relative to the current champion

This makes promotion conservative and avoids replacing the champion because of random small fluctuations.

### Promotion outcome

If challenger qualifies:

- old champion becomes inactive
- its `active_to` timestamp is closed
- new challenger is inserted as the active champion

If challenger does not qualify:

- current champion remains active
- challenger is still preserved in registry, metrics, and predictions for analysis

---

## Current implemented tables

### DuckDB ML layer

- `ml.model_registry` — metadata for trained models
- `ml.metrics` — evaluation metrics per model run
- `ml.predictions` — validation predictions per model run
- `ml.model_champion` — active champion lifecycle table

### DuckDB Gold layer

- `gold.housing_predictions_daily` — production daily predictions

### Postgres serving layer

- `public.daily_housing_raw`
- `public.housing_features`
- `public.housing_predictions_daily`

---

## Practical operating model

### For technical users

The platform provides:

- modular Python training and scoring logic
- orchestrated pipelines in Airflow
- lifecycle tracking of models
- export layer for downstream integration

### For non-technical users

The platform provides:

- daily updated prediction outputs
- stable delivery through CSV / Parquet
- ability to consume results in Excel
- controlled model updates rather than ad-hoc model changes

---

## Current limitations

At the current stage, the system is intentionally practical and lightweight.

Important current limitations:

- scoring output tables are operational current-state tables, not full immutable prediction history
- online inference inside the database is not yet implemented in Python
- Power BI / Windows environment deployment is planned as a later step
- R-based model translation and automation are planned as the next extension

These are known and intentional next milestones rather than architectural failures.

---

## Planned next milestones

### 1. R-based pipeline extension

Planned extension:

- containerized R execution
- orchestration of R scripts from Airflow
- support for legacy R workflows
- use of `recipes`, `orbital`, and `tidypredict`

Expected value:

- automation of existing R logic
- SQL-translatable prediction logic for simpler models
- potential online inference in the database for GLM-like models

### 2. Historical scoring

Planned extension:

- immutable historical prediction records
- proper point-in-time model scoring of past periods

### 3. Windows VM deployment

Planned extension:

- run the infrastructure in a Windows-oriented VM environment
- validate portability for infrastructure deployment
- support future BI tooling in a more realistic enterprise environment
