library(DBI)
library(duckdb)
library(tidymodels)
library(orbital)
library(jsonlite)
library(glue)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

duckdb_path <- "/project/data/duckdb/warehouse.duckdb"
model_dir <- "/project/R/models"

sql_output_path <- "/project/R/models/orbital_ridge_sql_expression.sql"
meta_output_path <- "/project/R/models/orbital_ridge_sql_metadata.json"

source_relation <- "silver.housing_score"
target_view <- "ml.ridge_housing_predictions_rt"

model_files <- list.files(
  model_dir,
  pattern = "^ridge_housing_medv__.*\\.rds$",
  full.names = TRUE
)

if (length(model_files) == 0) {
  stop("No ridge_housing_medv .rds models found in /project/R/models")
}

model_info <- file.info(model_files)
latest_idx <- which.max(model_info$mtime)
model_path <- model_files[latest_idx]

cat("Using model:", model_path, "\n")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

wf_fit <- readRDS(model_path)
orbital_obj <- orbital(wf_fit)

cat("Orbital object created successfully\n")

# --------------------------------------------------
# CONNECT TO DUCKDB
# --------------------------------------------------

con <- dbConnect(duckdb(), duckdb_path)

# --------------------------------------------------
# GENERATE SQL EXPRESSION DIRECTLY
# --------------------------------------------------

sql_parts <- orbital_sql(orbital_obj, con)

cat("\nNames returned by orbital_sql():\n")
print(names(sql_parts))

if (!".pred" %in% names(sql_parts)) {
  stop("orbital_sql() did not return a '.pred' expression")
}

ridge_pred_sql <- sql_parts[".pred"][[1]]

cat("\nGenerated SQL expression for .pred:\n")
cat(ridge_pred_sql, "\n")

writeLines(ridge_pred_sql, sql_output_path)

meta <- list(
  model_path = model_path,
  source_relation = source_relation,
  target_view = target_view,
  sql_output_path = sql_output_path
)

write_json(meta, meta_output_path, pretty = TRUE, auto_unbox = TRUE)

cat("\nSQL expression written to:", sql_output_path, "\n")
cat("Metadata written to:", meta_output_path, "\n")

# --------------------------------------------------
# DEPLOY REAL-TIME VIEW IN DUCKDB
# --------------------------------------------------

create_view_sql <- glue("
CREATE SCHEMA IF NOT EXISTS ml;

CREATE OR REPLACE VIEW {target_view} AS
SELECT
    run_date,
    house_id,
    {ridge_pred_sql} AS predicted_medv_r_sql
FROM {source_relation};
")

cat("\nDeploying DuckDB view:\n")
cat(create_view_sql, "\n")

dbExecute(con, create_view_sql)

cat("\nView deployed successfully:", target_view, "\n")

# --------------------------------------------------
# VALIDATE PREVIEW
# --------------------------------------------------

preview_df <- dbGetQuery(
  con,
  glue("
    SELECT *
    FROM {target_view}
    ORDER BY run_date DESC, house_id
    LIMIT 10
  ")
)

cat("\nPreview of deployed view:\n")
print(preview_df)

dbDisconnect(con, shutdown = TRUE)