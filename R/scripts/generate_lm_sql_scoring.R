library(DBI)
library(duckdb)
library(jsonlite)
library(glue)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

duckdb_path <- "/project/data/duckdb/warehouse.duckdb"
model_dir <- "/project/R/models"

sql_output_path <- "/project/R/models/lm_housing_sql_expression.sql"
meta_output_path <- "/project/R/models/lm_housing_sql_metadata.json"

source_relation <- "silver.housing_score"
target_view <- "ml.lm_housing_predictions_rt"

model_files <- list.files(
  model_dir,
  pattern = "^lm_housing_medv__.*\\.rds$",
  full.names = TRUE
)

if (length(model_files) == 0) {
  stop("No lm_housing_medv .rds models found in /project/R/models")
}

model_info <- file.info(model_files)
latest_idx <- which.max(model_info$mtime)
model_path <- model_files[latest_idx]

cat("Using model:", model_path, "\n")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

lm_fit <- readRDS(model_path)

coefs <- coef(lm_fit)

if (is.null(coefs) || length(coefs) == 0) {
  stop("No coefficients found in lm model")
}

# --------------------------------------------------
# BUILD SQL EXPRESSION
# --------------------------------------------------

intercept <- unname(coefs["(Intercept)"])
feature_coefs <- coefs[names(coefs) != "(Intercept)"]

# Drop undefined coefficients caused by perfect collinearity
feature_coefs <- feature_coefs[!is.na(feature_coefs)]

sql_terms <- c(sprintf("%.15f", intercept))

for (nm in names(feature_coefs)) {
  coef_val <- unname(feature_coefs[[nm]])
  sql_terms <- c(sql_terms, sprintf("(%.15f * %s)", coef_val, nm))
}

sql_expr <- paste(sql_terms, collapse = " + ")

cat("\nGenerated SQL expression:\n")
cat(sql_expr, "\n")

writeLines(sql_expr, sql_output_path)

meta <- list(
  model_path = model_path,
  source_relation = source_relation,
  target_view = target_view,
  sql_output_path = sql_output_path,
  coefficients = as.list(coefs)
)

write_json(meta, meta_output_path, pretty = TRUE, auto_unbox = TRUE)

cat("\nSQL expression written to:", sql_output_path, "\n")
cat("Metadata written to:", meta_output_path, "\n")

# --------------------------------------------------
# DEPLOY REAL-TIME VIEW IN DUCKDB
# --------------------------------------------------

con <- dbConnect(duckdb(), duckdb_path)

create_view_sql <- glue("
CREATE SCHEMA IF NOT EXISTS ml;

CREATE OR REPLACE VIEW {target_view} AS
SELECT
    run_date,
    house_id,
    {sql_expr} AS predicted_medv_lm_sql
FROM {source_relation};
")

cat("\nDeploying DuckDB view:\n")
cat(create_view_sql, "\n")

dbExecute(con, create_view_sql)

cat("\nView deployed successfully:", target_view, "\n")

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