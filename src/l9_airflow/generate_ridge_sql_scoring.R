library(tidymodels)
library(tidypredict)
library(jsonlite)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

model_path <- "/project/R/models/ridge_housing_medv__f4032ec3-2876-4845-bb8f-1735f5f9fc97.rds"
sql_output_path <- "/project/R/models/ridge_housing_sql_expression.sql"
meta_output_path <- "/project/R/models/ridge_housing_sql_metadata.json"

feature_cols <- c(
  "crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat",
  "rm_sq","crime_tax_ratio","lstat_ptratio_interact","dis_rm_interact","tax_norm"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

fit_model <- readRDS(model_path)

# --------------------------------------------------
# EXTRACT PARSNIP FIT
# --------------------------------------------------
# workflow -> parsnip fit -> glmnet model

parsnip_fit <- workflows::extract_fit_parsnip(fit_model)

# --------------------------------------------------
# GENERATE SQL EXPRESSION
# --------------------------------------------------
# This generates a SQL expression for the prediction itself.
# We use dbplyr-compatible SQL fragments.

sql_expr <- tidypredict_sql(parsnip_fit$fit)

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

writeLines(sql_expr, sql_output_path)

meta <- list(
  model_path = model_path,
  sql_output_path = sql_output_path,
  feature_cols = feature_cols
)

write_json(meta, meta_output_path, pretty = TRUE, auto_unbox = TRUE)

cat("SQL scoring expression written to:", sql_output_path, "\n")
cat("Metadata written to:", meta_output_path, "\n")
cat("Generated SQL expression:\n")
cat(sql_expr, "\n")