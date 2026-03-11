library(DBI)
library(duckdb)
library(dplyr)
library(jsonlite)
library(uuid)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

duckdb_path <- "/project/data/duckdb/warehouse.duckdb"

train_view <- "silver.housing_train"
valid_view <- "silver.housing_valid"

target_col <- "medv"

feature_cols <- c(
  "crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat",
  "rm_sq","crime_tax_ratio","lstat_ptratio_interact","dis_rm_interact","tax_norm"
)

model_dir <- "/project/R/models"
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

run_id <- UUIDgenerate()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

con <- dbConnect(duckdb(), duckdb_path)

train_df <- dbGetQuery(
  con,
  paste0(
    "SELECT run_date, house_id, ",
    paste(feature_cols, collapse = ", "),
    ", medv FROM ", train_view
  )
)

valid_df <- dbGetQuery(
  con,
  paste0(
    "SELECT run_date, house_id, ",
    paste(feature_cols, collapse = ", "),
    ", medv FROM ", valid_view
  )
)

dbDisconnect(con, shutdown = TRUE)

train_df <- train_df %>% tidyr::drop_na()
valid_df <- valid_df %>% tidyr::drop_na()

# --------------------------------------------------
# TRAIN LM
# --------------------------------------------------

formula_txt <- paste(target_col, "~", paste(feature_cols, collapse = " + "))
model_formula <- as.formula(formula_txt)

lm_fit <- lm(model_formula, data = train_df)

# --------------------------------------------------
# VALIDATE
# --------------------------------------------------

preds <- predict(lm_fit, newdata = valid_df)

mae <- mean(abs(valid_df[[target_col]] - preds))
rmse <- sqrt(mean((valid_df[[target_col]] - preds)^2))
r2 <- 1 - sum((valid_df[[target_col]] - preds)^2) / sum((valid_df[[target_col]] - mean(valid_df[[target_col]]))^2)

metrics <- list(
  mae = unname(mae),
  rmse = unname(rmse),
  r2 = unname(r2)
)

# --------------------------------------------------
# SAVE MODEL + METRICS
# --------------------------------------------------

model_path <- file.path(model_dir, paste0("lm_housing_medv__", run_id, ".rds"))
metrics_path <- file.path(model_dir, paste0("lm_housing_metrics__", run_id, ".json"))

saveRDS(lm_fit, model_path)
write_json(metrics, metrics_path, pretty = TRUE, auto_unbox = TRUE)

cat("LM model saved:", model_path, "\n")
cat("Metrics saved:", metrics_path, "\n")
cat("Metrics:\n")
print(metrics)