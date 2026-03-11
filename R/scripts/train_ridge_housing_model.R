library(DBI)
library(duckdb)
library(dplyr)
library(tidymodels)
library(readr)
library(jsonlite)

# ----------------------------------------
# CONFIG
# ----------------------------------------

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

run_id <- uuid::UUIDgenerate()

# ----------------------------------------
# LOAD DATA
# ----------------------------------------

con <- dbConnect(duckdb(), duckdb_path)

train_df <- dbGetQuery(con, paste0(
  "SELECT run_date, house_id, ",
  paste(feature_cols, collapse=", "),
  ", medv FROM ", train_view
))

valid_df <- dbGetQuery(con, paste0(
  "SELECT run_date, house_id, ",
  paste(feature_cols, collapse=", "),
  ", medv FROM ", valid_view
))

dbDisconnect(con)

train_df <- train_df %>% tidyr::drop_na()
valid_df <- valid_df %>% tidyr::drop_na()

# ----------------------------------------
# RECIPE (StandardScaler equivalent)
# ----------------------------------------

recipe_obj <- recipe(
  medv ~ ., 
  data = train_df %>% select(all_of(c(feature_cols, target_col)))
) %>%
  step_normalize(all_predictors())

# ----------------------------------------
# MODEL (Ridge)
# ----------------------------------------

ridge_model <- linear_reg(
  penalty = 1.0,
  mixture = 0
) %>%
set_engine("glmnet", standardize = FALSE)
 # set_engine("glmnet")

# ----------------------------------------
# WORKFLOW
# ----------------------------------------

wf <- workflow() %>%
  add_model(ridge_model) %>%
  add_recipe(recipe_obj)

fit_model <- fit(wf, data = train_df)

# ----------------------------------------
# PREDICTIONS
# ----------------------------------------

preds <- predict(fit_model, valid_df) %>%
  bind_cols(valid_df)

# ----------------------------------------
# METRICS
# ----------------------------------------

mae <- mae_vec(preds$medv, preds$.pred)
rmse <- rmse_vec(preds$medv, preds$.pred)
r2 <- rsq_vec(preds$medv, preds$.pred)

metrics <- list(
  mae = mae,
  rmse = rmse,
  r2 = r2
)

# ----------------------------------------
# SAVE MODEL
# ----------------------------------------

model_path <- paste0(
  model_dir,
  "/ridge_housing_medv__",
  run_id,
  ".rds"
)

saveRDS(fit_model, model_path)

# ----------------------------------------
# SAVE METRICS
# ----------------------------------------

metrics_path <- paste0(
  model_dir,
  "/ridge_housing_metrics__",
  run_id,
  ".json"
)

write_json(metrics, metrics_path, pretty = TRUE)

cat("Model saved:", model_path, "\n")
cat("Metrics:", "\n")
print(metrics)