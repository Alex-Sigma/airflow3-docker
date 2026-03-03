CREATE OR REPLACE VIEW public.v_housing_target_today AS
SELECT
  run_date,
  house_id,
  medv
FROM public.housing_features
WHERE run_date = CURRENT_DATE;