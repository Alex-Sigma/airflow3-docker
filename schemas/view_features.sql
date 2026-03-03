CREATE OR REPLACE VIEW public.v_housing_features_today AS
SELECT
  run_date,
  house_id,
  crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat,
  rm_sq, crime_tax_ratio, lstat_ptratio_interact, dis_rm_interact, tax_norm
FROM public.housing_features
WHERE run_date = CURRENT_DATE;