-- Prevent duplicates per day per house (enables safe backfill reruns)
ALTER TABLE public.daily_housing_raw
ADD CONSTRAINT uq_daily_housing_raw_run_date_house_id
UNIQUE (run_date, house_id);