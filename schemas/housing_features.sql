CREATE TABLE public.housing_features (
    run_date      DATE      NOT NULL,
    house_id      INTEGER   NOT NULL,
    -- raw features
    crim          DOUBLE PRECISION,
    zn            DOUBLE PRECISION,
    indus         DOUBLE PRECISION,
    chas          INTEGER,
    nox           DOUBLE PRECISION,
    rm            DOUBLE PRECISION,
    age           DOUBLE PRECISION,
    dis           DOUBLE PRECISION,
    rad           INTEGER,
    tax           DOUBLE PRECISION,
    ptratio       DOUBLE PRECISION,
    b             DOUBLE PRECISION,
    lstat         DOUBLE PRECISION,

    -- engineered
    rm_sq                DOUBLE PRECISION,
    crime_tax_ratio      DOUBLE PRECISION,
    lstat_ptratio_interact DOUBLE PRECISION,
    dis_rm_interact      DOUBLE PRECISION,
    tax_norm             DOUBLE PRECISION,

    -- target
    medv          DOUBLE PRECISION,

    PRIMARY KEY (run_date, house_id)
);