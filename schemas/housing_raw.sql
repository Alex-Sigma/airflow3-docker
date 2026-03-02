CREATE TABLE IF NOT EXISTS daily_housing_raw (
    id SERIAL PRIMARY KEY,
    crim FLOAT,
    zn FLOAT,
    indus FLOAT,
    chas INT,
    nox FLOAT,
    rm FLOAT,
    age FLOAT,
    dis FLOAT,
    rad INT,
    tax FLOAT,
    ptratio FLOAT,
    b FLOAT,
    lstat FLOAT,
    medv FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);