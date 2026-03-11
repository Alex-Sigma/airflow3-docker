FROM rocker/r-ver:4.5.2

WORKDIR /project

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    xz-utils \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c( \
    'DBI', \
    'dplyr', \
    'readr', \
    'recipes', \
    'parsnip', \
    'workflows', \
    'tidypredict', \
    'orbital', \
    'uuid', \
    'tidyr', \
    'glmnet', \
    'duckdb', \
    'jsonlite', \
    ,'tidymodels', \
    ,'dbplyr'\
    ), repos='https://cloud.r-project.org')"

CMD ["tail", "-f", "/dev/null"]