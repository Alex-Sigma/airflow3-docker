FROM rocker/r-ver:4.5.2

WORKDIR /project

# Install useful system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages
RUN R -e "install.packages(c( \
    'DBI', \
    'duckdb', \
    'dplyr', \
    'readr', \
    'recipes', \
    'parsnip', \
    'workflows', \
    'tidypredict', \
    'orbital' \
    ), repos='https://cloud.r-project.org')"

# Container will stay alive for Airflow execution
CMD ["tail", "-f", "/dev/null"]