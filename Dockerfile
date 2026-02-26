FROM apache/airflow:3.1.7

# Install additional Python dependencies
# (run as airflow user; installing as root may fail)
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt