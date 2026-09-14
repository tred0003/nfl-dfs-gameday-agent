# Slim image for the scheduled gameday report. The report path needs only
# pandas/numpy/requests -- xgboost, sklearn and pulp are training/optimizing
# dependencies and have no business in a container that posts to Discord.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY deploy/requirements-report.txt .
RUN pip install --no-cache-dir -r requirements-report.txt

# only the modules the report actually imports
COPY src/dk_client.py src/predict_slate.py src/watch_slate.py \
     src/gameday_report.py src/scoring.py ./src/

# Cloud Run jobs run to completion and exit -- no server, no port.
ENTRYPOINT ["python", "src/gameday_report.py"]
