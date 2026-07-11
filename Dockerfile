# Stage 1: Build & Python Dependency Wheel Cache
FROM python:3.10-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Clean Executable Runtime
FROM python:3.10-slim AS runner
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY ./src ./src
COPY ./src/mlruns ./mlruns

ENV PATH=/root/.local/bin:$PATH
ENV MODEL_PATH=/app/mlruns/0/

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]