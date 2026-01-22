# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/build/.venv

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a local folder
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    uv venv $UV_PROJECT_ENVIRONMENT && \
    uv pip install --no-cache \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt pyyaml

# --- STAGE 2: Final Runtime ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1 
ENV PATH="/app/.venv/bin:$PATH"

# Install necessary system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download Thai OCR Data
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata/ && \
    wget -O /usr/share/tesseract-ocr/5/tessdata/tha.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/tha.traineddata

WORKDIR /app

# Copy installed packages and code from the builder
COPY --from=builder /build/.venv /app/.venv
COPY . .

CMD ["python", "pipeline_test.py"]