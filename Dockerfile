# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a local folder
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --target=/build/deps \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt pyyaml

# --- STAGE 2: Final Runtime ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app/deps"

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
COPY --from=builder /build/deps /app/deps
COPY . .

CMD ["python", "pipeline_test.py"]