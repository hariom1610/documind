# ============================================================
# Document Analysis API — Dockerfile
# ============================================================
FROM python:3.11-slim

# Install system dependencies:
# - tesseract-ocr: OCR engine for image text extraction
# - tesseract-ocr-eng: English language pack for Tesseract
# - libgl1, libglib2.0-0: Required by PyMuPDF/OpenCV
# - libsm6, libxext6, libxrender-dev: Required for image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
