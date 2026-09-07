FROM python:3.11-slim

# tesseract-ocr for local image OCR (multimodal/ocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# ✅ FIX: Install CPU-only PyTorch BEFORE installing requirements
# This prevents sentence-transformers from pulling the full GPU version
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
# sentence-transformers will see that torch is already installed and skip it
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]