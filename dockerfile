FROM python:3.11-alpine

WORKDIR /app

# Install system dependencies for Alpine
RUN apk add --no-cache \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm \
    libxext \
    libxrender \
    ffmpeg \
    gcc \
    g++ \
    musl-dev \
    linux-headers

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run setup
RUN chmod +x setup.sh && ./setup.sh

# Environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

EXPOSE 8501

CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true
