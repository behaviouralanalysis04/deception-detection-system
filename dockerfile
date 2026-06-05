FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make setup script executable and run it
RUN chmod +x setup.sh && ./setup.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV OPENCV_IO_MAX_TRIES=1

# Expose port
EXPOSE 8501

# Run the application
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true
