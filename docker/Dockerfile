FROM python:3.10-slim

# Create non-root user
RUN useradd --create-home appuser
WORKDIR /workspace

# Install OS deps for PIL, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    gcc \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if present and install
COPY Models/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Fallback common packages
RUN pip install --no-cache-dir pillow numpy

# Set Python path for imports
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Set user and entrypoint
USER appuser
ENTRYPOINT ["python3", "Models/inference/cli.py"]
