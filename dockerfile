# Use the official PyTorch runtime image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set environment variables to reduce Python overhead
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CC=gcc \
    CXX=g++

# Set working directory
WORKDIR /workspace

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies and clean up to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy application code last to maximize caching
COPY train_foundation.py DNAMoleculeModel.py DNAMoleculeModelFoundation.py ./

# Set the default command
CMD ["python", "train_foundation.py"]
