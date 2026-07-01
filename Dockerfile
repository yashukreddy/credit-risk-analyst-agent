# Use the official Python slim image for a smaller footprint
FROM python:3.11-slim

# Set environment variables to optimize Python runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# Install system dependencies required for building some Python packages (like PyMuPDF)
# We clean up the apt cache immediately to save space
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# CRITICAL SIZE OPTIMIZATION: 
# sentence-transformers installs PyTorch. By default, pip pulls the massive CUDA (GPU) version.
# We explicitly point pip to the CPU-only PyTorch index first to save ~2GB of space,
# then we install the rest of the requirements without caching the downloaded files.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application
# Binds to 0.0.0.0 so it can be accessed from outside the container
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
