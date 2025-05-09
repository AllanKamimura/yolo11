# Use the full official Python 3.10 image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the app folder
COPY export/ export/

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libusb-1.0-0 \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# Install Python dependencies from ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r export/requirements.txt

# Default command
CMD ["python3"]
