FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy app code
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the server
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]