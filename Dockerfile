# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for some Python packages
# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Streamlit uses 8501 by default, but Render provides a dynamic $PORT
EXPOSE 8501

# Run the app and bind to the port Render provides
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]