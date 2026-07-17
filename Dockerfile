# Use standard python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies including git (required by GitPython)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements file first to leverage docker caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to the container
COPY app/ ./app/
COPY benchmark/ ./benchmark/
COPY streamlit_app.py .
COPY README.md .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
