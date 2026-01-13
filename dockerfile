# Use a specific version for stability
FROM python:3.13-slim-bookworm

# Prevent python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install ONLY necessary system dependencies (if any)
# If you don't need specific C-libraries, you can remove this block entirely
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

EXPOSE 8501

# Using the full path or module execution is often safer
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=127.0.0.1"]