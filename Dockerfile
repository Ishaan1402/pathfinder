FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose broker port
EXPOSE 8000

# Run uvicorn on 0.0.0.0
CMD ["python", "broker.py", "--daemon", "--host", "0.0.0.0", "--port", "8000"]
