FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Add API test script
COPY fmp_api_test.py /app/fmp_api_test.py

# Modificar el punto de entrada para incluir pruebas de API
COPY init_ingestion.sh /app/init_ingestion.sh
RUN chmod +x /app/init_ingestion.sh

# Ejecutar pruebas de API antes de iniciar el servicio
ENTRYPOINT ["/bin/bash", "-c", "python /app/fmp_api_test.py && /app/init_ingestion.sh"]