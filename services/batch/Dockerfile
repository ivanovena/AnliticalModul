FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para numpy, pandas y matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Flask y dependencias para API y Dashboard si no están en requirements.txt
RUN pip install --no-cache-dir flask prometheus_client

# Copiar el código fuente
COPY . .

# Crear directorios necesarios para plantillas y modelos
RUN mkdir -p /app/templates /models /app/logs

# Copiar script de inicialización
COPY init_batch.sh .
RUN chmod +x /app/init_batch.sh

# Exponer puertos para API y Prometheus
EXPOSE 8003 8080 8000

# Definir punto de entrada
ENTRYPOINT ["/app/init_batch.sh"]
