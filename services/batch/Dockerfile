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

# Copiar el código fuente
COPY . .

# Copiar script de inicialización
COPY init_batch.sh .
RUN chmod +x /app/init_batch.sh

# Exponer puerto para el API de health check si es necesario
EXPOSE 8003

# Definir punto de entrada
ENTRYPOINT ["/app/init_batch.sh"]
