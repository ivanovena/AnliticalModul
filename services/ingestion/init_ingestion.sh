#!/bin/sh
set -e

echo "Starting ingestion service initialization..."

# Verificar variables de entorno críticas
echo "Verificando variables de entorno..."
echo "KAFKA_BROKER: $KAFKA_BROKER"
echo "INGESTION_TOPIC: $INGESTION_TOPIC"
echo "FMP_BASE_URL: $FMP_BASE_URL"
echo "FMP_API_KEY: $FMP_API_KEY"
echo "DB_URI: $DB_URI"

# Esperar a que Kafka esté disponible
echo "Esperando a Kafka..."
timeout=60
counter=0
while ! nc -z -v kafka 9092 && [ $counter -lt $timeout ]; do
    echo "Waiting for Kafka... ($counter/$timeout)"
    sleep 1
    counter=$((counter+1))
done

if [ $counter -eq $timeout ]; then
    echo "Error: Kafka no disponible después de $timeout segundos"
    exit 1
fi

echo "Kafka está disponible!"

# Esperar a PostgreSQL
echo "Esperando a PostgreSQL..."
timeout=60
counter=0
while ! nc -z -v postgres 5432 && [ $counter -lt $timeout ]; do
    echo "Waiting for PostgreSQL... ($counter/$timeout)"
    sleep 1
    counter=$((counter+1))
done

if [ $counter -eq $timeout ]; then
    echo "Error: PostgreSQL no disponible después de $timeout segundos"
    exit 1
fi

echo "PostgreSQL está disponible!"

# Ejecutar pruebas de API
echo "Ejecutando pruebas de API..."
python /app/fmp_api_test.py
API_TEST_RESULT=$?

if [ $API_TEST_RESULT -ne 0 ]; then
    echo "❌ Las pruebas de API han fallado. Deteniendo la inicialización."
    exit 1
fi

echo "✅ Pruebas de API completadas con éxito"

# Iniciar aplicación
echo "Iniciando servicio de ingesta..."
python /app/app.py