#!/bin/bash
set -e

echo "Iniciando servicio de batch..."

# Esperar a que PostgreSQL esté disponible
echo "Esperando a que PostgreSQL esté disponible..."
until nc -z postgres 5432; do
  echo "PostgreSQL no está disponible - esperando..."
  sleep 2
done
echo "PostgreSQL está disponible"

# Esperar a que Kafka esté disponible
echo "Esperando a que Kafka esté disponible..."
until nc -z kafka 9092; do
  echo "Kafka no está disponible - esperando..."
  sleep 2
done
echo "Kafka está disponible"

# Crear directorios necesarios
# Asegurar que el directorio de modelos siempre exista con un valor por defecto
MODEL_DIR="${MODEL_STORAGE_PATH:-/app/models}"
mkdir -p "$MODEL_DIR"

# Si existe un argumento para modo programado
if [ "$1" = "scheduled" ]; then
    echo "Iniciando en modo programado - ejecutando cada 24 horas"
    while true; do
        echo "Ejecutando proceso de entrenamiento batch..."
        python app.py
        echo "Proceso completado. Próxima ejecución en 24 horas."
        sleep 86400  # 24 horas
    done
else
    # Modo normal - ejecutar una vez
    echo "Ejecutando proceso de entrenamiento batch..."
    python app.py
fi 