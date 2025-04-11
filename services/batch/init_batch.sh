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

# Asegurar que el registro de modelos esté actualizado antes de iniciar la API
echo "Actualizando registro de modelos..."
python -c "from app import create_model_registry; from config import MODEL_OUTPUT_PATH; create_model_registry(MODEL_OUTPUT_PATH)"

# Iniciar el servidor API para visualización de modelos y métricas
if [ "$1" != "scheduled" ] && [ "$1" != "batch_only" ]; then
    echo "Iniciando servidor API en segundo plano..."
    python api.py &
fi

# Si existe un argumento para modo programado
if [ "$1" = "scheduled" ]; then
    echo "Iniciando en modo programado - ejecutando cada 24 horas"
    while true; do
        echo "Ejecutando proceso de entrenamiento batch..."
        python app.py
        
        # Actualizar el registro de modelos después de entrenar
        python -c "from app import create_model_registry; from config import MODEL_OUTPUT_PATH; create_model_registry(MODEL_OUTPUT_PATH)"
        
        echo "Proceso completado. Próxima ejecución en 24 horas."
        sleep 86400  # 24 horas
    done
elif [ "$1" = "api_only" ]; then
    # Modo API solamente
    echo "Ejecutando solo el servidor API..."
    python api.py
else
    # Modo normal - ejecutar batch una vez y mantener API ejecutándose
    echo "Ejecutando proceso de entrenamiento batch..."
    python app.py
    
    # Si la API no está iniciada, iniciarla ahora
    if [ "$1" = "batch_only" ]; then
        echo "Entrenamiento completado. Servicio finalizado (modo batch_only)."
        exit 0
    else
        # Actualizar el registro de modelos después de entrenar
        python -c "from app import create_model_registry; from config import MODEL_OUTPUT_PATH; create_model_registry(MODEL_OUTPUT_PATH)"
        
        # Mantener el proceso vivo esperando señales
        echo "Entrenamiento completado. Manteniendo API activa..."
        wait
    fi
fi