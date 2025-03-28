#!/bin/bash

# Script para reiniciar correctamente los servicios del proyecto
# Autor: Ivanovena
# Fecha: $(date)

set -e  # Salir si hay error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Parando servicios actuales..."
docker-compose down

echo "Limpiando caché de Docker..."
docker system prune -f --volumes

echo "Reconstruyendo el frontend React..."
docker-compose build frontend

echo "Iniciando servicios en orden correcto..."
# Primero iniciamos los servicios de infraestructura
docker-compose up -d postgres zookeeper kafka
echo "Esperando a que los servicios de infraestructura estén listos..."
sleep 15

# Luego iniciamos los servicios de aplicación
echo "Iniciando servicios de aplicación..."
docker-compose up -d ingestion broker streaming

# Finalmente iniciamos el frontend
echo "Iniciando frontend..."
docker-compose up -d frontend

echo "Comprobando el estado de los servicios..."
docker-compose ps

echo "Verificando logs del frontend..."
docker-compose logs frontend | tail -n 20

echo "Todos los servicios han sido reiniciados correctamente."
echo "Accede a la aplicación en: http://localhost"
