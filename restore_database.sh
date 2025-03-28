#!/bin/bash

# Script para restaurar la base de datos desde un backup

# Verificar si se proporcionó un archivo de backup
if [ -z "$1" ]; then
    echo "Error: Debe proporcionar un archivo de backup."
    echo "Uso: $0 archivo_backup.sql.gz"
    
    echo "Backups disponibles:"
    ls -lh backups/postgres/
    exit 1
fi

BACKUP_FILE="$1"

# Verificar si el archivo existe
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: El archivo de backup no existe: $BACKUP_FILE"
    exit 1
fi

echo "==============================================="
echo "ADVERTENCIA: Esta acción eliminará todos los datos actuales"
echo "y los reemplazará con los datos del backup."
echo "==============================================="
read -p "¿Está seguro de que desea continuar? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Restauración cancelada."
    exit 0
fi

echo "Iniciando restauración desde: $BACKUP_FILE"

# Detener el servicio de ingestion para evitar conflictos
echo "Deteniendo el servicio de ingestion..."
docker-compose stop ingestion

# Restaurar la base de datos
echo "Restaurando la base de datos..."
gunzip -c "$BACKUP_FILE" | docker-compose exec -T postgres psql -U market_admin market_data

# Verificar si la restauración se realizó correctamente
if [ $? -eq 0 ]; then
    echo "Restauración completada exitosamente."
else
    echo "Error al restaurar la base de datos."
    echo "Reiniciando el servicio de ingestion..."
    docker-compose start ingestion
    exit 1
fi

# Reiniciar el servicio de ingestion
echo "Reiniciando el servicio de ingestion..."
docker-compose start ingestion

echo "Proceso de restauración completado."

