#!/bin/bash

# Script para backup manual de la base de datos
# Este script se puede ejecutar cuando se necesite realizar 
# un backup completo adicional al que se hace automáticamente.

# Obtener fecha y hora actual
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/postgres"
BACKUP_FILE="$BACKUP_DIR/market_data_manual_$TIMESTAMP.sql.gz"

# Asegurar que existe el directorio de backup
mkdir -p $BACKUP_DIR

echo "Iniciando backup manual de la base de datos..."

# Realizar el backup utilizando docker-compose
docker-compose exec -T postgres pg_dump -U market_admin market_data | gzip > $BACKUP_FILE

# Verificar si el backup se realizó correctamente
if [ $? -eq 0 ]; then
    echo "Backup completado exitosamente: $BACKUP_FILE"
    echo "Tamaño del backup: $(du -h $BACKUP_FILE | cut -f1)"
else
    echo "Error al realizar el backup"
    exit 1
fi

echo "Backup manual completado."

