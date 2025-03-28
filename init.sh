#!/bin/bash

# Ensure correct permissions
chmod +x init.sh

# Crear directorios necesarios
mkdir -p logs/broker logs/ingestion logs/streaming logs/batch
chmod -R 777 logs

# Verificar dependencias
pip install -r services/broker/requirements.txt
pip install -r services/ingestion/requirements.txt
pip install -r services/streaming/requirements.txt
pip install -r services/batch/requirements.txt

# Inicializar base de datos
python scripts/init_database.py

# Iniciar servicios
docker-compose up --build -d

# Mostrar logs
docker-compose logs -f
