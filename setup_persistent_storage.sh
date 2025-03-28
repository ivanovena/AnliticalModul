#!/bin/bash

# Script para crear la estructura de directorios necesaria para la persistencia de datos
# y realizar la configuración inicial

# Directorios de datos
mkdir -p data/postgres_data data/kafka_data data/zookeeper_data data/broker_models
mkdir -p backups/postgres

# Establecer permisos adecuados
chmod -R 777 data backups

echo "Estructura de directorios creada correctamente."
echo "Los datos se almacenarán de forma persistente en:"
echo "- data/postgres_data: Datos de PostgreSQL"
echo "- data/kafka_data: Datos de Kafka"
echo "- data/zookeeper_data: Datos de Zookeeper"
echo "- data/broker_models: Modelos del broker"
echo "- backups/postgres: Backups automáticos de la base de datos"

echo "Los backups se realizarán automáticamente cada 24 horas."

