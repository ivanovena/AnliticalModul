#!/bin/bash
echo "Reiniciando el servicio frontend..."
docker-compose stop frontend
docker-compose rm -f frontend
docker-compose build frontend
docker-compose up -d frontend
echo "Verificando estado..."
sleep 5
docker-compose ps frontend
