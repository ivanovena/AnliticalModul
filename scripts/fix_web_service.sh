#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Corrigiendo el servicio web...${NC}"

# Detener y eliminar el contenedor frontend existente
echo -e "${YELLOW}Deteniendo el servicio web existente...${NC}"
docker-compose stop frontend
docker-compose rm -f frontend

# Limpiar posibles artefactos de compilación anteriores
echo -e "${YELLOW}Limpiando archivos de compilación anteriores...${NC}"
if [ -d "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/node_modules" ]; then
  rm -rf "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/node_modules"
fi
if [ -d "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/build" ]; then
  rm -rf "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/build"
fi

# Reconstruir la imagen frontend
echo -e "${YELLOW}Reconstruyendo la imagen web...${NC}"
docker-compose build frontend

# Iniciar el servicio frontend nuevamente
echo -e "${YELLOW}Iniciando el servicio web...${NC}"
docker-compose up -d frontend

# Verificar el estado
echo -e "${YELLOW}Verificando el estado del servicio...${NC}"
sleep 5
docker-compose ps frontend

echo -e "${GREEN}¡Servicio web reiniciado!${NC}"
echo -e "${YELLOW}Para ver los logs en tiempo real, ejecuta:${NC}"
echo -e "docker-compose logs -f web"
