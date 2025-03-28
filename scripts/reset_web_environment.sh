#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Iniciando reinicio completo del entorno web...${NC}"

# Detener todos los contenedores
echo -e "${YELLOW}Deteniendo todos los servicios...${NC}"
docker-compose down

# Limpiar artefactos de compilación
echo -e "${YELLOW}Limpiando artefactos de compilación...${NC}"
if [ -d "web/node_modules" ]; then
  rm -rf web/node_modules
fi
if [ -d "web/build" ]; then
  rm -rf web/build
fi

# Verificar y crear los archivos de configuración de Tailwind si no existen
echo -e "${YELLOW}Verificando archivos de configuración de Tailwind...${NC}"
if [ ! -f "web/tailwind.config.js" ]; then
  echo "/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    \"./src/**/*.{js,jsx,ts,tsx}\",
    \"./public/index.html\"
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}" > web/tailwind.config.js
  echo -e "${GREEN}Creado tailwind.config.js${NC}"
fi

if [ ! -f "web/postcss.config.js" ]; then
  echo "module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}" > web/postcss.config.js
  echo -e "${GREEN}Creado postcss.config.js${NC}"
fi

# Reconstruir la imagen frontend
echo -e "${YELLOW}Reconstruyendo la imagen frontend...${NC}"
docker-compose build frontend

# Iniciar servicios en orden correcto
echo -e "${YELLOW}Iniciando servicios de infraestructura...${NC}"
docker-compose up -d postgres zookeeper kafka
echo -e "${YELLOW}Esperando inicialización (15s)...${NC}"
sleep 15

echo -e "${YELLOW}Iniciando servicios de aplicación...${NC}"
docker-compose up -d ingestion broker streaming
echo -e "${YELLOW}Esperando inicialización (10s)...${NC}"
sleep 10

echo -e "${YELLOW}Iniciando frontend...${NC}"
docker-compose up -d frontend

# Verificar estado
echo -e "${YELLOW}Verificando estado de los servicios...${NC}"
docker-compose ps

echo -e "${GREEN}¡Entorno reiniciado correctamente!${NC}"
echo -e "${YELLOW}Accede a la aplicación en: ${GREEN}http://localhost${NC}"
echo -e "${YELLOW}Para ver los logs del frontend: ${NC}docker-compose logs -f frontend"
echo -e "${YELLOW}Para ver todos los logs: ${NC}docker-compose logs -f"
