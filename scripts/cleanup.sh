#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ROOT=$(cd $(dirname $0)/..; pwd)
echo -e "${GREEN}Limpiando proyecto en${NC}: $PROJECT_ROOT"

# Eliminar archivos .DS_Store (macOS)
echo -e "${YELLOW}Eliminando archivos .DS_Store...${NC}"
find "$PROJECT_ROOT" -type f -name ".DS_Store" -delete
echo -e "${GREEN}✓ Archivos .DS_Store eliminados${NC}"

# Eliminar directorios web duplicados
if [ -d "$PROJECT_ROOT/services/web" ]; then
  echo -e "${YELLOW}Eliminando directorio web duplicado en services...${NC}"
  rm -rf "$PROJECT_ROOT/services/web"
  echo -e "${GREEN}✓ Directorio web duplicado eliminado${NC}"
else
  echo -e "${GREEN}✓ No hay directorio web duplicado para eliminar${NC}"
fi

# Eliminar directorios .git anidados
echo -e "${YELLOW}Buscando y eliminando directorios .git anidados...${NC}"
find "$PROJECT_ROOT" -path "$PROJECT_ROOT/.git" -prune -o -name ".git" -type d -print -exec rm -rf {} \; 2>/dev/null
echo -e "${GREEN}✓ Directorios .git anidados eliminados${NC}"

# Eliminar archivos temporales y de respaldo
echo -e "${YELLOW}Eliminando archivos temporales y de respaldo...${NC}"
find "$PROJECT_ROOT" -type f \( -name "*~" -o -name "*.bak" -o -name "*.swp" -o -name ".DS_Store" -o -name "*.pyc" \) -delete
echo -e "${GREEN}✓ Archivos temporales y de respaldo eliminados${NC}"

# Eliminar directorios __pycache__
echo -e "${YELLOW}Eliminando directorios __pycache__...${NC}"
find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo -e "${GREEN}✓ Directorios __pycache__ eliminados${NC}"

# Limpiar directorios logs vacíos
echo -e "${YELLOW}Limpiando directorios de logs vacíos...${NC}"
find "$PROJECT_ROOT/logs" -type d -empty -delete 2>/dev/null
echo -e "${GREEN}✓ Directorios de logs vacíos eliminados${NC}"

# Verificar la estructura después de la limpieza
echo -e "${YELLOW}Verificando estructura de directorios...${NC}"
echo -e "${GREEN}✓ Estructura básica de directorios:${NC}"
ls -la "$PROJECT_ROOT"

echo -e "${GREEN}✓ Servicios:${NC}"
ls -la "$PROJECT_ROOT/services"

echo -e "\n${GREEN}Limpieza completada con éxito!${NC}"
