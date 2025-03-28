#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Corrigiendo el servicio broker...${NC}"

# Detener y eliminar el contenedor broker existente
echo -e "${YELLOW}Deteniendo el servicio broker existente...${NC}"
docker-compose stop broker
docker-compose rm -f broker

# Reconstruir la imagen del broker
echo -e "${YELLOW}Reconstruyendo la imagen del broker...${NC}"
docker-compose build broker

# Iniciar el servicio broker nuevamente
echo -e "${YELLOW}Iniciando el servicio broker...${NC}"
docker-compose up -d broker

# Verificar el estado
echo -e "${YELLOW}Verificando el estado del servicio...${NC}"
sleep 10
docker-compose ps broker

echo -e "${GREEN}¡Servicio broker reiniciado!${NC}"
echo -e "${YELLOW}Para ver los logs en tiempo real, ejecuta:${NC}"
echo -e "docker-compose logs -f broker"

# Ahora vamos a verificar si el healthcheck funciona
echo -e "${YELLOW}Verificando el healthcheck del broker...${NC}"
sleep 20
HEALTH_STATUS=$(docker-compose ps broker | grep broker | grep -o "(healthy)" || echo "(unhealthy)")

if [[ $HEALTH_STATUS == "(healthy)" ]]; then
    echo -e "${GREEN}El broker está healthy ahora.${NC}"
else
    echo -e "${RED}El broker sigue en estado unhealthy.${NC}"
    echo -e "${YELLOW}Aplicando correcciones adicionales...${NC}"
    
    # Reiniciar el broker con un tiempo de espera más largo
    docker-compose restart broker
    echo -e "${YELLOW}Esperando 30 segundos para que inicie completamente...${NC}"
    sleep 30
    
    # Verificar nuevamente
    HEALTH_STATUS=$(docker-compose ps broker | grep broker | grep -o "(healthy)" || echo "(unhealthy)")
    
    if [[ $HEALTH_STATUS == "(healthy)" ]]; then
        echo -e "${GREEN}El broker está healthy ahora.${NC}"
    else
        echo -e "${RED}El broker sigue en estado unhealthy después de reintentar.${NC}"
        echo -e "${YELLOW}Revisa los logs para más detalles:${NC}"
        docker-compose logs broker | tail -n 50
    fi
fi
