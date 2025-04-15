#!/bin/bash

# Colores para mejorar la visualización
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Verificando conectividad entre host local y contenedores...${NC}"

# 1. Verificar que Docker está funcionando
echo -e "${YELLOW}Verificando que Docker esté en funcionamiento...${NC}"
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker no está en funcionamiento. Por favor, inicia Docker Desktop e intenta nuevamente.${NC}"
  exit 1
fi
echo -e "${GREEN}Docker está funcionando correctamente.${NC}"

# 2. Verificar que los servicios están en ejecución
echo -e "${YELLOW}Verificando que los servicios estén en ejecución...${NC}"
services=("kafka" "postgres" "redis" "broker" "ingestion" "streaming" "batch")
for service in "${services[@]}"; do
  if ! docker ps | grep -q "${COMPOSE_PROJECT_NAME:-project7}-$service"; then
    echo -e "${RED}El servicio $service no está en ejecución. Ejecuta 'docker-compose up -d' para iniciar todos los servicios.${NC}"
    exit 1
  fi
done
echo -e "${GREEN}Todos los servicios están en ejecución.${NC}"

# 3. Probar conectividad a los servicios mediante comandos curl
echo -e "${YELLOW}Probando conectividad a los servicios mediante curl...${NC}"

# Función para verificar la conexión a un servicio
check_service() {
  local name=$1
  local url=$2
  echo -n "Verificando conexión a $name ($url)... "
  if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|204\|301\|302\|307\|308"; then
    echo -e "${GREEN}OK${NC}"
    return 0
  else
    echo -e "${RED}Error${NC}"
    return 1
  fi
}

# Probar conexión a los servicios HTTP
check_service "broker API" "http://localhost:8001/health" || echo -e "${YELLOW}Broker no responde. Verifica los logs con 'docker logs ${COMPOSE_PROJECT_NAME:-project7}-broker-1'${NC}"
check_service "ingestion API" "http://localhost:8000/health" || echo -e "${YELLOW}Ingestion no responde. Verifica los logs con 'docker logs ${COMPOSE_PROJECT_NAME:-project7}-ingestion-1'${NC}"
check_service "streaming API" "http://localhost:8002/health" || echo -e "${YELLOW}Streaming no responde. Verifica los logs con 'docker logs ${COMPOSE_PROJECT_NAME:-project7}-streaming-1'${NC}"
check_service "batch API" "http://localhost:8003/health" || echo -e "${YELLOW}Batch no responde. Verifica los logs con 'docker logs ${COMPOSE_PROJECT_NAME:-project7}-batch-1'${NC}"

# 4. Verificar resolución de host.docker.internal
echo -e "${YELLOW}Verificando resolución de host.docker.internal desde los contenedores...${NC}"
if docker exec -it ${COMPOSE_PROJECT_NAME:-project7}-broker-1 ping -c 1 host.docker.internal > /dev/null 2>&1; then
  echo -e "${GREEN}host.docker.internal es accesible desde los contenedores.${NC}"
else
  echo -e "${RED}Error al resolver host.docker.internal. Es posible que la comunicación entre contenedores y host no funcione correctamente.${NC}"
  echo -e "${YELLOW}Recomendación: Asegúrate de que 'extra_hosts: - \"host.docker.internal:host-gateway\"' esté en tu docker-compose.yml para cada servicio.${NC}"
fi

# 5. Verificar configuración CORS en los archivos .env
echo -e "${YELLOW}Verificando configuración CORS...${NC}"
if grep -q "CORS_ORIGINS=.*localhost.*3000" /Users/ivangodo/Documents/Documentos\ personales/Model\ bursátil/project7/docker-compose.yml; then
  echo -e "${GREEN}La configuración CORS parece correcta para el frontend local.${NC}"
else
  echo -e "${RED}Advertencia: Es posible que la configuración CORS no incluya localhost:3000. Verifica que CORS_ORIGINS incluya http://localhost:3000 en docker-compose.yml.${NC}"
fi

# 6. Verificar puertos de websockets
echo -e "${YELLOW}Verificando puertos de WebSockets...${NC}"
ports=(8080 8090)
for port in "${ports[@]}"; do
  if nc -z localhost $port; then
    echo -e "${GREEN}El puerto WebSocket $port está abierto y accesible.${NC}"
  else
    echo -e "${RED}El puerto WebSocket $port no está accesible. Verifica la configuración del servicio correspondiente.${NC}"
  fi
done

# 7. Verificar configuración .env del frontend
echo -e "${YELLOW}Verificando configuración .env del frontend...${NC}"
env_file="/Users/ivangodo/Documents/Documentos personales/Model bursátil/project7/web/.env"
if [ -f "$env_file" ]; then
  if grep -q "REACT_APP_API_URL=http://localhost:8001" "$env_file" && \
     grep -q "REACT_APP_MARKET_WS_URL=ws://localhost:8080/ws" "$env_file" && \
     grep -q "REACT_APP_PREDICTIONS_WS_URL=ws://localhost:8090/ws" "$env_file"; then
    echo -e "${GREEN}La configuración del frontend parece correcta.${NC}"
  else
    echo -e "${RED}Advertencia: La configuración del frontend puede no estar correctamente configurada para localhost.${NC}"
    echo -e "${YELLOW}Verifica las siguientes líneas en $env_file:${NC}"
    echo "REACT_APP_API_URL=http://localhost:8001"
    echo "REACT_APP_MARKET_WS_URL=ws://localhost:8080/ws"
    echo "REACT_APP_PREDICTIONS_WS_URL=ws://localhost:8090/ws"
    echo "REACT_APP_RECOMMENDATIONS_WS_URL=ws://localhost:8001/ws"
  fi
else
  echo -e "${RED}No se encontró el archivo .env del frontend. Verifica que el archivo exista y esté correctamente configurado.${NC}"
fi

# 8. Resumen final
echo -e "\n${YELLOW}Resumen de verificación:${NC}"
echo -e "- Docker está en funcionamiento"
echo -e "- Los servicios están en ejecución"
echo -e "- Se ha verificado la conectividad a los servicios"
echo -e "- Se ha verificado la resolución de host.docker.internal"
echo -e "- Se ha verificado la configuración CORS"
echo -e "- Se ha verificado los puertos de WebSockets"
echo -e "- Se ha verificado la configuración del frontend"

echo -e "\n${GREEN}Para iniciar el dashboard, ejecuta:${NC}"
echo -e "${YELLOW}cd \"/Users/ivangodo/Documents/Documentos personales/Model bursátil/project7/web\" && npm start${NC}"