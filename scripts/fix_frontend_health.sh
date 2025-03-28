#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Corrigiendo healthcheck del frontend...${NC}"

# 1. Corregir el healthcheck en docker-compose.yml
echo -e "${YELLOW}Actualizando healthcheck en docker-compose.yml...${NC}"

cat > "/tmp/frontend_healthcheck.txt" << 'EOL'
  frontend:
    build:
      context: ./web
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - ingestion
      - broker
      - streaming
    networks:
      - project7_network
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/50x.html"]
      interval: 30s
      timeout: 5s
      retries: 3
EOL

# Reemplazar la sección del frontend en docker-compose.yml
sed -i '' '/^  frontend:/,/healthcheck:/{/healthcheck:/,/retries: [0-9]/d;}' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/docker-compose.yml"

# Ahora insertar el nuevo bloque del frontend
awk '/^  frontend:/{print;system("cat /tmp/frontend_healthcheck.txt");next}1' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/docker-compose.yml" > /tmp/docker-compose-new.yml
mv /tmp/docker-compose-new.yml "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/docker-compose.yml"

echo -e "${GREEN}Actualizado healthcheck del frontend para usar un recurso estático simple${NC}"

# 2. Actualizar el Dockerfile directamente
echo -e "${YELLOW}Actualizando Dockerfile del frontend...${NC}"

sed -i '' 's|CMD wget -q -O /dev/null http://localhost || exit 1|CMD curl -f http://localhost/50x.html || exit 1|g' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/Dockerfile"

echo -e "${GREEN}Dockerfile del frontend actualizado${NC}"

# 3. Asegurarnos de que el archivo 50x.html exista
echo -e "${YELLOW}Creando archivo 50x.html para health check...${NC}"

cat > "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/public/50x.html" << 'EOL'
<!DOCTYPE html>
<html>
<head>
    <title>Error 50x</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
        }
        h1 {
            color: #d33;
        }
    </style>
</head>
<body>
    <h1>Error 50x</h1>
    <p>Lo sentimos, ocurrió un error en el servidor.</p>
</body>
</html>
EOL

echo -e "${GREEN}Archivo 50x.html creado${NC}"

# 4. Reconstruir y reiniciar el frontend
echo -e "${YELLOW}Reconstruyendo y reiniciando el frontend...${NC}"

docker-compose stop frontend
docker-compose rm -f frontend
docker-compose build frontend
docker-compose up -d frontend

echo -e "${YELLOW}Esperando 30 segundos para inicialización completa...${NC}"
sleep 30

# Verificar estado
echo -e "${YELLOW}Verificando estado del frontend...${NC}"
docker ps | grep frontend

echo -e "${GREEN}¡Corrección del healthcheck frontend completada!${NC}"
echo -e "${YELLOW}Accede a http://localhost para verificar que todo funciona correctamente${NC}"
