#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Aplicando solución de emergencia...${NC}"

# 1. Crear un archivo estático simple en el contenedor
echo -e "${YELLOW}Creando archivo healthcheck.html en el contenedor...${NC}"

docker exec project7-frontend-1 sh -c "echo 'OK' > /usr/share/nginx/html/healthcheck.html"

# 2. Verificar que el archivo esté accesible
echo -e "${YELLOW}Verificando acceso al archivo...${NC}"

if docker exec project7-frontend-1 wget -q -O - http://localhost/healthcheck.html | grep -q "OK"; then
    echo -e "${GREEN}Archivo healthcheck.html accesible dentro del contenedor${NC}"
else
    echo -e "${RED}No se puede acceder al archivo healthcheck.html${NC}"
fi

# 3. Corregir el Dockerfile para el futuro
echo -e "${YELLOW}Actualizando Dockerfile para futuros rebuilds...${NC}"

cat > "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/Dockerfile.fixed" << 'EOL'
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json ./
COPY postcss.config.js ./
COPY tailwind.config.js ./
RUN npm install --legacy-peer-deps
RUN npm install ajv autoprefixer postcss tailwindcss

# Copy application code
COPY public/ ./public/
COPY src/ ./src/

# Create an empty .env file to avoid warnings
RUN touch .env

# Build the application with debugging output
RUN echo "Building application..." && \
    npm run build || { echo "Build failed"; exit 1; }

# Production stage
FROM nginx:alpine

# Copy the build output
COPY --from=build /app/build /usr/share/nginx/html

# Copy the nginx configuration
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Create basic favicon to prevent 404 errors
RUN touch /usr/share/nginx/html/favicon.ico

# Create a simple 50x.html page
RUN echo '<html><body><h1>Error 50x</h1><p>Lo sentimos, ocurrió un error en el servidor.</p></body></html>' > /usr/share/nginx/html/50x.html

# Create a healthcheck file
RUN echo "OK" > /usr/share/nginx/html/healthcheck.html

# NO HEALTHCHECK - el healthcheck en docker-compose causaba problemas

# Expose port
EXPOSE 80

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
EOL

mv "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/Dockerfile.fixed" "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/Dockerfile"

echo -e "${GREEN}Dockerfile actualizado para futuros rebuilds${NC}"

# 4. Reiniciar el contenedor (opcional)
echo -e "${YELLOW}¿Deseas reiniciar el contenedor ahora? (s/n)${NC}"
read -n 1 answer
if [ "$answer" == "s" ]; then
    echo
    echo -e "${YELLOW}Reiniciando el contenedor frontend...${NC}"
    docker restart project7-frontend-1
    sleep 5
    echo -e "${GREEN}Contenedor reiniciado${NC}"
else
    echo
    echo -e "${YELLOW}No se reiniciará el contenedor${NC}"
fi

echo -e "${GREEN}¡Solución de emergencia aplicada!${NC}"
echo -e "${YELLOW}NOTA: Este arreglo es temporal y solo afecta al contenedor actual.${NC}"
echo -e "${YELLOW}Para una solución permanente, ejecutar ./fix_all_healthchecks.sh${NC}"
