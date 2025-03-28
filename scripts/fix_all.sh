#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Iniciando reparación completa del sistema...${NC}"

# 1. Configurar Tailwind CSS
echo -e "${YELLOW}Configurando Tailwind CSS...${NC}"
# Crear tailwind.config.js
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

# Crear postcss.config.js
if [ ! -f "web/postcss.config.js" ]; then
    echo "module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}" > web/postcss.config.js
    echo -e "${GREEN}Creado postcss.config.js${NC}"
fi

# Crear respaldo CSS
if [ ! -f "web/public/backup-styles.css" ]; then
    echo "/* Backup styles in case Tailwind fails to load */
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  margin: 0;
  padding: 0;
  background-color: #f3f4f6;
  color: #1f2937;
}

/* Navbar styles */
nav {
  background-color: #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.bg-white {
  background-color: #ffffff;
}

.rounded-lg {
  border-radius: 0.5rem;
}

.shadow {
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
}

.p-6 {
  padding: 1.5rem;
}

.mb-6 {
  margin-bottom: 1.5rem;
}

.text-2xl {
  font-size: 1.5rem;
}

.text-lg {
  font-size: 1.125rem;
}

.font-semibold {
  font-weight: 600;
}

.font-medium {
  font-weight: 500;
}

.text-gray-700 {
  color: #374151;
}

.text-green-600 {
  color: #059669;
}

.text-blue-600 {
  color: #2563eb;
}

.text-purple-600 {
  color: #7c3aed;
}

/* Grid */
.grid {
  display: grid;
}

.grid-cols-1 {
  grid-template-columns: repeat(1, minmax(0, 1fr));
}

.gap-6 {
  gap: 1.5rem;
}

/* Table */
table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 0.75rem 0.5rem;
  text-align: left;
}

thead {
  background-color: #f9fafb;
}

tbody tr {
  border-bottom: 1px solid #e5e7eb;
}

/* Buttons */
button {
  cursor: pointer;
  padding: 0.375rem 0.75rem;
  border-radius: 0.25rem;
  font-weight: 500;
}

button:hover {
  opacity: 0.9;
}

/* Media Queries */
@media (min-width: 768px) {
  .grid-cols-1 {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}" > web/public/backup-styles.css
    echo -e "${GREEN}Creado backup-styles.css${NC}"
fi

# Actualizar index.html
if ! grep -q "backup-styles.css" web/public/index.html; then
    sed -i '' 's|</head>|    <link rel="stylesheet" href="%PUBLIC_URL%/backup-styles.css" />\n  </head>|g' web/public/index.html
    echo -e "${GREEN}Actualizado index.html con estilos de respaldo${NC}"
fi

# 2. Corregir el broker
echo -e "${YELLOW}Corrigiendo configuración del broker...${NC}"
# Actualizar healthcheck en docker-compose.yml
sed -i '' 's|test: \["CMD", "curl", "-f", "http://localhost:8001/health"\]|test: \["CMD", "wget", "-q", "--spider", "http://localhost:8001/health"\]|g' docker-compose.yml
echo -e "${GREEN}Actualizada configuración de healthcheck para broker${NC}"

# Asegurar que curl y wget estén instalados
if ! grep -q "wget" services/broker/Dockerfile; then
    sed -i '' 's|pkg-config \\|pkg-config \\\n    curl \\\n    wget \\|g' services/broker/Dockerfile
    echo -e "${GREEN}Agregadas dependencias curl y wget al broker${NC}"
fi

# 3. Reconstruir y reiniciar todos los servicios
echo -e "${YELLOW}Reconstruyendo y reiniciando todos los servicios...${NC}"

# Detener todo
echo -e "${YELLOW}Deteniendo todos los servicios...${NC}"
docker-compose down

# Reconstruir
echo -e "${YELLOW}Reconstruyendo imágenes...${NC}"
docker-compose build frontend broker

# Iniciar servicios en orden
echo -e "${YELLOW}Iniciando servicios de infraestructura...${NC}"
docker-compose up -d postgres zookeeper kafka
echo -e "${YELLOW}Esperando inicialización (15s)...${NC}"
sleep 15

echo -e "${YELLOW}Iniciando servicios de aplicación...${NC}"
docker-compose up -d ingestion broker streaming
echo -e "${YELLOW}Esperando inicialización (15s)...${NC}"
sleep 15

echo -e "${YELLOW}Iniciando frontend...${NC}"
docker-compose up -d frontend

# Verificar estado
echo -e "${YELLOW}Verificando estado de los servicios...${NC}"
docker-compose ps

echo -e "${GREEN}¡Reparación completada!${NC}"
echo -e "${YELLOW}Accede a la aplicación en: ${GREEN}http://localhost${NC}"
echo -e "${YELLOW}Para ver los logs del frontend: ${NC}docker-compose logs -f frontend"
echo -e "${YELLOW}Para ver los logs del broker: ${NC}docker-compose logs -f broker"
echo -e "${YELLOW}Para ver todos los logs: ${NC}docker-compose logs -f"
