#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Corrigiendo problemas de CSS...${NC}"

# Verificar y crear los archivos de configuración de Tailwind si no existen
echo -e "${YELLOW}Configurando Tailwind CSS...${NC}"

# Crear tailwind.config.js
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

# Crear postcss.config.js
echo "module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}" > web/postcss.config.js
echo -e "${GREEN}Creado postcss.config.js${NC}"

# Actualizar package.json con las dependencias necesarias
echo -e "${YELLOW}Actualizando package.json...${NC}"
cd web
if ! grep -q "autoprefixer" package.json; then
  # Reemplazar la línea de dependencias para incluir autoprefixer y postcss
  sed -i '' 's/"dependencies": {/"dependencies": {\n    "autoprefixer": "^10.4.14",\n    "postcss": "^8.4.24",/g' package.json
  echo -e "${GREEN}Añadidas dependencias de CSS${NC}"
fi
cd ..

# Crear estilos de respaldo
echo -e "${YELLOW}Creando estilos de respaldo...${NC}"
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

# Actualizar index.html para incluir los estilos de respaldo
echo -e "${YELLOW}Actualizando index.html...${NC}"
if ! grep -q "backup-styles.css" web/public/index.html; then
  # Insertar la línea de estilos de respaldo antes de </head>
  sed -i '' 's|</head>|    <link rel="stylesheet" href="%PUBLIC_URL%/backup-styles.css" />\n  </head>|g' web/public/index.html
  echo -e "${GREEN}Añadido enlace a estilos de respaldo${NC}"
fi

# Reconstruir la imagen frontend
echo -e "${YELLOW}Reconstruyendo la imagen frontend...${NC}"
docker-compose build frontend

# Iniciar el frontend
echo -e "${YELLOW}Iniciando el frontend...${NC}"
docker-compose up -d frontend

echo -e "${GREEN}¡Corrección de CSS completada!${NC}"
echo -e "${YELLOW}La aplicación debería estar accesible en: ${GREEN}http://localhost${NC}"
