#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Iniciando corrección completa del frontend...${NC}"

# 1. Corregir apiService.js para usar rutas relativas
echo -e "${YELLOW}Actualizando apiService.js...${NC}"
sed -i '' 's|const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '"'"''"'"';|const API_BASE_URL = '"'"''"'"';  // Usar rutas relativas para evitar problemas CORS|g' "./web/src/services/apiService.js"

# 2. Corregir WebSocketService para usar ruta relativa
echo -e "${YELLOW}Actualizando websocketService.js...${NC}"
sed -i '' 's|const WEBSOCKET_URL = '"'"'http://localhost:8001'"'"';|const WEBSOCKET_URL = '"'"'/ws'"'"';|g' "./web/src/services/websocketService.js"

# 3. Eliminar datos simulados de Dashboard.jsx
echo -e "${YELLOW}Eliminando datos simulados de Dashboard.jsx...${NC}"
# Usar archivos temporales para los reemplazos complejos
cat > /tmp/dashboard_fix1.sed << 'EOL'
/} else {/,/setSelectedSymbol('AAPL');/ c\
        } else {\
          console.error('Error al recibir datos del portafolio');\
          setError('Error al cargar los datos del portafolio.');\
        }
EOL

cat > /tmp/dashboard_fix2.sed << 'EOL'
/} catch (err) {/,/setSelectedSymbol('AAPL');/ c\
      } catch (err) {\
        console.error('Error fetching portfolio:', err);\
        setError('Error al cargar los datos del portafolio. Intentando conectar con el servidor...');\
      }
EOL

sed -i '' -f /tmp/dashboard_fix1.sed "./web/src/components/Dashboard.jsx"
sed -i '' -f /tmp/dashboard_fix2.sed "./web/src/components/Dashboard.jsx"

# 4. Corregir variables en Dashboard
sed -i '' 's|  const dashboardData = portfolio || {|  const dashboardData = portfolio || {\n    cash: 0,\n    positions: {},\n    total_value: 0|g' "./web/src/components/Dashboard.jsx"

# 5. Configurar Tailwind CSS
echo -e "${YELLOW}Configurando Tailwind CSS...${NC}"
mkdir -p "./web/public"

# Crear tailwind.config.js
cat > "./web/tailwind.config.js" << 'EOL'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
EOL

# Crear postcss.config.js
cat > "./web/postcss.config.js" << 'EOL'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOL

# Crear estilos de respaldo
cat > "./web/public/backup-styles.css" << 'EOL'
/* Backup styles in case Tailwind fails to load */
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
  margin: 0;
  padding: 0;
  background-color: #f3f4f6;
  color: #1f2937;
}

nav {
  background-color: #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.bg-white { background-color: #ffffff; }
.rounded-lg { border-radius: 0.5rem; }
.shadow { box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); }
.p-6 { padding: 1.5rem; }
.mb-6 { margin-bottom: 1.5rem; }
.text-2xl { font-size: 1.5rem; }
.text-lg { font-size: 1.125rem; }
.font-semibold { font-weight: 600; }
.font-medium { font-weight: 500; }
.text-gray-700 { color: #374151; }
.text-green-600 { color: #059669; }
.text-blue-600 { color: #2563eb; }
.text-purple-600 { color: #7c3aed; }

.grid {
  display: grid;
}
.grid-cols-1 {
  grid-template-columns: repeat(1, minmax(0, 1fr));
}
.gap-6 {
  gap: 1.5rem;
}

/* Media Queries */
@media (min-width: 768px) {
  .grid-cols-1 {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

/* Tables */
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
EOL

# 6. Crear archivos estáticos necesarios
echo -e "${YELLOW}Creando archivos estáticos...${NC}"

# Crear index.html
cat > "./web/public/index.html" << 'EOL'
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#000000" />
  <meta
    name="description"
    content="Plataforma de predicción y trading de mercado de valores"
  />
  <link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />
  <link rel="manifest" href="%PUBLIC_URL%/manifest.json" />
  <link rel="stylesheet" href="%PUBLIC_URL%/backup-styles.css" />
  <title>Modelo Bursátil</title>
</head>
<body>
  <noscript>Necesitas habilitar JavaScript para ejecutar esta aplicación.</noscript>
  <div id="root"></div>
</body>
</html>
EOL

# Crear manifest.json
cat > "./web/public/manifest.json" << 'EOL'
{
  "short_name": "Modelo Bursátil",
  "name": "Plataforma de Predicción y Trading de Mercado de Valores",
  "icons": [
    {
      "src": "favicon.ico",
      "sizes": "64x64 32x32 24x24 16x16",
      "type": "image/x-icon"
    }
  ],
  "start_url": ".",
  "display": "standalone",
  "theme_color": "#000000",
  "background_color": "#ffffff"
}
EOL

# Crear favicon.ico vacío
touch "./web/public/favicon.ico"

# Crear página de error 50x
cat > "./web/public/50x.html" << 'EOL'
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

# 7. Corregir Dockerfile para incluir healthcheck
echo -e "${YELLOW}Actualizando Dockerfile...${NC}"
# Crear archivo temporal de sed para el reemplazo
cat > /tmp/dockerfile_fix.sed << 'EOL'
/RUN touch \/usr\/share\/nginx\/html\/favicon.ico/,/# NO HEALTHCHECK/ c\
# Create a healthcheck file\
RUN echo "OK" > /usr/share/nginx/html/healthcheck.html\
\
# Healthcheck to make sure nginx is running\
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\\
  CMD wget -q -O /dev/null http://localhost/healthcheck.html || exit 1
EOL

sed -i '' -f /tmp/dockerfile_fix.sed "./web/Dockerfile"

# 8. Actualizar docker-compose.yml para incluir healthcheck
echo -e "${YELLOW}Actualizando docker-compose.yml...${NC}"
# Crear archivo temporal de sed para el reemplazo
cat > /tmp/docker_compose_fix.sed << 'EOL'
/  frontend:/,/    restart: always/ c\
  frontend:\
    build:\
      context: ./web\
      dockerfile: Dockerfile\
    ports:\
      - "80:80"\
    depends_on:\
      - ingestion\
      - broker\
      - streaming\
    networks:\
      - project7_network\
    restart: always\
    healthcheck:\
      test: ["CMD", "wget", "-q", "--spider", "http://localhost/healthcheck.html", "||", "exit", "1"]\
      interval: 30s\
      timeout: 5s\
      retries: 3
EOL

sed -i '' -f /tmp/docker_compose_fix.sed "./docker-compose.yml"

# 9. Crear archivo de resumen
echo -e "${YELLOW}Creando archivo de resumen...${NC}"
cat > "./frontend_fix_summary.md" << 'EOL'
# Resumen de Correcciones del Frontend

Se han implementado las siguientes correcciones para solucionar los problemas del frontend:

## 1. Corrección de Referencias a API

- Modificado `apiService.js` para usar rutas relativas en lugar de URLs hardcodeadas a localhost
- Actualizado `websocketService.js` para usar la ruta `/ws` en lugar de `http://localhost:8001`

## 2. Eliminación de Datos Simulados

- Eliminados todos los datos simulados del componente Dashboard.jsx
- Implementada lógica de manejo de errores adecuada en lugar de caer en datos simulados

## 3. Configuración de Tailwind CSS

- Creado `tailwind.config.js` con la configuración apropiada
- Creado `postcss.config.js` para la compilación de CSS
- Añadido archivo de estilos de respaldo `backup-styles.css` para garantizar que la interfaz sea utilizable incluso si Tailwind falla

## 4. Correcciones de Estructura del Proyecto React

- Creada estructura básica de `/public` con los archivos necesarios:
  - `index.html` con las referencias correctas
  - `manifest.json`
  - `favicon.ico`
  - `50x.html` para páginas de error

## 5. Mejoras en Dockerfile y Docker Compose

- Actualizado Dockerfile para incluir un healthcheck funcional
- Corregida la configuración en docker-compose.yml para garantizar reinicio apropiado del frontend

## Reinicio del Sistema

Para aplicar todos estos cambios, ejecute los siguientes comandos:

```bash
# Detener todos los contenedores
docker-compose down

# Reconstruir el frontend
docker-compose build frontend

# Iniciar servicios en el orden correcto
docker-compose up -d postgres zookeeper kafka
sleep 15
docker-compose up -d ingestion broker streaming
sleep 15
docker-compose up -d frontend

# Verificar el estado
docker-compose ps
```

## Verificación

Para verificar que el frontend está funcionando correctamente:

1. Acceda a http://localhost en su navegador
2. Confirme que puede ver el Panel de Control sin errores
3. Verifique que se conecta correctamente a los servicios de backend

Si encuentra algún error, puede consultar los logs con:
```bash
docker-compose logs -f frontend
```
EOL

echo -e "${GREEN}¡Correcciones completadas!${NC}"
echo -e "${YELLOW}Para aplicar los cambios, reinicie los servicios con:${NC}"
echo -e "docker-compose down && docker-compose build frontend && docker-compose up -d"
echo -e "${YELLOW}Consulte el archivo frontend_fix_summary.md para más detalles.${NC}"
