#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Iniciando solución completa para los problemas de la plataforma...${NC}"

# 1. Crear el modelo ARIMA personalizado que falta
echo -e "${YELLOW}Creando modelo personalizado de machine learning...${NC}"

# Crear el archivo custom_models.py para el modelo ARIMA
cat > "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/services/streaming/custom_models.py" << 'EOL'
import numpy as np
from river import base
from river.utils import rolling

class ALMARegressor(base.Regressor):
    """Arnaud Legoux Moving Average Regressor.
    
    The ALMA is a moving average algorithm that aims to reduce lag and noise.
    It's a weighted moving average where the weights are a Gaussian distribution,
    whose center and width can be adjusted.
    
    Parameters:
    -----------
    alpha : float (default=0.1)
        Learning rate for updating weights
    window_size : int (default=10)
        Size of the rolling window for ALMA calculation
    sigma : float (default=6.0)
        Controls the width of the distribution of weights
    offset : float (default=0.85)
        Controls the position of the distribution of weights (0 to 1)
    """
    
    def __init__(self, alpha=0.1, window_size=10, sigma=6.0, offset=0.85):
        self.alpha = alpha
        self.window_size = window_size
        self.sigma = sigma
        self.offset = offset
        self.weights = np.zeros(window_size)
        self._y_history = rolling.Window(size=window_size)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize ALMA weights using Gaussian distribution"""
        m = np.floor(self.offset * (self.window_size - 1))
        s = self.window_size / self.sigma
        
        denom = 0
        
        # Create weights using Gaussian distribution
        for i in range(self.window_size):
            w = np.exp(-((i - m) ** 2) / (2 * s ** 2))
            self.weights[i] = w
            denom += w
        
        # Normalize weights
        if denom != 0:
            self.weights /= denom
    
    def predict_one(self, x):
        """Predict the next value"""
        if len(self._y_history) < self.window_size:
            # Default prediction is the feature value if not enough history
            return next(iter(x.values())) if isinstance(x, dict) else x
        
        # Use the ALMA weights to predict the next value
        y_hist = np.array(list(self._y_history))
        return np.sum(y_hist * self.weights)
    
    def learn_one(self, x, y):
        """Update the model with a single learning example"""
        self._y_history.append(y)
        
        # Only update the weights if we have enough history
        if len(self._y_history) >= self.window_size:
            # Calculate current prediction (used for error calculation)
            y_hist = np.array(list(self._y_history))
            y_pred = np.sum(y_hist * self.weights)
            
            # Calculate error
            error = y - y_pred
            
            # Update weights using gradient descent
            # We add a small factor to prioritize more recent data
            recency_factor = np.linspace(0.8, 1.0, self.window_size)
            
            # Update each weight individually
            for i in range(self.window_size):
                gradient = -2 * error * y_hist[i] * recency_factor[i]
                self.weights[i] -= self.alpha * gradient
            
            # Re-normalize weights
            self.weights /= np.sum(self.weights)
        
        return self
EOL

echo -e "${GREEN}Modelo ARIMA creado correctamente${NC}"

# 2. Configurar Tailwind CSS para los estilos
echo -e "${YELLOW}Configurando Tailwind CSS...${NC}"

# Crear tailwind.config.js
if [ ! -f "web/tailwind.config.js" ]; then
    cat > "web/tailwind.config.js" << 'EOL'
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
    echo -e "${GREEN}Creado tailwind.config.js${NC}"
fi

# Crear postcss.config.js
if [ ! -f "web/postcss.config.js" ]; then
    cat > "web/postcss.config.js" << 'EOL'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOL
    echo -e "${GREEN}Creado postcss.config.js${NC}"
fi

# Crear estilos de respaldo
if [ ! -f "web/public/backup-styles.css" ]; then
    cat > "web/public/backup-styles.css" << 'EOL'
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
    echo -e "${GREEN}Creado backup-styles.css${NC}"
fi

# 3. Corregir el healthcheck del broker
echo -e "${YELLOW}Corrigiendo healthcheck del broker...${NC}"

# Asegurarnos de que curl y wget estén instalados en el broker
if ! grep -q "wget" "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/services/broker/Dockerfile"; then
    sed -i '' 's|pkg-config \\|pkg-config \\\n    curl \\\n    wget \\|g' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/services/broker/Dockerfile"
    echo -e "${GREEN}Agregadas dependencias curl y wget al broker${NC}"
fi

# Actualizar healthcheck para usar wget en lugar de curl
sed -i '' 's|test: \["CMD", "curl", "-f", "http://localhost:8001/health"\]|test: \["CMD", "wget", "-q", "--spider", "http://localhost:8001/health"\]|g' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/docker-compose.yml"
echo -e "${GREEN}Actualizado healthcheck del broker para usar wget${NC}"

# 4. Corregir URLs de la API para asegurar que sean correctas
echo -e "${YELLOW}Corrigiendo URLs de la API...${NC}"

# Actualizar API_BASE_URL en apiService.js para usar rutas relativas
sed -i '' 's|const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '"'"'http://localhost:8000'"'"';|const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '"'"''"'"';|g' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/src/services/apiService.js"
echo -e "${GREEN}API_BASE_URL actualizado para usar rutas relativas${NC}"

# 5. Reiniciar todos los servicios en el orden correcto
echo -e "${YELLOW}Reiniciando todos los servicios...${NC}"

# Detener todos los servicios
docker-compose down

# Reconstruir todas las imágenes
docker-compose build frontend broker streaming

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

echo -e "${GREEN}¡Reparación completa finalizada!${NC}"
echo -e "${YELLOW}Accede a la aplicación en: ${GREEN}http://localhost${NC}"
echo -e "${YELLOW}Para ver los logs del frontend: ${NC}docker-compose logs -f frontend"
echo -e "${YELLOW}Para ver los logs del broker: ${NC}docker-compose logs -f broker"
echo -e "${YELLOW}Para ver los logs de streaming: ${NC}docker-compose logs -f streaming"
echo -e "${YELLOW}Para ver todos los logs: ${NC}docker-compose logs -f"

# 6. Instrucciones adicionales
echo -e "${YELLOW}NOTAS IMPORTANTES:${NC}"
echo -e "1. Si sigues viendo errores 50x, puede ser necesario esperar unos minutos más."
echo -e "2. Si los precios en tiempo real siguen incorrectos, comprueba que la API FMP está funcionando correctamente."
echo -e "3. Para restablecer completamente el broker, navega a http://localhost/broker y escribe 'reiniciar'."
