#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Corrigiendo problemas de healthcheck y conexión entre servicios...${NC}"

# 1. Corregir el healthcheck del broker
echo -e "${YELLOW}Corrigiendo el healthcheck del broker...${NC}"

# Cambiar método HEAD por GET en el healthcheck del broker
sed -i '' 's|test: \["CMD", "wget", "-q", "--spider", "http://localhost:8001/health"\]|test: \["CMD", "wget", "-q", "-O", "/dev/null", "http://localhost:8001/health"\]|g' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/docker-compose.yml"

# 2. Corregir el rolling.Window en custom_models.py
echo -e "${YELLOW}Corrigiendo custom_models.py para streaming...${NC}"

# Crear la implementación correcta de la clase Window para rolling
cat > "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/services/streaming/custom_models.py" << 'EOL'
import numpy as np
from river import base

# Implementación de Window para rolling
class Window:
    """Rolling window."""

    def __init__(self, size):
        self.size = size
        self.items = []

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def append(self, x):
        self.items.append(x)
        if len(self.items) > self.size:
            self.items.pop(0)
        return self

# Namespace para mantener compatibilidad de importación
class rolling:
    Window = Window

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
        self._y_history = Window(size=window_size)
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

echo -e "${GREEN}Corregido custom_models.py${NC}"

# 3. Corregir la configuración de Nginx para conectar correctamente a streaming
echo -e "${YELLOW}Corrigiendo configuración de NGINX para conectar a streaming...${NC}"

sed -i '' 's|proxy_pass http://streaming:8001/;|proxy_pass http://streaming:8090/;|g' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/nginx/nginx.conf"

echo -e "${GREEN}Actualizada configuración de NGINX${NC}"

# 4. Reiniciar los servicios
echo -e "${YELLOW}Reiniciando servicios...${NC}"

docker-compose stop frontend broker streaming
docker-compose rm -f frontend broker streaming
docker-compose build broker streaming frontend
docker-compose up -d broker streaming frontend

echo -e "${YELLOW}Esperando a que los servicios se inicien (30s)...${NC}"
sleep 30

echo -e "${YELLOW}Verificando estado...${NC}"
docker ps | grep "broker\|frontend\|streaming"

echo -e "${GREEN}¡Correcciones aplicadas!${NC}"
echo -e "${YELLOW}Prueba acceder a http://localhost para verificar que todo funciona correctamente${NC}"
