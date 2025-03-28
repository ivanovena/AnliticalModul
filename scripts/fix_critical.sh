#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Aplicando correcciones críticas...${NC}"

# 1. Corregir ruta streaming en nginx.conf
echo -e "${YELLOW}Corrigiendo configuración de nginx para streaming...${NC}"

cat > "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/web/nginx/nginx.conf" << 'EOL'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip Settings
    gzip on;
    gzip_disable "msie6";
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_min_length 256;
    gzip_types
        application/atom+xml
        application/javascript
        application/json
        application/ld+json
        application/manifest+json
        application/rss+xml
        application/vnd.geo+json
        application/vnd.ms-fontobject
        application/x-font-ttf
        application/x-web-app-manifest+json
        application/xhtml+xml
        application/xml
        font/opentype
        image/bmp
        image/svg+xml
        image/x-icon
        text/cache-manifest
        text/css
        text/plain
        text/vcard
        text/vnd.rim.location.xloc
        text/vtt
        text/x-component
        text/x-cross-domain-policy;

    # Cache control for static assets
    location ~* \.(jpg|jpeg|png|gif|ico|css|js|svg)$ {
        expires 30d;
        add_header Cache-Control "public, no-transform";
        # Asegurarnos de que la cabecera content-type está correctamente configurada
        types {
            text/css css;
            application/javascript js;
            image/svg+xml svg;
            image/jpeg jpg jpeg;
            image/png png;
            image/gif gif;
            image/x-icon ico;
        }
    }

    # API Proxy para el broker
    location /api/ {
        proxy_pass http://broker:8001/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # API Proxy para ingestion
    location /ingestion/ {
        proxy_pass http://ingestion:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API Proxy para streaming - CORREGIDO PARA USAR EL PUERTO 8090
    location /streaming/ {
        proxy_pass http://streaming:8090/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Endpoint directo para predicciones
    location /prediction/ {
        proxy_pass http://streaming:8090/prediction/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Endpoint específico para el chat
    location /chat {
        proxy_pass http://broker:8001/chat;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket Proxy
    location /ws/ {
        proxy_pass http://broker:8001/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # React Router Support
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-XSS-Protection "1; mode=block";
    add_header X-Content-Type-Options "nosniff";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' ws: wss: http: https:;";

    # Error handling
    error_page 404 /index.html;
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
EOL

echo -e "${GREEN}Configuración de nginx actualizada${NC}"

# 2. Corregir el healthcheck del broker
echo -e "${YELLOW}Corrigiendo healthcheck del broker en docker-compose.yml...${NC}"

# Cambiar a un healthcheck más básico usando GET en lugar de HEAD
sed -i '' 's|test: \["CMD", "wget", "-q", "--spider", "http://localhost:8001/health"\]|test: \["CMD", "curl", "-f", "http://localhost:8001/health"\]|g' "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/docker-compose.yml"

echo -e "${GREEN}Healthcheck del broker actualizado${NC}"

# 3. Corregir el custom_models.py para el servicio streaming
echo -e "${YELLOW}Corrigiendo custom_models.py con implementación adecuada de Window...${NC}"

cat > "/Users/ivanovena/Documents/Documentos personales/Model bursátil/project7/services/streaming/custom_models.py" << 'EOL'
import numpy as np
from river import base

# Implementar nuestra propia clase Window
class Window:
    """Implementación de ventana deslizante para almacenar datos históricos."""
    
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

echo -e "${GREEN}custom_models.py corregido${NC}"

# 4. Reiniciar los servicios afectados
echo -e "${YELLOW}Reiniciando servicios...${NC}"

docker-compose stop frontend broker streaming
docker-compose rm -f frontend broker streaming

echo -e "${YELLOW}Reconstruyendo imágenes...${NC}"
docker-compose build broker streaming frontend

echo -e "${YELLOW}Iniciando servicios...${NC}"
docker-compose up -d broker streaming frontend

echo -e "${YELLOW}Esperando 30 segundos para inicialización completa...${NC}"
sleep 30

# Verificar estado
echo -e "${YELLOW}Verificando estado de los servicios...${NC}"
docker ps

echo -e "${GREEN}¡Correcciones críticas aplicadas!${NC}"
echo -e "${YELLOW}Accede a http://localhost para verificar los resultados${NC}"
