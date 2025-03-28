#!/bin/sh
set -e

echo "Starting streaming service initialization..."

# Create necessary directories if they don't exist
mkdir -p /app/logs /app/models

# Check environment variables
echo "Checking environment variables..."
echo "KAFKA_BROKER: $KAFKA_BROKER"
echo "INGESTION_TOPIC: $INGESTION_TOPIC"
echo "STREAMING_TOPIC: $STREAMING_TOPIC"
echo "FMP_API_KEY: [HIDDEN]" # Security measure: don't show API key

# Check if required modules are installed
echo "Checking required modules..."
if ! python -c "import flask, werkzeug" 2>/dev/null; then
    echo "Installing missing dependencies..."
    pip install Flask==2.2.3 Werkzeug==2.2.3
fi

# Wait for Kafka to be ready
echo "Waiting for Kafka to be available..."
timeout=60
counter=0
while ! nc -z -v kafka 9092 2>/dev/null && [ $counter -lt $timeout ]; do
    echo "Waiting for Kafka... ($counter/$timeout)"
    sleep 1
    counter=$((counter+1))
done

if [ $counter -eq $timeout ]; then
    echo "Error: Kafka is not available after $timeout seconds"
    exit 1
fi

echo "Kafka is available! Starting the application..."

# Find where app.py is located
echo "Looking for app.py..."
find /app -name "app.py" | sort

# Verify custom_models.py exists
if [ ! -f "/app/custom_models.py" ]; then
    echo "custom_models.py not found, creating it..."
    cat > /app/custom_models.py << 'EOF'
from river import base
import numpy as np

class ALMARegressor(base.Regressor):
    """
    Arnaud Legoux Moving Average Regressor.
    
    A regression model based on the ALMA indicator commonly used in technical analysis.
    The ALMA indicator provides a smoother price line than other moving averages while
    maintaining better responsiveness to price changes.
    
    Parameters
    ----------
    alpha: float
        The smoothing factor. Higher values make the moving average more responsive,
        but less smooth. Default is 0.1.
    window_size: int
        The number of observations to consider. Default is 10.
    sigma: float
        Controls the smoothness of the ALMA. Default is 6.
    offset: float
        Controls the responsiveness of the ALMA. Default is 0.85.
    """
    
    def __init__(self, alpha=0.1, window_size=10, sigma=6, offset=0.85):
        self.alpha = alpha
        self.window_size = window_size
        self.sigma = sigma
        self.offset = offset
        self.weights = self._calculate_weights()
        self.buffer = []  # Store last 'window_size' observations
        self._last_prediction = 0
        
    def _calculate_weights(self):
        """Calculate ALMA weights for the given parameters."""
        m = np.floor(self.offset * (self.window_size - 1))
        s = self.window_size / self.sigma
        weights = np.zeros(self.window_size)
        
        for i in range(self.window_size):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s ** 2))
            
        # Normalize weights
        weights /= np.sum(weights)
        return weights
        
    def learn_one(self, x, y):
        """Update the model with a single observation."""
        # Add new observation to buffer
        self.buffer.append(y)
        
        # Keep only the last window_size observations
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
            
        # Update last prediction if we have enough data
        if len(self.buffer) == self.window_size:
            weighted_sum = sum(w * val for w, val in zip(self.weights, self.buffer))
            self._last_prediction = weighted_sum
            
        return self
        
    def predict_one(self, x):
        """Predict the target value for a single observation."""
        # If we don't have enough data, use simple average
        if len(self.buffer) < self.window_size:
            if not self.buffer:
                return 0
            return sum(self.buffer) / len(self.buffer)
            
        return self._last_prediction
EOF
fi

# Verify config.py exists
if [ ! -f "/app/config.py" ]; then
    echo "config.py not found, creating it..."
    cat > /app/config.py << 'EOF'
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Kafka configuration
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
INGESTION_TOPIC = os.getenv('INGESTION_TOPIC', 'ingestion_events')
STREAMING_TOPIC = os.getenv('STREAMING_TOPIC', 'streaming_events')

# API configuration
FMP_API_KEY = os.getenv('FMP_API_KEY', '')
FMP_BASE_URL = os.getenv('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Model configuration
MODEL_OUTPUT_PATH = os.getenv('MODEL_OUTPUT_PATH', '/models')
EOF
fi

# Try to run the main application, if it fails use the fallback
if [ -f "/app/app.py" ]; then
    echo "Trying to run /app/app.py..."
    if python -c "import sys; from importlib.util import find_spec; sys.exit(0 if find_spec('flask') and find_spec('werkzeug') else 1)" 2>/dev/null; then
        echo "Running main app.py..."
        python /app/app.py || {
            echo "Main app failed, switching to fallback..."
            if [ -f "/app/app_fallback.py" ]; then
                python /app/app_fallback.py
            else
                echo "ERROR: Fallback app not found!"
                exit 1
            fi
        }
    else
        echo "Flask or Werkzeug not properly installed, using fallback app..."
        if [ -f "/app/app_fallback.py" ]; then
            python /app/app_fallback.py
        else
            echo "ERROR: Fallback app not found!"
            exit 1
        fi
    fi
elif [ -f "/app/services/streaming/app.py" ]; then
    echo "Trying to run /app/services/streaming/app.py..."
    python /app/services/streaming/app.py || {
        echo "Main app failed, switching to fallback..."
        if [ -f "/app/app_fallback.py" ]; then
            python /app/app_fallback.py
        else
            echo "ERROR: Fallback app not found!"
            exit 1
        fi
    }
else
    echo "Error: app.py not found! Trying fallback..."
    if [ -f "/app/app_fallback.py" ]; then
        python /app/app_fallback.py
    else
        echo "ERROR: Neither main nor fallback app found!"
        exit 1
    fi
fi
