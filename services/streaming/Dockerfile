FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Instalar explícitamente fastapi y sus dependencias
RUN pip install --no-cache-dir fastapi uvicorn websockets

# Copy application code
COPY . .

# Fix for rolling.Window issue
RUN sed -i 's/self._y_history = rolling.Window(size=window_size)/self._y_history = []\n        self._max_window_size = window_size/' custom_models.py
RUN sed -i 's/y_hist = np.array(list(self._y_history))/y_hist = np.array(self._y_history)/' custom_models.py
RUN sed -i 's/self._y_history.append(y)/self._y_history.append(y)\n        # Maintain the window size by removing old elements\n        if len(self._y_history) > self._max_window_size:\n            self._y_history.pop(0)/' custom_models.py

# Comentar el HEALTHCHECK aquí, se define en docker-compose.yml
# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
#   CMD python -c \"import requests; requests.get('http://localhost:8090/health', timeout=5)\" || exit 1

# Command to run the FastAPI application using Uvicorn on port 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
