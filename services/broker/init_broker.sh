#!/bin/bash
set -e

# Configuración de variables de entorno con valores por defecto
KAFKA_HOST=${KAFKA_HOST:-kafka}
KAFKA_PORT=${KAFKA_PORT:-9092}
POSTGRES_HOST=${POSTGRES_HOST:-postgres}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres}
POSTGRES_DB=${POSTGRES_DB:-trading}
MAX_RETRIES=${MAX_RETRIES:-30}
INITIAL_BACKOFF=${INITIAL_BACKOFF:-2}

echo "Iniciando servicio broker..."
echo "KAFKA_HOST=$KAFKA_HOST"
echo "POSTGRES_HOST=$POSTGRES_HOST"

# Verificar e instalar dependencias necesarias
echo "Verificando dependencias..."
pip install --no-cache-dir -r requirements.txt

# Verificar si existe el directorio de modelos y crearlo si no existe
MODEL_STORAGE_PATH=${MODEL_STORAGE_PATH:-"/app/models"}
if [ ! -d "$MODEL_STORAGE_PATH" ]; then
    echo "Creando directorio de modelos en $MODEL_STORAGE_PATH"
    mkdir -p "$MODEL_STORAGE_PATH"
fi

# Función para verificar si Kafka está disponible
check_kafka() {
    python -c "
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import time
import socket

def is_port_open(host, port, timeout=5):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

try:
    # Primero comprobar si el puerto está abierto
    if not is_port_open('$KAFKA_HOST', $KAFKA_PORT):
        print('Puerto de Kafka no accesible')
        exit(1)
    
    # Luego intentar crear un productor para verificar la conexión
    producer = KafkaProducer(bootstrap_servers=['$KAFKA_HOST:$KAFKA_PORT'], 
                            api_version=(0, 10, 1),
                            request_timeout_ms=5000,
                            security_protocol='PLAINTEXT')
    producer.close()
    print('Kafka está disponible')
    exit(0)
except NoBrokersAvailable:
    print('No se puede conectar a Kafka: NoBrokersAvailable')
    exit(1)
except Exception as e:
    print(f'Error conectando a Kafka: {e}')
    exit(1)
"
    return $?
}

# Función para verificar si PostgreSQL está disponible
check_postgres() {
    python -c "
import psycopg2
import socket

def is_port_open(host, port, timeout=5):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

try:
    # Primero comprobar si el puerto está abierto
    if not is_port_open('$POSTGRES_HOST', $POSTGRES_PORT):
        print('Puerto de PostgreSQL no accesible')
        exit(1)
    
    # Luego intentar conectarse a la base de datos
    conn = psycopg2.connect(
        host='$POSTGRES_HOST',
        port=$POSTGRES_PORT,
        user='$POSTGRES_USER',
        password='$POSTGRES_PASSWORD',
        dbname='$POSTGRES_DB',
        connect_timeout=5
    )
    conn.close()
    print('PostgreSQL está disponible')
    exit(0)
except Exception as e:
    print(f'Error conectando a PostgreSQL: {e}')
    exit(1)
"
    return $?
}

# Esperar a que Kafka esté disponible usando backoff exponencial
echo "Esperando a que Kafka esté disponible en $KAFKA_HOST:$KAFKA_PORT"
backoff=$INITIAL_BACKOFF
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    if check_kafka; then
        echo "Kafka está disponible después de $retry_count intentos"
        break
    fi
    
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $MAX_RETRIES ]; then
        echo "Error: No se pudo conectar a Kafka después de $MAX_RETRIES intentos"
        echo "Procediendo de todos modos para permitir que el servicio funcione en modo de respaldo"
    else
        echo "Reintentando en $backoff segundos (intento $retry_count de $MAX_RETRIES)..."
        sleep $backoff
        backoff=$((backoff * 2))
    fi
done

# Esperar a que PostgreSQL esté disponible
echo "Esperando a que PostgreSQL esté disponible en $POSTGRES_HOST:$POSTGRES_PORT"
backoff=$INITIAL_BACKOFF
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    if check_postgres; then
        echo "PostgreSQL está disponible después de $retry_count intentos"
        break
    fi
    
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $MAX_RETRIES ]; then
        echo "Error: No se pudo conectar a PostgreSQL después de $MAX_RETRIES intentos"
        echo "Procediendo de todos modos para permitir que el servicio funcione en modo de respaldo"
    else
        echo "Reintentando en $backoff segundos (intento $retry_count de $MAX_RETRIES)..."
        sleep $backoff
        backoff=$((backoff * 2))
    fi
done

# Crear archivo config.py si no existe
if [ ! -f "config.py" ]; then
    echo "Creando archivo config.py..."
    cat > config.py << EOL
# Configuración para el servicio broker
KAFKA_CONFIG = {
    'bootstrap_servers': ['$KAFKA_HOST:$KAFKA_PORT'],
    'api_version': (0, 10, 1),
    'client_id': 'broker-service',
    'auto_offset_reset': 'latest',
    'security_protocol': 'PLAINTEXT'
}

# Configuración de PostgreSQL
DB_CONFIG = {
    'host': '$POSTGRES_HOST',
    'port': $POSTGRES_PORT,
    'user': '$POSTGRES_USER',
    'password': '$POSTGRES_PASSWORD',
    'dbname': '$POSTGRES_DB'
}

# Configuración API
API_HOST = '0.0.0.0'
API_PORT = 8001
API_DEBUG = False
API_WORKERS = 4

# Configuración de tópicos Kafka
TOPICS = {
    'input': 'streaming_events',
    'output': 'broker_events',
    'batch_updates': 'batch_events'
}

# Configuración de modelos
MODEL_STORAGE_PATH = '$MODEL_STORAGE_PATH'
DEFAULT_ENSEMBLE_WEIGHTS = {
    'random_forest': 0.4,
    'gradient_boosting': 0.4,
    'elastic_net': 0.2
}

# Configuración de cartera (inicial)
INITIAL_CASH = 100000.0
RISK_FACTOR = 0.02  # 2% de riesgo por operación 

# Configuración de logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configuración de caché
CACHE_EXPIRY = 300  # segundos

# Configuración de fallback cuando los servicios están caídos
FALLBACK_ENABLED = True
EOL
fi

# Descargar modelo de LLM si es necesario
LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-"$MODEL_STORAGE_PATH/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"}
if [ ! -f "$LLAMA_MODEL_PATH" ]; then
    echo "Descargando modelo TinyLlama..."
    mkdir -p "$(dirname "$LLAMA_MODEL_PATH")"
    # Usar curl para descargar el modelo (URL de ejemplo)
    if curl -L "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -o "$LLAMA_MODEL_PATH"; then
        echo "Modelo descargado correctamente"
    else
        echo "Error al descargar el modelo LLM, creando archivo de placeholder..."
        echo "ERROR: Modelo no disponible, por favor descargue manualmente" > "$LLAMA_MODEL_PATH"
    fi
fi

# Comprobar si tenemos que crear tablas en la base de datos
echo "Verificando tablas en la base de datos..."
python -c "
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    # Conectar a la base de datos
    conn = psycopg2.connect(
        host='$POSTGRES_HOST',
        port=$POSTGRES_PORT,
        user='$POSTGRES_USER',
        password='$POSTGRES_PASSWORD',
        dbname='$POSTGRES_DB'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Verificar si existe la tabla 'portfolio'
    cursor.execute(\"\"\"
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name = 'portfolio'
        );
    \"\"\")
    table_exists = cursor.fetchone()[0]
    
    # Si la tabla no existe, crearla junto con otras tablas necesarias
    if not table_exists:
        print('Creando tablas necesarias...')
        
        # Crear tabla de portfolio
        cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS portfolio (
                id SERIAL PRIMARY KEY,
                cash DECIMAL(15,2) NOT NULL,
                initial_cash DECIMAL(15,2) NOT NULL,
                last_update TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        \"\"\")
        
        # Crear tabla de posiciones
        cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolio(id),
                symbol VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                average_price DECIMAL(15,2) NOT NULL,
                current_price DECIMAL(15,2) NOT NULL,
                last_update TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        \"\"\")
        
        # Crear tabla de órdenes
        cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                action VARCHAR(4) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(15,2) NOT NULL,
                total_value DECIMAL(15,2) NOT NULL,
                profit DECIMAL(15,2),
                status VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        \"\"\")
        
        # Insertar portfolio inicial
        cursor.execute(\"\"\"
            INSERT INTO portfolio (cash, initial_cash)
            VALUES ($INITIAL_CASH, $INITIAL_CASH);
        \"\"\")
        
        print('Tablas creadas exitosamente')
    else:
        print('Las tablas ya existen')
    
    # Cerrar conexión
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f'Error al verificar/crear tablas: {e}')
"

# Iniciar aplicación principal
echo "Iniciando aplicación principal broker con FastAPI..."
exec uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4 --reload
