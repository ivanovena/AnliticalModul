#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Directorio del proyecto
PROJECT_ROOT=$(cd $(dirname $0)/..; pwd)
cd $PROJECT_ROOT

# Verificar que .env existe
if [ ! -f ".env" ]; then
  echo -e "${RED}Error: Archivo .env no encontrado.${NC}"
  echo -e "Por favor, crea el archivo .env basado en .env.example"
  exit 1
fi

# Cargar variables de entorno
source .env

# Verificar variables críticas
if [ -z "$FMP_API_KEY" ]; then
  echo -e "${RED}Error: FMP_API_KEY no está configurada en .env${NC}"
  exit 1
fi

# Crear directorios necesarios
mkdir -p logs/{ingestion,broker,streaming,batch} models

# Funciones para cada servicio
run_ingestion() {
  echo -e "${YELLOW}Iniciando servicio de ingestion...${NC}"
  cd $PROJECT_ROOT/services/ingestion
  python app.py > $PROJECT_ROOT/logs/ingestion/ingestion.log 2>&1 &
  echo $! > $PROJECT_ROOT/logs/ingestion/ingestion.pid
  echo -e "${GREEN}Servicio de ingestion iniciado. PID: $(cat $PROJECT_ROOT/logs/ingestion/ingestion.pid)${NC}"
}

run_streaming() {
  echo -e "${YELLOW}Iniciando servicio de streaming...${NC}"
  cd $PROJECT_ROOT/services/streaming
  python app.py > $PROJECT_ROOT/logs/streaming/streaming.log 2>&1 &
  echo $! > $PROJECT_ROOT/logs/streaming/streaming.pid
  echo -e "${GREEN}Servicio de streaming iniciado. PID: $(cat $PROJECT_ROOT/logs/streaming/streaming.pid)${NC}"
}

run_batch() {
  echo -e "${YELLOW}Iniciando servicio de batch...${NC}"
  cd $PROJECT_ROOT/services/batch
  python app.py > $PROJECT_ROOT/logs/batch/batch.log 2>&1 &
  echo $! > $PROJECT_ROOT/logs/batch/batch.pid
  echo -e "${GREEN}Servicio de batch iniciado. PID: $(cat $PROJECT_ROOT/logs/batch/batch.pid)${NC}"
}

run_broker() {
  echo -e "${YELLOW}Iniciando servicio de broker...${NC}"
  cd $PROJECT_ROOT/services/broker
  uvicorn app:app --host 0.0.0.0 --port 8000 > $PROJECT_ROOT/logs/broker/broker.log 2>&1 &
  echo $! > $PROJECT_ROOT/logs/broker/broker.pid
  echo -e "${GREEN}Servicio de broker iniciado. PID: $(cat $PROJECT_ROOT/logs/broker/broker.pid)${NC}"
}

run_web() {
  echo -e "${YELLOW}Iniciando servicio web...${NC}"
  cd $PROJECT_ROOT/web
  npm start > $PROJECT_ROOT/logs/web/web.log 2>&1 &
  echo $! > $PROJECT_ROOT/logs/web/web.pid
  echo -e "${GREEN}Servicio web iniciado. PID: $(cat $PROJECT_ROOT/logs/web/web.pid)${NC}"
}

stop_services() {
  echo -e "${YELLOW}Deteniendo servicios...${NC}"
  
  # Verificar servicios en ejecución
  for service in ingestion streaming batch broker web; do
    pid_file="$PROJECT_ROOT/logs/$service/$service.pid"
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file")
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Deteniendo $service (PID: $pid)...${NC}"
        kill $pid
        rm "$pid_file"
      else
        echo -e "${YELLOW}El servicio $service no está en ejecución.${NC}"
        rm "$pid_file"
      fi
    else
      echo -e "${YELLOW}No se encontró PID para $service.${NC}"
    fi
  done
  
  echo -e "${GREEN}Todos los servicios han sido detenidos.${NC}"
}

show_status() {
  echo -e "${YELLOW}Estado de los servicios:${NC}"
  
  for service in ingestion streaming batch broker web; do
    pid_file="$PROJECT_ROOT/logs/$service/$service.pid"
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file")
      if ps -p $pid > /dev/null; then
        echo -e "${GREEN}$service: En ejecución (PID: $pid)${NC}"
      else
        echo -e "${RED}$service: Detenido (PID inválido: $pid)${NC}"
      fi
    else
      echo -e "${RED}$service: No iniciado${NC}"
    fi
  done
}

view_logs() {
  service=$1
  if [ -z "$service" ]; then
    echo -e "${RED}Especifique un servicio: ingestion, streaming, batch, broker, web${NC}"
    return
  fi
  
  log_file="$PROJECT_ROOT/logs/$service/$service.log"
  if [ -f "$log_file" ]; then
    tail -f "$log_file"
  else
    echo -e "${RED}Archivo de log no encontrado: $log_file${NC}"
  fi
}

# Función principal
case "$1" in
  start)
    # Iniciar servicios en el orden correcto
    run_ingestion
    sleep 2
    run_streaming
    sleep 2
    run_batch
    sleep 2
    run_broker
    sleep 5
    run_web
    echo -e "${GREEN}Todos los servicios iniciados. Accede a la interfaz web en http://localhost:3000${NC}"
    ;;
  
  stop)
    stop_services
    ;;
  
  restart)
    stop_services
    sleep 5
    # Iniciar servicios en el orden correcto
    run_ingestion
    sleep 2
    run_streaming
    sleep 2
    run_batch
    sleep 2
    run_broker
    sleep 5
    run_web
    echo -e "${GREEN}Todos los servicios reiniciados. Accede a la interfaz web en http://localhost:3000${NC}"
    ;;
  
  status)
    show_status
    ;;
  
  logs)
    view_logs $2
    ;;
  
  *)
    echo -e "${YELLOW}Uso: $0 {start|stop|restart|status|logs [service]}${NC}"
    echo -e "  start   - Inicia todos los servicios"
    echo -e "  stop    - Detiene todos los servicios"
    echo -e "  restart - Reinicia todos los servicios"
    echo -e "  status  - Muestra el estado de los servicios"
    echo -e "  logs    - Muestra los logs de un servicio específico"
    echo -e "            use: $0 logs [ingestion|streaming|batch|broker|web]"
    exit 1
    ;;
esac

exit 0
