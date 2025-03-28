# AI Broker Agent

Este módulo proporciona un agente IA para analizar predicciones de modelos ensemble y proporcionar estrategias de trading.

## Características

- **Chat interactivo**: Interfaz de conversación natural para consultar estrategias de inversión
- **Análisis en tiempo real**: Incorpora datos de mercado actualizados
- **Optimizado para 16GB RAM**: Utiliza modelos LLM livianos (Llama 3.2 1B Instruct)
- **Integración con modelos ensemble**: Combina predicciones de modelos online y offline
- **Análisis técnico automatizado**: Cálculo de indicadores y patrones de mercado
- **Estrategias personalizadas**: Recomendaciones específicas por símbolo

## Uso

### Chat con el Broker

Puedes interactuar con el broker IA a través de la interfaz web o mediante la API REST:

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué estrategia recomiendas para AAPL?"}'
```

### Ejemplos de consultas

- "¿Qué estrategia recomiendas para TSLA?"
- "¿Cómo está mi portafolio actual?"
- "Muéstrame mis métricas de riesgo"
- "Quiero comprar 10 acciones de AMZN"
- "¿Cuál es la predicción para GOOG?"
- "Dame datos de mercado de MSFT"

## API REST

### Endpoints principales

- `/chat` - Interacción con el broker IA
- `/portfolio` - Estado actual del portafolio
- `/orders` - Historial y creación de órdenes
- `/metrics` - Métricas de rendimiento y riesgo
- `/market-data/{symbol}` - Datos de mercado de un símbolo
- `/predictions/{symbol}` - Predicciones de modelos para un símbolo
- `/strategy/{symbol}` - Estrategia de inversión para un símbolo

## Arquitectura

El broker IA combina varias tecnologías:

1. **LLM para interfaz conversacional**: Procesamiento de lenguaje natural para entender consultas
2. **Modelos Ensemble**: Combinación ponderada de predicciones de múltiples modelos
3. **Transfer Learning**: Transferencia de conocimiento entre modelos online y offline
4. **Análisis Técnico**: Cálculo automático de indicadores técnicos
5. **Gestión de Portafolio**: Simulación y seguimiento de posiciones y rentabilidad

## Requisitos

- Docker y Docker Compose
- 16GB RAM mínimo recomendado
- Conexión a Internet para datos de mercado en tiempo real

## Configuración

Las principales opciones de configuración están disponibles como variables de entorno:

- `LLAMA_MODEL_PATH` - Ruta al modelo LLM (por defecto usa LLama 3.2 1B)
- `INITIAL_CASH` - Efectivo inicial para el portafolio (por defecto 100,000)
- `FMP_API_KEY` - API key para Financial Modeling Prep
- `KAFKA_BROKER` - Dirección del broker Kafka

## Inicio rápido

```bash
# Iniciar el servicio de broker IA
cd project7
docker-compose up broker
```

Accede a la interfaz web en http://localhost:8001/docs o interactúa a través de la interfaz de chat.
