# Frontend de la Plataforma de Trading - Documentación

Este documento detalla la arquitectura del frontend de la plataforma de trading, sus componentes principales y cómo conectarlos con los microservicios de backend.

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Componentes Principales](#componentes-principales)
4. [Contextos y Gestión de Estado](#contextos-y-gestión-de-estado)
5. [Servicios API](#servicios-api)
6. [Conectando con Microservicios](#conectando-con-microservicios)
7. [Configuración y Despliegue](#configuración-y-despliegue)
8. [Solución de Problemas](#solución-de-problemas)

## Descripción General

El frontend es una aplicación React moderna construida con:
- React Router para navegación
- Tailwind CSS para estilos
- Recharts para visualizaciones
- Contextos para gestión de estado
- WebSockets para actualizaciones en tiempo real

Proporciona una interfaz profesional para el análisis de mercado, predicciones de precios, simulación de trading, y comunicación con un broker IA.

## Estructura del Proyecto

```
/web
├── public/                # Archivos estáticos
├── src/
│   ├── components/        # Componentes reutilizables
│   │   ├── charts/        # Componentes de gráficos
│   │   ├── layouts/       # Layouts de la aplicación
│   │   ├── models/        # Componentes relacionados con modelos de ML
│   │   ├── trading/       # Componentes de trading
│   │   └── ui/            # Componentes de UI básicos
│   ├── contexts/          # Contextos para gestión de estado
│   ├── hooks/             # Hooks personalizados
│   ├── pages/             # Páginas principales
│   ├── services/          # Servicios API
│   ├── utils/             # Utilidades
│   ├── App.jsx            # Componente principal
│   └── index.js           # Punto de entrada
├── nginx/                 # Configuración de Nginx
│   └── nginx.conf         # Configuración para proxy y rutas
├── Dockerfile             # Docker para entorno de producción
├── package.json           # Dependencias
└── tailwind.config.js     # Configuración de Tailwind
```

## Componentes Principales

### Pages

#### 1. Dashboard (`/src/pages/Dashboard.jsx`)

El Dashboard es la página principal de la aplicación, mostrando:
- Resumen de cartera
- Gráfico de evolución del valor
- Predicciones destacadas
- Enlaces rápidos a otras secciones

#### 2. TradingSimulator (`/src/pages/TradingSimulator.jsx`)

El simulador de trading profesional incluye:
- Gráfico de precios en tiempo real con predicciones
- Panel de cartera
- Panel de órdenes (compra/venta)
- Tabla de predicciones con verificación de precisión
- Historial de operaciones
- Indicador de salud de modelos

#### 3. BrokerChat (`/src/pages/BrokerChat.jsx`)

Interfaz para comunicarse con el broker IA:
- Chat en tiempo real
- Prompts rápidos
- Selección de símbolo para consultas específicas
- Resumen de cartera

#### 4. ModelPerformance (`/src/pages/ModelPerformance.jsx`)

Análisis del rendimiento de los modelos predictivos:
- Estado actual de los modelos (online, batch, ensemble)
- Métricas históricas
- Comparación por horizonte temporal
- Información técnica

#### 5. PredictionAnalysis (`/src/pages/PredictionAnalysis.jsx`)

Análisis detallado de las predicciones:
- Predicciones por símbolo y horizonte temporal
- Verificación de predicciones pasadas
- Gráficos de error por predicción
- Comparación de precisión entre símbolos

#### 6. MarketAnalysis (`/src/pages/MarketAnalysis.jsx`)

Análisis de mercado general:
- Información detallada de símbolos
- Comparación de índices
- Rendimiento por sector
- Gestión de watchlist

### Componentes

#### Componentes de Trading

1. **PortfolioSummary** (`/src/components/trading/PortfolioSummary.jsx`)
   - Muestra el resumen de la cartera, incluyendo efectivo, valor total, y ganancia.

2. **OrderPanel** (`/src/components/trading/OrderPanel.jsx`)
   - Panel para crear órdenes de compra/venta, con soporte para órdenes de mercado y límite.

3. **PredictionTable** (`/src/components/trading/PredictionTable.jsx`)
   - Tabla que muestra predicciones para diferentes horizontes temporales y su verificación.

4. **TransactionHistory** (`/src/components/trading/TransactionHistory.jsx`)
   - Historial de operaciones realizadas.

5. **StrategyAdvisor** (`/src/components/trading/StrategyAdvisor.jsx`)
   - Muestra recomendaciones de estrategia del broker IA.

#### Componentes de Gráficos

1. **TradingChart** (`/src/components/charts/TradingChart.jsx`)
   - Gráfico avanzado de precios que incluye visualización de predicciones.

#### Componentes de Modelos

1. **ModelHealthIndicator** (`/src/components/models/ModelHealthIndicator.jsx`)
   - Indicador del estado de salud de los modelos de ML.

#### Layouts

1. **MainLayout** (`/src/components/layouts/MainLayout.jsx`)
   - Layout principal con sidebar de navegación.

## Contextos y Gestión de Estado

### 1. PortfolioContext (`/src/contexts/PortfolioContext.jsx`)

Gestiona el estado de la cartera:
- Efectivo disponible
- Posiciones actuales
- Métricas de rendimiento
- Historial de transacciones

**API:**
```javascript
const { 
  portfolio,        // Estado actual de la cartera
  metrics,          // Métricas de rendimiento
  transactions,     // Historial de transacciones
  placeOrder        // Función para colocar órdenes
} = usePortfolio();
```

### 2. PredictionContext (`/src/contexts/PredictionContext.jsx`)

Gestiona las predicciones y su verificación:
- Predicciones para diferentes símbolos y horizontes
- Historial de precisión
- Verificación de predicciones pasadas

**API:**
```javascript
const { 
  predictions,           // Predicciones actuales
  verificationResults,   // Resultados de verificación
  activeSymbols,         // Símbolos activos
  fetchPredictions       // Obtener predicciones para un símbolo
} = usePrediction();
```

### 3. MarketDataContext (`/src/contexts/MarketDataContext.jsx`)

Gestiona los datos de mercado:
- Precios y datos históricos
- Watchlist
- Timeframes

**API:**
```javascript
const { 
  marketData,           // Datos de mercado por símbolo
  watchlist,            // Lista de símbolos en seguimiento
  selectedTimeframe,    // Timeframe seleccionado
  changeTimeframe,      // Cambiar timeframe
  fetchMarketData       // Obtener datos para un símbolo
} = useMarketData();
```

### 4. ModelStatusContext (`/src/contexts/ModelStatusContext.jsx`)

Gestiona el estado de los modelos de predicción:
- Estado de salud
- Métricas de rendimiento
- Configuración de health check

**API:**
```javascript
const { 
  modelStatus,            // Estado de los modelos
  fetchModelStatus        // Obtener estado actualizado
} = useModelStatus();
```

### 5. NotificationContext (`/src/contexts/NotificationContext.jsx`)

Gestiona notificaciones del sistema:
- Éxito/error en operaciones
- Alertas de sistema
- Cambios en el estado de los modelos

**API:**
```javascript
const { 
  notifications,           // Lista de notificaciones
  addNotification,         // Añadir notificación
  dismissNotification      // Descartar notificación
} = useNotification();
```

## Servicios API

El archivo `/src/services/api.js` centraliza todas las llamadas a la API, proporcionando:

- Manejo uniforme de errores
- Simulación de datos para desarrollo
- WebSockets para actualizaciones en tiempo real

### Principales endpoints:

```javascript
api.getPortfolio()            // Obtener cartera actual
api.getPortfolioMetrics()     // Obtener métricas de cartera
api.getOrders()               // Obtener historial de órdenes
api.placeOrder(orderData)     // Ejecutar orden
api.getQuote(symbol)          // Obtener cotización de símbolo
api.getHistoricalData(symbol) // Obtener datos históricos
api.getPredictions(symbol)    // Obtener predicciones
api.getStrategy(symbol)       // Obtener estrategia recomendada
api.sendChatMessage(message)  // Enviar mensaje al broker IA
api.getModelStatus()          // Obtener estado de modelos
```

## Conectando con Microservicios

Esta sección explica cómo conectar el frontend con cada microservicio backend.

### Configuración de Proxy en Nginx

La configuración en `nginx/nginx.conf` establece el enrutamiento de las peticiones API a los microservicios correspondientes:

```nginx
# API Proxy para el broker
location /api/ {
    proxy_pass http://broker:8001/;
}

# API Proxy para ingestion
location /ingestion/ {
    proxy_pass http://ingestion:8000/;
}

# API Proxy para streaming
location /streaming/ {
    proxy_pass http://streaming:8090/;
}

# Endpoint directo para predicciones
location /prediction/ {
    proxy_pass http://streaming:8090/prediction/;
}

# Endpoint específico para el chat
location /chat {
    proxy_pass http://broker:8001/chat;
}

# WebSocket Proxy
location /ws/ {
    proxy_pass http://broker:8001/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
}
```

### Conexión con Broker Service

El Broker Service proporciona:
- Gestión de cartera
- Ejecución de órdenes
- Estrategias de trading
- Asistente IA (chat)

**Endpoints principales:**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/portfolio` | GET | Obtener estado de cartera |
| `/api/orders` | GET | Obtener historial de órdenes |
| `/api/orders` | POST | Colocar nueva orden |
| `/api/metrics` | GET | Obtener métricas de cartera |
| `/api/chat` | POST | Interactuar con el broker IA |
| `/api/market-data/:symbol` | GET | Obtener datos para un símbolo |
| `/api/strategy/:symbol` | GET | Obtener estrategia para un símbolo |
| `/ws` | WebSocket | Actualizaciones en tiempo real |

**Ejemplo de uso:**

```javascript
// Obtener cartera
const portfolio = await api.getPortfolio();

// Colocar orden
const orderData = {
  symbol: 'AAPL',
  action: 'buy',
  quantity: 10,
  price: 155.50,
  orderType: 'market'
};
const result = await api.placeOrder(orderData);

// Chat con broker IA
const response = await api.sendChatMessage("¿Qué opinas sobre AAPL?");
```

### Conexión con Streaming Service

El Streaming Service proporciona:
- Predicciones en tiempo real
- Estado de modelos online
- Métricas de rendimiento

**Endpoints principales:**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/prediction/:symbol` | GET | Obtener predicciones para un símbolo |
| `/health` | GET | Verificar estado del servicio con métricas |

**Ejemplo de uso:**

```javascript
// Obtener predicciones
const predictions = await api.getPredictions('AAPL');
```

### Conexión con Ingestion Service

El Ingestion Service proporciona:
- Datos históricos
- Búsqueda de símbolos
- Datos fundamentales

**Endpoints principales:**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/ingestion/historical/:symbol` | GET | Obtener datos históricos |
| `/ingestion/search` | GET | Buscar símbolos |

**Ejemplo de uso:**

```javascript
// Obtener datos históricos
const historicalData = await api.getHistoricalData('AAPL', '1h');
```

## Configuración y Despliegue

### Variables de Entorno

Configure las siguientes variables en `.env`:

```
REACT_APP_API_URL=http://localhost 
REACT_APP_WS_URL=ws://localhost/ws
```

### Compilación para Producción

```bash
# Instalar dependencias
npm install

# Compilar para producción
npm run build

# El resultado estará en la carpeta /build
```

### Docker

El proyecto incluye un Dockerfile para construir una imagen optimizada:

```bash
# Construir imagen
docker build -t stock-market-frontend .

# Ejecutar contenedor
docker run -p 80:80 stock-market-frontend
```

## Solución de Problemas

### Problemas Comunes

**1. No se reciben datos en tiempo real**

Verificar:
- Que el servicio WebSocket está funcionando
- Que la configuración de proxy de Nginx es correcta
- Que las URLs de WebSocket son accesibles

**2. Errores al cargar predicciones**

Verificar:
- Que el Streaming Service está activo
- Respuesta correcta del endpoint `/prediction/:symbol`
- Formato de datos correcto en las respuestas

**3. Errores al colocar órdenes**

Verificar:
- Que el Broker Service está activo
- Validación correcta de datos en el frontend
- Formato de payload correcto

### Logging y Depuración

Todos los componentes incluyen logs detallados que se pueden ver en la consola del navegador.

Para un modo de depuración más detallado, añadir a la URL el parámetro `?debug=true`.

### Modo Fallback

El frontend incluye un modo fallback que simula datos cuando los servicios backend no están disponibles, útil para desarrollo y pruebas.

Para deshabilitar este comportamiento, establecer en `.env`:

```
REACT_APP_DISABLE_FALLBACK=true
```

---

## Conclusión

Esta documentación proporciona las guías necesarias para entender y conectar el frontend con los microservicios backend de la plataforma de trading. Para más detalles sobre cada componente, referirse a los comentarios en el código fuente.
