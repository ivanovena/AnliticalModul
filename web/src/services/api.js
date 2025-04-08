import axios from 'axios';
// import io from 'socket.io-client'; // Comentado: No usamos Socket.IO

// Configurar axios con valores predeterminados
const api = axios.create({
  timeout: 10000, // 10 segundos
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
});

// Añadir interceptor para reintentar solicitudes fallidas
api.interceptors.response.use(undefined, async (error) => {
  // Configuración de reintento
  const maxRetries = 3;
  const retryDelay = 1000; // 1 segundo
  
  // Obtener la configuración de la solicitud original
  const { config, response = {} } = error;
  
  // Si no hay configuración o ya se ha reintentado el máximo de veces, rechazar
  if (!config || !config.retry) {
    config.retry = 0;
  }
  
  // Si ya se ha reintentado el máximo de veces, rechazar
  if (config.retry >= maxRetries) {
    return Promise.reject(error);
  }
  
  // Incrementar el contador de reintentos
  config.retry += 1;
  
  // Crear una nueva promesa para esperar antes de reintentar
  const delayPromise = new Promise((resolve) => {
    setTimeout(resolve, retryDelay * config.retry);
  });
  
  // Esperar y luego reintentar la solicitud
  await delayPromise;
  return api(config);
});

// Datos de fallback simulados para desarrollo
const fallbackMarketData = {
  AAPL: {
    quote: { price: 175.43, percentChange: 1.25 },
    historical: [/* datos históricos simulados */]
  },
  MSFT: {
    quote: { price: 334.65, percentChange: 0.89 },
    historical: [/* datos históricos simulados */]
  },
  GOOGL: {
    quote: { price: 135.60, percentChange: -0.45 },
    historical: [/* datos históricos simulados */]
  },
  AMZN: {
    quote: { price: 145.24, percentChange: 2.10 },
    historical: [/* datos históricos simulados */]
  },
  TSLA: {
    quote: { price: 254.85, percentChange: -1.32 },
    historical: [/* datos históricos simulados */]
  }
};

const fallbackPredictions = {
  AAPL: {
    currentPrice: 175.43,
    predictions: {
      '1d': 177.85,
      '1w': 180.20,
      '1m': 185.60
    },
    accuracy: 0.82
  },
  MSFT: {
    currentPrice: 334.65,
    predictions: {
      '1d': 336.20,
      '1w': 340.50,
      '1m': 350.75
    },
    accuracy: 0.85
  }
};

const fallbackPortfolio = {
  cash: 85000,
  initialCash: 100000,
  positions: {
    AAPL: {
      symbol: 'AAPL',
      quantity: 50,
      averagePrice: 170.25,
      currentPrice: 175.43
    },
    MSFT: {
      symbol: 'MSFT',
      quantity: 15,
      averagePrice: 320.10,
      currentPrice: 334.65
    }
  },
  totalValue: 95000,
  lastUpdate: new Date().toISOString()
};

// Variable para rastrear si estamos en modo de desarrollo
const isDevelopment = process.env.NODE_ENV === 'development';

// Funciones de API (sin 'export' individual)

const getMarketData = async (symbol) => {
  try {
    const response = await api.get(`/market-data/${symbol}`);
    return response.data;
  } catch (error) {
    console.error(`Error al obtener datos de mercado para ${symbol}:`, error);
    
    if (isDevelopment && fallbackMarketData[symbol]) {
      console.log(`Usando datos simulados para ${symbol}`);
      return fallbackMarketData[symbol];
    }
    
    throw new Error(`No se pudieron obtener datos de mercado para ${symbol}`);
  }
};

const getHistoricalData = async (symbol, timeframe = '1d') => {
  try {
    const response = await api.get(`/historical/${symbol}?timeframe=${timeframe}`);
    return response.data;
  } catch (error) {
    console.error(`Error al obtener datos históricos para ${symbol}:`, error);
    
    if (isDevelopment) {
      const now = new Date();
      const data = [];
      for (let i = 30; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        data.push({
          date: date.toISOString().split('T')[0],
          open: Math.random() * 10 + 150,
          high: Math.random() * 10 + 155,
          low: Math.random() * 10 + 145,
          close: Math.random() * 10 + 150,
          volume: Math.floor(Math.random() * 10000000)
        });
      }
      return data;
    }
    
    throw new Error(`No se pudieron obtener datos históricos para ${symbol}`);
  }
};

const getPrediction = async (symbol) => {
  try {
    const response = await api.get(`/predictions/${symbol}`);
    return response.data;
  } catch (error) {
    console.error(`Error al obtener predicciones para ${symbol}:`, error);
    
    if (isDevelopment && fallbackPredictions[symbol]) {
      console.log(`Usando predicciones simuladas para ${symbol}`);
      return fallbackPredictions[symbol];
    } else if (isDevelopment) {
      const currentPrice = fallbackMarketData[symbol]?.quote?.price || 100 + Math.random() * 50;
      return {
        currentPrice,
        predictions: {
          '1d': currentPrice * (1 + (Math.random() * 0.04 - 0.02)),
          '1w': currentPrice * (1 + (Math.random() * 0.08 - 0.03)),
          '1m': currentPrice * (1 + (Math.random() * 0.15 - 0.05))
        },
        accuracy: 0.7 + Math.random() * 0.2
      };
    }
    
    throw new Error(`No se pudieron obtener predicciones para ${symbol}`);
  }
};

const getAllPredictions = async () => {
  try {
    const response = await api.get('/predictions');
    return response.data;
  } catch (error) {
    console.error('Error al obtener todas las predicciones:', error);
    
    if (isDevelopment) {
      return fallbackPredictions;
    }
    
    throw new Error('No se pudieron obtener las predicciones');
  }
};

const getActiveSymbols = async () => {
  try {
    const response = await api.get('/symbols');
    return response.data;
  } catch (error) {
    console.error('Error al obtener símbolos activos:', error);
    
    if (isDevelopment) {
      return Object.keys(fallbackMarketData);
    }
    
    throw new Error('No se pudieron obtener los símbolos activos');
  }
};

const getPortfolio = async () => {
  try {
    const response = await api.get('/portfolio');
    return response.data;
  } catch (error) {
    console.error('Error al obtener portafolio:', error);
    
    if (isDevelopment) {
      console.log('Usando portafolio simulado');
      const positions = { ...fallbackPortfolio.positions };
      let totalPositionValue = 0;
      Object.keys(positions).forEach(symbol => {
        const position = positions[symbol];
        position.currentPrice = fallbackMarketData[symbol]?.quote?.price || position.currentPrice;
        totalPositionValue += position.currentPrice * position.quantity;
      });
      const updatedPortfolio = {
        ...fallbackPortfolio,
        positions,
        totalValue: fallbackPortfolio.cash + totalPositionValue,
        lastUpdate: new Date().toISOString()
      };
      return updatedPortfolio;
    }
    
    throw new Error('No se pudo obtener el portafolio');
  }
};

const getPortfolioMetrics = async () => {
  try {
    const response = await api.get('/portfolio/metrics');
    return response.data;
  } catch (error) {
    console.error('Error al obtener métricas del portafolio:', error);
    
    if (isDevelopment) {
      return {
        totalReturn: 5.67,
        dailyReturn: 0.82,
        weeklyReturn: 2.34,
        monthlyReturn: 5.67,
        sharpeRatio: 1.23,
        volatility: 12.5
      };
    }
    
    throw new Error('No se pudieron obtener las métricas del portafolio');
  }
};

const getOrders = async () => {
  try {
    const response = await api.get('/portfolio/orders');
    return response.data;
  } catch (error) {
    console.error('Error al obtener órdenes:', error);
    
    if (isDevelopment) {
      const now = new Date();
      return [
        {
          id: '12345',
          symbol: 'AAPL',
          quantity: 10,
          price: 170.25,
          action: 'buy',
          timestamp: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          status: 'completed'
        },
        {
          id: '12346',
          symbol: 'MSFT',
          quantity: 5,
          price: 320.10,
          action: 'buy',
          timestamp: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          status: 'completed'
        }
      ];
    }
    
    throw new Error('No se pudieron obtener las órdenes');
  }
};

const placeOrder = async (orderData) => {
  try {
    const response = await api.post('/portfolio/orders', orderData);
    return response.data;
  } catch (error) {
    console.error('Error al colocar orden:', error);
    
    if (isDevelopment) {
      return {
        id: Math.floor(Math.random() * 100000).toString(),
        ...orderData,
        timestamp: new Date().toISOString(),
        status: 'completed'
      };
    }
    
    throw new Error('No se pudo colocar la orden');
  }
};

const getModelStatus = async () => {
  try {
    const response = await api.get('/models/status');
    return response.data;
  } catch (error) {
    console.error('Error al obtener estado de modelos:', error);
    
    if (isDevelopment) {
      return {
        ensemble: {
          status: 'healthy',
          accuracy: 85.2,
          lastUpdate: new Date().toISOString(),
          models: ['random_forest', 'gradient_boost', 'elastic_net']
        },
        models: {
          random_forest: {
            status: 'healthy',
            accuracy: 82.1,
            lastUpdate: new Date().toISOString()
          },
          gradient_boost: {
            status: 'healthy',
            accuracy: 86.5,
            lastUpdate: new Date().toISOString()
          },
          elastic_net: {
            status: 'degraded',
            accuracy: 78.9,
            lastUpdate: new Date().toISOString()
          }
        }
      };
    }
    
    throw new Error('No se pudo obtener el estado de los modelos');
  }
};

// --- NUEVA FUNCION PARA WEBSOCKET DE DATOS DE MERCADO ---
const connectMarketDataWebSocket = (symbols, onOpen, onMessage, onError, onClose) => {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/market`;
  console.log(`Conectando a WebSocket de Mercado: ${wsUrl}`);

  const socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log('WebSocket de Mercado conectado.');
    if (onOpen) onOpen();
    // Ejemplo:
    // if (symbols && symbols.length > 0) {
    //   socket.send(JSON.stringify({ type: 'subscribe', symbols: symbols }));
    // }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.debug('Mensaje WebSocket recibido (Mercado):', data);
      if (onMessage) onMessage(data);
    } catch (e) {
      console.error('Error parseando mensaje WebSocket (Mercado):', e);
      if (onError) onError(e);
    }
  };

  socket.onerror = (error) => {
    console.error('Error en WebSocket de Mercado:', error);
    if (onError) onError(error);
  };

  socket.onclose = (event) => {
    console.log('WebSocket de Mercado desconectado:', event.code, event.reason);
    if (onClose) onClose(event);
  };

  return socket;
};
// --- FIN NUEVA FUNCION ---

// Exportar la instancia de axios y todas las funciones de la API
export {
  api,
  getMarketData,
  getHistoricalData,
  getPrediction,
  getAllPredictions,
  getActiveSymbols,
  getPortfolio,
  getPortfolioMetrics,
  getOrders,
  placeOrder,
  getModelStatus,
  connectMarketDataWebSocket
};
