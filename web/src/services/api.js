import axios from 'axios';
import io from 'socket.io-client';

// Configuración de axios
const axiosInstance = axios.create({
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Interceptor para manejo de errores
axiosInstance.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API Error:', error.response || error);
    return Promise.reject(error.response?.data || error);
  }
);

// Servicio de API para la plataforma de trading
export const api = {
  // ===== Portfolio API =====
  getPortfolio: async () => {
    try {
      return await axiosInstance.get('/api/portfolio');
    } catch (error) {
      console.error('Error fetching portfolio:', error);
      // Fallback para desarrollo/demo
      return {
        cash: 100000 + Math.random() * 5000,
        initialCash: 100000,
        positions: {
          AAPL: {
            symbol: 'AAPL',
            quantity: 10,
            avgCost: 152.35,
            currentPrice: 155.80 + (Math.random() * 2 - 1),
            marketValue: 1558.00
          },
          MSFT: {
            symbol: 'MSFT',
            quantity: 5,
            avgCost: 305.22,
            currentPrice: 310.65 + (Math.random() * 3 - 1.5),
            marketValue: 1553.25
          }
        },
        totalValue: 103111.25,
        lastUpdate: new Date().toISOString()
      };
    }
  },
  
  getPortfolioMetrics: async () => {
    try {
      return await axiosInstance.get('/api/metrics');
    } catch (error) {
      console.error('Error fetching portfolio metrics:', error);
      // Fallback para desarrollo/demo
      return {
        totalReturn: parseFloat((Math.random() * 8 - 2).toFixed(2)),
        dailyReturn: parseFloat((Math.random() * 2 - 0.5).toFixed(2)),
        weeklyReturn: parseFloat((Math.random() * 4 - 1).toFixed(2)),
        monthlyReturn: parseFloat((Math.random() * 6 - 1.5).toFixed(2)),
        sharpeRatio: parseFloat((Math.random() * 0.5 + 0.8).toFixed(2)),
        volatility: parseFloat((Math.random() * 0.5 + 0.5).toFixed(2))
      };
    }
  },
  
  getOrders: async () => {
    try {
      return await axiosInstance.get('/api/orders');
    } catch (error) {
      console.error('Error fetching orders:', error);
      // Fallback para desarrollo/demo
      return [
        {
          id: '1',
          symbol: 'AAPL',
          action: 'BUY',
          quantity: 10,
          price: 152.35,
          totalValue: 1523.50,
          timestamp: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString()
        },
        {
          id: '2',
          symbol: 'MSFT',
          action: 'BUY',
          quantity: 5,
          price: 305.22,
          totalValue: 1526.10,
          timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
        },
        {
          id: '3',
          symbol: 'GOOGL',
          action: 'BUY',
          quantity: 3,
          price: 123.45,
          totalValue: 370.35,
          timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString()
        },
        {
          id: '4',
          symbol: 'GOOGL',
          action: 'SELL',
          quantity: 3,
          price: 126.78,
          totalValue: 380.34,
          profit: 9.99,
          timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
        }
      ];
    }
  },
  
  placeOrder: async (orderData) => {
    try {
      // Convertir acción a mayúsculas para que coincida con el backend
      const formattedOrderData = {
        ...orderData,
        action: orderData.action.toUpperCase()  // Convertir "buy" a "BUY", "sell" a "SELL"
      };
      
      return await axiosInstance.post('/api/orders', formattedOrderData);
    } catch (error) {
      console.error('Error placing order:', error);
      // Manejar error específico de fondos insuficientes
      if (error.detail && error.detail.includes('Fondos insuficientes')) {
        throw new Error('Fondos insuficientes para esta orden');
      } else if (error.detail && error.detail.includes('No hay suficientes acciones')) {
        throw new Error('No hay suficientes acciones para vender');
      }
      
      // Propagar error para que el componente lo maneje
      throw error;
    }
  },
  
  // ===== Market Data API =====
  getQuote: async (symbol) => {
    try {
      const response = await axiosInstance.get(`/api/market-data/${symbol}`);
      return response;
    } catch (error) {
      console.error(`Error fetching quote for ${symbol}:`, error);
      
      // Manejar errores específicos
      if (error.response) {
        if (error.response.status === 404) {
          console.warn(`No se encontraron datos para ${symbol}`);
        }
      }
      
      // Fallback para desarrollo/demo
      const basePrice = symbol === 'AAPL' ? 155 : 
                        symbol === 'MSFT' ? 310 : 
                        symbol === 'GOOGL' ? 125 : 
                        symbol === 'AMZN' ? 140 : 
                        symbol === 'TSLA' ? 220 : 
                        symbol === 'IAG.MC' ? 5.25 : 
                        symbol === 'PHM.MC' ? 70.30 : 
                        symbol === 'BKY.MC' ? 0.85 : 
                        symbol === 'AENA.MC' ? 180.45 : 
                        symbol === 'BA' ? 180.20 : 
                        symbol === 'NLGO' ? 25.75 : 
                        symbol === 'CAR' ? 105.30 : 
                        symbol === 'DLTR' ? 115.50 : 
                        symbol === 'CANTE.IS' ? 45.20 : 
                        symbol === 'SASA.IS' ? 32.85 : 100;
      
      return {
        symbol,
        price: parseFloat((basePrice + (Math.random() * 4 - 2)).toFixed(2)),
        change: parseFloat((Math.random() * 3 - 1.5).toFixed(2)),
        percentChange: parseFloat((Math.random() * 2 - 1).toFixed(2)),
        volume: Math.floor(Math.random() * 5000000 + 1000000),
        timestamp: new Date().toISOString()
      };
    }
  },
  
  getHistoricalData: async (symbol, timeframe = '1h') => {
    try {
      // Usar el endpoint correcto que hemos implementado en el broker
      return await axiosInstance.get(`/historical/${symbol}?timeframe=${timeframe}`);
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      // Fallback para desarrollo/demo
      const basePrice = symbol === 'AAPL' ? 155 : 
                        symbol === 'MSFT' ? 310 : 
                        symbol === 'GOOGL' ? 125 : 
                        symbol === 'AMZN' ? 140 : 
                        symbol === 'TSLA' ? 220 : 
                        symbol === 'IAG.MC' ? 5.25 : 
                        symbol === 'PHM.MC' ? 70.30 : 
                        symbol === 'BKY.MC' ? 0.85 : 
                        symbol === 'AENA.MC' ? 180.45 : 
                        symbol === 'BA' ? 180.20 : 
                        symbol === 'NLGO' ? 25.75 : 
                        symbol === 'CAR' ? 105.30 : 
                        symbol === 'DLTR' ? 115.50 : 
                        symbol === 'CANTE.IS' ? 45.20 : 
                        symbol === 'SASA.IS' ? 32.85 : 100;
      
      const volatility = symbol === 'TSLA' ? 0.05 : 0.02;
      
      // Determinar número de puntos según timeframe
      let points;
      switch (timeframe) {
        case '1m': points = 60; break;
        case '5m': points = 60; break;
        case '15m': points = 60; break;
        case '30m': points = 48; break;
        case '1h': points = 24; break;
        case '4h': points = 30; break;
        case '1d': points = 30; break;
        default: points = 30;
      }
      
      // Intervalo en milisegundos
      const intervalMap = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
      };
      
      const interval = intervalMap[timeframe] || 60 * 60 * 1000;
      
      // Generar datos históricos
      const now = new Date();
      const data = [];
      let price = basePrice;
      
      for (let i = points - 1; i >= 0; i--) {
        const date = new Date(now.getTime() - i * interval);
        
        // Simular movimiento de precio
        const change = (Math.random() - 0.5) * volatility * price;
        price = Math.max(price + change, basePrice * 0.5);
        
        // Generar OHLC
        const open = price;
        const high = price * (1 + Math.random() * volatility * 0.5);
        const low = price * (1 - Math.random() * volatility * 0.5);
        const close = price * (1 + (Math.random() - 0.5) * volatility * 0.3);
        
        data.push({
          date: date.toISOString(),
          open: parseFloat(open.toFixed(2)),
          high: parseFloat(high.toFixed(2)),
          low: parseFloat(low.toFixed(2)),
          close: parseFloat(close.toFixed(2)),
          volume: Math.floor(Math.random() * 5000000 + 1000000)
        });
      }
      
      return data;
    }
  },
  
  // ===== Predictions API =====
  getPredictions: async (symbol) => {
    try {
      const response = await axiosInstance.get(`/prediction/${symbol}`);
      return response;
    } catch (error) {
      console.error(`Error fetching predictions for ${symbol}:`, error);
      
      // Si hay un error específico, manejarlo adecuadamente
      if (error.response && error.response.status === 404) {
        console.warn(`No se encontraron predicciones para ${symbol}`);
      }
      
      // Fallback para desarrollo/demo
      const basePrice = symbol === 'AAPL' ? 155 : 
                        symbol === 'MSFT' ? 310 : 
                        symbol === 'GOOGL' ? 125 : 
                        symbol === 'AMZN' ? 140 : 
                        symbol === 'TSLA' ? 220 : 100;
      
      // Crear predicciones con sesgo ligeramente alcista
      const currentPrice = basePrice + (Math.random() * 4 - 2);
      const predictions = {
        '15m': parseFloat((currentPrice * (1 + (Math.random() * 0.01 - 0.003))).toFixed(2)),
        '30m': parseFloat((currentPrice * (1 + (Math.random() * 0.015 - 0.005))).toFixed(2)),
        '1h': parseFloat((currentPrice * (1 + (Math.random() * 0.02 - 0.007))).toFixed(2)),
        '3h': parseFloat((currentPrice * (1 + (Math.random() * 0.03 - 0.01))).toFixed(2)),
        '1d': parseFloat((currentPrice * (1 + (Math.random() * 0.04 - 0.01))).toFixed(2))
      };
      
      return {
        symbol,
        currentPrice: parseFloat(currentPrice.toFixed(2)),
        predictions,
        timestamp: new Date().toISOString(),
        modelMetrics: {
          MAPE: parseFloat((Math.random() * 2 + 1).toFixed(2)),
          RMSE: parseFloat((Math.random() * 3 + 1.5).toFixed(2)),
          accuracy: parseFloat((Math.random() * 10 + 85).toFixed(2))
        }
      };
    }
  },
  
  getPredictionHistory: async (symbol) => {
    try {
      const response = await axiosInstance.get(`/prediction/history/${symbol}`);
      return response;
    } catch (error) {
      console.error(`Error fetching prediction history for ${symbol}:`, error);
      
      // Si hay un error específico, manejarlo adecuadamente
      if (error.response && error.response.status === 404) {
        console.warn(`No se encontró historial de predicciones para ${symbol}`);
      }
      
      // Fallback para desarrollo/demo
      const basePrice = symbol === 'AAPL' ? 155 : 
                        symbol === 'MSFT' ? 310 : 
                        symbol === 'GOOGL' ? 125 : 
                        symbol === 'AMZN' ? 140 : 
                        symbol === 'TSLA' ? 220 : 100;
      
      // Crear historial de verificaciones de predicciones
      const history = [];
      const horizons = ['15m', '30m', '1h', '3h', '1d'];
      
      // Generar datos para las últimas 24 horas con ciclos de 1h
      for (let i = 0; i < 24; i++) {
        const timestamp = new Date(Date.now() - i * 60 * 60 * 1000).toISOString();
        
        // Generar verificaciones para cada horizonte
        horizons.forEach(horizon => {
          const predictedPrice = basePrice * (1 + (Math.random() * 0.04 - 0.01));
          const actualPrice = basePrice * (1 + (Math.random() * 0.04 - 0.01));
          const error = Math.abs(actualPrice - predictedPrice);
          const errorPct = (error / actualPrice) * 100;
          
          history.push({
            timestamp,
            horizon,
            predictedPrice: parseFloat(predictedPrice.toFixed(2)),
            actualPrice: parseFloat(actualPrice.toFixed(2)),
            error: parseFloat(error.toFixed(2)),
            errorPct: parseFloat(errorPct.toFixed(2))
          });
        });
      }
      
      return history;
    }
  },
  
  // ===== Model Status API =====
  getModelStatus: async () => {
    try {
      return await axiosInstance.get('/api/model-status');
    } catch (error) {
      console.error('Error fetching model status:', error);
      // Si hay un error específico, manéjalo adecuadamente
      if (error.response && error.response.status === 404) {
        console.warn('El endpoint de model-status no está disponible');
      }
      
      // Usar fallback
      return {
        online: {
          status: ['healthy', 'degraded', 'critical'][Math.floor(Math.random() * 3)],
          accuracy: parseFloat((Math.random() * 10 + 85).toFixed(2)),
          metrics: {
            MAPE: parseFloat((Math.random() * 3 + 1).toFixed(2)),
            RMSE: parseFloat((Math.random() * 2 + 1).toFixed(2)),
            accuracy: parseFloat((Math.random() * 10 + 85).toFixed(2))
          },
          lastUpdated: new Date().toISOString()
        },
        batch: {
          status: ['healthy', 'degraded', 'critical'][Math.floor(Math.random() * 3)],
          accuracy: parseFloat((Math.random() * 7 + 88).toFixed(2)),
          metrics: {
            MAPE: parseFloat((Math.random() * 2 + 0.8).toFixed(2)),
            RMSE: parseFloat((Math.random() * 1.5 + 0.8).toFixed(2)),
            accuracy: parseFloat((Math.random() * 7 + 88).toFixed(2))
          },
          lastUpdated: new Date().toISOString()
        },
        ensemble: {
          status: ['healthy', 'degraded', 'critical'][Math.floor(Math.random() * 3)],
          accuracy: parseFloat((Math.random() * 5 + 90).toFixed(2)),
          metrics: {
            MAPE: parseFloat((Math.random() * 1.5 + 0.5).toFixed(2)),
            RMSE: parseFloat((Math.random() * 1 + 0.5).toFixed(2)),
            accuracy: parseFloat((Math.random() * 5 + 90).toFixed(2))
          },
          lastUpdated: new Date().toISOString()
        },
        lastUpdated: new Date().toISOString()
      };
    }
  },
  
  // ===== Broker Chat API =====
  sendChatMessage: async (message, conversationId = null) => {
    try {
      // Asegurarse de que el formato de datos coincide con lo que espera el backend
      const chatRequest = {
        message: message,
        conversation_id: conversationId
      };
      
      // Usar directamente el endpoint /chat que configuramos en Nginx
      return await axiosInstance.post('/chat', chatRequest);
    } catch (error) {
      console.error('Error sending chat message:', error);
      
      // Manejar errores específicos
      if (error.response) {
        if (error.response.status === 404) {
          console.warn('El servicio de chat no está disponible');
        } else if (error.response.status === 400) {
          console.warn('Solicitud de chat inválida:', error.response.data);
        }
      }
      
      // Fallback para desarrollo/demo
      const responses = [
        "Basado en el análisis técnico actual, veo señales mixtas en el mercado. Los principales índices muestran divergencias, con tecnología mostrando fortaleza mientras otros sectores consolidan. Recomendaría mantener posiciones diversificadas con sesgo defensivo a corto plazo.",
        "El análisis de volumen muestra acumulación institucional en el sector tecnológico, particularmente en semiconductores y cloud computing. Esto sugiere potencial alcista en estas áreas durante las próximas semanas.",
        "Las condiciones macroeconómicas actuales favorecen empresas con balances sólidos y generación de caja estable. Recomiendo enfocarse en calidad sobre crecimiento en este entorno.",
        "Los indicadores técnicos para AAPL sugieren soporte en $150. El RSI está en zona neutral (55) y el MACD muestra divergencia alcista. Considero que es un buen momento para acumular posiciones con stop loss en $145.",
        "La diversificación es clave en este mercado. Recomiendo una distribución de 40% tecnología, 20% consumo básico, 15% salud, 15% financieras y 10% en efectivo para aprovechar oportunidades."
      ];
      
      return {
        response: responses[Math.floor(Math.random() * responses.length)],
        conversation_id: conversationId || `conv-${Math.random().toString(36).substring(2, 9)}`
      };
    }
  },
  
  getStrategy: async (symbol) => {
    try {
      const response = await axiosInstance.get(`/api/strategy/${symbol}`);
      return response;
    } catch (error) {
      console.error(`Error fetching strategy for ${symbol}:`, error);
      
      // Manejar errores específicos
      if (error.response) {
        if (error.response.status === 404) {
          console.warn(`No se encontró estrategia para ${symbol}`);
        }
      }
      
      // Fallback para desarrollo/demo
      const basePrice = symbol === 'AAPL' ? 155 : 
                        symbol === 'MSFT' ? 310 : 
                        symbol === 'GOOGL' ? 125 : 
                        symbol === 'AMZN' ? 140 : 
                        symbol === 'TSLA' ? 220 : 100;
      
      const actions = ['comprar', 'vender', 'mantener'];
      // Sesgo hacia comprar para AAPL y MSFT
      const actionIndex = (symbol === 'AAPL' || symbol === 'MSFT') ? 
                         Math.floor(Math.random() * 1.5) : 
                         Math.floor(Math.random() * 3);
      
      return {
        symbol,
        summary: `Análisis estratégico para ${symbol} basado en patrones técnicos, indicadores y condiciones de mercado actuales.`,
        recommendation: {
          action: actions[actionIndex],
          price: parseFloat((basePrice + (Math.random() * 4 - 2)).toFixed(2)),
          quantity: Math.floor(Math.random() * 10) + 1,
          stopLoss: parseFloat((basePrice * 0.95).toFixed(2)),
          takeProfit: parseFloat((basePrice * 1.1).toFixed(2)),
          confidence: Math.floor(Math.random() * 15) + 75,
          timeframe: "corto plazo"
        },
        factors: [
          "Tendencia alcista confirmada en múltiples timeframes",
          "Soportes sólidos establecidos en niveles recientes",
          "Volumen creciente en movimientos alcistas",
          "Indicadores técnicos en zonas favorables"
        ],
        technicalMetrics: {
          RSI: parseFloat((Math.random() * 20 + 45).toFixed(2)),
          MACD: parseFloat((Math.random() * 2 - 0.5).toFixed(2)),
          Volumen: Math.floor(Math.random() * 5000000 + 2000000),
          SMA50: parseFloat((basePrice * (1 + (Math.random() * 0.05 - 0.02))).toFixed(2)),
          SMA200: parseFloat((basePrice * (1 + (Math.random() * 0.1 - 0.05))).toFixed(2))
        }
      };
    }
  },
  
  // ===== WebSocket para datos en tiempo real =====
  connectToMarketSocket: (symbols) => {
    // En un entorno real, esto se conectaría a un WebSocket del servidor
    // Para desarrollo/demo, simulamos los eventos
    
    const mockSocket = {
      listeners: {},
      
      on: function(event, callback) {
        this.listeners[event] = callback;
      },
      
      emit: function(event, data) {
        if (this.listeners[event]) {
          this.listeners[event](data);
        }
      },
      
      disconnect: function() {
        clearInterval(this.interval);
        console.log('Mock WebSocket disconnected');
      }
    };
    
    // Simular actualizaciones de mercado cada segundo
    mockSocket.interval = setInterval(() => {
      if (symbols && symbols.length > 0) {
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        const basePrice = symbol === 'AAPL' ? 155 : 
                          symbol === 'MSFT' ? 310 : 
                          symbol === 'GOOGL' ? 125 : 
                          symbol === 'AMZN' ? 140 : 
                          symbol === 'TSLA' ? 220 : 100;
        
        const change = (Math.random() - 0.5) * 0.5;
        const percentChange = (change / basePrice) * 100;
        
        mockSocket.emit('market_update', {
          symbol,
          price: parseFloat((basePrice + change).toFixed(2)),
          change: parseFloat(change.toFixed(2)),
          percentChange: parseFloat(percentChange.toFixed(2)),
          volume: Math.floor(Math.random() * 10000 + 5000),
          timestamp: new Date().toISOString()
        });
      }
    }, 1000);
    
    return mockSocket;
  }
};
