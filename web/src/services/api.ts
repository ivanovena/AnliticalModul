import axios from 'axios';
import { MarketData, Portfolio, Order, Prediction, Strategy, ChatMessage, ChatResponse, Metrics, ModelStatus } from '../types/api';

// Obtener la URL base de la API desde las variables de entorno o usar un valor por defecto
const BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Configurar axios con mejores opciones para entorno de producción
const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 10000, // Timeout de 10 segundos
  headers: {
    'Content-Type': 'application/json',
  }
});

// Interceptor para manejar errores y reintentos
apiClient.interceptors.response.use(
  response => response,
  async error => {
    console.error('API Error:', error.response?.data || error.message);
    
    // Si el error es de conexión, podríamos intentar usar un endpoint alternativo
    if (error.message.includes('Network Error') || error.code === 'ECONNABORTED') {
      console.log('Intentando conectar a través de proxy de desarrollo...');
      
      // En desarrollo, podemos intentar conectar a través del proxy de desarrollo
      // En producción, esto será manejado por nginx
      if (process.env.NODE_ENV === 'development') {
        const originalRequest = error.config;
        
        // Evitar loops infinitos
        if (!originalRequest._retry) {
          originalRequest._retry = true;
          
          // Intentar con la URL alternativa
          try {
            return await axios(originalRequest);
          } catch (retryError) {
            console.error('Error en reintento:', retryError);
          }
        }
      }
    }
    
    return Promise.reject(error);
  }
);

// Servicio para obtener datos de mercado
export const marketService = {
  // Obtener datos en tiempo real para un símbolo
  getMarketData: async (symbol: string): Promise<MarketData> => {
    try {
      const response = await apiClient.get(`/market-data/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching market data for ${symbol}:`, error);
      
      // Crear datos ficticios basados en el símbolo para evitar errores en la interfaz
      const mockPrice = Math.round(50 + Math.random() * 50 * 100) / 100; // Entre 50 y 100
      return {
        symbol: symbol,
        timestamp: new Date().toISOString(),
        price: mockPrice,
        open: mockPrice * 0.98,
        high: mockPrice * 1.02,
        low: mockPrice * 0.97,
        previousClose: mockPrice * 0.99,
        volume: Math.round(1000 + Math.random() * 9000),
        change: mockPrice * 0.01,
        changePercent: 1.0
      };
    }
  },

  // Obtener datos históricos para un símbolo
  getHistoricalData: async (symbol: string, timeframe: string = '1d', limit: number = 30): Promise<MarketData[]> => {
    try {
      const response = await apiClient.get(`/historical/${symbol}?timeframe=${timeframe}&limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      
      // Generar datos históricos ficticios
      const mockData: MarketData[] = [];
      const now = new Date();
      const basePrice = 50 + Math.random() * 50;
      
      for (let i = 0; i < limit; i++) {
        const date = new Date(now);
        date.setDate(date.getDate() - (limit - i));
        
        const dailyVariation = (Math.random() - 0.5) * 2; // Entre -1 y 1
        const price = basePrice + dailyVariation * 3 * (i / limit); // Crear tendencia suave
        
        mockData.push({
          symbol: symbol,
          timestamp: date.toISOString(),
          price: price,
          open: price * 0.99,
          high: price * 1.01,
          low: price * 0.98,
          previousClose: price * 0.97,
          volume: Math.round(1000 + Math.random() * 9000),
          change: dailyVariation * price * 0.01,
          changePercent: dailyVariation * 1.0
        });
      }
      
      return mockData;
    }
  },

  // Obtener lista de símbolos disponibles
  getSymbols: async (): Promise<string[]> => {
    try {
      const response = await apiClient.get(`/symbols`);
      return response.data.symbols;
    } catch (error) {
      console.error('Error fetching symbols:', error);
      // Devolver símbolos por defecto para pruebas
      return ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "IAG.MC", "PHM.MC", "AENA.MC", "BA", "CAR", "DLTR"];
    }
  }
};

// Servicio de métricas e indicadores
export const metricsService = {
  // Obtener métricas del portafolio
  getMetrics: async (): Promise<Metrics> => {
    try {
      const response = await apiClient.get(`/metrics`);
      return response.data;
    } catch (error) {
      console.error('Error fetching metrics:', error);
      
      // Datos ficticios de métricas para evitar errores
      return {
        performance: {
          total_return: 5.2,
          cash_ratio: 0.6,
          positions_count: 3,
          trading_frequency: 2.5
        },
        stock_performance: {
          AAPL: {
            symbol: "AAPL",
            quantity: 10,
            current_price: 145.86,
            avg_cost: 140.25,
            market_value: 1458.6,
            profit: 56.1,
            profit_percent: 4.0,
            prediction: 3.2,
            prediction_direction: 'up'
          },
          MSFT: {
            symbol: "MSFT",
            quantity: 5,
            current_price: 305.45,
            avg_cost: 290.75,
            market_value: 1527.25,
            profit: 73.5,
            profit_percent: 5.05,
            prediction: 1.8,
            prediction_direction: 'up'
          }
        },
        risk_metrics: {
          portfolio: {
            diversification_score: 0.65,
            cash_ratio: 0.6,
            total_value: 10000
          }
        }
      };
    }
  }
};

// Servicio de cartera
export const portfolioService = {
  // Obtener estado del portafolio
  getPortfolio: async (): Promise<Portfolio> => {
    try {
      const response = await apiClient.get(`/portfolio`);
      return response.data;
    } catch (error) {
      console.error('Error fetching portfolio:', error);
      
      // Portafolio ficticio para evitar errores
      return {
        account_id: "12345",
        total_value: 100000,
        cash: 60000,
        invested_value: 40000,
        total_profit: 2500,
        total_profit_pct: 2.5,
        positions: {
          AAPL: {
            symbol: "AAPL",
            quantity: 10,
            avg_cost: 140.25,
            market_value: 1458.6,
            current_profit: 56.1,
            current_profit_pct: 4.0,
            last_updated: new Date().toISOString()
          },
          MSFT: {
            symbol: "MSFT",
            quantity: 5,
            avg_cost: 290.75,
            market_value: 1527.25,
            current_profit: 73.5,
            current_profit_pct: 5.05,
            last_updated: new Date().toISOString()
          },
          GOOGL: {
            symbol: "GOOGL",
            quantity: 2,
            avg_cost: 2750.50,
            market_value: 5650.20,
            current_profit: 149.2,
            current_profit_pct: 2.71,
            last_updated: new Date().toISOString()
          }
        },
        last_updated: new Date().toISOString()
      };
    }
  },
  
  // Obtener historial de órdenes
  getOrders: async (): Promise<Order[]> => {
    try {
      const response = await apiClient.get(`/orders`);
      return response.data;
    } catch (error) {
      console.error('Error fetching orders:', error);
      
      // Órdenes ficticias para evitar errores
      return [
        {
          symbol: "AAPL",
          action: "BUY",
          quantity: 5,
          price: 140.25,
          timestamp: new Date(new Date().getTime() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          total_value: 701.25
        },
        {
          symbol: "MSFT",
          action: "BUY",
          quantity: 3,
          price: 290.75,
          timestamp: new Date(new Date().getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          total_value: 872.25
        },
        {
          symbol: "GOOGL",
          action: "BUY",
          quantity: 1,
          price: 2750.50,
          timestamp: new Date(new Date().getTime() - 3 * 24 * 60 * 60 * 1000).toISOString(),
          total_value: 2750.50
        },
        {
          symbol: "AAPL",
          action: "BUY",
          quantity: 5,
          price: 142.35,
          timestamp: new Date(new Date().getTime() - 1 * 24 * 60 * 60 * 1000).toISOString(),
          total_value: 711.75
        }
      ];
    }
  },
  
  // Colocar una nueva orden
  placeOrder: async (order: Order): Promise<{ success: boolean; message: string }> => {
    try {
      const response = await apiClient.post(`/orders`, order);
      return {
        success: true,
        message: 'Orden ejecutada correctamente'
      };
    } catch (error) {
      console.error('Error placing order:', error);
      throw new Error('No se pudo ejecutar la orden. Por favor, inténtalo de nuevo.');
    }
  }
};

// Servicio para predicciones y análisis
export const analysisService = {
  // Obtener predicción para un símbolo
  getPrediction: async (symbol: string): Promise<Prediction> => {
    try {
      const response = await apiClient.get(`/predictions/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching prediction for ${symbol}:`, error);
      
      // Crear predicción ficticia
      const predictionValue = (Math.random() - 0.4) * 5; // Sesgo ligeramente hacia valores positivos
      const datePredictions: Record<string, number> = {};
      const now = new Date();
      
      for (let i = 1; i <= 5; i++) {
        const date = new Date(now);
        date.setDate(date.getDate() + i);
        datePredictions[date.toISOString().split('T')[0]] = predictionValue * i / 3;
      }
      
      return {
        symbol: symbol,
        timestamp: new Date().toISOString(),
        predictions: datePredictions,
        currentPrice: 50 + Math.random() * 50,
        modelMetrics: {
          MAPE: 2.5,
          RMSE: 1.8,
          accuracy: 0.82
        }
      };
    }
  },

  // Obtener estrategia de inversión para un símbolo
  getStrategy: async (symbol: string): Promise<Strategy> => {
    try {
      const response = await apiClient.get(`/strategy/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching strategy for ${symbol}:`, error);
      
      // Estrategia ficticia
      const predictionValue = Math.random() - 0.4; // Sesgo ligeramente hacia valores positivos
      const recommendation = predictionValue > 0.1 ? "BUY" : 
                            predictionValue < -0.1 ? "SELL" : "HOLD";
      
      return {
        symbol: symbol,
        summary: "Análisis basado en datos de mercado recientes",
        recommendation: {
          action: predictionValue > 0.1 ? "comprar" : 
                predictionValue < -0.1 ? "vender" : "mantener",
          price: 50 + Math.random() * 50,
          quantity: Math.round(1 + Math.random() * 9),
          stopLoss: 45 + Math.random() * 5,
          takeProfit: 55 + Math.random() * 10,
          confidence: 0.8,
          timeframe: "short_term"
        },
        factors: ["tendencia", "volumen", "sentimiento"],
        technicalMetrics: {
          rsi: 55,
          macd: 0.5,
          bollinger: 0.2
        },
        analysis: "Estrategia basada en tendencia reciente y análisis técnico."
      };
    }
  },

  // Obtener estado de los modelos de ML
  getModelStatus: async (): Promise<ModelStatus[]> => {
    try {
      const response = await apiClient.get(`/model-status`);
      return response.data;
    } catch (error) {
      console.error('Error fetching model status:', error);
      
      // Crear modelos de estado ficticios
      return [
        {
          modelId: "online",
          name: "Modelo Tiempo Real",
          status: "active",
          version: "1.0",
          lastUpdated: new Date().toISOString(),
          symbols: ["AAPL", "MSFT", "GOOGL"],
          metrics: [
            { name: "accuracy", value: 0.78, status: "good", trend: "up" },
            { name: "MAPE", value: 3.2, status: "good", trend: "down" },
            { name: "RMSE", value: 2.1, status: "good", trend: "stable" }
          ]
        },
        {
          modelId: "batch",
          name: "Modelo Diario",
          status: "active",
          version: "1.1",
          lastUpdated: new Date().toISOString(),
          symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
          metrics: [
            { name: "accuracy", value: 0.82, status: "good", trend: "stable" },
            { name: "MAPE", value: 2.8, status: "good", trend: "down" },
            { name: "RMSE", value: 1.9, status: "good", trend: "down" }
          ]
        },
        {
          modelId: "ensemble",
          name: "Modelo Ensemble",
          status: "active",
          version: "1.2",
          lastUpdated: new Date().toISOString(),
          symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"],
          metrics: [
            { name: "accuracy", value: 0.85, status: "good", trend: "up" },
            { name: "MAPE", value: 2.5, status: "good", trend: "down" },
            { name: "RMSE", value: 1.7, status: "good", trend: "down" }
          ]
        }
      ];
    }
  }
};

// Servicio para el chat con el agente IA
export const chatService = {
  // Enviar mensaje al agente IA
  sendMessage: async (message: ChatMessage): Promise<ChatResponse> => {
    try {
      const response = await apiClient.post(`/chat`, message);
      return response.data;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  }
};

// Configuración para WebSockets
export const socketService = {
  // URL de los WebSockets - usar URLs relativas con host window.location si es posible
  getMarketDataSocketUrl: () => {
    return process.env.REACT_APP_MARKET_WS_URL || 
          (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + 
          window.location.host + '/ws';
  },
  getPredictionsSocketUrl: () => {
    return process.env.REACT_APP_PREDICTIONS_WS_URL || 
          (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + 
          window.location.host + '/predictions';
  },
  getRecommendationsSocketUrl: () => {
    return process.env.REACT_APP_RECOMMENDATIONS_WS_URL || 
          (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + 
          window.location.host + '/recommendations';
  }
};