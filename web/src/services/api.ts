import axios from 'axios';
import { MarketData, Portfolio, Order, Prediction, Strategy, ChatMessage, ChatResponse, Metrics, ModelStatus } from '../types/api';

const BASE_URL = process.env.REACT_APP_API_URL || '';

// Configurar axios para manejar errores
axios.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Servicio para obtener datos de mercado
export const marketService = {
  // Obtener datos en tiempo real para un símbolo
  getMarketData: async (symbol: string): Promise<MarketData> => {
    try {
      const response = await axios.get(`${BASE_URL}/market-data/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching market data for ${symbol}:`, error);
      throw error;
    }
  },

  // Obtener datos históricos para un símbolo
  getHistoricalData: async (symbol: string, timeframe: string = '1d', limit: number = 30): Promise<MarketData[]> => {
    try {
      const response = await axios.get(`${BASE_URL}/historical/${symbol}?timeframe=${timeframe}&limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      throw error;
    }
  },

  // Obtener lista de símbolos disponibles
  getSymbols: async (): Promise<string[]> => {
    try {
      const response = await axios.get(`${BASE_URL}/symbols`);
      return response.data.symbols;
    } catch (error) {
      console.error('Error fetching symbols:', error);
      throw error;
    }
  }
};

// Servicio para gestionar el portafolio
export const portfolioService = {
  // Obtener el portafolio actual
  getPortfolio: async (): Promise<Portfolio> => {
    try {
      const response = await axios.get(`${BASE_URL}/portfolio`);
      return response.data;
    } catch (error) {
      console.error('Error fetching portfolio:', error);
      throw error;
    }
  },

  // Obtener historial de órdenes
  getOrders: async (): Promise<Order[]> => {
    try {
      const response = await axios.get(`${BASE_URL}/orders`);
      return response.data;
    } catch (error) {
      console.error('Error fetching orders:', error);
      throw error;
    }
  },

  // Colocar una nueva orden
  placeOrder: async (order: Omit<Order, 'timestamp' | 'total_value'>): Promise<Order> => {
    try {
      const response = await axios.post(`${BASE_URL}/orders`, order);
      return response.data;
    } catch (error) {
      console.error('Error placing order:', error);
      throw error;
    }
  },

  // Obtener métricas del portafolio
  getMetrics: async (): Promise<Metrics> => {
    try {
      const response = await axios.get(`${BASE_URL}/metrics`);
      return response.data;
    } catch (error) {
      console.error('Error fetching metrics:', error);
      throw error;
    }
  },

  // Obtener rendimiento detallado por acción
  getStockPerformance: async () => {
    try {
      const response = await axios.get(`${BASE_URL}/stock-performance`);
      return response.data;
    } catch (error) {
      console.error('Error fetching stock performance:', error);
      throw error;
    }
  }
};

// Servicio para predicciones y análisis
export const analysisService = {
  // Obtener predicción para un símbolo
  getPrediction: async (symbol: string): Promise<Prediction> => {
    try {
      const response = await axios.get(`${BASE_URL}/predictions/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching prediction for ${symbol}:`, error);
      throw error;
    }
  },

  // Obtener estrategia de inversión para un símbolo
  getStrategy: async (symbol: string): Promise<Strategy> => {
    try {
      const response = await axios.get(`${BASE_URL}/strategy/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching strategy for ${symbol}:`, error);
      throw error;
    }
  },

  // Obtener estado de los modelos de ML
  getModelStatus: async (): Promise<ModelStatus[]> => {
    try {
      const response = await axios.get(`${BASE_URL}/model-status`);
      return response.data;
    } catch (error) {
      console.error('Error fetching model status:', error);
      throw error;
    }
  }
};

// Servicio para el chat con el agente IA
export const chatService = {
  // Enviar mensaje al agente IA
  sendMessage: async (message: ChatMessage): Promise<ChatResponse> => {
    try {
      const response = await axios.post(`${BASE_URL}/chat`, message);
      return response.data;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  }
};

// Configuración para WebSockets
export const socketService = {
  // URL de los WebSockets
  MARKET_DATA_SOCKET_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8080/ws/market',
  PREDICTIONS_SOCKET_URL: process.env.REACT_APP_PREDICTIONS_WS_URL || 'ws://localhost:8090/ws/predictions'
};