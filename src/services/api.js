// src/services/api.js
import axios from 'axios';

// Configuración por defecto para todas las solicitudes
axios.defaults.baseURL = 'http://localhost:8001';
axios.defaults.headers.common['Content-Type'] = 'application/json';

// Interceptor para manejo de errores global
axios.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// API para datos de mercado
export const marketAPI = {
  /**
   * Obtiene la cotización actual de un símbolo
   * @param {string} symbol - Símbolo de la acción
   * @returns {Promise} Datos de la cotización
   */
  getQuote: async (symbol) => {
    try {
      const response = await axios.get(`/api/market/quote/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching quote for ${symbol}:`, error);
      throw error;
    }
  },
  
  /**
   * Obtiene datos históricos de un símbolo
   * @param {string} symbol - Símbolo de la acción
   * @param {string} startDate - Fecha de inicio (YYYY-MM-DD)
   * @param {string} endDate - Fecha de fin (YYYY-MM-DD)
   * @returns {Promise} Datos históricos
   */
  getHistoricalData: async (symbol, startDate, endDate) => {
    try {
      const response = await axios.get(
        `/api/market/historical/${symbol}?start=${startDate}&end=${endDate}`
      );
      return response.data;
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      throw error;
    }
  },
  
  /**
   * Busca símbolos que coincidan con un término
   * @param {string} query - Término de búsqueda
   * @returns {Promise} Lista de símbolos coincidentes
   */
  searchSymbols: async (query) => {
    try {
      const response = await axios.get(`/api/market/search?q=${query}`);
      return response.data;
    } catch (error) {
      console.error(`Error searching symbols for "${query}":`, error);
      throw error;
    }
  }
};

// API para predicciones
export const predictionsAPI = {
  /**
   * Obtiene la última predicción para un símbolo
   * @param {string} symbol - Símbolo de la acción
   * @returns {Promise} Datos de la predicción
   */
  getLatestPrediction: async (symbol) => {
    try {
      const response = await axios.get(`/api/predictions/${symbol}/latest`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching latest prediction for ${symbol}:`, error);
      throw error;
    }
  },
  
  /**
   * Obtiene predicciones históricas para un símbolo
   * @param {string} symbol - Símbolo de la acción
   * @param {string} startDate - Fecha de inicio (YYYY-MM-DD)
   * @param {string} endDate - Fecha de fin (YYYY-MM-DD)
   * @returns {Promise} Predicciones históricas
   */
  getHistoricalPredictions: async (symbol, startDate, endDate) => {
    try {
      const response = await axios.get(
        `/api/predictions/${symbol}?start=${startDate}&end=${endDate}`
      );
      return response.data;
    } catch (error) {
      console.error(`Error fetching historical predictions for ${symbol}:`, error);
      throw error;
    }
  },
  
  /**
   * Obtiene las últimas predicciones para todos los símbolos
   * @param {number} limit - Límite de resultados
   * @returns {Promise} Lista de predicciones recientes
   */
  getLatestPredictions: async (limit = 10) => {
    try {
      const response = await axios.get(`/api/predictions/latest?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching latest predictions:`, error);
      throw error;
    }
  }
};

// API para portfolio y broker
export const portfolioAPI = {
  /**
   * Obtiene el estado actual del portfolio
   * @returns {Promise} Datos del portfolio
   */
  getPortfolio: async () => {
    try {
      const response = await axios.get('/api/portfolio');
      return response.data;
    } catch (error) {
      console.error('Error fetching portfolio:', error);
      throw error;
    }
  },
  
  /**
   * Obtiene el historial de órdenes
   * @param {string} status - Estado de las órdenes (all, active, completed)
   * @param {number} limit - Límite de resultados
   * @returns {Promise} Historial de órdenes
   */
  getOrders: async (status = 'all', limit = 50) => {
    try {
      const response = await axios.get(`/api/orders?status=${status}&limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching orders with status ${status}:`, error);
      throw error;
    }
  },
  
  /**
   * Envía una nueva orden
   * @param {Object} orderData - Datos de la orden
   * @returns {Promise} Resultado de la orden
   */
  placeOrder: async (orderData) => {
    try {
      const response = await axios.post('/api/order', orderData);
      return response.data;
    } catch (error) {
      console.error('Error placing order:', error);
      throw error;
    }
  },
  
  /**
   * Obtiene métricas del portfolio
   * @returns {Promise} Métricas del portfolio
   */
  getMetrics: async () => {
    try {
      const response = await axios.get('/api/metrics');
      return response.data;
    } catch (error) {
      console.error('Error fetching portfolio metrics:', error);
      throw error;
    }
  }
};

// API para el asistente de IA
export const advisorAPI = {
  /**
   * Envía un mensaje al asistente de IA
   * @param {string} message - Mensaje del usuario
   * @returns {Promise} Respuesta del asistente
   */
  sendMessage: async (message) => {
    try {
      const response = await axios.post('/api/chat', { message });
      return response.data;
    } catch (error) {
      console.error('Error sending message to AI advisor:', error);
      throw error;
    }
  },
  
  /**
   * Envía retroalimentación sobre una recomendación
   * @param {string} predictionId - ID de la predicción
   * @param {boolean} wasHelpful - Si la recomendación fue útil
   * @param {string} feedback - Comentarios adicionales
   * @returns {Promise} Confirmación
   */
  sendFeedback: async (predictionId, wasHelpful, feedback = '') => {
    try {
      const response = await axios.post('/api/feedback', {
        predictionId,
        wasHelpful,
        feedback
      });
      return response.data;
    } catch (error) {
      console.error('Error sending feedback:', error);
      throw error;
    }
  },
  
  /**
   * Solicita un plan de inversión basado en predicciones
   * @param {Object} criteria - Criterios para el plan
   * @returns {Promise} Plan de inversión
   */
  generatePlan: async (criteria) => {
    try {
      const response = await axios.post('/api/plan', criteria);
      return response.data;
    } catch (error) {
      console.error('Error generating investment plan:', error);
      throw error;
    }
  }
};

// Exportación de todas las APIs
export default {
  market: marketAPI,
  predictions: predictionsAPI,
  portfolio: portfolioAPI,
  advisor: advisorAPI
};