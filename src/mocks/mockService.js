// src/mocks/mockService.js
import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { 
  portfolioData, 
  ordersData, 
  quoteData, 
  predictionData, 
  historicalData,
  historicalPredictions
} from './mockData';

// Crear instancia de mock para axios
const mock = new MockAdapter(axios, { delayResponse: 500 });

// Mock para portfolio
mock.onGet('/api/portfolio').reply(200, portfolioData);

// Mock para órdenes
mock.onGet(/\/api\/orders\?status=(.+)/).reply((config) => {
  const status = config.url.split('status=')[1].split('&')[0];
  const filteredOrders = ordersData.filter(order => 
    status === 'all' ? true : order.status === status
  );
  return [200, filteredOrders];
});

mock.onGet('/api/orders').reply(200, ordersData);

// Mock para colocación de órdenes
mock.onPost('/api/order').reply((config) => {
  const orderData = JSON.parse(config.data);
  const newOrder = {
    id: `ord-${Math.floor(Math.random() * 10000)}`,
    symbol: orderData.symbol,
    quantity: orderData.quantity,
    price: orderData.price,
    orderType: orderData.orderType,
    status: 'completed',
    timestamp: new Date().toISOString()
  };
  
  // Actualizar portfolio simulado (no persistente)
  if (orderData.orderType === 'buy') {
    portfolioData.cash -= orderData.quantity * orderData.price;
    
    // Buscar si ya hay una posición para este símbolo
    const existingPosition = portfolioData.positions.find(p => p.symbol === orderData.symbol);
    if (existingPosition) {
      // Actualizar posición existente
      const newTotalQuantity = existingPosition.quantity + orderData.quantity;
      const newTotalCost = (existingPosition.quantity * existingPosition.averagePrice) + 
                          (orderData.quantity * orderData.price);
      existingPosition.quantity = newTotalQuantity;
      existingPosition.averagePrice = newTotalCost / newTotalQuantity;
    } else {
      // Crear nueva posición
      portfolioData.positions.push({
        symbol: orderData.symbol,
        quantity: orderData.quantity,
        averagePrice: orderData.price,
        currentPrice: quoteData[orderData.symbol]?.price || orderData.price
      });
    }
  } else if (orderData.orderType === 'sell') {
    portfolioData.cash += orderData.quantity * orderData.price;
    
    // Buscar la posición para este símbolo
    const existingPosition = portfolioData.positions.find(p => p.symbol === orderData.symbol);
    if (existingPosition) {
      // Actualizar posición existente
      existingPosition.quantity -= orderData.quantity;
      
      // Eliminar posición si la cantidad es 0
      if (existingPosition.quantity <= 0) {
        portfolioData.positions = portfolioData.positions.filter(p => p.symbol !== orderData.symbol);
      }
    }
  }
  
  return [200, newOrder];
});

// Mock para cotizaciones
mock.onGet(/\/api\/market\/quote\/(.+)/).reply((config) => {
  const symbol = config.url.split('/').pop();
  return [200, quoteData[symbol] || {}];
});

// Mock para datos históricos
mock.onGet(/\/api\/market\/historical\/(.+)/).reply((config) => {
  const symbol = config.url.split('/')[4].split('?')[0];
  const queryParams = new URLSearchParams(config.url.split('?')[1]);
  const start = queryParams.get('start');
  const end = queryParams.get('end');
  
  if (!historicalData[symbol]) {
    return [404, { message: 'Symbol not found' }];
  }
  
  // Filtrar por fecha si se proporcionan los parámetros
  let filteredData = historicalData[symbol];
  if (start) {
    filteredData = filteredData.filter(item => item.date >= start);
  }
  if (end) {
    filteredData = filteredData.filter(item => item.date <= end);
  }
  
  return [200, filteredData];
});

// Mock para predicciones
mock.onGet(/\/api\/predictions\/(.+)\/latest/).reply((config) => {
  const symbol = config.url.split('/')[3];
  return [200, predictionData[symbol] || {}];
});

// Mock para predicciones históricas
mock.onGet(/\/api\/predictions\/(.+)/).reply((config) => {
  const symbol = config.url.split('/')[3].split('?')[0];
  const queryParams = new URLSearchParams(config.url.split('?')[1]);
  const start = queryParams.get('start');
  const end = queryParams.get('end');
  
  if (!historicalPredictions[symbol]) {
    return [404, { message: 'Symbol not found' }];
  }
  
  // Filtrar por fecha si se proporcionan los parámetros
  let filteredData = historicalPredictions[symbol];
  if (start) {
    filteredData = filteredData.filter(item => item.date >= start);
  }
  if (end) {
    filteredData = filteredData.filter(item => item.date <= end);
  }
  
  return [200, filteredData];
});

// Mock para últimas predicciones
mock.onGet('/api/predictions/latest').reply((config) => {
  const queryParams = new URLSearchParams(config.url.split('?')[1]);
  const limit = parseInt(queryParams.get('limit')) || 10;
  
  const latestPredictions = Object.values(predictionData).slice(0, limit);
  return [200, latestPredictions];
});

// Mock para métricas del portfolio
mock.onGet('/api/metrics').reply(200, {
  totalReturn: 12.5,
  dailyReturn: 0.8,
  weeklyReturn: 2.3,
  monthlyReturn: 5.7,
  sharpeRatio: 1.2,
  volatility: 15.3,
  maxDrawdown: -8.2,
  winRate: 65.4
});

// Exportar función para inicializar mocks
export const initMockServices = () => {
  console.log('Mock services initialized');
  return mock;
};

export default initMockServices;