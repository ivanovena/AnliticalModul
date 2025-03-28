// src/mocks/mockData.js

// Datos de ejemplo para el portfolio
export const portfolioData = {
  cash: 100000.00,
  positions: [
    {
      symbol: 'AAPL',
      quantity: 10,
      averagePrice: 160.50,
      currentPrice: 165.00
    },
    {
      symbol: 'MSFT',
      quantity: 5,
      averagePrice: 280.75,
      currentPrice: 290.25
    },
    {
      symbol: 'GOOGL',
      quantity: 3,
      averagePrice: 2720.80,
      currentPrice: 2800.50
    }
  ]
};

// Datos de ejemplo para órdenes
export const ordersData = [
  {
    id: 'ord-001',
    symbol: 'AAPL',
    quantity: 10,
    price: 160.50,
    orderType: 'buy',
    status: 'completed',
    timestamp: '2023-05-10T14:30:00Z'
  },
  {
    id: 'ord-002',
    symbol: 'MSFT',
    quantity: 5,
    price: 280.75,
    orderType: 'buy',
    status: 'completed',
    timestamp: '2023-06-15T09:45:00Z'
  },
  {
    id: 'ord-003',
    symbol: 'GOOGL',
    quantity: 3,
    price: 2720.80,
    orderType: 'buy',
    status: 'completed',
    timestamp: '2023-07-20T11:15:00Z'
  }
];

// Datos de ejemplo para cotizaciones
export const quoteData = {
  'AAPL': {
    symbol: 'AAPL',
    price: 165.00,
    change: 2.50,
    changePercent: 1.52,
    volume: 32500000,
    marketCap: 2650000000000
  },
  'MSFT': {
    symbol: 'MSFT',
    price: 290.25,
    change: 3.75,
    changePercent: 1.31,
    volume: 22700000,
    marketCap: 2150000000000
  },
  'GOOGL': {
    symbol: 'GOOGL',
    price: 2800.50,
    change: 35.80,
    changePercent: 1.29,
    volume: 1500000,
    marketCap: 1850000000000
  },
  'AMZN': {
    symbol: 'AMZN',
    price: 130.25,
    change: -1.20,
    changePercent: -0.91,
    volume: 28900000,
    marketCap: 1320000000000
  },
  'TSLA': {
    symbol: 'TSLA',
    price: 250.75,
    change: 5.50,
    changePercent: 2.24,
    volume: 42500000,
    marketCap: 780000000000
  }
};

// Datos de ejemplo para predicciones
export const predictionData = {
  'AAPL': {
    symbol: 'AAPL',
    currentPrice: 165.00,
    predictedPrice: 168.00,
    confidence: 0.87,
    trend: 'bullish',
    recommendation: 'buy',
    targetPrice: 185.00
  },
  'MSFT': {
    symbol: 'MSFT',
    currentPrice: 290.25,
    predictedPrice: 305.50,
    confidence: 0.82,
    trend: 'bullish',
    recommendation: 'buy',
    targetPrice: 320.00
  },
  'GOOGL': {
    symbol: 'GOOGL',
    currentPrice: 2800.50,
    predictedPrice: 2900.00,
    confidence: 0.75,
    trend: 'bullish',
    recommendation: 'hold',
    targetPrice: 3000.00
  },
  'AMZN': {
    symbol: 'AMZN',
    currentPrice: 130.25,
    predictedPrice: 125.00,
    confidence: 0.68,
    trend: 'bearish',
    recommendation: 'sell',
    targetPrice: 120.00
  },
  'TSLA': {
    symbol: 'TSLA',
    currentPrice: 250.75,
    predictedPrice: 280.00,
    confidence: 0.78,
    trend: 'bullish',
    recommendation: 'buy',
    targetPrice: 300.00
  }
};

// Datos de ejemplo para históricos
export const historicalData = {
  'AAPL': Array.from({ length: 180 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (180 - i));
    const basePrice = 150;
    const noise = Math.sin(i / 20) * 20 + Math.random() * 10 - 5;
    const trend = i / 10;
    const price = basePrice + noise + trend;
    
    return {
      date: date.toISOString().split('T')[0],
      price: Math.max(price, 120).toFixed(2)
    };
  }),
  'MSFT': Array.from({ length: 180 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (180 - i));
    const basePrice = 260;
    const noise = Math.sin(i / 25) * 25 + Math.random() * 12 - 6;
    const trend = i / 8;
    const price = basePrice + noise + trend;
    
    return {
      date: date.toISOString().split('T')[0],
      price: Math.max(price, 220).toFixed(2)
    };
  })
};

// Datos de ejemplo para predicciones históricas
export const historicalPredictions = {
  'AAPL': Array.from({ length: 180 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (180 - i));
    const basePrice = 150;
    const noise = Math.sin(i / 20) * 20 + Math.random() * 10 - 5;
    const trend = i / 10;
    const price = basePrice + noise + trend;
    const predictedPrice = price * (1 + (Math.random() * 0.06 - 0.02));
    
    return {
      date: date.toISOString().split('T')[0],
      predictedPrice: Math.max(predictedPrice, 120).toFixed(2)
    };
  }),
  'MSFT': Array.from({ length: 180 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (180 - i));
    const basePrice = 260;
    const noise = Math.sin(i / 25) * 25 + Math.random() * 12 - 6;
    const trend = i / 8;
    const price = basePrice + noise + trend;
    const predictedPrice = price * (1 + (Math.random() * 0.06 - 0.02));
    
    return {
      date: date.toISOString().split('T')[0],
      predictedPrice: Math.max(predictedPrice, 220).toFixed(2)
    };
  })
};

export default {
  portfolioData,
  ordersData,
  quoteData,
  predictionData,
  historicalData,
  historicalPredictions
};