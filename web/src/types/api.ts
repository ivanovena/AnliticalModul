export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  timestamp: string;
}

export interface Portfolio {
  account_id: string;
  total_value: number;
  cash: number;
  invested_value: number;
  total_profit: number;
  total_profit_pct: number;
  positions: Record<string, Position>;
  last_updated: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  market_value: number;
  current_profit?: number;
  current_profit_pct?: number;
  last_updated?: string;
}

export interface Order {
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp?: string;
  total_value?: number;
}

export interface Prediction {
  symbol: string;
  currentPrice: number;
  predictions: Record<string, number>;
  timestamp: string;
  modelMetrics: {
    MAPE: number;
    RMSE: number;
    accuracy: number;
  };
}

export interface ModelMetric {
  name: string;
  value: number;
  status: 'good' | 'warning' | 'error';
  trend: 'up' | 'down' | 'stable';
}

export interface ModelStatus {
  modelId: string;
  name: string;
  version: string;
  lastUpdated: string;
  status: 'active' | 'training' | 'error';
  metrics: ModelMetric[];
  symbols: string[];
}

export interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
}

export interface Strategy {
  symbol: string;
  summary: string;
  recommendation: {
    action: 'comprar' | 'vender' | 'mantener';
    price: number;
    quantity: number;
    stopLoss: number;
    takeProfit: number;
    confidence: number;
    timeframe: string;
  };
  factors: string[];
  technicalMetrics: Record<string, any>;
  analysis: string;
}

export interface ChatMessage {
  message: string;
  conversation_id?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  timestamp: string;
}

export interface StockPerformance {
  symbol: string;
  quantity: number;
  current_price: number;
  avg_cost: number;
  market_value: number;
  profit: number;
  profit_percent: number;
  prediction: number;
  prediction_direction: 'up' | 'down' | 'neutral';
}

export interface Metrics {
  performance: {
    total_return: number;
    cash_ratio: number;
    positions_count: number;
    trading_frequency: number;
  };
  stock_performance: Record<string, StockPerformance>;
  risk_metrics: {
    portfolio: {
      diversification_score: number;
      cash_ratio: number;
      total_value: number;
    };
  };
}