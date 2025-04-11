import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { marketService, portfolioService, analysisService } from '../services/api';
import { MarketData, Portfolio, Order, Prediction, ModelStatus, Metrics } from '../types/api';

// Interfaz para el contexto
interface TradingContextType {
  // Estado
  portfolio: Portfolio | null;
  selectedSymbol: string;
  marketData: Record<string, MarketData>;
  predictions: Record<string, Prediction>;
  orders: Order[];
  metrics: Metrics | null;
  availableSymbols: string[];
  modelsStatus: ModelStatus[] | null;
  isLoading: boolean;
  error: string | null;
  
  // Acciones
  setSelectedSymbol: (symbol: string) => void;
  placeOrder: (order: Omit<Order, 'timestamp' | 'total_value'>) => Promise<void>;
  fetchMarketData: (symbol: string) => Promise<void>;
  fetchPrediction: (symbol: string) => Promise<void>;
  refreshPortfolio: () => Promise<void>;
  refreshMetrics: () => Promise<void>;
}

// Crear contexto con valores por defecto
const TradingContext = createContext<TradingContextType | undefined>(undefined);

// Props para el proveedor de contexto
interface TradingProviderProps {
  children: ReactNode;
}

// Proveedor de contexto
export const TradingProvider: React.FC<TradingProviderProps> = ({ children }) => {
  // Estado
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('AAPL');
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  const [predictions, setPredictions] = useState<Record<string, Prediction>>({});
  const [orders, setOrders] = useState<Order[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [modelsStatus, setModelsStatus] = useState<ModelStatus[] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Cargar datos iniciales
  useEffect(() => {
    const loadInitialData = async () => {
      setIsLoading(true);
      try {
        // Cargar lista de símbolos
        const symbols = await marketService.getSymbols();
        setAvailableSymbols(symbols);
        
        // Cargar portafolio
        await refreshPortfolio();
        
        // Cargar órdenes
        const ordersData = await portfolioService.getOrders();
        setOrders(ordersData);
        
        // Cargar métricas
        await refreshMetrics();
        
        // Cargar estado de modelos
        const status = await analysisService.getModelStatus();
        setModelsStatus(status);
        
        // Cargar datos de mercado y predicciones para el símbolo seleccionado
        if (selectedSymbol) {
          await fetchMarketData(selectedSymbol);
          await fetchPrediction(selectedSymbol);
        }
        
        setError(null);
      } catch (err) {
        setError(`Error cargando datos iniciales: ${err instanceof Error ? err.message : String(err)}`);
        console.error('Error cargando datos iniciales:', err);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadInitialData();
    
    // Configurar intervalo para actualizar datos automáticamente
    const intervalId = setInterval(() => {
      if (selectedSymbol) {
        fetchMarketData(selectedSymbol).catch(console.error);
        fetchPrediction(selectedSymbol).catch(console.error);
      }
      refreshPortfolio().catch(console.error);
    }, 30000); // Actualizar cada 30 segundos
    
    return () => clearInterval(intervalId);
  }, []);

  // Actualizar datos de mercado cuando cambia el símbolo seleccionado
  useEffect(() => {
    if (selectedSymbol) {
      fetchMarketData(selectedSymbol).catch(console.error);
      fetchPrediction(selectedSymbol).catch(console.error);
    }
  }, [selectedSymbol]);

  // Funciones para interactuar con la API
  const fetchMarketData = async (symbol: string) => {
    try {
      const data = await marketService.getMarketData(symbol);
      setMarketData(prev => ({
        ...prev,
        [symbol]: data
      }));
    } catch (err) {
      console.error(`Error fetching market data for ${symbol}:`, err);
      throw err;
    }
  };

  const fetchPrediction = async (symbol: string) => {
    try {
      const prediction = await analysisService.getPrediction(symbol);
      setPredictions(prev => ({
        ...prev,
        [symbol]: prediction
      }));
    } catch (err) {
      console.error(`Error fetching prediction for ${symbol}:`, err);
      throw err;
    }
  };

  const refreshPortfolio = async () => {
    try {
      const portfolioData = await portfolioService.getPortfolio();
      setPortfolio(portfolioData);
    } catch (err) {
      console.error('Error refreshing portfolio:', err);
      throw err;
    }
  };

  const refreshMetrics = async () => {
    try {
      const metricsData = await portfolioService.getMetrics();
      setMetrics(metricsData);
    } catch (err) {
      console.error('Error refreshing metrics:', err);
      throw err;
    }
  };

  const placeOrder = async (order: Omit<Order, 'timestamp' | 'total_value'>) => {
    try {
      await portfolioService.placeOrder(order);
      
      // Actualizar portfolio y órdenes después de colocar una orden
      await refreshPortfolio();
      const ordersData = await portfolioService.getOrders();
      setOrders(ordersData);
      await refreshMetrics();
      
      // Actualizar datos de mercado para el símbolo de la orden
      await fetchMarketData(order.symbol);
    } catch (err) {
      console.error('Error placing order:', err);
      throw err;
    }
  };

  // Valor del contexto
  const value: TradingContextType = {
    portfolio,
    selectedSymbol,
    marketData,
    predictions,
    orders,
    metrics,
    availableSymbols,
    modelsStatus,
    isLoading,
    error,
    setSelectedSymbol,
    placeOrder,
    fetchMarketData,
    fetchPrediction,
    refreshPortfolio,
    refreshMetrics
  };

  return (
    <TradingContext.Provider value={value}>
      {children}
    </TradingContext.Provider>
  );
};

// Hook para usar el contexto
export const useTradingContext = () => {
  const context = useContext(TradingContext);
  if (context === undefined) {
    throw new Error('useTradingContext must be used within a TradingProvider');
  }
  return context;
};