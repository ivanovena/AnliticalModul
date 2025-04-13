import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { marketService, portfolioService, analysisService, metricsService } from '../services/api';
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
  placeOrder: (order: Order) => Promise<void>;
  fetchMarketData: (symbol: string) => Promise<void>;
  fetchPrediction: (symbol: string) => Promise<void>;
  refreshPortfolio: () => Promise<void>;
  refreshMetrics: () => Promise<void>;
  refreshModelsStatus: () => Promise<void>;
}

// Crear contexto con valores por defecto
const TradingContext = createContext<TradingContextType | undefined>(undefined);

// Props para el proveedor de contexto
interface TradingProviderProps {
  children: ReactNode;
}

// Proveedor de contexto
export const TradingProvider: React.FC<TradingProviderProps> = ({ children }) => {
  // Estado para símbolos disponibles
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  // Estado para símbolo seleccionado
  const [selectedSymbol, setSelectedSymbol] = useState<string>('AAPL');
  // Estado para datos de mercado (precio, etc.)
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  // Estado para predicciones
  const [predictions, setPredictions] = useState<Record<string, Prediction>>({});
  // Estado para portafolio
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  // Estado para órdenes
  const [orders, setOrders] = useState<Order[]>([]);
  // Estado para métricas
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  // Estado para modelos
  const [modelsStatus, setModelsStatus] = useState<ModelStatus[] | null>(null);
  // Estado para indicar carga
  const [isLoading, setIsLoading] = useState<boolean>(true);
  // Estado para gestionar errores
  const [error, setError] = useState<string | null>(null);

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
      const metricsData = await metricsService.getMetrics();
      setMetrics(metricsData);
    } catch (err) {
      console.error('Error refreshing metrics:', err);
      throw err;
    }
  };

  const refreshModelsStatus = async () => {
    try {
      const status = await analysisService.getModelStatus();
      setModelsStatus(status);
    } catch (err) {
      console.error('Error refreshing models status:', err);
      throw err;
    }
  };

  const placeOrder = async (order: Order) => {
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

  // Obtener datos iniciales cuando se monta el componente
  useEffect(() => {
    const fetchInitialData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Obtener símbolos disponibles
        const symbols = await marketService.getSymbols();
        setAvailableSymbols(symbols);
        
        // Si no hay símbolos seleccionados, seleccionar el primero
        if (!selectedSymbol && symbols.length > 0) {
          setSelectedSymbol(symbols[0]);
        }
        
        // Obtener datos de mercado para todos los símbolos
        const marketDataPromises = symbols.map(async (symbol) => {
          try {
            const data = await marketService.getMarketData(symbol);
            return { symbol, data };
          } catch (err) {
            console.error(`Error fetching market data for ${symbol}:`, err);
            // Devolver un objeto con datos de mercado por defecto en caso de error
            return { symbol, data: null };
          }
        });
        
        const marketDataResults = await Promise.all(marketDataPromises);
        const marketDataMap: Record<string, MarketData> = {};
        
        marketDataResults.forEach(({ symbol, data }) => {
          if (data) {
            marketDataMap[symbol] = data;
          }
        });
        
        setMarketData(marketDataMap);
        
        // Obtener predicciones para todos los símbolos
        const predictionsPromises = symbols.map(async (symbol) => {
          try {
            const prediction = await analysisService.getPrediction(symbol);
            return { symbol, prediction };
          } catch (err) {
            console.error(`Error fetching prediction for ${symbol}:`, err);
            // Crear un objeto de predicción por defecto en caso de error
            return { symbol, prediction: null };
          }
        });
        
        const predictionsResults = await Promise.all(predictionsPromises);
        const predictionsMap: Record<string, Prediction> = {};
        
        predictionsResults.forEach(({ symbol, prediction }) => {
          if (prediction) {
            predictionsMap[symbol] = prediction;
          }
        });
        
        setPredictions(predictionsMap);
        
        // Obtener portafolio
        try {
          const portfolioData = await portfolioService.getPortfolio();
          setPortfolio(portfolioData);
        } catch (err) {
          console.error('Error fetching portfolio:', err);
          // Crear un portafolio por defecto en caso de error
          setPortfolio(null);
        }
        
        // Obtener órdenes
        try {
          const ordersData = await portfolioService.getOrders();
          setOrders(ordersData);
        } catch (err) {
          console.error('Error fetching orders:', err);
          setOrders([]);
        }
        
      } catch (err) {
        console.error('Error initializing trading data:', err);
        setError('Error al cargar los datos iniciales. Por favor, recarga la página.');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchInitialData();
    
    // Cleanup
    return () => {
      // Código de limpieza si es necesario
    };
  }, []);

  // Actualizar datos de mercado cuando cambia el símbolo seleccionado
  useEffect(() => {
    if (selectedSymbol) {
      fetchMarketData(selectedSymbol).catch(console.error);
      fetchPrediction(selectedSymbol).catch(console.error);
    }
  }, [selectedSymbol]);

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
    refreshMetrics,
    refreshModelsStatus
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