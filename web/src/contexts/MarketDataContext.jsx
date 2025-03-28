import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

// Crear contexto
const MarketDataContext = createContext();

// Hook personalizado para usar el contexto
export const useMarketData = () => useContext(MarketDataContext);

export const MarketDataProvider = ({ children }) => {
  const [marketData, setMarketData] = useState({});
  const [watchlist, setWatchlist] = useState(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'IAG.MC', 'PHM.MC', 'BKY.MC', 'AENA.MC', 'BA', 'NLGO', 'CAR', 'DLTR', 'CANTE.IS', 'SASA.IS']);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [realTimeEnabled, setRealTimeEnabled] = useState(true);
  const [websocket, setWebsocket] = useState(null);

  // Conectar WebSocket para datos en tiempo real
  useEffect(() => {
    if (realTimeEnabled && watchlist.length > 0) {
      const connectWebSocket = () => {
        try {
          // En un sistema real, esto se conectaría al servidor WebSocket
          // Para este ejemplo, simulamos con la API de socket.io-client
          const socket = api.connectToMarketSocket(watchlist);
          
          socket.on('market_update', (data) => {
            if (data && data.symbol) {
              setMarketData(prev => ({
                ...prev,
                [data.symbol]: {
                  ...prev[data.symbol],
                  price: data.price,
                  change: data.change,
                  percentChange: data.percentChange,
                  volume: data.volume,
                  lastUpdate: new Date().toISOString()
                }
              }));
            }
          });
          
          socket.on('error', (err) => {
            console.error('Error en WebSocket:', err);
            setError('Se perdió la conexión con el mercado en tiempo real');
          });
          
          socket.on('disconnect', () => {
            console.log('Desconectado del WebSocket');
            // Intentar reconectar después de 3 segundos
            setTimeout(connectWebSocket, 3000);
          });
          
          setWebsocket(socket);
          
          // Limpiar al desmontar
          return () => {
            if (socket) {
              socket.disconnect();
            }
          };
        } catch (err) {
          console.error('Error al conectar WebSocket:', err);
          setError('No se pudo establecer conexión en tiempo real');
          // Intentar reconectar después de 5 segundos
          setTimeout(connectWebSocket, 5000);
        }
      };
      
      connectWebSocket();
    } else {
      // Desconectar si está deshabilitado
      if (websocket) {
        websocket.disconnect();
        setWebsocket(null);
      }
    }
    
    return () => {
      if (websocket) {
        websocket.disconnect();
        setWebsocket(null);
      }
    };
  }, [realTimeEnabled, watchlist]);

  // Cargar datos de mercado para los símbolos en la watchlist
  useEffect(() => {
    if (watchlist.length > 0) {
      watchlist.forEach(symbol => {
        fetchMarketData(symbol);
      });
      
      if (!realTimeEnabled) {
        // Si no hay tiempo real, actualizar cada minuto
        const intervalId = setInterval(() => {
          watchlist.forEach(symbol => {
            fetchMarketData(symbol);
          });
        }, 60000);
        
        return () => clearInterval(intervalId);
      }
    }
  }, [watchlist, realTimeEnabled]);

  // Obtener datos de mercado para un símbolo
  const fetchMarketData = useCallback(async (symbol, timeframe = selectedTimeframe) => {
    if (!symbol) return;
    
    setLoading(true);
    try {
      const [quote, historicalData] = await Promise.all([
        api.getQuote(symbol),
        api.getHistoricalData(symbol, timeframe)
      ]);
      
      setMarketData(prev => ({
        ...prev,
        [symbol]: {
          ...prev[symbol],
          quote,
          historicalData,
          lastUpdate: new Date().toISOString()
        }
      }));
      
      setLoading(false);
    } catch (err) {
      console.error(`Error al obtener datos de mercado para ${symbol}:`, err);
      setError(`No se pudieron cargar los datos de mercado para ${symbol}`);
      setLoading(false);
    }
  }, [selectedTimeframe]);

  // Cambiar timeframe seleccionado
  const changeTimeframe = (timeframe) => {
    setSelectedTimeframe(timeframe);
    // Actualizar datos para el nuevo timeframe
    watchlist.forEach(symbol => {
      fetchMarketData(symbol, timeframe);
    });
  };

  // Agregar o quitar símbolo de la watchlist
  const toggleWatchlistSymbol = (symbol) => {
    if (watchlist.includes(symbol)) {
      setWatchlist(prev => prev.filter(s => s !== symbol));
    } else {
      setWatchlist(prev => [...prev, symbol]);
      fetchMarketData(symbol);
    }
  };

  // Habilitar/deshabilitar datos en tiempo real
  const toggleRealTime = () => {
    setRealTimeEnabled(!realTimeEnabled);
  };

  return (
    <MarketDataContext.Provider 
      value={{
        marketData,
        watchlist,
        selectedTimeframe,
        loading,
        error,
        realTimeEnabled,
        fetchMarketData,
        changeTimeframe,
        toggleWatchlistSymbol,
        toggleRealTime
      }}
    >
      {children}
    </MarketDataContext.Provider>
  );
};
