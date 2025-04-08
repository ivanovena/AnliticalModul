import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { connectMarketDataWebSocket, getHistoricalData, getMarketData } from '../services/api';

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
  const socketRef = React.useRef(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  // Obtener datos de mercado para un símbolo
  const fetchMarketData = useCallback(async (symbol, timeframe = selectedTimeframe) => {
    if (!symbol) return;

    setLoading(true);
    try {
      // Llamar directamente a las funciones importadas
      // ASUNCIÓN: getMarketData devuelve { quote: {...}, ... } o similar
      const marketInfo = await getMarketData(symbol);
      const historical = await getHistoricalData(symbol, timeframe);

      setMarketData(prev => ({
        ...prev,
        [symbol]: {
          ...prev[symbol],
          quote: marketInfo, // Asumiendo que marketInfo contiene los datos de quote
          historicalData: historical,
          lastUpdate: new Date().toISOString()
        }
      }));

      setLoading(false);
    } catch (err) {
      console.error(`Error al obtener datos de mercado para ${symbol}:`, err);
      setError(`Error al cargar datos para ${symbol}`);
      setMarketData(prev => ({
        ...prev,
        [symbol]: {
          ...(prev && prev[symbol] ? prev[symbol] : {}),
          quote: null,
          error: err?.response?.data?.detail || err?.message || 'Error desconocido',
          lastUpdate: new Date().toISOString()
        }
      }));
      setLoading(false);
    }
  }, [selectedTimeframe]);

  // Conectar WebSocket para datos en tiempo real
  const connectWebSocket = useCallback(() => {
    console.log("Intentando conectar WebSocket de Mercado con watchlist:", watchlist);
    // Cerrar conexión existente si la hay
    if (socketRef.current) {
      socketRef.current.close();
    }

    // Llamar directamente a la función importada
    const newSocket = connectMarketDataWebSocket( // Ya no necesitamos `api.`
      watchlist, // Símbolos a suscribir (puede necesitar ajuste según backend)
      () => {
        console.log("Callback onOpen llamado desde MarketDataContext");
        setConnectionStatus('connected');
      },
      (data) => {
        // Procesar mensaje
        if (data && data.symbol) {
          console.debug("Actualizando datos para:", data.symbol, data);
          // Actualizar SOLO la información que llega del WS (precio, cambio, etc.)
          // Evitar sobreescribir historicalData
          setMarketData(prevData => ({
            ...prevData,
            [data.symbol]: {
              ...prevData[data.symbol], // Mantener datos existentes como historicalData
              ...data, // Actualizar con los nuevos datos del WS
              lastUpdate: new Date().toISOString()
            }
           }));
        }
      },
      (error) => {
        console.error("Error WebSocket en MarketDataContext:", error);
        setConnectionStatus('error');
      },
      () => {
        console.log("Callback onClose llamado desde MarketDataContext");
        setConnectionStatus('disconnected');
        socketRef.current = null;
        // Opcional: intentar reconectar aquí
        // setTimeout(connectWebSocket, 5000); // Reintentar después de 5 segundos
      }
    );

    socketRef.current = newSocket;

  }, [watchlist]);

  useEffect(() => {
    if (watchlist.length > 0) {
      connectWebSocket();
    }

    // Limpieza al desmontar el componente
    return () => {
      if (socketRef.current) {
        console.log("Desmontando MarketDataContext, cerrando WebSocket.");
        socketRef.current.close();
      }
    };
  }, [watchlist, connectWebSocket]); // Ejecutar cuando watchlist o la función cambien

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
        toggleRealTime,
        connectionStatus
      }}
    >
      {children}
    </MarketDataContext.Provider>
  );
};
