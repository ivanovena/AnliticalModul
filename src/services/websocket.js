// src/services/websocket.js
import { useState, useEffect, useRef, useCallback } from 'react';

// Constantes para estado de la conexión
export const CONNECTION_STATE = {
  CONNECTING: 'connecting',
  OPEN: 'open',
  CLOSING: 'closing',
  CLOSED: 'closed',
  ERROR: 'error'
};

/**
 * Hook para gestionar una conexión WebSocket
 * @param {string} url - URL del WebSocket
 * @param {Object} options - Opciones de configuración
 * @returns {Object} Estado de la conexión y funciones
 */
export const useWebSocket = (url, options = {}) => {
  const {
    onOpen,
    onMessage,
    onClose,
    onError,
    reconnectInterval = 5000,
    reconnectAttempts = 10,
    autoReconnect = true,
    topics = []
  } = options;

  const [connectionState, setConnectionState] = useState(CONNECTION_STATE.CLOSED);
  const [lastMessage, setLastMessage] = useState(null);
  const [reconnectCount, setReconnectCount] = useState(0);
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const subscribedTopicsRef = useRef(topics);

  // Guardar la URL actual como ref
  const urlRef = useRef(url);
  useEffect(() => {
    urlRef.current = url;
  }, [url]);

  // Función para conectar al WebSocket
  const connect = useCallback(() => {
    // Limpiar timeout de reconexión si existe
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Establecer estado como conectando
    setConnectionState(CONNECTION_STATE.CONNECTING);

    // Crear nueva instancia de WebSocket
    const ws = new WebSocket(urlRef.current);
    wsRef.current = ws;

    // Manejador de apertura de conexión
    ws.onopen = (event) => {
      console.log('WebSocket connection established');
      setConnectionState(CONNECTION_STATE.OPEN);
      setReconnectCount(0);
      
      // Suscribirse a los tópicos iniciales
      subscribedTopicsRef.current.forEach(topic => {
        subscribe(topic);
      });
      
      // Ejecutar callback de apertura si existe
      if (onOpen) {
        onOpen(event);
      }
    };

    // Manejador de recepción de mensajes
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
        
        // Ejecutar callback de mensaje si existe
        if (onMessage) {
          onMessage(data);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    // Manejador de cierre de conexión
    ws.onclose = (event) => {
      console.log('WebSocket connection closed');
      setConnectionState(CONNECTION_STATE.CLOSED);
      
      // Ejecutar callback de cierre si existe
      if (onClose) {
        onClose(event);
      }
      
      // Intentar reconexión si está habilitado
      if (autoReconnect && reconnectCount < reconnectAttempts) {
        const nextReconnectCount = reconnectCount + 1;
        setReconnectCount(nextReconnectCount);
        
        console.log(`Attempting to reconnect (${nextReconnectCount}/${reconnectAttempts})...`);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectInterval);
      }
    };

    // Manejador de errores
    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setConnectionState(CONNECTION_STATE.ERROR);
      
      // Ejecutar callback de error si existe
      if (onError) {
        onError(event);
      }
    };
  }, [autoReconnect, onClose, onError, onMessage, onOpen, reconnectAttempts, reconnectCount, reconnectInterval]);

  // Conectar al montar el componente
  useEffect(() => {
    connect();
    
    // Limpiar al desmontar
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Función para enviar un mensaje
  const sendMessage = useCallback((data) => {
    if (wsRef.current && connectionState === CONNECTION_STATE.OPEN) {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      wsRef.current.send(message);
      return true;
    }
    return false;
  }, [connectionState]);

  // Función para suscribirse a un tópico
  const subscribe = useCallback((topic) => {
    if (!subscribedTopicsRef.current.includes(topic)) {
      subscribedTopicsRef.current.push(topic);
    }
    
    if (connectionState === CONNECTION_STATE.OPEN) {
      sendMessage({
        action: 'subscribe',
        topic
      });
    }
  }, [connectionState, sendMessage]);

  // Función para desuscribirse de un tópico
  const unsubscribe = useCallback((topic) => {
    subscribedTopicsRef.current = subscribedTopicsRef.current.filter(t => t !== topic);
    
    if (connectionState === CONNECTION_STATE.OPEN) {
      sendMessage({
        action: 'unsubscribe',
        topic
      });
    }
  }, [connectionState, sendMessage]);

  // Función para cerrar la conexión
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      setConnectionState(CONNECTION_STATE.CLOSING);
      wsRef.current.close();
    }
  }, []);

  // Devolver el estado y funciones
  return {
    connectionState,
    lastMessage,
    sendMessage,
    subscribe,
    unsubscribe,
    connect,
    disconnect
  };
};

/**
 * Hook específico para el WebSocket de actualizaciones del mercado
 * @param {string} symbol - Símbolo a suscribir (opcional)
 * @returns {Object} Estado de la conexión y datos
 */
export const useMarketWebSocket = (symbol = null) => {
  const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8001';
  const [marketData, setMarketData] = useState({});

  // Definir tópicos iniciales
  const initialTopics = ['real-time-stock-update'];
  if (symbol) {
    initialTopics.push(`predictions-${symbol}`);
  }

  // Callback para procesar mensajes
  const handleMessage = useCallback((data) => {
    if (data.type === 'market-update') {
      setMarketData(prevData => ({
        ...prevData,
        [data.symbol]: data
      }));
    } else if (data.type === 'prediction-update') {
      setMarketData(prevData => {
        const symbolData = prevData[data.symbol] || {};
        return {
          ...prevData,
          [data.symbol]: {
            ...symbolData,
            prediction: data.prediction
          }
        };
      });
    }
  }, []);

  // Usar el hook general de WebSocket
  const { 
    connectionState, 
    subscribe, 
    unsubscribe 
  } = useWebSocket(wsUrl, {
    onMessage: handleMessage,
    topics: initialTopics,
    reconnectInterval: 3000
  });

  // Efecto para suscribirse/desuscribirse a cambios en el símbolo
  useEffect(() => {
    if (symbol) {
      subscribe(`predictions-${symbol}`);
      return () => {
        unsubscribe(`predictions-${symbol}`);
      };
    }
  }, [symbol, subscribe, unsubscribe]);

  return {
    connectionState,
    marketData,
    symbolData: symbol ? marketData[symbol] : null
  };
};

/**
 * Hook para el WebSocket de actualizaciones del portfolio
 * @returns {Object} Estado de la conexión y datos del portfolio
 */
export const usePortfolioWebSocket = () => {
  const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8001';
  const [portfolioData, setPortfolioData] = useState(null);
  const [orderUpdates, setOrderUpdates] = useState([]);

  // Callback para procesar mensajes
  const handleMessage = useCallback((data) => {
    if (data.type === 'portfolio-update') {
      setPortfolioData(data.portfolio);
    } else if (data.type === 'order-update') {
      setOrderUpdates(prev => [data.order, ...prev].slice(0, 50));
    }
  }, []);

  // Usar el hook general de WebSocket
  const { 
    connectionState
  } = useWebSocket(wsUrl, {
    onMessage: handleMessage,
    topics: ['portfolio-update', 'order-update'],
    reconnectInterval: 3000
  });

  return {
    connectionState,
    portfolioData,
    orderUpdates
  };
};

export default {
  useWebSocket,
  useMarketWebSocket,
  usePortfolioWebSocket,
  CONNECTION_STATE
};