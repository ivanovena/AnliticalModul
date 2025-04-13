import { useState, useEffect, useRef, useCallback } from 'react';
import { socketService } from '../services/api';

interface UseWebSocketOptions {
  onMessage?: (data: any) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

/**
 * Hook personalizado para manejar conexiones WebSocket con reconexión automática
 */
export const useWebSocket = (url: string, options: UseWebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<Event | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [reconnectStatus, setReconnectStatus] = useState<string | null>(null);
  
  const webSocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10
  } = options;
  
  // Función para conectar al WebSocket
  const connect = useCallback(() => {
    try {
      // Limpiar conexión anterior si existe
      if (webSocketRef.current) {
        webSocketRef.current.close();
      }
      
      // Crear nueva conexión
      const ws = new WebSocket(url);
      webSocketRef.current = ws;
      
      // Configurar handlers
      ws.onopen = () => {
        console.log(`WebSocket connected to ${url}`);
        setIsConnected(true);
        setError(null);
        setReconnectAttempts(0);
        setReconnectStatus(null);
        if (onOpen) onOpen();
      };
      
      ws.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          setLastMessage(parsedData);
          if (onMessage) onMessage(parsedData);
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
          setLastMessage(event.data);
          if (onMessage) onMessage(event.data);
        }
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket disconnected from ${url}`, event);
        setIsConnected(false);
        if (onClose) onClose();
        
        // Intentar reconectar si no fue cerrado intencionalmente
        if (!event.wasClean && reconnectAttempts < maxReconnectAttempts) {
          const nextReconnectAttempt = reconnectAttempts + 1;
          setReconnectAttempts(nextReconnectAttempt);
          setReconnectStatus(`Reconnecting (${nextReconnectAttempt}/${maxReconnectAttempts})...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (reconnectAttempts >= maxReconnectAttempts) {
          setReconnectStatus('Max reconnect attempts reached. Please try again later.');
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(event);
        if (onError) onError(event);
      };
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      setError(error as Event);
      if (onError) onError(error as Event);
    }
  }, [url, onMessage, onOpen, onClose, onError, reconnectAttempts, maxReconnectAttempts, reconnectInterval]);
  
  // Función para enviar mensajes
  const sendMessage = useCallback((data: string | object) => {
    if (!webSocketRef.current || webSocketRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected');
      return false;
    }
    
    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      webSocketRef.current.send(message);
      return true;
    } catch (error) {
      console.error('Error sending message:', error);
      return false;
    }
  }, []);
  
  // Función para cerrar la conexión
  const disconnect = useCallback(() => {
    if (webSocketRef.current) {
      webSocketRef.current.close();
      webSocketRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    setIsConnected(false);
  }, []);
  
  // Conectar al montar el componente
  useEffect(() => {
    connect();
    
    // Limpiar al desmontar
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);
  
  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    disconnect,
    reconnect: connect,
    reconnectStatus,
    reconnectAttempts
  };
};

/**
 * Hook para suscribirse a los datos de mercado en tiempo real
 */
export const useMarketDataWebSocket = (symbol: string, onData?: (data: any) => void) => {
  const wsUrl = socketService.getMarketDataSocketUrl();
  
  const handleMessage = useCallback((data: any) => {
    // Verificar si los datos son para el símbolo que nos interesa
    if (data && data.symbol === symbol) {
      if (onData) onData(data);
    }
  }, [symbol, onData]);
  
  return useWebSocket(wsUrl, {
    onMessage: handleMessage,
    reconnectInterval: 3000,
    maxReconnectAttempts: 20
  });
};

/**
 * Hook para suscribirse a las predicciones en tiempo real
 */
export const usePredictionsWebSocket = (symbol: string, onData?: (data: any) => void) => {
  const wsUrl = socketService.getPredictionsSocketUrl();
  
  const handleMessage = useCallback((data: any) => {
    // Verificar si los datos son para el símbolo que nos interesa
    if (data && data.symbol === symbol) {
      if (onData) onData(data);
    }
  }, [symbol, onData]);
  
  return useWebSocket(wsUrl, {
    onMessage: handleMessage,
    reconnectInterval: 3000,
    maxReconnectAttempts: 20
  });
};

/**
 * Hook para suscribirse a las recomendaciones de trading en tiempo real
 */
export const useRecommendationsWebSocket = (symbol: string, onData?: (data: any) => void) => {
  const wsUrl = socketService.getRecommendationsSocketUrl();
  
  const handleMessage = useCallback((data: any) => {
    // Verificar si los datos son para el símbolo que nos interesa o si son recomendaciones generales
    if (data && (data.symbol === symbol || data.symbol === '*')) {
      if (onData) onData(data);
    }
  }, [symbol, onData]);
  
  return useWebSocket(wsUrl, {
    onMessage: handleMessage,
    reconnectInterval: 3000,
    maxReconnectAttempts: 20
  });
};