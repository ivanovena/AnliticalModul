import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { useNotification } from './NotificationContext';

// Crear contexto
const PortfolioContext = createContext();

// Hook personalizado para usar el contexto
export const usePortfolio = () => useContext(PortfolioContext);

export const PortfolioProvider = ({ children }) => {
  const { addNotification } = useNotification();
  const [portfolio, setPortfolio] = useState({
    cash: 100000,
    initialCash: 100000,
    positions: {},
    totalValue: 100000,
    lastUpdate: new Date().toISOString()
  });
  const [metrics, setMetrics] = useState({
    totalReturn: 0,
    dailyReturn: 0,
    weeklyReturn: 0,
    monthlyReturn: 0,
    sharpeRatio: 0,
    volatility: 0
  });
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Cargar datos iniciales
  useEffect(() => {
    fetchPortfolio();
    fetchMetrics();
    fetchTransactions();
    
    // Actualizar datos cada minuto
    const intervalId = setInterval(() => {
      fetchPortfolio();
    }, 60000);
    
    return () => clearInterval(intervalId);
  }, []);

  // Obtener portafolio
  const fetchPortfolio = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.getPortfolio();
      setPortfolio(response);
      setLoading(false);
    } catch (err) {
      console.error('Error al cargar el portafolio:', err);
      setError('No se pudo cargar la información del portafolio');
      setLoading(false);
    }
  }, []);

  // Obtener métricas
  const fetchMetrics = useCallback(async () => {
    try {
      const response = await api.getPortfolioMetrics();
      setMetrics(response);
    } catch (err) {
      console.error('Error al cargar métricas:', err);
      setError('No se pudieron cargar las métricas del portafolio');
    }
  }, []);

  // Obtener historial de transacciones
  const fetchTransactions = useCallback(async () => {
    try {
      const response = await api.getOrders();
      setTransactions(response);
    } catch (err) {
      console.error('Error al cargar transacciones:', err);
      setError('No se pudo cargar el historial de transacciones');
    }
  }, []);

  // Colocar orden
  const placeOrder = useCallback(async (orderData) => {
    setLoading(true);
    try {
      const response = await api.placeOrder(orderData);
      
      // Actualizar portafolio y transacciones
      await Promise.all([
        fetchPortfolio(),
        fetchTransactions()
      ]);
      
      // Notificar al usuario
      addNotification({
        type: 'success',
        title: 'Orden ejecutada',
        message: `${orderData.action === 'buy' ? 'Compra' : 'Venta'} de ${orderData.quantity} ${orderData.symbol} a $${orderData.price.toFixed(2)}`,
      });
      
      setLoading(false);
      return response;
    } catch (err) {
      console.error('Error al colocar orden:', err);
      
      // Notificar error
      addNotification({
        type: 'error',
        title: 'Error en la orden',
        message: err.message || 'No se pudo procesar la orden',
      });
      
      setError(err.message || 'Error al procesar la orden');
      setLoading(false);
      throw err;
    }
  }, [fetchPortfolio, fetchTransactions, addNotification]);

  return (
    <PortfolioContext.Provider 
      value={{
        portfolio,
        metrics,
        transactions,
        loading,
        error,
        fetchPortfolio,
        fetchMetrics,
        fetchTransactions,
        placeOrder
      }}
    >
      {children}
    </PortfolioContext.Provider>
  );
};
