import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getPortfolio, getPortfolioMetrics, getOrders, placeOrder } from '../services/api';
import { useNotification } from './NotificationContext';

// Crear contexto
const PortfolioContext = createContext();

// Hook personalizado para usar el contexto
export const usePortfolio = () => useContext(PortfolioContext);

export const PortfolioProvider = ({ children }) => {
  const { addNotification } = useNotification();
  
  // Comprobar si hay una cantidad inicial guardada en localStorage
  const savedInitialCash = localStorage.getItem('initialCash');
  const defaultInitialCash = savedInitialCash ? parseFloat(savedInitialCash) : 100000;
  
  const [initialCash, setInitialCash] = useState(defaultInitialCash);
  const [portfolio, setPortfolio] = useState({
    cash: defaultInitialCash,
    initialCash: defaultInitialCash,
    positions: {},
    totalValue: defaultInitialCash,
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
      // Llamar directamente a la función importada
      const portfolioData = await getPortfolio(); 
      setPortfolio(portfolioData);
      setError(null);
    } catch (err) {
      console.error('Error al cargar el portafolio:', err);
      setError('No se pudo cargar el portafolio.');
      // Podrías establecer un portafolio vacío o de fallback aquí si es necesario
      setPortfolio(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Obtener métricas
  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    try {
      const metricsData = await getPortfolioMetrics();
      setMetrics(metricsData);
    } catch (err) {
      console.error('Error al cargar métricas:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Obtener historial de transacciones
  const fetchTransactions = useCallback(async () => {
    setLoading(true);
    try {
      // Llamar directamente a la función importada
      const ordersData = await getOrders();
      setTransactions(ordersData);
    } catch (err) {
      console.error('Error al cargar transacciones:', err);
      // Manejar error de órdenes
    } finally {
      setLoading(false);
    }
  }, []);

  // Colocar orden
  const executeOrder = useCallback(async (orderData) => {
    try {
      // Llamar directamente a la función importada
      const newOrder = await placeOrder(orderData);
      // Actualizar estado local (órdenes, portafolio) después de una orden exitosa
      fetchTransactions();
      fetchPortfolio(); 
      fetchMetrics();
      return { success: true, order: newOrder };
    } catch (err) {
      console.error('Error al ejecutar orden:', err);
      return { success: false, error: err.message || 'Error desconocido al colocar la orden' };
    }
  }, [fetchTransactions, fetchPortfolio, fetchMetrics]);

  // Modificar el dinero inicial
  const updateInitialCash = useCallback((newAmount) => {
    if (newAmount <= 0) {
      addNotification({
        type: 'error',
        title: 'Valor inválido',
        message: 'El capital inicial debe ser mayor que cero',
      });
      return;
    }
    
    setInitialCash(newAmount);
    localStorage.setItem('initialCash', newAmount.toString());
    
    // Resetear cartera si no hay posiciones abiertas
    if (Object.keys(portfolio.positions).length === 0) {
      setPortfolio(prev => ({
        ...prev,
        cash: newAmount,
        initialCash: newAmount,
        totalValue: newAmount
      }));
      
      addNotification({
        type: 'success',
        title: 'Capital actualizado',
        message: `Capital inicial establecido en $${newAmount.toLocaleString('es-ES')}`,
      });
    } else {
      addNotification({
        type: 'warning',
        title: 'Capital actualizado parcialmente',
        message: 'El nuevo capital inicial se aplicará cuando cierres todas tus posiciones',
      });
    }
  }, [portfolio, addNotification]);

  // Resetear cartera
  const resetPortfolio = useCallback(() => {
    const newPortfolio = {
      cash: initialCash,
      initialCash: initialCash,
      positions: {},
      totalValue: initialCash,
      lastUpdate: new Date().toISOString()
    };
    
    setPortfolio(newPortfolio);
    setTransactions([]);
    
    addNotification({
      type: 'info',
      title: 'Cartera reiniciada',
      message: 'Se ha reiniciado tu cartera con el capital inicial configurado',
    });
  }, [initialCash, addNotification]);

  return (
    <PortfolioContext.Provider 
      value={{
        portfolio,
        metrics,
        transactions,
        initialCash,
        loading,
        error,
        fetchPortfolio,
        fetchMetrics,
        fetchTransactions,
        executeOrder,
        updateInitialCash,
        resetPortfolio
      }}
    >
      {children}
    </PortfolioContext.Provider>
  );
};
