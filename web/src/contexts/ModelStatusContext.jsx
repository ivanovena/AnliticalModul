import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { useNotification } from './NotificationContext';
import { getModelStatus } from '../services/api';

// Crear contexto
const ModelStatusContext = createContext();

// Hook personalizado para usar el contexto
export const useModelStatus = () => useContext(ModelStatusContext);

export const ModelStatusProvider = ({ children }) => {
  const { addNotification } = useNotification();
  const [modelStatus, setModelStatus] = useState({
    online: {
      status: 'unknown',
      accuracy: 0,
      metrics: {
        MAPE: 0,
        RMSE: 0,
        accuracy: 0,
      },
      lastUpdated: null
    },
    batch: {
      status: 'unknown',
      accuracy: 0,
      metrics: {
        MAPE: 0,
        RMSE: 0,
        accuracy: 0,
      },
      lastUpdated: null
    },
    ensemble: {
      status: 'unknown',
      accuracy: 0,
      metrics: {
        MAPE: 0,
        RMSE: 0,
        accuracy: 0,
      },
      lastUpdated: null
    },
    lastUpdated: new Date().toISOString()
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [healthCheckInterval, setHealthCheckInterval] = useState(5 * 60 * 1000); // 5 minutos por defecto

  // Cargar estado inicial de los modelos
  useEffect(() => {
    fetchModelStatus();
    
    // Actualizar según el intervalo configurado
    const intervalId = setInterval(fetchModelStatus, healthCheckInterval);
    
    return () => clearInterval(intervalId);
  }, [healthCheckInterval]);

  // Obtener estado de los modelos
  const fetchModelStatus = useCallback(async () => {
    setLoading(true);
    try {
      // Llamar directamente a la función importada
      const statusData = await getModelStatus();
      setModelStatus(statusData);
      setError(null);
      
      // Verificar si el estado de algún modelo cambió a degradado o crítico
      const previousStatus = modelStatus;
      Object.entries(statusData).forEach(([modelType, status]) => {
        if (modelType !== 'lastUpdated' && status.status !== previousStatus[modelType]?.status) {
          if (status.status === 'degraded' || status.status === 'critical') {
            addNotification({
              type: status.status === 'degraded' ? 'warning' : 'error',
              title: `Modelo ${modelType} ${status.status === 'degraded' ? 'degradado' : 'crítico'}`,
              message: `El rendimiento del modelo ha disminuido. Precisión actual: ${status.accuracy.toFixed(2)}%`,
            });
          } else if (status.status === 'healthy' && previousStatus[modelType]?.status !== 'unknown') {
            addNotification({
              type: 'success',
              title: `Modelo ${modelType} recuperado`,
              message: `El modelo ha vuelto a estado normal. Precisión actual: ${status.accuracy.toFixed(2)}%`,
            });
          }
        }
      });
    } catch (err) {
      console.error('Error al obtener estado de modelos:', err);
      setError('No se pudo obtener el estado de los modelos.');
    } finally {
      setLoading(false);
    }
  }, [addNotification, modelStatus]);

  // Configurar el intervalo de health check
  const setHealthCheckFrequency = (minutes) => {
    const milliseconds = minutes * 60 * 1000;
    setHealthCheckInterval(milliseconds);
  };

  return (
    <ModelStatusContext.Provider 
      value={{
        modelStatus,
        loading,
        error,
        fetchModelStatus,
        setHealthCheckFrequency
      }}
    >
      {children}
    </ModelStatusContext.Provider>
  );
};
