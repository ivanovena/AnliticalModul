import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { useNotification } from './NotificationContext';
import { getAllPredictions, getPrediction } from '../services/api';

// Crear contexto
const PredictionContext = createContext();

// Hook personalizado para usar el contexto
export const usePrediction = () => useContext(PredictionContext);

export const PredictionProvider = ({ children }) => {
  const { addNotification } = useNotification();
  const [predictions, setPredictions] = useState({});
  const [predictionHistory, setPredictionHistory] = useState({});
  const [verificationResults, setVerificationResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeSymbols, setActiveSymbols] = useState(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']);
  const [loadingSymbol, setLoadingSymbol] = useState(null);

  // Cargar predicciones iniciales para símbolos activos
  useEffect(() => {
    activeSymbols.forEach(symbol => {
      fetchSinglePrediction(symbol);
    });
    
    // Actualizar cada 5 minutos
    const intervalId = setInterval(() => {
      activeSymbols.forEach(symbol => {
        fetchSinglePrediction(symbol);
      });
    }, 5 * 60 * 1000);
    
    return () => clearInterval(intervalId);
  }, [activeSymbols]);

  // Obtener predicciones para un símbolo
  const fetchSinglePrediction = useCallback(async (symbol) => {
    if (!symbol) return;
    setLoadingSymbol(symbol);
    try {
      // Llamar directamente a la función importada
      const data = await getPrediction(symbol);
      setPredictions(prev => ({ ...prev, [symbol]: data }));
      setError(null);
    } catch (err) {
      console.error(`Error al obtener predicción para ${symbol}:`, err);
      setError(`No se pudo cargar la predicción para ${symbol}.`);
      // Mantener la predicción anterior si existe
    } finally {
      setLoadingSymbol(null);
    }
  }, []);

  // Cargar historial de predicciones
  const fetchPredictionHistory = useCallback(async (symbol) => {
    if (!symbol) return;
    
    try {
      const response = await api.getPredictionHistory(symbol);
      
      setPredictionHistory(prev => ({
        ...prev,
        [symbol]: response
      }));
    } catch (err) {
      console.error(`Error al obtener historial de predicciones para ${symbol}:`, err);
      setError(`No se pudo cargar el historial de predicciones para ${symbol}`);
    }
  }, []);

  // Verificar precisión de predicciones
  const verifyPrediction = useCallback((symbol, horizon, predictedPrice, actualPrice) => {
    // Calcular error
    const errorValue = Math.abs(actualPrice - predictedPrice);
    const errorPercentage = (errorValue / actualPrice) * 100;
    
    // Agregar al historial de verificaciones
    const timestamp = new Date().toISOString();
    
    setVerificationResults(prev => {
      const symbolResults = prev[symbol] || {};
      const horizonResults = symbolResults[horizon] || [];
      
      const updatedResults = [
        ...horizonResults,
        {
          timestamp,
          predictedPrice,
          actualPrice,
          errorValue,
          errorPercentage,
        }
      ];
      
      // Calcular MAPE (Mean Absolute Percentage Error)
      const mape = updatedResults.reduce((sum, r) => sum + r.errorPercentage, 0) / updatedResults.length;
      
      // Si el error es superior al 10%, enviar notificación
      if (errorPercentage > 10) {
        addNotification({
          type: 'warning',
          title: 'Error alto en predicción',
          message: `La predicción para ${symbol} (${horizon}) tiene un error del ${errorPercentage.toFixed(2)}%`,
        });
      }
      
      return {
        ...prev,
        [symbol]: {
          ...symbolResults,
          [horizon]: updatedResults,
          mape: mape
        }
      };
    });
  }, [addNotification]);

  // Agregar o eliminar un símbolo de la lista de seguimiento
  const toggleSymbol = useCallback((symbol) => {
    setActiveSymbols(prev => {
      if (prev.includes(symbol)) {
        return prev.filter(s => s !== symbol);
      } else {
        const newSymbols = [...prev, symbol];
        fetchSinglePrediction(symbol);
        return newSymbols;
      }
    });
  }, [fetchSinglePrediction]);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    try {
      // Llamar directamente a la función importada
      const data = await getAllPredictions();
      setPredictions(data);
      setError(null);
    } catch (err) {
      console.error('Error al obtener predicciones:', err);
      setError('No se pudieron cargar las predicciones.');
      setPredictions({}); // En caso de error, establecer un objeto vacío
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <PredictionContext.Provider 
      value={{
        predictions,
        predictionHistory,
        verificationResults,
        loading,
        error,
        activeSymbols,
        fetchPredictions,
        fetchPredictionHistory,
        verifyPrediction,
        toggleSymbol
      }}
    >
      {children}
    </PredictionContext.Provider>
  );
};
