import React, { useState, useEffect } from 'react';
import { usePortfolio } from '../../contexts/PortfolioContext';
import { api } from '../../services/api';

const StrategyAdvisor = ({ symbol }) => {
  const { placeOrder } = usePortfolio();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [strategy, setStrategy] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Cargar estrategia para el símbolo seleccionado
  useEffect(() => {
    const fetchStrategy = async () => {
      if (!symbol) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const response = await api.getStrategy(symbol);
        setStrategy(response);
        setLoading(false);
      } catch (err) {
        console.error(`Error al obtener estrategia para ${symbol}:`, err);
        setError(`No se pudo cargar la estrategia para ${symbol}`);
        setLoading(false);
      }
    };
    
    fetchStrategy();
  }, [symbol]);
  
  // Ejecutar estrategia recomendada
  const executeStrategy = async () => {
    if (!strategy || !strategy.recommendation) return;
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Convertir la recomendación en una orden
      const orderData = {
        symbol,
        action: strategy.recommendation.action.toLowerCase() === 'comprar' ? 'buy' : 'sell',
        quantity: strategy.recommendation.quantity,
        price: strategy.recommendation.price,
        orderType: 'market'
      };
      
      // Validar que sea una acción real (comprar o vender)
      if (orderData.action !== 'buy' && orderData.action !== 'sell') {
        throw new Error('La recomendación actual es mantener, no se puede ejecutar una orden');
      }
      
      await placeOrder(orderData);
      setSuccess('Estrategia ejecutada con éxito');
      setLoading(false);
    } catch (err) {
      console.error('Error al ejecutar estrategia:', err);
      setError(err.message || 'Error al ejecutar la estrategia');
      setLoading(false);
    }
  };
  
  // Obtener badge según la acción recomendada
  const getActionBadge = (action) => {
    if (!action) return null;
    
    const actionLower = action.toLowerCase();
    if (actionLower === 'comprar' || actionLower === 'buy') {
      return <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-semibold">Comprar</span>;
    } else if (actionLower === 'vender' || actionLower === 'sell') {
      return <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs font-semibold">Vender</span>;
    } else {
      return <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs font-semibold">Mantener</span>;
    }
  };
  
  // Formatear metros técnicos
  const formatMetric = (value) => {
    if (typeof value === 'number') {
      return value.toLocaleString('es-ES', { maximumFractionDigits: 2 });
    }
    return value;
  };
  
  if (loading && !strategy) {
    return (
      <div className="flex justify-center items-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500"></div>
        <span className="ml-2 text-indigo-700">Cargando estrategia...</span>
      </div>
    );
  }
  
  if (error && !strategy) {
    return (
      <div className="p-4 bg-red-100 text-red-800 rounded-md">
        <p>{error}</p>
      </div>
    );
  }
  
  if (!strategy) {
    return (
      <div className="p-4 bg-gray-100 text-gray-700 rounded-md">
        <p>No hay estrategia disponible para {symbol}</p>
      </div>
    );
  }
  
  return (
    <div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Columna 1: Recomendación principal */}
        <div className="lg:col-span-2">
          <div className="mb-4">
            <div className="flex justify-between items-center">
              <h3 className="text-md font-semibold">Recomendación para {symbol}</h3>
              {getActionBadge(strategy.recommendation?.action)}
            </div>
            <p className="mt-2 text-sm">{strategy.summary}</p>
          </div>
          
          {/* Factores clave */}
          {strategy.factors && strategy.factors.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold mb-2">Factores clave:</h4>
              <ul className="text-sm list-disc list-inside space-y-1">
                {strategy.factors.map((factor, index) => (
                  <li key={index}>{factor}</li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Detalles de la recomendación */}
          {strategy.recommendation && (
            <div className="mb-4 p-3 bg-gray-50 rounded-md">
              <h4 className="text-sm font-semibold mb-2">Detalles de la operación:</h4>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Acción:</span>
                  <span className="font-medium">{strategy.recommendation.action}</span>
                </div>
                
                {strategy.recommendation.price && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Precio objetivo:</span>
                    <span className="font-medium">${strategy.recommendation.price.toFixed(2)}</span>
                  </div>
                )}
                
                {strategy.recommendation.quantity && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cantidad:</span>
                    <span className="font-medium">{strategy.recommendation.quantity}</span>
                  </div>
                )}
                
                {strategy.recommendation.confidence && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Confianza:</span>
                    <span className="font-medium">{strategy.recommendation.confidence}%</span>
                  </div>
                )}
                
                {strategy.recommendation.stopLoss && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Stop Loss:</span>
                    <span className="font-medium">${strategy.recommendation.stopLoss.toFixed(2)}</span>
                  </div>
                )}
                
                {strategy.recommendation.takeProfit && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Take Profit:</span>
                    <span className="font-medium">${strategy.recommendation.takeProfit.toFixed(2)}</span>
                  </div>
                )}
                
                {strategy.recommendation.timeframe && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Horizonte:</span>
                    <span className="font-medium">{strategy.recommendation.timeframe}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Columna 2: Indicadores técnicos y botón de ejecución */}
        <div className="lg:col-span-1">
          {/* Indicadores técnicos */}
          {strategy.technicalMetrics && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold mb-2">Indicadores técnicos:</h4>
              <div className="p-3 bg-gray-50 rounded-md space-y-2">
                {Object.entries(strategy.technicalMetrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-gray-600">{key}:</span>
                    <span className="font-medium">{formatMetric(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Mensajes de éxito/error */}
          {success && (
            <div className="p-3 mb-4 bg-green-100 text-green-800 text-sm rounded-md">
              {success}
            </div>
          )}
          
          {error && (
            <div className="p-3 mb-4 bg-red-100 text-red-800 text-sm rounded-md">
              {error}
            </div>
          )}
          
          {/* Botón de ejecución */}
          {strategy.recommendation && 
           (strategy.recommendation.action.toLowerCase() === 'comprar' || 
            strategy.recommendation.action.toLowerCase() === 'vender') && (
            <button
              className={`w-full py-2 px-4 rounded-md text-white font-medium
                        ${strategy.recommendation.action.toLowerCase() === 'comprar'
                          ? 'bg-green-600 hover:bg-green-700'
                          : 'bg-red-600 hover:bg-red-700'}`}
              onClick={executeStrategy}
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Procesando...
                </span>
              ) : (
                `Ejecutar ${strategy.recommendation.action} de ${strategy.recommendation.quantity} ${symbol}`
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default StrategyAdvisor;
