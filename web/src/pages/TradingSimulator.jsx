import React, { useState, useEffect } from 'react';
import { usePortfolio } from '../contexts/PortfolioContext';
import { useMarketData } from '../contexts/MarketDataContext';
import { usePrediction } from '../contexts/PredictionContext';
import { useModelStatus } from '../contexts/ModelStatusContext';

// Componentes
import PortfolioSummary from '../components/trading/PortfolioSummary';
import TradingChart from '../components/charts/TradingChart';
import PredictionTable from '../components/trading/PredictionTable';
import OrderPanel from '../components/trading/OrderPanel';
import ModelHealthIndicator from '../components/models/ModelHealthIndicator';
import TransactionHistory from '../components/trading/TransactionHistory';
import StrategyAdvisor from '../components/trading/StrategyAdvisor';

const TradingSimulator = () => {
  const { portfolio, metrics, transactions, loading: portfolioLoading } = usePortfolio();
  const { marketData, watchlist, selectedTimeframe, changeTimeframe } = useMarketData();
  const { predictions, verificationResults, fetchPredictions } = usePrediction();
  const { modelStatus } = useModelStatus();
  
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [viewMode, setViewMode] = useState('split'); // split, chart, predictions
  const [showStrategyPanel, setShowStrategyPanel] = useState(false);
  
  // Cargar datos para el símbolo seleccionado
  useEffect(() => {
    if (selectedSymbol) {
      fetchPredictions(selectedSymbol);
    }
  }, [selectedSymbol, fetchPredictions]);
  
  // Calcular métricas para el símbolo seleccionado
  const position = portfolio.positions[selectedSymbol] || null;
  const currentPrice = marketData[selectedSymbol]?.quote?.price || 
                     predictions[selectedSymbol]?.currentPrice || 0;
  
  const symbolPredictions = predictions[selectedSymbol]?.predictions || {};

  // Función para cambiar el símbolo seleccionado
  const handleSymbolChange = (event) => {
    setSelectedSymbol(event.target.value);
  };

  return (
    <div className="space-y-4">
      {/* Cabecera con resumen general */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 bg-white p-4 rounded-lg shadow">
        <div className="col-span-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold">Simulador de Trading</h1>
              <select 
                className="border rounded-md p-2 bg-white"
                value={selectedSymbol}
                onChange={handleSymbolChange}
              >
                {watchlist.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
              <div className="text-xl font-semibold">
                ${currentPrice.toFixed(2)}
                {marketData[selectedSymbol]?.quote?.change && (
                  <span className={`ml-2 text-sm ${marketData[selectedSymbol]?.quote?.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {marketData[selectedSymbol]?.quote?.change >= 0 ? '+' : ''}
                    {marketData[selectedSymbol]?.quote?.change.toFixed(2)} 
                    ({marketData[selectedSymbol]?.quote?.percentChange.toFixed(2)}%)
                  </span>
                )}
              </div>
            </div>
            <div className="flex space-x-2">
              <button 
                className={`px-3 py-1 rounded-md ${viewMode === 'split' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
                onClick={() => setViewMode('split')}
              >
                Vista Dividida
              </button>
              <button 
                className={`px-3 py-1 rounded-md ${viewMode === 'chart' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
                onClick={() => setViewMode('chart')}
              >
                Gráfico
              </button>
              <button 
                className={`px-3 py-1 rounded-md ${viewMode === 'predictions' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
                onClick={() => setViewMode('predictions')}
              >
                Predicciones
              </button>
            </div>
          </div>
          
          <div className="mt-4 flex items-center justify-between">
            <div className="flex space-x-2">
              <span className="text-sm text-gray-500">Timeframe:</span>
              {['15m', '30m', '1h', '4h', '1d'].map((tf) => (
                <button
                  key={tf}
                  className={`px-2 py-1 text-xs rounded ${selectedTimeframe === tf ? 'bg-indigo-100 text-indigo-800 font-medium' : 'bg-gray-100'}`}
                  onClick={() => changeTimeframe(tf)}
                >
                  {tf}
                </button>
              ))}
            </div>
            <ModelHealthIndicator status={modelStatus} />
          </div>
        </div>
        <div className="col-span-1">
          <PortfolioSummary portfolio={portfolio} metrics={metrics} />
        </div>
      </div>
      
      {/* Panel de estrategia (expandible/colapsable) */}
      {showStrategyPanel && (
        <div className="bg-white rounded-lg shadow p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-lg font-semibold">Estrategia de Trading IA</h2>
            <button 
              className="text-gray-500 hover:text-gray-700"
              onClick={() => setShowStrategyPanel(false)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
          <StrategyAdvisor symbol={selectedSymbol} />
        </div>
      )}
      
      {/* Contenido principal basado en el modo de vista */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Panel izquierdo - Gráfico y predicciones */}
        <div className={`${viewMode === 'split' ? 'lg:col-span-3' : 'lg:col-span-4'} space-y-4`}>
          {/* Gráfico de trading */}
          {(viewMode === 'split' || viewMode === 'chart') && (
            <div className="bg-white rounded-lg shadow">
              <TradingChart 
                symbol={selectedSymbol}
                timeframe={selectedTimeframe}
                data={marketData[selectedSymbol]?.historicalData}
                predictions={predictions[selectedSymbol]}
              />
            </div>
          )}
          
          {/* Tabla de predicciones */}
          {(viewMode === 'split' || viewMode === 'predictions') && (
            <div className="bg-white rounded-lg shadow p-4">
              <PredictionTable 
                symbol={selectedSymbol}
                predictions={symbolPredictions}
                verificationResults={verificationResults[selectedSymbol]}
              />
            </div>
          )}
          
          {/* Historial de transacciones */}
          {viewMode !== 'split' && (
            <div className="bg-white rounded-lg shadow p-4">
              <TransactionHistory 
                transactions={transactions} 
                selectedSymbol={selectedSymbol}
              />
            </div>
          )}
        </div>
        
        {/* Panel derecho - Órdenes y posición (solo en vista dividida) */}
        {viewMode === 'split' && (
          <div className="lg:col-span-1 space-y-4">
            {/* Panel de órdenes */}
            <div className="bg-white rounded-lg shadow p-4">
              <OrderPanel 
                symbol={selectedSymbol}
                currentPrice={currentPrice}
                position={position}
                cash={portfolio.cash}
              />
            </div>
            
            {/* Botón para mostrar estrategia */}
            {!showStrategyPanel && (
              <button 
                className="w-full py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
                onClick={() => setShowStrategyPanel(true)}
              >
                Ver Estrategia IA
              </button>
            )}
            
            {/* Historial de transacciones */}
            <div className="bg-white rounded-lg shadow p-4">
              <TransactionHistory 
                transactions={transactions} 
                selectedSymbol={selectedSymbol}
                compact={true}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradingSimulator;
