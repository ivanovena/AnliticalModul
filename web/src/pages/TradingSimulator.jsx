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
  console.log('TradingSimulator: Component rendering/re-rendering.');
  const { portfolio, metrics, transactions, initialCash, loading: portfolioLoading, updateInitialCash, resetPortfolio } = usePortfolio();
  const { marketData, watchlist, selectedTimeframe, changeTimeframe } = useMarketData();
  const { predictions, verificationResults, fetchPredictions } = usePrediction();
  const { modelStatus } = useModelStatus();
  
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [viewMode, setViewMode] = useState('split'); // split, chart, predictions
  const [showStrategyPanel, setShowStrategyPanel] = useState(false);
  const [showPortfolioSettings, setShowPortfolioSettings] = useState(false);
  const [newInitialCash, setNewInitialCash] = useState(initialCash);
  
  // Cargar datos para el símbolo seleccionado
  useEffect(() => {
    if (selectedSymbol) {
      fetchPredictions(selectedSymbol);
    }
  }, [selectedSymbol, fetchPredictions]);
  
  // Comprobación principal para estado de carga o error del portfolio
  if (portfolioLoading) {
    return <div className="p-4 text-center">Cargando datos de la cartera...</div>;
  }

  // Nota: Dejamos la comprobación !portfolio aquí por si acaso, aunque el contexto debería manejarlo.
  if (!portfolio) {
    return <div className="p-4 text-center text-red-600">Error al cargar la cartera. No se puede mostrar el simulador.</div>;
  }

  // Calcular métricas para el símbolo seleccionado (con protecciones)
  console.log('TradingSimulator: About to calculate position. Portfolio:', JSON.stringify(portfolio, null, 2));
  const position = (portfolio && portfolio.positions) ? (portfolio.positions[selectedSymbol] || null) : null;
  const currentPrice = marketData[selectedSymbol]?.quote?.price || 
                     predictions[selectedSymbol]?.currentPrice || 0;
  
  const symbolPredictions = predictions[selectedSymbol]?.predictions || {};

  // Función para cambiar el símbolo seleccionado
  const handleSymbolChange = (event) => {
    setSelectedSymbol(event.target.value);
  };

  // Función para actualizar el capital inicial
  const handleUpdateInitialCash = () => {
    updateInitialCash(parseFloat(newInitialCash));
    setShowPortfolioSettings(false);
  };

  // Modal de configuración de cartera
  const PortfolioSettingsModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold">Configuración de Cartera</h2>
          <button 
            className="text-gray-500 hover:text-gray-700"
            onClick={() => setShowPortfolioSettings(false)}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm text-gray-600 mb-2">Capital Inicial ($)</label>
          <input
            type="number"
            className="w-full p-2 border rounded"
            value={newInitialCash}
            onChange={(e) => setNewInitialCash(Math.max(1000, parseFloat(e.target.value) || 1000))}
            min="1000"
            step="1000"
          />
          <p className="text-xs text-gray-500 mt-1">Valor mínimo: $1,000</p>
        </div>
        
        <div className="flex space-x-2">
          <button
            className="flex-1 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
            onClick={handleUpdateInitialCash}
          >
            Guardar
          </button>
          <button
            className="flex-1 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200"
            onClick={() => {
              if (window.confirm('¿Estás seguro? Esto eliminará todas tus posiciones y transacciones.')) {
                resetPortfolio();
                setShowPortfolioSettings(false);
              }
            }}
          >
            Reiniciar Cartera
          </button>
        </div>
      </div>
    </div>
  );

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
                {marketData[selectedSymbol]?.quote?.change != null && (
                  <span className={`ml-2 text-sm ${marketData[selectedSymbol]?.quote?.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {marketData[selectedSymbol]?.quote?.change >= 0 ? '+' : ''}
                    {marketData[selectedSymbol]?.quote?.change.toFixed(2)} 
                    ({(marketData[selectedSymbol]?.quote?.percentChange ?? 0).toFixed(2)}%)
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
          <div className="flex flex-col h-full">
            {/* Restaurado: PortfolioSummary comentado por precaución mientras probamos */} 
            {/* <PortfolioSummary portfolio={portfolio} metrics={metrics} /> */}
            <button
              className="mt-2 text-xs text-indigo-600 hover:text-indigo-800 self-end"
              onClick={() => setShowPortfolioSettings(true)}
            >
              Configurar Cartera
            </button>
          </div>
        </div>
      </div>
      
      {showPortfolioSettings && <PortfolioSettingsModal />}
      
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
          {(viewMode === 'split' || viewMode === 'predictions') && (
            <div className="bg-white rounded-lg shadow p-4">
              <PredictionTable 
                symbol={selectedSymbol}
                predictions={symbolPredictions}
                verificationResults={verificationResults[selectedSymbol]}
              />
            </div>
          )}
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
                cash={portfolio?.cash ?? 0} // Protección añadida aquí
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
