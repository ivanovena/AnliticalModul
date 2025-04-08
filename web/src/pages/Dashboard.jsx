import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { usePortfolio } from '../contexts/PortfolioContext';
import { useMarketData } from '../contexts/MarketDataContext';
import { usePrediction } from '../contexts/PredictionContext';
import { useModelStatus } from '../contexts/ModelStatusContext';
import { 
  LineChart, Line, 
  XAxis, YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell
} from 'recharts';
import { CircularProgress, Alert } from '@mui/material';
import PortfolioSummary from '../components/dashboard/PortfolioSummary';

// Componente para mostrar un valor de mercado con cambio
const MarketValue = ({ symbol, price, change, onClick }) => {
  const isPositive = change >= 0;
  
  return (
    <div className="p-4 border rounded-lg cursor-pointer hover:shadow-md transition-shadow" onClick={onClick}>
      <div className="flex justify-between items-center">
        <div className="font-medium">{symbol}</div>
        <div className={`px-2 py-1 text-xs rounded-full ${isPositive ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {isPositive ? '+' : ''}{change.toFixed(2)}%
        </div>
      </div>
      <div className="text-2xl font-semibold mt-1">${price.toFixed(2)}</div>
    </div>
  );
};

const Dashboard = () => {
  const { portfolio, metrics, loading: portfolioLoading, error: portfolioError } = usePortfolio();
  const { marketData, watchlist, loading: marketLoading, error: marketError } = useMarketData();
  const { predictions, activeSymbols, loading: predictionLoading, error: predictionError } = usePrediction();
  const { modelStatus, loading: modelLoading, error: modelError } = useModelStatus();
  
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [topPredictions, setTopPredictions] = useState([]);
  
  // Verificar si hay datos disponibles
  const hasPortfolioData = portfolio && Object.keys(portfolio).length > 0;
  const hasMarketData = Object.keys(marketData).length > 0;
  const hasPredictionData = Object.keys(predictions).length > 0;
  
  // Actualizar lista de mejores predicciones
  useEffect(() => {
    if (!activeSymbols || !predictions || activeSymbols.length === 0) return;
    
    // Obtener predicciones para todos los símbolos activos
    const allPredictions = [];
    
    activeSymbols.forEach(symbol => {
      if (predictions[symbol]) {
        const currentPrice = predictions[symbol].currentPrice;
        const prediction1d = predictions[symbol].predictions?.['1d'];
        
        if (currentPrice && prediction1d) {
          const changePct = ((prediction1d - currentPrice) / currentPrice) * 100;
          
          allPredictions.push({
            symbol,
            currentPrice,
            predictedPrice: prediction1d,
            changePct
          });
        }
      }
    });
    
    // Ordenar por cambio porcentual (mayor a menor)
    const sorted = [...allPredictions].sort((a, b) => b.changePct - a.changePct);
    
    // Tomar los 3 mejores y los 2 peores
    const top3 = sorted.slice(0, 3);
    const bottom2 = sorted.slice(-2);
    
    setTopPredictions([...top3, ...bottom2]);
  }, [predictions, activeSymbols]);
  
  // Obtener color basado en el estado
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return '#10b981';
      case 'degraded': return '#f59e0b';
      case 'critical': return '#ef4444';
      default: return '#6b7280';
    }
  };
  
  // Preparar datos para el gráfico de cartera
  const portfolioData = 
    portfolio?.positions
    ? Object.entries(portfolio.positions).map(([symbol, position]) => ({
        name: symbol,
        value: (position?.currentPrice ?? 0) * (position?.quantity ?? 0)
      }))
    : [];
  
  // Agregar efectivo al gráfico de cartera si hay datos disponibles
  if (portfolio?.cash) {
    portfolioData.push({
      name: 'Efectivo',
      value: portfolio.cash
    });
  }
  
  // Colores para el gráfico de cartera
  const PORTFOLIO_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
  
  // Generar datos históricos simulados para el gráfico de valor de cartera
  const generatePortfolioHistory = () => {
    if (!portfolio?.totalValue) return [];
    
    const data = [];
    const now = new Date();
    const days = 30; // Último mes
    
    // Valor actual
    const currentValue = portfolio.totalValue || 100000;
    
    // Generar datos
    for (let i = days; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      
      // Generar un valor que converge hacia el valor actual
      const randomFactor = 1 + (Math.random() - 0.5) * 0.01; // ±0.5% diario
      const dayValue = currentValue * (1 - (i / days) * 0.1) * randomFactor; // Tendencia alcista
      
      data.push({
        date: date.toLocaleDateString(),
        value: dayValue
      });
    }
    
    return data;
  };
  
  // Datos para el gráfico de valor de cartera
  const portfolioHistoryData = generatePortfolioHistory();
  
  // Estado general de carga
  const isLoading = portfolioLoading || marketLoading || predictionLoading || modelLoading;
  
  // Combinar errores para mostrar alertas
  const errors = [];
  if (portfolioError) errors.push(`Error de cartera: ${portfolioError}`);
  if (marketError) errors.push(`Error de mercado: ${marketError}`);
  if (predictionError) errors.push(`Error de predicciones: ${predictionError}`);
  if (modelError) errors.push(`Error de modelos: ${modelError}`);
  
  // Modificación: Comprobación temprana si portfolio es null después de cargar
  if (!portfolioLoading && !portfolio && !portfolioError) {
    return (
      <Alert severity="error">
        No se pudo cargar la información de la cartera. Inténtalo de nuevo más tarde.
      </Alert>
    );
  }
  
  return (
    <div className="space-y-6 relative">
      {isLoading && (
        <div className="fixed top-1/2 right-4 z-50">
          <CircularProgress size={30} />
        </div>
      )}
      
      {errors.length > 0 && (
        <Alert severity="warning" className="mb-4">
          Algunos servicios no están disponibles. Se muestra información guardada previamente.
        </Alert>
      )}
      
      {/* Resumen de cartera */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-500">Valor Total</div>
          <div className="text-2xl font-bold mt-1">
            ${(portfolio?.totalValue ?? 0).toLocaleString('es-ES', { minimumFractionDigits: 2 })}
          </div>
          <div className={`text-sm ${metrics?.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {metrics?.totalReturn >= 0 ? '+' : ''}{(metrics?.totalReturn ?? 0).toFixed(2)}% total
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-500">Efectivo Disponible</div>
          <div className="text-2xl font-bold mt-1">
            ${(portfolio?.cash ?? 0).toLocaleString('es-ES', { minimumFractionDigits: 2 })}
          </div>
          <div className="text-sm text-gray-500">
            {(portfolio?.totalValue ?? 0) > 0 
              ? (((portfolio?.cash ?? 0) / portfolio.totalValue) * 100).toFixed(1) 
              : "0.0"}% del total
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-500">Rendimiento Diario</div>
          <div className="text-2xl font-bold mt-1">
            <span className={metrics?.dailyReturn >= 0 ? 'text-green-600' : 'text-red-600'}>
              {metrics?.dailyReturn >= 0 ? '+' : ''}{(metrics?.dailyReturn ?? 0).toFixed(2)}%
            </span>
          </div>
          <div className="text-sm text-gray-500">
            ${((portfolio?.totalValue ?? 0) * (metrics?.dailyReturn ?? 0) / 100).toLocaleString('es-ES', { minimumFractionDigits: 2 })}
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-500">Estado de Modelos</div>
          <div className="flex items-center mt-1">
            <div className={`w-3 h-3 rounded-full mr-2`} 
                style={{ backgroundColor: getStatusColor(modelStatus?.ensemble?.status || 'unknown') }}></div>
            <div className="text-lg font-semibold">
              {modelStatus?.ensemble?.status === 'healthy' ? 'Normal' : 
               modelStatus?.ensemble?.status === 'degraded' ? 'Degradado' : 
               modelStatus?.ensemble?.status === 'critical' ? 'Crítico' : 'Desconocido'}
            </div>
          </div>
          <div className="text-sm text-gray-500">
            Precisión: {modelStatus?.ensemble?.accuracy?.toFixed(1) || "N/A"}%
          </div>
        </div>
      </div>
      
      {/* Componente PortfolioSummary mejorado */}
      <div className="bg-white rounded-lg shadow">
        <PortfolioSummary />
      </div>
      
      {/* Gráfico de valor de cartera y distribución */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-4">Evolución del Valor de Cartera</h2>
          <div className="h-80">
            {portfolioHistoryData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={portfolioHistoryData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip formatter={(value) => ['$' + value.toLocaleString('es-ES', { minimumFractionDigits: 2 }), 'Valor']} />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    name="Valor de Cartera" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-500">
                No hay datos históricos disponibles
              </div>
            )}
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-4">Distribución de Cartera</h2>
          <div className="h-80">
            {portfolioData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={portfolioData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {portfolioData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={PORTFOLIO_COLORS[index % PORTFOLIO_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => ['$' + value.toLocaleString('es-ES', { minimumFractionDigits: 2 }), 'Valor']} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-500">
                No hay posiciones en la cartera
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Mejores predicciones y valores de mercado */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Mejores predicciones */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">Predicciones Destacadas</h2>
            <Link to="/predictions" className="text-indigo-600 hover:underline text-sm">
              Ver más predicciones →
            </Link>
          </div>
          
          <div className="overflow-x-auto">
            {topPredictions.length > 0 ? (
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Símbolo
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Precio Actual
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Precio a 1d
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Cambio Esperado
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Acción
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {topPredictions.map((prediction, index) => (
                    <tr key={prediction.symbol}>
                      <td className="px-6 py-4 whitespace-nowrap font-medium">
                        {prediction.symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        ${prediction.currentPrice.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        ${prediction.predictedPrice.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={prediction.changePct >= 0 ? 'text-green-600' : 'text-red-600'}>
                          {prediction.changePct >= 0 ? '+' : ''}{prediction.changePct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <Link 
                          to={`/trading?symbol=${prediction.symbol}`}
                          className="text-indigo-600 hover:text-indigo-900"
                        >
                          Operar
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="py-8 text-center text-gray-500">
                {predictionLoading ? 'Cargando predicciones...' : 'No hay predicciones disponibles'}
              </div>
            )}
          </div>
        </div>
        
        {/* Valores de mercado */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">Mercado en Tiempo Real</h2>
            <Link to="/market" className="text-indigo-600 hover:underline text-sm">
              Ver más →
            </Link>
          </div>
          
          <div className="grid grid-cols-1 gap-3">
            {hasMarketData && watchlist && watchlist.length > 0 ? (
              watchlist.slice(0, 5).map(symbol => {
                const dataEntry = marketData[symbol];
                const quoteData = dataEntry?.quote;
                const symbolError = dataEntry?.error;

                // Si hay un error específico para este símbolo, mostrarlo
                if (symbolError) {
                  return (
                    <div key={symbol} className="p-4 border rounded-lg bg-red-50 text-red-700">
                      <div className="font-medium">{symbol}</div>
                      <div className="text-xs mt-1">Error: {symbolError}</div>
                    </div>
                  );
                }
                
                // Si no hay error, mostrar MarketValue (asegurando valores por defecto)
                return (
                  <MarketValue 
                    key={symbol}
                    symbol={symbol}
                    price={quoteData?.price ?? 0}
                    change={quoteData?.percentChange ?? 0}
                    onClick={() => setSelectedSymbol(symbol)}
                  />
                );
              })
            ) : (
              <div className="py-8 text-center text-gray-500">
                {marketLoading ? 'Cargando datos de mercado...' : 'No hay datos de mercado disponibles'}
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Enlaces rápidos */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Link to="/trading" className="bg-white hover:bg-indigo-50 transition-colors p-6 rounded-lg shadow text-center">
          <div className="text-indigo-600 text-4xl mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <h3 className="text-lg font-medium">Simulador de Trading</h3>
          <p className="text-sm text-gray-500 mt-1">Opera en tiempo real con gráficos y predicciones</p>
        </Link>
        
        <Link to="/predictions" className="bg-white hover:bg-indigo-50 transition-colors p-6 rounded-lg shadow text-center">
          <div className="text-indigo-600 text-4xl mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium">Análisis de Predicciones</h3>
          <p className="text-sm text-gray-500 mt-1">Verifica precisión y horizonte temporal</p>
        </Link>
        
        <Link to="/models" className="bg-white hover:bg-indigo-50 transition-colors p-6 rounded-lg shadow text-center">
          <div className="text-indigo-600 text-4xl mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium">Estado de Modelos</h3>
          <p className="text-sm text-gray-500 mt-1">Monitorea rendimiento y métricas</p>
        </Link>
        
        <Link to="/broker-chat" className="bg-white hover:bg-indigo-50 transition-colors p-6 rounded-lg shadow text-center">
          <div className="text-indigo-600 text-4xl mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium">Chat con Broker IA</h3>
          <p className="text-sm text-gray-500 mt-1">Consulta estrategias y recomendaciones</p>
        </Link>
      </div>
    </div>
  );
};

export default Dashboard;
