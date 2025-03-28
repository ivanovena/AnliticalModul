import React, { useState, useEffect } from 'react';
import { useMarketData } from '../contexts/MarketDataContext';
import { usePrediction } from '../contexts/PredictionContext';
import { Link } from 'react-router-dom';
import {
  LineChart, Line,
  XAxis, YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts';

const MarketAnalysis = () => {
  const { marketData, watchlist, selectedTimeframe, changeTimeframe, fetchMarketData, toggleWatchlistSymbol } = useMarketData();
  const { predictions } = usePrediction();
  
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [searchTerm, setSearchTerm] = useState('');
  const [availableSymbols] = useState(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'WMT', 'IAG.MC', 'PHM.MC', 'BKY.MC', 'AENA.MC', 'BA', 'NLGO', 'CAR', 'DLTR', 'CANTE.IS', 'SASA.IS']);
  
  // Cargar datos de mercado para el símbolo seleccionado
  useEffect(() => {
    if (selectedSymbol) {
      fetchMarketData(selectedSymbol, selectedTimeframe);
    }
  }, [selectedSymbol, selectedTimeframe, fetchMarketData]);
  
  // Generar datos para gráfico de comparación de sectores (simulado)
  const generateSectorComparisonData = () => {
    const sectors = [
      { name: 'Tecnología', change: 1.2 + (Math.random() * 0.5 - 0.25) },
      { name: 'Financiero', change: 0.5 + (Math.random() * 0.5 - 0.25) },
      { name: 'Salud', change: 0.8 + (Math.random() * 0.5 - 0.25) },
      { name: 'Energía', change: -0.7 + (Math.random() * 0.5 - 0.25) },
      { name: 'Consumo', change: 0.3 + (Math.random() * 0.5 - 0.25) },
      { name: 'Industrial', change: -0.2 + (Math.random() * 0.5 - 0.25) },
      { name: 'Materiales', change: -0.5 + (Math.random() * 0.5 - 0.25) },
    ];
    
    return sectors;
  };
  
  // Generar datos para índices de mercado (simulado)
  const generateMarketIndicesData = () => {
    return [
      { name: 'S&P 500', value: 4892.32, change: 0.68 + (Math.random() * 0.2 - 0.1) },
      { name: 'NASDAQ', value: 15982.45, change: 1.12 + (Math.random() * 0.2 - 0.1) },
      { name: 'Dow Jones', value: 38648.92, change: -0.21 + (Math.random() * 0.2 - 0.1) },
      { name: 'Russell 2000', value: 2046.18, change: 0.32 + (Math.random() * 0.2 - 0.1) },
    ];
  };
  
  // Datos para gráficos
  const sectorComparisonData = generateSectorComparisonData();
  const marketIndicesData = generateMarketIndicesData();
  const marketHistoricalData = marketData[selectedSymbol]?.historicalData || [];
  
  // Función para obtener color según el cambio
  const getChangeColor = (change) => {
    return change >= 0 ? '#10b981' : '#ef4444';
  };
  
  // Filtrar símbolos según búsqueda
  const filteredSymbols = availableSymbols.filter(symbol => 
    symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-xl font-semibold">Análisis de Mercado</h1>
          
          <div className="flex space-x-4">
            <div>
              <label className="text-sm text-gray-500 mr-2">Símbolo:</label>
              <select
                className="border rounded-md px-2 py-1"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
              >
                {watchlist.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="text-sm text-gray-500 mr-2">Timeframe:</label>
              <select
                className="border rounded-md px-2 py-1"
                value={selectedTimeframe}
                onChange={(e) => changeTimeframe(e.target.value)}
              >
                <option value="15m">15 minutos</option>
                <option value="30m">30 minutos</option>
                <option value="1h">1 hora</option>
                <option value="4h">4 horas</option>
                <option value="1d">1 día</option>
              </select>
            </div>
          </div>
        </div>
        
        {/* Información del símbolo actual */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          <div>
            <div className="flex items-end">
              <h2 className="text-2xl font-bold">{selectedSymbol}</h2>
              <span className="ml-2 text-gray-500 pb-1">{marketData[selectedSymbol]?.quote?.name || selectedSymbol}</span>
            </div>
            
            <div className="flex items-center mt-2">
              <div className="text-3xl font-bold">
                ${marketData[selectedSymbol]?.quote?.price.toFixed(2) || '--.--'}
              </div>
              
              {marketData[selectedSymbol]?.quote?.change && (
                <div className="ml-3">
                  <div className={`${marketData[selectedSymbol]?.quote?.change >= 0 ? 'text-green-600' : 'text-red-600'} font-semibold`}>
                    {marketData[selectedSymbol]?.quote?.change >= 0 ? '+' : ''}
                    {marketData[selectedSymbol]?.quote?.change.toFixed(2)} 
                    ({marketData[selectedSymbol]?.quote?.percentChange.toFixed(2)}%)
                  </div>
                </div>
              )}
            </div>
            
            <div className="grid grid-cols-2 gap-x-8 gap-y-2 mt-4">
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Apertura:</span>
                <span className="text-sm">${marketData[selectedSymbol]?.historicalData?.[0]?.open.toFixed(2) || '--.--'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Máximo:</span>
                <span className="text-sm">${marketData[selectedSymbol]?.historicalData?.[0]?.high.toFixed(2) || '--.--'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Mínimo:</span>
                <span className="text-sm">${marketData[selectedSymbol]?.historicalData?.[0]?.low.toFixed(2) || '--.--'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Volumen:</span>
                <span className="text-sm">{marketData[selectedSymbol]?.historicalData?.[0]?.volume.toLocaleString() || '--'}</span>
              </div>
            </div>
            
            <div className="mt-4">
              <Link 
                to={`/trading?symbol=${selectedSymbol}`}
                className="inline-block px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
              >
                Operar {selectedSymbol}
              </Link>
            </div>
          </div>
          
          <div>
            <div className="p-3 bg-indigo-50 rounded-lg">
              <h3 className="font-medium mb-2">Predicción para {selectedSymbol}</h3>
              
              {predictions[selectedSymbol]?.predictions ? (
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(predictions[selectedSymbol].predictions)
                   .filter(([key]) => ['1h', '3h', '1d'].includes(key))
                   .map(([horizon, price]) => {
                    const currentPrice = predictions[selectedSymbol].currentPrice;
                    const changePct = ((price - currentPrice) / currentPrice) * 100;
                    const isPositive = changePct >= 0;
                    
                    return (
                      <div key={horizon} className="text-center p-2 bg-white rounded">
                        <div className="text-xs text-gray-500">{horizon}</div>
                        <div className="font-medium">${price.toFixed(2)}</div>
                        <div className={`text-xs ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                          {isPositive ? '+' : ''}{changePct.toFixed(2)}%
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-3 text-gray-500">
                  No hay predicciones disponibles
                </div>
              )}
              
              <div className="mt-2 text-xs text-right text-gray-500">
                <Link to={`/predictions?symbol=${selectedSymbol}`} className="text-indigo-600 hover:underline">
                  Ver análisis completo →
                </Link>
              </div>
            </div>
          </div>
        </div>
        
        {/* Gráfico histórico */}
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={marketHistoricalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 10 }}
                tickFormatter={(date) => new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} />
              <Tooltip 
                formatter={(value) => ['$' + value.toFixed(2), 'Precio']}
                labelFormatter={(label) => new Date(label).toLocaleString()}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="close" 
                name="Precio" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Índices de mercado */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Índices de Mercado</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {marketIndicesData.map(index => (
            <div key={index.name} className="p-4 border rounded-lg">
              <div className="flex justify-between items-center">
                <div className="font-medium">{index.name}</div>
                <div 
                  className={`px-2 py-1 text-xs rounded-full 
                              ${index.change >= 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}
                >
                  {index.change >= 0 ? '+' : ''}{index.change.toFixed(2)}%
                </div>
              </div>
              <div className="text-2xl font-semibold mt-1">{index.value.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Rendimiento por sector */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Rendimiento por Sector</h2>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={sectorComparisonData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" />
              <Tooltip formatter={(value) => [`${value.toFixed(2)}%`, 'Cambio']} />
              <Bar dataKey="change" name="Cambio %">
                {sectorComparisonData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getChangeColor(entry.change)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Búsqueda y gestión de watchlist */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-4">Buscar Símbolos</h2>
          
          <div className="flex mb-4">
            <input 
              type="text"
              className="flex-grow border rounded-l-md px-4 py-2"
              placeholder="Buscar símbolo..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <button className="bg-indigo-600 text-white px-4 py-2 rounded-r-md">
              Buscar
            </button>
          </div>
          
          <div className="border rounded-md overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Símbolo
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Acciones
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredSymbols.length > 0 ? (
                  filteredSymbols.map(symbol => (
                    <tr key={symbol}>
                      <td className="px-6 py-4 whitespace-nowrap font-medium">
                        {symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button 
                          className="text-indigo-600 hover:text-indigo-900 mr-3"
                          onClick={() => setSelectedSymbol(symbol)}
                        >
                          Ver
                        </button>
                        <button 
                          className={`${watchlist.includes(symbol) ? 'text-red-600 hover:text-red-900' : 'text-green-600 hover:text-green-900'}`}
                          onClick={() => toggleWatchlistSymbol(symbol)}
                        >
                          {watchlist.includes(symbol) ? 'Quitar' : 'Seguir'}
                        </button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="2" className="px-6 py-4 text-center text-sm text-gray-500">
                      No se encontraron símbolos
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-4">Lista de Seguimiento</h2>
          
          <div className="border rounded-md overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Símbolo
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Precio
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Cambio
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Acciones
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {watchlist.length > 0 ? (
                  watchlist.map(symbol => {
                    const quote = marketData[symbol]?.quote || {};
                    
                    return (
                      <tr key={symbol}>
                        <td className="px-6 py-4 whitespace-nowrap font-medium">
                          {symbol}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          ${quote.price?.toFixed(2) || '--.--'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {quote.change && (
                            <span className={quote.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                              {quote.change >= 0 ? '+' : ''}{quote.change.toFixed(2)}%
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <button 
                            className="text-indigo-600 hover:text-indigo-900 mr-3"
                            onClick={() => setSelectedSymbol(symbol)}
                          >
                            Ver
                          </button>
                          <button 
                            className="text-red-600 hover:text-red-900"
                            onClick={() => toggleWatchlistSymbol(symbol)}
                          >
                            Quitar
                          </button>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan="4" className="px-6 py-4 text-center text-sm text-gray-500">
                      No hay símbolos en la lista de seguimiento
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      {/* Información adicional */}
      <div className="bg-white rounded-lg shadow p-4 text-sm text-gray-600">
        <h2 className="text-lg font-semibold mb-2">Información sobre Datos de Mercado</h2>
        
        <p className="mb-2">
          Todos los datos de mercado mostrados en esta plataforma son proporcionados por el servicio 
          Financial Modeling Prep API. Los precios pueden tener un retraso de hasta 15 minutos 
          respecto al mercado en tiempo real.
        </p>
        
        <p className="mb-2">
          Las predicciones generadas por nuestros modelos de aprendizaje automático están basadas 
          en análisis técnico y patrones históricos. Deben considerarse como herramientas de apoyo 
          a la decisión y no como recomendaciones definitivas.
        </p>
        
        <p className="font-medium">
          Recuerde: Las inversiones en mercados financieros conllevan riesgos. Nunca invierta más 
          dinero del que puede permitirse perder.
        </p>
      </div>
    </div>
  );
};

export default MarketAnalysis;
