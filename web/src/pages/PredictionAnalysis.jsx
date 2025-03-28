import React, { useState, useEffect } from 'react';
import { usePrediction } from '../contexts/PredictionContext';
import { useMarketData } from '../contexts/MarketDataContext';
import { useModelStatus } from '../contexts/ModelStatusContext';
import {
  LineChart, Line,
  XAxis, YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Cell,
  ReferenceLine,
  BarChart,
  Bar
} from 'recharts';

const PredictionAnalysis = () => {
  const { predictions, verificationResults, activeSymbols, fetchPredictions } = usePrediction();
  const { marketData } = useMarketData();
  const { modelStatus } = useModelStatus();
  
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [selectedHorizon, setSelectedHorizon] = useState('1d');
  
  // Cargar datos para el símbolo seleccionado
  useEffect(() => {
    if (selectedSymbol) {
      fetchPredictions(selectedSymbol);
    }
  }, [selectedSymbol, fetchPredictions]);
  
  // Generar datos para gráfico de precisión por símbolo
  const generateSymbolAccuracyData = () => {
    const data = [];
    
    activeSymbols.forEach(symbol => {
      if (verificationResults[symbol]) {
        // Calcular promedio de error por símbolo a través de todos los horizontes
        let totalError = 0;
        let count = 0;
        
        Object.values(verificationResults[symbol]).forEach(horizonData => {
          if (horizonData.avgError) {
            totalError += horizonData.avgError;
            count++;
          }
        });
        
        const avgError = count > 0 ? totalError / count : null;
        
        if (avgError !== null) {
          data.push({
            symbol,
            error: avgError,
            accuracy: 100 - avgError // Simplificación de precisión
          });
        }
      }
    });
    
    return data;
  };
  
  // Generar datos para predicciones vs. resultados reales
  const generatePredictionVsActualData = () => {
    const data = [];
    
    // Usar datos de verificación si están disponibles
    if (verificationResults[selectedSymbol]?.[selectedHorizon]?.records) {
      const records = verificationResults[selectedSymbol][selectedHorizon].records;
      
      records.forEach(record => {
        data.push({
          timestamp: new Date(record.timestamp).toLocaleString(),
          predicted: record.predictedPrice,
          actual: record.actualPrice,
          error: record.errorPct
        });
      });
    } else {
      // Generar datos simulados si no hay datos reales
      const now = new Date();
      
      for (let i = 10; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000); // Cada hora
        const basePrice = 150 + Math.random() * 10;
        const predicted = basePrice * (1 + (Math.random() * 0.04 - 0.02));
        const actual = basePrice * (1 + (Math.random() * 0.04 - 0.02));
        const error = Math.abs((predicted - actual) / actual * 100);
        
        data.push({
          timestamp: timestamp.toLocaleString(),
          predicted,
          actual,
          error
        });
      }
    }
    
    return data;
  };
  
  // Datos para gráficos
  const symbolAccuracyData = generateSymbolAccuracyData();
  const predictionVsActualData = generatePredictionVsActualData();
  
  // Función para determinar color según precisión
  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 95) return '#10b981'; // verde
    if (accuracy >= 90) return '#3b82f6'; // azul
    if (accuracy >= 85) return '#f59e0b'; // amarillo
    return '#ef4444'; // rojo
  };
  
  // Función para determinar color según error
  const getErrorColor = (error) => {
    if (error < 1) return '#10b981'; // verde
    if (error < 3) return '#3b82f6'; // azul
    if (error < 5) return '#f59e0b'; // amarillo
    return '#ef4444'; // rojo
  };
  
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-xl font-semibold">Análisis de Predicciones</h1>
          
          <div className="flex space-x-4">
            <div>
              <label className="text-sm text-gray-500 mr-2">Símbolo:</label>
              <select
                className="border rounded-md px-2 py-1"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
              >
                {activeSymbols.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="text-sm text-gray-500 mr-2">Horizonte:</label>
              <select
                className="border rounded-md px-2 py-1"
                value={selectedHorizon}
                onChange={(e) => setSelectedHorizon(e.target.value)}
              >
                <option value="15m">15 minutos</option>
                <option value="30m">30 minutos</option>
                <option value="1h">1 hora</option>
                <option value="3h">3 horas</option>
                <option value="1d">1 día</option>
              </select>
            </div>
          </div>
        </div>
        
        {/* Resumen de predicciones para el símbolo seleccionado */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
          {predictions[selectedSymbol]?.predictions && 
           Object.entries(predictions[selectedSymbol].predictions).map(([horizon, price]) => {
            const currentPrice = predictions[selectedSymbol].currentPrice;
            const changePct = ((price - currentPrice) / currentPrice) * 100;
            const isPositive = changePct >= 0;
            
            return (
              <div key={horizon} className="p-4 border rounded-lg text-center">
                <div className="text-xs text-gray-500">Precio a {horizon}</div>
                <div className="text-xl font-semibold">${price.toFixed(2)}</div>
                <div className={`text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                  {isPositive ? '+' : ''}{changePct.toFixed(2)}%
                </div>
              </div>
            );
          })}
        </div>
        
        <div className="text-sm text-right text-gray-500">
          Última actualización: {predictions[selectedSymbol]?.timestamp ? 
            new Date(predictions[selectedSymbol].timestamp).toLocaleString() : 'No disponible'}
        </div>
      </div>
      
      {/* Gráfico de predicciones vs. resultados reales */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Predicciones vs. Resultados Reales ({selectedHorizon})</h2>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={predictionVsActualData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="predicted"
                name="Precio Predicho"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 5 }}
                activeDot={{ r: 8 }}
              />
              <Line
                type="monotone"
                dataKey="actual"
                name="Precio Real"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ r: 5 }}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Gráfico de error por predicción */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Error de Predicción por Verificación ({selectedHorizon})</h2>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={predictionVsActualData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 'auto']} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="error" name="Error %" fill="#3b82f6">
                {predictionVsActualData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getErrorColor(entry.error)} />
                ))}
              </Bar>
              <ReferenceLine y={2} stroke="#10b981" strokeDasharray="3 3" />
              <ReferenceLine y={5} stroke="#f59e0b" strokeDasharray="3 3" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="mt-2 text-sm text-gray-500 text-center">
          <span className="inline-block w-3 h-3 bg-red-500 rounded-full mr-1"></span> Error alto (>5%)
          <span className="inline-block ml-3 w-3 h-3 bg-yellow-500 rounded-full mr-1"></span> Error medio (3-5%)
          <span className="inline-block ml-3 w-3 h-3 bg-blue-500 rounded-full mr-1"></span> Error bajo (1-3%)
          <span className="inline-block ml-3 w-3 h-3 bg-green-500 rounded-full mr-1"></span> Error mínimo (<1%)
        </div>
      </div>
      
      {/* Comparación de precisión por símbolo */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Precisión por Símbolo</h2>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={symbolAccuracyData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[80, 100]} />
              <YAxis dataKey="symbol" type="category" />
              <Tooltip />
              <Legend />
              <Bar dataKey="accuracy" name="Precisión %" background={{ fill: '#eee' }}>
                {symbolAccuracyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getAccuracyColor(entry.accuracy)} />
                ))}
              </Bar>
              <ReferenceLine x={90} stroke="#3b82f6" strokeDasharray="3 3" />
              <ReferenceLine x={95} stroke="#10b981" strokeDasharray="3 3" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Tabla de verificación de predicciones */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Registro de Verificaciones ({selectedSymbol})</h2>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Horizonte
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Predicción
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Real
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Error %
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Estado
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Fecha
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {predictionVsActualData.map((record, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {selectedHorizon}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    ${record.predicted.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    ${record.actual.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {record.error.toFixed(2)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span 
                      className={`px-2 py-1 text-xs rounded-full
                                ${record.error < 1 ? 'bg-green-100 text-green-800' : 
                                  record.error < 3 ? 'bg-blue-100 text-blue-800' :
                                  record.error < 5 ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-red-100 text-red-800'}`}
                    >
                      {record.error < 1 ? 'Excelente' :
                       record.error < 3 ? 'Bueno' :
                       record.error < 5 ? 'Regular' :
                       'Deficiente'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {record.timestamp}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Información sobre métricas */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-2">Acerca de las Métricas de Predicción</h2>
        
        <div className="text-sm text-gray-600 space-y-2">
          <p>
            Las predicciones se evalúan automáticamente comparando el precio previsto con el precio real
            al alcanzar el horizonte temporal correspondiente. El proceso de verificación se realiza en ciclos de 1 hora.
          </p>
          
          <p>
            <strong>Error Porcentual:</strong> Diferencia entre el precio predicho y el real, expresada como porcentaje
            del precio real. Un menor error indica mayor precisión.
          </p>
          
          <p>
            <strong>Clasificación de precisión:</strong>
          </p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li><span className="text-green-600 font-medium">Excelente:</span> Error menor al 1% - Alta precisión para trading.</li>
            <li><span className="text-blue-600 font-medium">Bueno:</span> Error entre 1% y 3% - Adecuado para la mayoría de operaciones.</li>
            <li><span className="text-yellow-600 font-medium">Regular:</span> Error entre 3% y 5% - Considerar con precaución.</li>
            <li><span className="text-red-600 font-medium">Deficiente:</span> Error mayor al 5% - Evitar basar decisiones en estas predicciones.</li>
          </ul>
          
          <p className="mt-2 text-xs text-gray-500">
            Nota: Los modelos de aprendizaje automático están en continua mejora y adaptación.
            La precisión varía según el símbolo, condiciones de mercado y horizonte temporal.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionAnalysis;
