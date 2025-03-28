import React, { useState, useEffect } from 'react';
import { useModelStatus } from '../contexts/ModelStatusContext';
import { usePrediction } from '../contexts/PredictionContext';
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

const ModelPerformance = () => {
  const { modelStatus, fetchModelStatus } = useModelStatus();
  const { verificationResults, activeSymbols } = usePrediction();
  const [selectedModel, setSelectedModel] = useState('ensemble'); // ensemble, online, batch
  const [selectedMetric, setSelectedMetric] = useState('accuracy'); // accuracy, mape, rmse
  const [selectedTimeRange, setSelectedTimeRange] = useState('1d'); // 1d, 1w, 1m
  
  // Actualizar datos de modelos
  useEffect(() => {
    fetchModelStatus();
    
    // Actualizar cada 5 minutos
    const intervalId = setInterval(fetchModelStatus, 5 * 60 * 1000);
    return () => clearInterval(intervalId);
  }, [fetchModelStatus]);
  
  // Determinar el color según el estado
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return '#10b981'; // green
      case 'degraded': return '#f59e0b'; // yellow
      case 'critical': return '#ef4444'; // red
      default: return '#6b7280'; // gray
    }
  };
  
  // Determinar el color según el error
  const getErrorColor = (error) => {
    if (error < 1) return '#10b981'; // green
    if (error < 3) return '#3b82f6'; // blue
    if (error < 5) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };
  
  // Formatear estado
  const formatStatus = (status) => {
    switch (status) {
      case 'healthy': return 'Normal';
      case 'degraded': return 'Degradado';
      case 'critical': return 'Crítico';
      default: return 'Desconocido';
    }
  };
  
  // Generar datos para el gráfico de métricas históricas (simulados)
  const generateHistoricalMetrics = () => {
    const now = new Date();
    const data = [];
    
    // Determinar número de puntos según el rango de tiempo
    let points;
    let interval;
    
    switch (selectedTimeRange) {
      case '1d':
        points = 24;
        interval = 60 * 60 * 1000; // 1 hora
        break;
      case '1w':
        points = 7;
        interval = 24 * 60 * 60 * 1000; // 1 día
        break;
      case '1m':
        points = 30;
        interval = 24 * 60 * 60 * 1000; // 1 día
        break;
      default:
        points = 24;
        interval = 60 * 60 * 1000;
    }
    
    // Generar datos simulados
    for (let i = points - 1; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * interval);
      
      // Valores base para los modelos
      const baseValues = {
        ensemble: { accuracy: 90, mape: 1.5, rmse: 1.0 },
        online: { accuracy: 85, mape: 2.3, rmse: 1.5 },
        batch: { accuracy: 88, mape: 1.8, rmse: 1.2 }
      };
      
      // Agregar variación aleatoria
      const dataPoint = {
        timestamp: timestamp.toISOString(),
        date: timestamp.toLocaleDateString(),
        time: timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      
      // Añadir métricas para cada modelo
      Object.entries(baseValues).forEach(([model, metrics]) => {
        Object.entries(metrics).forEach(([metric, value]) => {
          // Añadir variación aleatoria
          const variation = (Math.random() - 0.5) * 2;
          
          // Si es accuracy, la variación debe ser menor para no salir de rangos realistas
          const variationFactor = metric === 'accuracy' ? 2 : (metric === 'mape' ? 0.5 : 0.3);
          
          dataPoint[`${model}_${metric}`] = value + variation * variationFactor;
        });
      });
      
      data.push(dataPoint);
    }
    
    return data;
  };
  
  // Generar datos para la comparación de horizonte temporal
  const generateHorizonComparison = () => {
    const horizons = ['15m', '30m', '1h', '3h', '1d'];
    
    // Usar datos de verificación reales si están disponibles
    if (verificationResults && Object.keys(verificationResults).length > 0) {
      return horizons.map(horizon => {
        // Calcular promedio de error a través de todos los símbolos activos
        let totalError = 0;
        let count = 0;
        
        activeSymbols.forEach(symbol => {
          if (verificationResults[symbol]?.[horizon]?.avgError) {
            totalError += verificationResults[symbol][horizon].avgError;
            count++;
          }
        });
        
        const avgError = count > 0 ? totalError / count : null;
        
        return {
          horizon,
          error: avgError !== null ? avgError : Math.random() * 4 + 1, // Fallback a datos simulados
          count: count
        };
      });
    }
    
    // Datos simulados si no hay datos reales
    return horizons.map(horizon => {
      // Error aumenta con el horizonte temporal
      const baseError = horizon === '15m' ? 1.2 :
                        horizon === '30m' ? 1.8 :
                        horizon === '1h' ? 2.5 :
                        horizon === '3h' ? 3.5 :
                        4.5; // 1d
      
      return {
        horizon,
        error: baseError + (Math.random() - 0.5), // Añadir variación
        count: Math.floor(Math.random() * 20) + 5 // Número simulado de verificaciones
      };
    });
  };
  
  // Generar datos
  const historicalData = generateHistoricalMetrics();
  const horizonData = generateHorizonComparison();
  
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-4">
        <h1 className="text-xl font-semibold mb-4">Rendimiento de los Modelos de Predicción</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
          {/* Métrica para modelo online */}
          <div className="p-4 border rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-medium">Modelo Online</h3>
              <div 
                className={`px-2 py-1 text-xs rounded-full 
                            ${modelStatus.online.status === 'healthy' ? 'bg-green-100 text-green-800' : 
                              modelStatus.online.status === 'degraded' ? 'bg-yellow-100 text-yellow-800' : 
                              modelStatus.online.status === 'critical' ? 'bg-red-100 text-red-800' : 
                              'bg-gray-100 text-gray-800'}`}
              >
                {formatStatus(modelStatus.online.status)}
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="text-sm text-gray-500">Precisión</div>
                <div className="text-lg font-semibold">{modelStatus.online.accuracy?.toFixed(1)}%</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">MAPE</div>
                <div className="text-lg font-semibold">{modelStatus.online.metrics?.MAPE?.toFixed(2)}</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">RMSE</div>
                <div className="text-lg font-semibold">{modelStatus.online.metrics?.RMSE?.toFixed(2)}</div>
              </div>
            </div>
          </div>
          
          {/* Métrica para modelo batch */}
          <div className="p-4 border rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-medium">Modelo Batch</h3>
              <div 
                className={`px-2 py-1 text-xs rounded-full 
                            ${modelStatus.batch.status === 'healthy' ? 'bg-green-100 text-green-800' : 
                              modelStatus.batch.status === 'degraded' ? 'bg-yellow-100 text-yellow-800' : 
                              modelStatus.batch.status === 'critical' ? 'bg-red-100 text-red-800' : 
                              'bg-gray-100 text-gray-800'}`}
              >
                {formatStatus(modelStatus.batch.status)}
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="text-sm text-gray-500">Precisión</div>
                <div className="text-lg font-semibold">{modelStatus.batch.accuracy?.toFixed(1)}%</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">MAPE</div>
                <div className="text-lg font-semibold">{modelStatus.batch.metrics?.MAPE?.toFixed(2)}</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">RMSE</div>
                <div className="text-lg font-semibold">{modelStatus.batch.metrics?.RMSE?.toFixed(2)}</div>
              </div>
            </div>
          </div>
          
          {/* Métrica para modelo ensemble */}
          <div className="p-4 border rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-medium">Modelo Ensemble</h3>
              <div 
                className={`px-2 py-1 text-xs rounded-full 
                            ${modelStatus.ensemble.status === 'healthy' ? 'bg-green-100 text-green-800' : 
                              modelStatus.ensemble.status === 'degraded' ? 'bg-yellow-100 text-yellow-800' : 
                              modelStatus.ensemble.status === 'critical' ? 'bg-red-100 text-red-800' : 
                              'bg-gray-100 text-gray-800'}`}
              >
                {formatStatus(modelStatus.ensemble.status)}
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="text-sm text-gray-500">Precisión</div>
                <div className="text-lg font-semibold">{modelStatus.ensemble.accuracy?.toFixed(1)}%</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">MAPE</div>
                <div className="text-lg font-semibold">{modelStatus.ensemble.metrics?.MAPE?.toFixed(2)}</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">RMSE</div>
                <div className="text-lg font-semibold">{modelStatus.ensemble.metrics?.RMSE?.toFixed(2)}</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="text-sm text-right text-gray-500">
          Última actualización: {modelStatus.lastUpdated ? new Date(modelStatus.lastUpdated).toLocaleString() : 'No disponible'}
        </div>
      </div>
      
      {/* Gráfico de rendimiento histórico */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Rendimiento Histórico</h2>
          
          <div className="flex space-x-4">
            <div>
              <label className="text-sm text-gray-500 mr-2">Modelo:</label>
              <select 
                className="border rounded px-2 py-1"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="ensemble">Ensemble</option>
                <option value="online">Online</option>
                <option value="batch">Batch</option>
              </select>
            </div>
            
            <div>
              <label className="text-sm text-gray-500 mr-2">Métrica:</label>
              <select 
                className="border rounded px-2 py-1"
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
              >
                <option value="accuracy">Precisión (%)</option>
                <option value="mape">MAPE</option>
                <option value="rmse">RMSE</option>
              </select>
            </div>
            
            <div>
              <label className="text-sm text-gray-500 mr-2">Periodo:</label>
              <select 
                className="border rounded px-2 py-1"
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
              >
                <option value="1d">Último día</option>
                <option value="1w">Última semana</option>
                <option value="1m">Último mes</option>
              </select>
            </div>
          </div>
        </div>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey={selectedTimeRange === '1d' ? 'time' : 'date'} 
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                domain={
                  selectedMetric === 'accuracy' ? [70, 100] :
                  selectedMetric === 'mape' ? [0, 5] :
                  [0, 3]
                }
                tick={{ fontSize: 12 }}
              />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey={`${selectedModel}_${selectedMetric}`} 
                name={selectedMetric === 'accuracy' ? 'Precisión' :
                       selectedMetric === 'mape' ? 'Error Porcentual Medio' :
                       'Error Cuadrático Medio'}
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Comparación por horizonte temporal */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Error por Horizonte Temporal</h2>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={horizonData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="horizon" type="category" />
              <Tooltip formatter={(value) => [value.toFixed(2) + '%', 'Error']} />
              <Legend />
              <Bar dataKey="error" name="Error %" barSize={30}>
                {horizonData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getErrorColor(entry.error)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="mt-2 text-xs text-gray-500 text-right">
          El error tiende a aumentar con la extensión del horizonte temporal de predicción
        </div>
      </div>
      
      {/* Información adicional */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Información de Modelos</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="p-4 border rounded-lg">
            <h3 className="font-medium mb-2">Modelo Online</h3>
            <p className="text-sm text-gray-600 mb-2">
              Modelo de aprendizaje en tiempo real que se adapta continuamente a los cambios del mercado.
            </p>
            <ul className="text-xs space-y-1 text-gray-500">
              <li>• Actualización: Tiempo real</li>
              <li>• Algoritmo: Regresión online adaptativa</li>
              <li>• Características: Indicadores técnicos, volumen</li>
              <li>• Fortaleza: Adaptación rápida</li>
            </ul>
          </div>
          
          <div className="p-4 border rounded-lg">
            <h3 className="font-medium mb-2">Modelo Batch</h3>
            <p className="text-sm text-gray-600 mb-2">
              Modelo entrenado periódicamente con grandes volúmenes de datos históricos.
            </p>
            <ul className="text-xs space-y-1 text-gray-500">
              <li>• Actualización: Diaria</li>
              <li>• Algoritmo: Gradient Boosting, redes neuronales</li>
              <li>• Características: Patrones históricos, fundamentales</li>
              <li>• Fortaleza: Precisión a medio-largo plazo</li>
            </ul>
          </div>
          
          <div className="p-4 border rounded-lg">
            <h3 className="font-medium mb-2">Modelo Ensemble</h3>
            <p className="text-sm text-gray-600 mb-2">
              Combinación ponderada de los modelos online y batch para maximizar la precisión.
            </p>
            <ul className="text-xs space-y-1 text-gray-500">
              <li>• Actualización: Tiempo real + diaria</li>
              <li>• Algoritmo: Meta-modelo adaptativo</li>
              <li>• Características: Combinadas de ambos modelos</li>
              <li>• Fortaleza: Equilibrio precisión/adaptabilidad</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-4 p-3 bg-blue-50 text-blue-800 text-sm rounded-md">
          <p className="font-medium">Nota sobre las métricas:</p>
          <ul className="list-disc list-inside mt-1 space-y-1 text-sm">
            <li><strong>Precisión:</strong> Porcentaje de predicciones dentro del margen de error aceptable.</li>
            <li><strong>MAPE:</strong> Error Porcentual Absoluto Medio - medida del error de predicción en porcentaje.</li>
            <li><strong>RMSE:</strong> Raíz del Error Cuadrático Medio - medida del error de predicción en unidades absolutas.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ModelPerformance;
