import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, 
  XAxis, YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart
} from 'recharts';

const TradingChart = ({ symbol, timeframe, data, predictions }) => {
  const [showPredictions, setShowPredictions] = useState(true);
  const [chartType, setChartType] = useState('line'); // 'line' o 'candle'
  const [combinedData, setCombinedData] = useState([]);
  
  // Procesar datos para el gráfico
  useEffect(() => {
    if (!data || data.length === 0) return;
    
    // Formatear datos históricos
    const chartData = data.map(point => ({
      time: new Date(point.date).getTime(),
      price: point.close,
      open: point.open,
      high: point.high,
      low: point.low,
      close: point.close,
      formattedTime: new Date(point.date).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}),
      volume: point.volume
    }));
    
    // Si hay predicciones y están habilitadas, añadirlas al gráfico
    if (showPredictions && predictions && predictions.predictions) {
      const lastTimestamp = chartData[chartData.length - 1].time;
      const lastPrice = chartData[chartData.length - 1].price;
      
      // Crear puntos para las predicciones
      const predictionPoints = [];
      
      // Mapear horizontes a milisegundos
      const horizonMap = {
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '3h': 3 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
      };
      
      // Añadir punto actual (para conectar línea de predicción)
      predictionPoints.push({
        time: lastTimestamp,
        price: lastPrice,
        formattedTime: new Date(lastTimestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}),
        isPrediction: false,
        isCurrentPrice: true
      });
      
      // Añadir las predicciones
      Object.entries(predictions.predictions).forEach(([horizon, price]) => {
        if (horizonMap[horizon]) {
          const predictionTime = lastTimestamp + horizonMap[horizon];
          
          predictionPoints.push({
            time: predictionTime,
            predictedPrice: price,
            horizon,
            formattedTime: new Date(predictionTime).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}),
            isPrediction: true
          });
        }
      });
      
      // Combinar datos históricos con predicciones
      setCombinedData([...chartData, ...predictionPoints]);
    } else {
      setCombinedData(chartData);
    }
  }, [data, predictions, showPredictions]);
  
  // Si no hay datos, mostrar mensaje de carga
  if (!data || data.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500"></div>
        <span className="ml-2 text-indigo-700">Cargando datos...</span>
      </div>
    );
  }
  
  // Componente personalizado para tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <div className="bg-white p-3 border shadow-md rounded-md">
          <p className="font-medium text-sm">{data.formattedTime}</p>
          
          {data.isPrediction ? (
            <div className="space-y-1 mt-1">
              <p className="text-sm text-indigo-600">
                Predicción ({data.horizon}): ${data.predictedPrice?.toFixed(2)}
              </p>
            </div>
          ) : (
            <div className="space-y-1 mt-1">
              <p className="text-sm">Precio: ${data.price?.toFixed(2)}</p>
              {chartType === 'candle' && (
                <>
                  <p className="text-sm">Apertura: ${data.open?.toFixed(2)}</p>
                  <p className="text-sm">Máximo: ${data.high?.toFixed(2)}</p>
                  <p className="text-sm">Mínimo: ${data.low?.toFixed(2)}</p>
                </>
              )}
              {data.volume && (
                <p className="text-sm text-gray-600">Vol: {data.volume.toLocaleString()}</p>
              )}
            </div>
          )}
        </div>
      );
    }
    
    return null;
  };
  
  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center space-x-4">
          <h2 className="text-lg font-semibold">{symbol} - {timeframe}</h2>
          
          {/* Selector de tipo de gráfico */}
          <div className="flex bg-gray-100 rounded-md">
            <button
              className={`px-3 py-1 text-sm rounded-md 
                        ${chartType === 'line' ? 'bg-indigo-600 text-white' : ''}`}
              onClick={() => setChartType('line')}
            >
              Línea
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md 
                        ${chartType === 'candle' ? 'bg-indigo-600 text-white' : ''}`}
              onClick={() => setChartType('candle')}
            >
              Velas
            </button>
          </div>
        </div>
        
        {/* Control para mostrar/ocultar predicciones */}
        <div className="flex items-center">
          <label className="flex items-center text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={showPredictions}
              onChange={() => setShowPredictions(!showPredictions)}
              className="mr-2"
            />
            Mostrar Predicciones
          </label>
        </div>
      </div>
      
      {/* Gráfico */}
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'line' ? (
            <ComposedChart data={combinedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                domain={['auto', 'auto']}
                tickFormatter={(time) => new Date(time).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}
                type="number"
              />
              <YAxis 
                domain={['auto', 'auto']}
                tickFormatter={(value) => `$${value.toFixed(2)}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              {/* Volumen como barras */}
              <Area
                dataKey="volume"
                type="monotone"
                name="Volumen"
                fill="rgba(200, 200, 200, 0.2)"
                stroke="none"
                yAxisId={1}
                hide={false}
              />
              
              {/* Precio histórico */}
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#2563eb" 
                dot={false}
                name="Precio"
                activeDot={{ r: 8 }}
                strokeWidth={2}
              />
              
              {/* Predicciones */}
              {showPredictions && (
                <Line 
                  type="monotone" 
                  dataKey="predictedPrice" 
                  stroke="#10b981" 
                  strokeDasharray="5 5" 
                  name="Predicción"
                  strokeWidth={2}
                  dot={(props) => {
                    const { cx, cy, payload } = props;
                    return payload.isPrediction 
                      ? (
                        <g>
                          <circle cx={cx} cy={cy} r={5} fill="#10b981" stroke="white" strokeWidth={1} />
                          <text x={cx} y={cy-10} textAnchor="middle" fill="#10b981" fontSize={10}>{payload.horizon}</text>
                        </g>
                      ) 
                      : null;
                  }}
                />
              )}
            </ComposedChart>
          ) : (
            <ComposedChart data={combinedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                domain={['auto', 'auto']}
                tickFormatter={(time) => new Date(time).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}
                type="number"
              />
              <YAxis 
                domain={['auto', 'auto']}
                tickFormatter={(value) => `$${value.toFixed(2)}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              {/* Volumen como barras */}
              <Area
                dataKey="volume"
                type="monotone"
                name="Volumen"
                fill="rgba(200, 200, 200, 0.2)"
                stroke="none"
                yAxisId={1}
                hide={false}
              />
              
              {/* Velas representadas como líneas personalizadas */}
              {combinedData.map((candle, index) => {
                if (candle.open && candle.close) {
                  const isUp = candle.close >= candle.open;
                  const color = isUp ? '#10b981' : '#ef4444';
                  
                  return (
                    <g key={`candle-${index}`}>
                      {/* Línea de sombra */}
                      <line 
                        x1={index} 
                        x2={index} 
                        y1={candle.high} 
                        y2={candle.low} 
                        stroke={color}
                        strokeWidth={1}
                      />
                      {/* Cuerpo de la vela */}
                      <rect
                        x={index - 0.3}
                        y={isUp ? candle.open : candle.close}
                        width={0.6}
                        height={Math.abs(candle.close - candle.open)}
                        fill={color}
                        stroke={color}
                      />
                    </g>
                  );
                }
                return null;
              })}
              
              {/* Predicciones */}
              {showPredictions && (
                <Line 
                  type="monotone" 
                  dataKey="predictedPrice" 
                  stroke="#10b981" 
                  strokeDasharray="5 5" 
                  name="Predicción"
                  strokeWidth={2}
                  dot={(props) => {
                    const { cx, cy, payload } = props;
                    return payload.isPrediction 
                      ? (
                        <g>
                          <circle cx={cx} cy={cy} r={5} fill="#10b981" stroke="white" strokeWidth={1} />
                          <text x={cx} y={cy-10} textAnchor="middle" fill="#10b981" fontSize={10}>{payload.horizon}</text>
                        </g>
                      ) 
                      : null;
                  }}
                />
              )}
            </ComposedChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TradingChart;
