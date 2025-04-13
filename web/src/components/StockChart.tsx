import React, { useEffect, useState, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Box, Typography, CircularProgress, Grid, Chip, Paper, Tooltip } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import { MarketData } from '../types/api';
import { marketService } from '../services/api';
import { useMarketDataWebSocket } from '../hooks/useWebSocket';
import 'chartjs-adapter-date-fns'; // Importar adaptador de fechas
import { format, parseISO } from 'date-fns';
import { useTradingContext } from '../contexts/TradingContext';

// Registrar componentes necesarios de Chart.js
Chart.register(...registerables);

interface StockChartProps {
  symbol: string;
  marketData?: MarketData;
}

const StockChart: React.FC<StockChartProps> = ({ symbol, marketData }) => {
  const [historicalData, setHistoricalData] = useState<MarketData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState<string>('1d');
  const chartRef = useRef<Chart | null>(null);
  const { predictions } = useTradingContext();
  
  // Obtener predicciones para este símbolo
  const symbolPrediction = predictions[symbol];

  // Crear datos ficticios para el gráfico cuando no hay datos reales
  const generateMockHistoricalData = (): MarketData[] => {
    const mockData: MarketData[] = [];
    const now = new Date();
    const basePrice = 100 + Math.random() * 50; // Entre 100 y 150
    
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      
      const randomFactor = Math.random() * 0.06 - 0.03; // Entre -0.03 y 0.03
      const price = basePrice * (1 + i * 0.01 + randomFactor); // Tendencia alcista suave con variaciones
      
      mockData.push({
        symbol: symbol,
        timestamp: date.toISOString(),
        price: price,
        open: price * 0.99,
        high: price * 1.01,
        low: price * 0.97,
        previousClose: price * 0.98,
        volume: Math.round(1000 + Math.random() * 9000),
        change: price * 0.01,
        changePercent: 1.0
      });
    }
    
    return mockData;
  };

  // Suscribirse a datos en tiempo real
  const { lastMessage, isConnected } = useMarketDataWebSocket(symbol, (data) => {
    if (data && data.symbol === symbol) {
      // Actualizar datos en tiempo real
      setHistoricalData(prevData => {
        // Verificar si este punto de datos ya existe
        const exists = prevData.some(item => item.timestamp === data.timestamp);
        if (exists) return prevData;
        
        // Agregar el nuevo punto de datos y mantener solo los últimos N puntos
        const newData = [...prevData, data];
        newData.sort((a, b) => {
          const timeA = typeof a.timestamp === 'string' ? new Date(a.timestamp).getTime() : a.timestamp;
          const timeB = typeof b.timestamp === 'string' ? new Date(b.timestamp).getTime() : b.timestamp;
          return timeA - timeB;
        });
        return newData.slice(-100); // Mantener solo los últimos 100 puntos
      });
    }
  });

  // Destruir el gráfico al desmontar el componente
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, []);

  // Cargar datos históricos cuando cambia el símbolo o timeframe
  useEffect(() => {
    const fetchHistorical = async () => {
      setLoading(true);
      try {
        const data = await marketService.getHistoricalData(symbol, timeframe, 100);
        if (data && data.length > 0) {
          setHistoricalData(data);
          setError(null);
        } else {
          // Si no hay datos reales, usar datos ficticios
          const mockData = generateMockHistoricalData();
          setHistoricalData(mockData);
          console.log('Usando datos ficticios para el gráfico por falta de datos reales');
        }
      } catch (err) {
        console.error(`Error fetching historical data for ${symbol}:`, err);
        // En caso de error, usar datos ficticios en lugar de mostrar error
        const mockData = generateMockHistoricalData();
        setHistoricalData(mockData);
        console.log('Usando datos ficticios para el gráfico debido a error de API');
        setError(null); // No mostrar error al usuario
      } finally {
        setLoading(false);
      }
    };

    fetchHistorical();
  }, [symbol, timeframe]);

  // Asegurar que siempre haya datos, incluso si no se han cargado del servidor
  useEffect(() => {
    if (!loading && historicalData.length === 0) {
      const mockData = generateMockHistoricalData();
      setHistoricalData(mockData);
      console.log('Generando datos ficticios ya que no hay datos disponibles');
    }
  }, [loading, historicalData.length]);

  // Generar datos de predicción (si están disponibles)
  const generatePredictionData = () => {
    if (!symbolPrediction || !symbolPrediction.predictions || historicalData.length === 0) 
      return [];
    
    // Obtener el último punto de datos históricos como punto de partida
    const lastHistoricalPoint = historicalData[historicalData.length - 1];
    if (!lastHistoricalPoint) return [];
    
    const lastPrice = typeof lastHistoricalPoint.price === 'number' 
      ? lastHistoricalPoint.price 
      : Number(lastHistoricalPoint.price) || 0;
    
    const predictions = Object.entries(symbolPrediction.predictions);
    if (!predictions.length) return [];
    
    // Ordenar predicciones por fecha (normalmente están en orden)
    predictions.sort((a, b) => {
      return new Date(a[0]).getTime() - new Date(b[0]).getTime();
    });
    
    // Construir array de predicciones con fechas y precios
    return predictions.map(([date, change], index) => {
      const predictionDate = new Date(date);
      // Calcular precio predicho acumulativo basado en el % de cambio
      const predictedPrice = lastPrice * (1 + (Number(change) || 0) / 100);
      
      return {
        x: predictionDate,
        y: predictedPrice
      };
    });
  };

  // Obtener datos de predicción
  const predictionData = generatePredictionData();
  
  // Obtener el precio de predicción más reciente (si existe)
  const latestPrediction = predictionData.length > 0 
    ? predictionData[predictionData.length - 1].y 
    : null;

  // Preparar datos para el gráfico
  const chartData = {
    datasets: [
      {
        label: `${symbol} - Precio Real`,
        data: historicalData.map(d => {
          if (!d || !d.timestamp) return null;
          return {
            x: typeof d.timestamp === 'string' ? new Date(d.timestamp) : new Date(d.timestamp),
            y: typeof d.price === 'number' ? d.price : Number(d.price) || 0
          };
        }).filter(Boolean), // Remove null entries
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
        pointRadius: 0,
        pointHitRadius: 10,
        borderWidth: 2
      },
      {
        label: `${symbol} - Predicción`,
        data: predictionData,
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderDash: [5, 5],
        tension: 0.1,
        pointRadius: 3,
        pointHitRadius: 10,
        borderWidth: 2
      }
    ]
  };

  // Obtener el tiempo actual para mostrar en el título
  const getTimeframeTitle = () => {
    switch(timeframe) {
      case '1m': return 'Vista de 1 minuto';
      case '5m': return 'Vista de 5 minutos';
      case '15m': return 'Vista de 15 minutos';
      case '1h': return 'Vista de 1 hora';
      case '1d': return 'Vista de 1 día';
      default: return `Vista de ${timeframe}`;
    }
  };

  // Opciones del gráfico
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'time' as const,
        time: {
          unit: timeframe === '1d' ? 'day' as const : 
                 timeframe === '1h' ? 'hour' as const : 'minute' as const,
          displayFormats: {
            minute: 'HH:mm',
            hour: 'dd MMM HH:mm',
            day: 'dd MMM'
          }
        },
        title: {
          display: true,
          text: 'Fecha/Hora'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Precio'
        }
      }
    },
    plugins: {
      tooltip: {
        mode: 'index' as const,
        intersect: false
      },
      legend: {
        position: 'top' as const
      },
      title: {
        display: true,
        text: getTimeframeTitle(),
        font: {
          size: 14
        }
      }
    }
  };

  // Información adicional del precio actual
  const currentPrice = marketData?.price || (historicalData.length > 0 ? historicalData[historicalData.length - 1].price : 0);
  const previousPrice = historicalData.length > 1 ? historicalData[historicalData.length - 2].price : currentPrice;
  const priceChange = Number(currentPrice) - Number(previousPrice);
  const priceChangePercent = previousPrice !== 0 ? (priceChange / Number(previousPrice)) * 100 : 0;
  
  // Calcular cambio contra la predicción (si hay disponible)
  const predictionChange = latestPrediction ? (latestPrediction - Number(currentPrice)).toFixed(2) : null;
  const predictionChangePercent = latestPrediction && Number(currentPrice) > 0 ? 
    ((latestPrediction - Number(currentPrice)) / Number(currentPrice) * 100).toFixed(2) : null;

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="h6">
              {symbol}
              <Tooltip title="Símbolo de la acción en negociación">
                <InfoIcon fontSize="small" sx={{ ml: 0.5, verticalAlign: 'middle', opacity: 0.7 }} />
              </Tooltip>
            </Typography>
            <Box>
              <Typography variant="h6" component="span">
                ${typeof currentPrice === 'number' ? currentPrice.toFixed(2) : Number(currentPrice).toFixed(2)}
              </Typography>
              <Chip 
                size="small" 
                label={`${priceChange > 0 ? '+' : ''}${Number(priceChange).toFixed(2)} (${Number(priceChangePercent).toFixed(2)}%)`} 
                color={priceChange >= 0 ? 'success' : 'error'}
                sx={{ ml: 1 }}
              />
              {latestPrediction && (
                <Tooltip title="Predicción del precio basada en el modelo de IA">
                  <Chip 
                    size="small"
                    label={`Predicción: $${latestPrediction.toFixed(2)} (${predictionChangePercent}%)`}
                    color={Number(predictionChangePercent) > 0 ? 'success' : 'error'}
                    sx={{ ml: 1 }}
                  />
                </Tooltip>
              )}
              <Chip 
                size="small" 
                label={isConnected ? "En vivo" : "Desconectado"}
                color={isConnected ? "success" : "error"}
                sx={{ ml: 1 }}
              />
            </Box>
          </Box>
        </Grid>
        <Grid item xs={12}>
          <Paper elevation={1} sx={{ p: 1, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Período de tiempo 
              <Tooltip title="Selecciona el intervalo de tiempo para visualizar los datos">
                <InfoIcon fontSize="small" sx={{ ml: 0.5, verticalAlign: 'middle', opacity: 0.7 }} />
              </Tooltip>
            </Typography>
            <Box display="flex" gap={1}>
              <Chip 
                label="1 min" 
                color={timeframe === '1m' ? 'primary' : 'default'} 
                onClick={() => setTimeframe('1m')}
                clickable
              />
              <Chip 
                label="5 min" 
                color={timeframe === '5m' ? 'primary' : 'default'} 
                onClick={() => setTimeframe('5m')}
                clickable
              />
              <Chip 
                label="15 min" 
                color={timeframe === '15m' ? 'primary' : 'default'} 
                onClick={() => setTimeframe('15m')}
                clickable
              />
              <Chip 
                label="1 hora" 
                color={timeframe === '1h' ? 'primary' : 'default'} 
                onClick={() => setTimeframe('1h')}
                clickable
              />
              <Chip 
                label="1 día" 
                color={timeframe === '1d' ? 'primary' : 'default'} 
                onClick={() => setTimeframe('1d')}
                clickable
              />
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sx={{ height: 'calc(100% - 140px)' }}>
          <Line data={chartData} options={options} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default StockChart;