import React, { useEffect, useState } from 'react';
import { Chart, registerables } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Box, Typography, CircularProgress, Grid, Chip } from '@mui/material';
import { MarketData } from '../types/api';
import { marketService } from '../services/api';
import { useMarketDataWebSocket } from '../hooks/useWebSocket';

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
        newData.sort((a, b) => a.timestamp - b.timestamp);
        return newData.slice(-100); // Mantener solo los últimos 100 puntos
      });
    }
  });

  // Cargar datos históricos cuando cambia el símbolo o timeframe
  useEffect(() => {
    const fetchHistorical = async () => {
      setLoading(true);
      try {
        const data = await marketService.getHistoricalData(symbol, timeframe, 100);
        setHistoricalData(data);
        setError(null);
      } catch (err) {
        console.error(`Error fetching historical data for ${symbol}:`, err);
        setError(`No se pudieron obtener los datos históricos para ${symbol}`);
      } finally {
        setLoading(false);
      }
    };

    fetchHistorical();
  }, [symbol, timeframe]);

  // Preparar datos para el gráfico
  const chartData = {
    labels: historicalData.map(d => new Date(d.timestamp)), // Cambiado de d.datetime a d.timestamp
    datasets: [
      {
        label: `${symbol} - Precio`,
        data: historicalData.map(d => d.price), // Cambiado de d.close a d.price
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
        pointRadius: 0,
        pointHitRadius: 10,
        borderWidth: 2
      }
    ]
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
      }
    }
  };

  // Información adicional del precio actual
  const currentPrice = marketData?.price || (historicalData.length > 0 ? historicalData[historicalData.length - 1].price : 0); // Cambiado de .close a .price
  const previousPrice = historicalData.length > 1 ? historicalData[historicalData.length - 2].price : currentPrice; // Cambiado de .close a .price
  const priceChange = currentPrice - previousPrice;
  const priceChangePercent = (priceChange / previousPrice) * 100;

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
            <Typography variant="h6">{symbol}</Typography>
            <Box>
              <Typography variant="h6" component="span">
                ${currentPrice.toFixed(2)}
              </Typography>
              <Chip 
                size="small" 
                label={`${priceChange > 0 ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePercent.toFixed(2)}%)`} 
                color={priceChange >= 0 ? 'success' : 'error'}
                sx={{ ml: 1 }}
              />
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
          <Box display="flex" gap={1} mb={2}>
            <Chip 
              label="1m" 
              color={timeframe === '1m' ? 'primary' : 'default'} 
              onClick={() => setTimeframe('1m')}
              clickable
            />
            <Chip 
              label="5m" 
              color={timeframe === '5m' ? 'primary' : 'default'} 
              onClick={() => setTimeframe('5m')}
              clickable
            />
            <Chip 
              label="15m" 
              color={timeframe === '15m' ? 'primary' : 'default'} 
              onClick={() => setTimeframe('15m')}
              clickable
            />
            <Chip 
              label="1h" 
              color={timeframe === '1h' ? 'primary' : 'default'} 
              onClick={() => setTimeframe('1h')}
              clickable
            />
            <Chip 
              label="1d" 
              color={timeframe === '1d' ? 'primary' : 'default'} 
              onClick={() => setTimeframe('1d')}
              clickable
            />
          </Box>
        </Grid>
        <Grid item xs={12} sx={{ height: 'calc(100% - 80px)' }}>
          <Line data={chartData} options={options} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default StockChart;